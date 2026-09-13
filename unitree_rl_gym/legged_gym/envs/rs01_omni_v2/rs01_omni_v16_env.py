"""Keep the identified plant; model the deployed policy-boundary guard."""
import math
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from .rs01_omni_v15_env import Rs01OmniV15StandRobot


def guarded_target(target, q, dq, kp, kd, limit, lower, upper):
    raw = kp * (target - q) - kd * dq
    safe = torch.maximum(torch.minimum(raw, limit), -limit)
    projected = torch.maximum(torch.minimum(q + (safe + kd * dq) / kp, upper), lower)
    return projected, raw, safe


def update_guard(rms_sq, torque, dt, continuous=6.0, peak=14.0, full=8.0, tau=2.0):
    alpha = math.exp(-dt / tau)
    rms_sq = alpha * rms_sq + (1.0 - alpha) * torque.square()
    fraction = ((rms_sq.sqrt() - continuous) / (full - continuous)).clamp(0.0, 1.0)
    return rms_sq, peak - fraction * (peak - continuous)


class Rs01OmniV16Robot(Rs01OmniV15StandRobot):
    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.guard_joint_lower = torch.tensor(props['lower'].copy(), device=self.device)
            self.guard_joint_upper = torch.tensor(props['upper'].copy(), device=self.device)
        return super()._process_dof_props(props, env_id)

    def step(self, actions):
        self._v16_project_pending = True
        return super().step(actions)

    def _compute_torques(self, actions):
        if self._rs01_actuator_ready and getattr(self, '_v16_project_pending', False):
            cfg = self.cfg.rs01_actuator
            if not hasattr(self, 'guard_rms_sq'):
                self.guard_rms_sq = torch.zeros_like(self.dof_pos)
                self.guard_active_limit = torch.full_like(self.dof_pos, cfg.peak_torque_limit_nm)
            self.guard_target, self.guard_raw_pd, self.guard_safe_pd = guarded_target(
                self.rs01_limited_position_target_rad, self.dof_pos, self.dof_vel,
                self.p_gains, self.d_gains, self.guard_active_limit,
                self.guard_joint_lower, self.guard_joint_upper)
            # Same order as the deployed core: project with previous limit,
            # then update for NEXT policy tick using safe demand + feedback.
            thermal_input = torch.maximum(self.guard_safe_pd.abs(),
                                          self.motor_electromagnetic_torques.abs())
            self.guard_rms_sq, self.guard_active_limit = update_guard(
                self.guard_rms_sq, thermal_input, self.dt,
                cfg.continuous_torque_nm, cfg.peak_torque_limit_nm,
                cfg.guard_derate_full_rms_nm, cfg.guard_time_constant_s)
            self._v16_project_pending = False
        return super()._compute_torques(actions)

    def _advance_rs01_response(self):
        # Protect the SENT target, not the actor's limiter history. The
        # identified delay/filter and the 2.5 ms PD feedback remain intact.
        original = self.rs01_limited_position_target_rad
        self.rs01_limited_position_target_rad = getattr(self, 'guard_target', original)
        try:
            super()._advance_rs01_response()
        finally:
            self.rs01_limited_position_target_rad = original

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        if hasattr(self, 'guard_target'):
            self.guard_target[env_ids] = self.default_dof_pos
        # Deliberately retain thermal history across falls/timeouts. A reset
        # cannot give a struggling policy a free cold motor.

    def _handoff_mask(self):
        return self._desired_contact_mask().sum(dim=1) == 4

    def _reward_odd_feet_contact(self):
        count = self.get_foot_contact_mask().sum(dim=1)
        # Three feet during load handoff is preferable to rushing a lift.
        illegal = (count == 1) | ((count == 3) & ~self._handoff_mask())
        return illegal.float() * self._walking_command_gate()

    def _support_structure_error(self):
        phase = self._gait_phase()
        duty = self.cfg.rewards.gait_stance_ratio
        overlap = duty - 0.5
        local = torch.remainder(phase, 0.5)
        blend = (local / overlap).clamp(0.0, 1.0)
        blend = blend.square() * (3.0 - 2.0 * blend)
        a_share = torch.where(phase < 0.5, blend, 1.0 - blend)
        force = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
        total = force.sum(dim=1).clamp(min=1.0e-6)
        actual_a = (force * self.diagonal_a_contact_mask).sum(dim=1) / total
        # Only diagonal GROUP load transfer; never equalize the two motors
        # inside a group. This leaves lateral/turn/COM compensation free.
        load_error = 2.0 * (actual_a - a_share).square()
        actual = self.get_foot_contact_mask()
        desired = self._desired_contact_mask()
        mismatch = (actual != desired).float().mean(dim=1)
        return load_error + mismatch

    def _footprint_quality(self):
        relative = self.feet_pos - self.root_states[:, None, :3]
        quat = self.base_quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
        body = quat_rotate_inverse(quat, relative.reshape(-1, 3)).reshape(-1, 4, 3)
        # Resolve by name, not rigid-body iteration order.
        ids = [self.foot_slot_by_leg[leg] for leg in ('FL', 'FR', 'RL', 'RR')]
        widths = torch.stack((body[:, ids[0], 1] - body[:, ids[1], 1],
                              body[:, ids[2], 1] - body[:, ids[3], 1]), dim=1)
        raw = self.commands[:, :3]  # V13 stores raw intent here, before correction.
        straight = (1.0 - raw[:, 1].abs() / 0.08 - raw[:, 2].abs() / 0.20).clamp(0.0, 1.0)
        straight *= torch.where(raw[:, 0] < -0.02, 0.35, 1.0)
        cfg = self.cfg.rewards
        minimum = cfg.footprint_omni_min_width_m + straight * (
            cfg.footprint_straight_min_width_m - cfg.footprint_omni_min_width_m)
        shortfall = ((minimum[:, None] - widths) / cfg.footprint_soft_scale_m).clamp(min=0.0)
        self.v16_foot_widths = widths.detach()
        return torch.exp(-shortfall.square().mean(dim=1))

    def _reward_phase_two_contact_quality(self):
        reward = torch.exp(-self._support_structure_error() / self.cfg.rewards.phase_support_sigma)
        reward *= self._footprint_quality() * self._walking_command_gate()
        self.v11_contact_quality_reward = reward.detach()
        return reward
