"""RS01 dog environment with a deterministic diagonal CPG base policy."""

import torch

from legged_gym.envs.fanfan.fanfan_env import FanfanRobot
from .rs01_cpg import RS01DiagonalCPG, RS01FootTrajectory


class DogRs01Robot(FanfanRobot):
    """Quadruped environment driven by the measured RS01 actuator chain."""

    def __init__(self, *args, **kwargs):
        # FanfanRobot construction invokes reset hooks, so the attribute must
        # exist before its constructor begins.
        self.rs01_cpg = None
        self.rs01_foot_trajectory = None
        self.cpg_leg_z_feedback = None
        super().__init__(*args, **kwargs)
        if getattr(self.cfg.control, "use_rs01_diagonal_cpg", False):
            self.rs01_cpg = RS01DiagonalCPG(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt,
                radial_rate=float(getattr(
                    self.cfg.control, "cpg_radial_rate", 30.0
                )),
                coupling_gain=float(getattr(
                    self.cfg.control, "cpg_coupling_gain", 18.0
                )),
            )
            self.rs01_foot_trajectory = RS01FootTrajectory()
            self.cpg_leg_z_feedback = torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                dtype=torch.float,
                device=self.device,
            )
            self.rs01_cpg.synchronize(self.gait_phase)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.rs01_cpg is not None and len(env_ids) > 0:
            self.rs01_cpg.reset(env_ids, self.gait_phase[env_ids])
            self.cpg_leg_z_feedback[env_ids] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cpg_leg_z_feedback is not None:
            self._update_cpg_diagonal_force_balance()

    def _update_cpg_diagonal_force_balance(self):
        """Equalize real vertical load inside each physical diagonal.

        Equal joint targets do not yield equal toe forces with the measured
        per-motor gain/tau/friction and the real link masses.  A millimetre-
        scale, low-pass foot-height correction prevents the lightly loaded
        member from leaving early without turning swing release into a slow
        binary contact gate.
        """
        vertical_force = self.contact_forces[
            :, self.feet_indices, 2
        ].clip(min=0.0)
        nominal_weight = max(float(getattr(
            self.cfg.rewards, "transition_nominal_weight_n", 115.1
        )), 1.0)
        normalized_gain = float(getattr(
            self.cfg.control,
            "cpg_force_balance_gain_m_per_weight",
            0.08,
        ))
        max_correction = float(getattr(
            self.cfg.control, "cpg_force_balance_max_m", 0.010
        ))
        target = self.cpg_leg_z_feedback.clone()
        for first, second in (("FL", "RR"), ("FR", "RL")):
            first_slot = self.foot_slot_by_leg[first]
            second_slot = self.foot_slot_by_leg[second]
            first_force = vertical_force[:, first_slot]
            second_force = vertical_force[:, second_slot]
            pair_loaded = (first_force + second_force) > 10.0
            correction = (
                normalized_gain
                * (second_force - first_force)
                / nominal_weight
            ).clip(min=-max_correction, max=max_correction)
            # Negative z extends a leg. If the first foot is lightly loaded,
            # extend it and retract the overloaded diagonal partner equally.
            target[:, first_slot] = torch.where(
                pair_loaded, -correction, target[:, first_slot]
            )
            target[:, second_slot] = torch.where(
                pair_loaded, correction, target[:, second_slot]
            )
        time_constant = max(float(getattr(
            self.cfg.control, "cpg_force_balance_time_constant_s", 0.12
        )), self.dt)
        blend = self.dt / (time_constant + self.dt)
        self.cpg_leg_z_feedback += blend * (
            target - self.cpg_leg_z_feedback
        )
        self.cpg_leg_z_feedback.clamp_(
            min=-max_correction, max=max_correction
        )

    def _contact_aware_gait_phase(self, proposed_phase):
        """Advance the coupled oscillator, then apply the load handoff hold."""
        if self.rs01_cpg is None:
            return super()._contact_aware_gait_phase(proposed_phase)
        phase_increment = torch.remainder(
            proposed_phase - self.gait_phase, 1.0
        )
        cpg_phase = self.rs01_cpg.step(phase_increment)
        accepted_phase = super()._contact_aware_gait_phase(cpg_phase)
        self.rs01_cpg.synchronize(accepted_phase)
        return accepted_phase

    def _compute_specialized_gait_offset(
        self, phase, stance_ratio, gait_amplitude_fraction
    ):
        """Generate the nominal foot-space CPG and convert it through URDF IK."""
        if self.rs01_foot_trajectory is None:
            return None

        speed = self._command_equivalent_speed()
        period_blend = ((speed - 0.01) / 0.29).clip(0.0, 1.0)
        period = self.gait_period_low_speed + period_blend * (
            self.gait_period_high_speed - self.gait_period_low_speed
        )
        stance = stance_ratio[:, 0]

        ramp_duration = max(float(getattr(
            self.cfg.control, "gait_transition_ramp_s", 0.20
        )), 1.0e-4)
        transition = (self.command_transition_age / ramp_duration).clip(
            0.0, 1.0
        )
        transition = transition * transition * (3.0 - 2.0 * transition)

        stride_gain = float(getattr(
            self.cfg.control, "cpg_stride_gain", 1.0
        ))
        max_stride = float(getattr(
            self.cfg.control, "cpg_max_stride_m", 0.085
        ))
        signed_stride = (
            self.commands[:, 0] * period * stance * stride_gain * transition
        ).clip(min=-max_stride, max=max_stride)

        clearance = float(getattr(
            self.cfg.control, "cpg_swing_clearance_m", 0.035
        ))
        clearance_speed = max(float(getattr(
            self.cfg.control, "cpg_full_clearance_speed_m_s", 0.12
        )), 1.0e-4)
        clearance_gate = (torch.abs(self.commands[:, 0]) / clearance_speed).clip(
            0.0, 1.0
        )
        clearance_target = clearance * clearance_gate * transition

        foot_x, foot_z = self.rs01_foot_trajectory.sample(
            phase=phase,
            stance_ratio=stance_ratio,
            signed_stride_m=signed_stride,
            clearance_m=clearance_target,
            # Apply the footprint trim below through the same smooth command
            # transition as stride/clearance.  Passing it here directly would
            # move the supplied zero-command standing pose.
            nominal_x_m=0.0,
            nominal_z_m=float(getattr(
                self.cfg.control, "cpg_nominal_foot_z_m", -0.300
            )),
            lift_fraction=float(getattr(
                self.cfg.control, "cpg_lift_fraction", 0.18
            )),
            lower_start_fraction=float(getattr(
                self.cfg.control, "cpg_lower_start_fraction", 0.62
            )),
        )
        foot_x += (
            float(getattr(
                self.cfg.control, "cpg_nominal_foot_x_m", 0.0
            ))
            * transition.unsqueeze(1)
        )
        # Small Cartesian contact-following correction for the heavy trunk.
        # With the selected negative gain, a rising body extends the stance
        # legs just enough to keep the support toes on the ground instead of
        # entering a ballistic interval. Swing clearance and phase are
        # untouched, and the correction remains inside a 6 mm bound.
        vertical_damping = float(getattr(
            self.cfg.control, "cpg_vertical_velocity_damping_s", 0.0
        ))
        if abs(vertical_damping) > 1.0e-8:
            max_vertical_correction = float(getattr(
                self.cfg.control,
                "cpg_vertical_velocity_damping_max_m",
                0.006,
            ))
            vertical_correction = (
                vertical_damping * self.base_lin_vel[:, 2] * transition
            ).clip(
                min=-max_vertical_correction,
                max=max_vertical_correction,
            )
            commanded_stance = phase < stance_ratio
            foot_z += (
                commanded_stance.float()
                * vertical_correction.unsqueeze(1)
            )
        nominal_foot_z = float(getattr(
            self.cfg.control, "cpg_nominal_foot_z_m", -0.300
        ))
        clearance_scales = getattr(
            self.cfg.control, "cpg_swing_clearance_scale_by_leg", None
        )
        if clearance_scales is not None:
            lift = foot_z - nominal_foot_z
            for leg in ("FL", "FR", "RL", "RR"):
                foot_slot = self.foot_slot_by_leg[leg]
                foot_z[:, foot_slot] = (
                    nominal_foot_z
                    + float(clearance_scales[leg]) * lift[:, foot_slot]
                )
        foot_z = foot_z + self.cpg_leg_z_feedback
        # The complete URDF has a rear-biased supported load even though the
        # nominal foot geometry is symmetric. During locomotion, extend both
        # front legs and retract both rear legs by the same bounded amount.
        # The transition gate keeps the supplied zero-command stand unchanged.
        fore_aft_bias = float(getattr(
            self.cfg.control, "cpg_front_rear_load_bias_m", 0.0
        )) * transition
        for leg in ("FL", "FR"):
            foot_z[:, self.foot_slot_by_leg[leg]] -= fore_aft_bias
        for leg in ("RL", "RR"):
            foot_z[:, self.foot_slot_by_leg[leg]] += fore_aft_bias
        # Static URDF inertia and the measured per-motor chain can bias total
        # load toward one physical diagonal even with a level trunk. A small
        # common-mode pair preload balances the two CPG oscillators without
        # changing their phase or the within-pair trajectory.
        diagonal_bias = float(getattr(
            self.cfg.control, "cpg_diagonal_load_bias_m", 0.0
        )) * transition
        for leg in ("FL", "RR"):
            foot_z[:, self.foot_slot_by_leg[leg]] -= diagonal_bias
        for leg in ("FR", "RL"):
            foot_z[:, self.foot_slot_by_leg[leg]] += diagonal_bias
        thigh_target, calf_target = (
            self.rs01_foot_trajectory.inverse_kinematics(foot_x, foot_z)
        )

        gait_offset = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=self.dof_pos.dtype,
            device=self.device,
        )
        for leg in ("FL", "FR", "RL", "RR"):
            foot_slot = self.foot_slot_by_leg[leg]
            thigh = self.leg_dof_indices[leg]["thigh"]
            calf = self.leg_dof_indices[leg]["calf"]
            gait_offset[:, thigh] = (
                thigh_target[:, foot_slot] - self.default_dof_pos[:, thigh]
            )
            gait_offset[:, calf] = (
                calf_target[:, foot_slot] - self.default_dof_pos[:, calf]
            )
        return gait_offset
