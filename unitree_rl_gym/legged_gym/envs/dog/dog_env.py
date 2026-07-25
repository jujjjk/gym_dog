"""RS01 dog environment: measured actuator chain; optional CPG offset."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.fanfan.fanfan_env import FanfanRobot
from .rs01_cpg import RS01DiagonalCPG, RS01FootTrajectory


class DogRs01Robot(FanfanRobot):
    """12 joint actions → targets; CPG offset only if explicitly enabled."""

    def __init__(self, *args, **kwargs):
        # FanfanRobot construction invokes reset hooks, so the attribute must
        # exist before its constructor begins.
        self.rs01_cpg = None
        self.rs01_foot_trajectory = None
        self.cpg_leg_z_feedback = None
        self.episode_start_xy = None
        super().__init__(*args, **kwargs)
        # World-frame episode origin for straight-path displacement rewards.
        # It is training state only and never modifies the 12 motor outputs.
        self.episode_start_xy = self.root_states[:, :2].clone()
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
        if self.episode_start_xy is not None and len(env_ids) > 0:
            self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        if self.rs01_cpg is not None and len(env_ids) > 0:
            self.rs01_cpg.reset(env_ids, self.gait_phase[env_ids])
            self.cpg_leg_z_feedback[env_ids] = 0.0

    def check_termination(self):
        # Illegal diagonal patterns are already covered by the overridden
        # `_get_non_diagonal_swing_mask()` and the parent debounce counter.
        # Do not add single-frame rear/overlap resets here: that bypasses the
        # configured termination_steps and truncates long straight cycles.
        super().check_termination()

    def _get_foot_air_mask(self):
        threshold = float(getattr(
            self.cfg.rewards, "foot_contact_force_threshold", 1.0
        ))
        return self.contact_forces[:, self.feet_indices, 2] <= threshold

    def _get_rear_pair_air_mask(self):
        air = self._get_foot_air_mask()
        return (
            air[:, self.foot_slot_by_leg["RL"]]
            & air[:, self.foot_slot_by_leg["RR"]]
        )

    def _get_front_pair_air_mask(self):
        air = self._get_foot_air_mask()
        return (
            air[:, self.foot_slot_by_leg["FL"]]
            & air[:, self.foot_slot_by_leg["FR"]]
        )

    def _get_overlapping_diagonal_air_mask(self):
        """True when both diagonals have at least one airborne foot.

        Legal walk states are only:
          - four-foot support, or
          - exactly one diagonal swinging (FL+RR or FR+RL), with the other
            diagonal fully planted.
        Starting the next diagonal before the previous has landed is illegal.
        """
        air = self._get_foot_air_mask()
        fl_rr_busy = (
            air[:, self.foot_slot_by_leg["FL"]]
            | air[:, self.foot_slot_by_leg["RR"]]
        )
        fr_rl_busy = (
            air[:, self.foot_slot_by_leg["FR"]]
            | air[:, self.foot_slot_by_leg["RL"]]
        )
        return fl_rr_busy & fr_rl_busy

    def _get_non_diagonal_swing_mask(self):
        """Stricter than Fanfan: also reject rear/front bounds and overlap."""
        base = super()._get_non_diagonal_swing_mask()
        return (
            base
            | self._get_rear_pair_air_mask()
            | self._get_front_pair_air_mask()
            | self._get_overlapping_diagonal_air_mask()
        )

    def _reward_rear_pair_air(self):
        return (
            self._get_rear_pair_air_mask().float()
            * (1.0 - self._stand_command_gate())
        )

    def _reward_overlapping_diagonal_air(self):
        return (
            self._get_overlapping_diagonal_air_mask().float()
            * (1.0 - self._stand_command_gate())
        )

    def _scheduled_diagonal_swing_state(self):
        """Return scheduled swing pairs and opposite-pair support scores."""
        desired_air = ~self._get_desired_foot_contacts()
        fl = self.foot_slot_by_leg["FL"]
        fr = self.foot_slot_by_leg["FR"]
        rl = self.foot_slot_by_leg["RL"]
        rr = self.foot_slot_by_leg["RR"]
        swing_fl_rr = (
            desired_air[:, fl] & desired_air[:, rr]
            & ~desired_air[:, fr] & ~desired_air[:, rl]
        )
        swing_fr_rl = (
            desired_air[:, fr] & desired_air[:, rl]
            & ~desired_air[:, fl] & ~desired_air[:, rr]
        )
        support_fl_rr, support_fr_rl = self._get_diagonal_support_scores()
        opposite_support = torch.where(
            swing_fl_rr,
            support_fr_rl,
            torch.where(
                swing_fr_rl,
                support_fl_rr,
                torch.zeros_like(support_fl_rr),
            ),
        )
        return swing_fl_rr, swing_fr_rl, opposite_support

    def _foot_velocity_relative_to_trunk(self):
        """Foot linear velocity in the trunk frame."""
        relative_world = (
            self.feet_state[:, :, 7:10]
            - self.root_states[:, None, 7:10]
        )
        foot_count = len(self.feet_indices)
        base_quat = self.base_quat[:, None, :].expand(
            -1, foot_count, -1
        ).reshape(-1, 4)
        return quat_rotate_inverse(
            base_quat, relative_world.reshape(-1, 3)
        ).reshape(self.num_envs, foot_count, 3)

    def _reward_forward_diagonal_pair_swing(self):
        """Dense intent reward for advancing both scheduled diagonal toes.

        The minimum over the two toes prevents a one-foot step from earning
        the pair reward.  The opposite diagonal must carry load, so advancing
        feet by hopping or unloading all four feet earns no credit.
        """
        swing_fl_rr, swing_fr_rl, opposite_support = (
            self._scheduled_diagonal_swing_state()
        )
        fl = self.foot_slot_by_leg["FL"]
        fr = self.foot_slot_by_leg["FR"]
        rl = self.foot_slot_by_leg["RL"]
        rr = self.foot_slot_by_leg["RR"]
        relative_velocity = self._foot_velocity_relative_to_trunk()

        command_x = self.commands[:, 0].clip(min=0.0)
        target_swing_speed = (1.8 * command_x).clip(min=0.12)
        fl_rr_forward = (
            torch.minimum(
                relative_velocity[:, fl, 0],
                relative_velocity[:, rr, 0],
            ) / target_swing_speed
        ).clip(0.0, 1.0)
        fr_rl_forward = (
            torch.minimum(
                relative_velocity[:, fr, 0],
                relative_velocity[:, rl, 0],
            ) / target_swing_speed
        ).clip(0.0, 1.0)
        pair_forward = torch.where(
            swing_fl_rr,
            fl_rr_forward,
            torch.where(
                swing_fr_rl,
                fr_rl_forward,
                torch.zeros_like(fl_rr_forward),
            ),
        )

        lift_start = float(getattr(
            self.cfg.rewards, "diagonal_pair_lift_start_height", 0.015
        ))
        lift_target = max(float(getattr(
            self.cfg.rewards, "diagonal_pair_lift_target_height", 0.030
        )), lift_start + 1.0e-4)
        lift = (
            (self.feet_pos[:, :, 2] - lift_start)
            / (lift_target - lift_start)
        ).clip(0.0, 1.0)
        fl_rr_lift = torch.minimum(lift[:, fl], lift[:, rr])
        fr_rl_lift = torch.minimum(lift[:, fr], lift[:, rl])
        pair_lift = torch.where(
            swing_fl_rr,
            fl_rr_lift,
            torch.where(
                swing_fr_rl,
                fr_rl_lift,
                torch.zeros_like(fl_rr_lift),
            ),
        )

        # A small dense pre-liftoff component gives the direct policy a usable
        # phase signal; full credit still requires paired clearance/support.
        readiness = 0.25 + 0.75 * pair_lift
        return (
            pair_forward
            * readiness
            * opposite_support
            * (1.0 - self._stand_command_gate())
        )

    def _reward_forward_progress_with_diagonal_swing(self):
        """Confirm that a legal scheduled swing actually propels the trunk."""
        swing_fl_rr, swing_fr_rl, opposite_support = (
            self._scheduled_diagonal_swing_state()
        )
        _, actual_fl_rr, actual_fr_rl = self._get_diagonal_swing_masks()
        exact_scheduled = (
            (swing_fl_rr & actual_fl_rr)
            | (swing_fr_rl & actual_fr_rl)
        )
        command_x = self.commands[:, 0].clip(min=0.0)
        normalized_progress = (
            self.base_lin_vel[:, 0].clip(min=0.0)
            / command_x.clip(min=0.08)
        ).clip(0.0, 1.2)
        return (
            exact_scheduled.float()
            * normalized_progress
            * opposite_support
            * (1.0 - self._stand_command_gate())
        )

    def _reward_body_angular_velocity(self):
        """Penalize visible whole-body twisting during straight locomotion.

        Normalization keeps this term well conditioned: yaw may naturally be
        somewhat faster than roll/pitch, but the current 1+ rad/s oscillation
        cannot hide behind a near-zero signed mean.
        """
        scale = self.base_ang_vel.new_tensor((0.8, 0.8, 1.5))
        normalized = self.base_ang_vel / scale
        return (
            torch.sum(torch.square(normalized), dim=1).clip(max=6.0)
            * self._straight_motion_gate()
        )

    def _reward_body_angular_acceleration(self):
        """Suppress rapid left/right reversals of trunk angular velocity."""
        current_world_ang_vel = self.root_states[:, 10:13]
        previous_world_ang_vel = self.last_root_vel[:, 3:6]
        angular_acceleration = (
            current_world_ang_vel - previous_world_ang_vel
        ) / self.dt
        normalized = angular_acceleration / 20.0
        valid = (self.episode_length_buf > 1).float()
        return (
            torch.sum(torch.square(normalized), dim=1).clip(max=6.0)
            * self._straight_motion_gate()
            * valid
        )

    def _sagittal_torque_headroom(self):
        """Return dense torque headroom for the thigh/calf motors."""
        indices = []
        for leg in ("FL", "FR", "RL", "RR"):
            indices.extend((
                self.leg_dof_indices[leg]["thigh"],
                self.leg_dof_indices[leg]["calf"],
            ))
        ratio = (
            torch.abs(self.raw_torques[:, indices])
            / self._active_episode_torque_limits()[:, indices]
        )
        usage = ((ratio - 0.55) / 0.80).clip(0.0, 1.0)
        return (1.0 - torch.mean(usage, dim=1)).clip(0.0, 1.0)

    def _body_angular_headroom(self):
        """Return one at a quiet trunk and zero at large visible twisting."""
        roll_scale = max(float(getattr(
            self.cfg.rewards, "body_headroom_roll_rate_rad_s", 1.5
        )), 1.0e-4)
        pitch_scale = max(float(getattr(
            self.cfg.rewards, "body_headroom_pitch_rate_rad_s", 1.2
        )), 1.0e-4)
        yaw_scale = max(float(getattr(
            self.cfg.rewards, "body_headroom_yaw_rate_rad_s", 2.2
        )), 1.0e-4)
        normalized = torch.stack((
            torch.abs(self.base_ang_vel[:, 0]) / roll_scale,
            torch.abs(self.base_ang_vel[:, 1]) / pitch_scale,
            torch.abs(self.base_ang_vel[:, 2]) / yaw_scale,
        ), dim=1)
        return (
            1.0 - torch.mean(normalized.clip(0.0, 1.0), dim=1)
        ).clip(0.0, 1.0)

    def _body_angular_acceleration_headroom(self):
        """Return smooth headroom for phase-to-phase trunk reversals."""
        current = self.root_states[:, 10:13]
        previous = self.last_root_vel[:, 3:6]
        acceleration = torch.abs(current - previous) / self.dt
        scales = acceleration.new_tensor((
            max(float(getattr(
                self.cfg.rewards,
                "body_headroom_roll_accel_rad_s2",
                25.0,
            )), 1.0e-4),
            max(float(getattr(
                self.cfg.rewards,
                "body_headroom_pitch_accel_rad_s2",
                25.0,
            )), 1.0e-4),
            max(float(getattr(
                self.cfg.rewards,
                "body_headroom_yaw_accel_rad_s2",
                40.0,
            )), 1.0e-4),
        ))
        usage = torch.mean((acceleration / scales).clip(0.0, 1.0), dim=1)
        valid = (self.episode_length_buf > 1).float()
        return (1.0 - usage).clip(0.0, 1.0) * valid

    def _forward_progress_gate(self):
        command_x = self.commands[:, 0].clip(min=0.0)
        return (
            self.base_lin_vel[:, 0].clip(min=0.0)
            / command_x.clip(min=0.08)
        ).clip(0.0, 1.0) * (1.0 - self._stand_command_gate())

    def _reward_smooth_low_torque_forward(self):
        """Positive credit only for efficient, quiet commanded progress.

        This supplies a useful gradient under positive-reward clipping:
        standing, twisting in place, and high-torque shuffling all score zero.
        """
        return (
            self._forward_progress_gate()
            * self._sagittal_torque_headroom()
            * self._body_angular_headroom()
        )

    def _desired_double_support_gate(self):
        desired_contact = self._get_desired_foot_contacts()
        return (
            torch.all(desired_contact, dim=1).float()
            * (1.0 - self._stand_command_gate())
        )

    def _reward_smooth_diagonal_handoff(self):
        """Reward a quiet, low-torque handoff between physical diagonals."""
        fl_rr_support, fr_rl_support = self._get_diagonal_support_scores()
        support = torch.maximum(fl_rr_support, fr_rl_support)
        return (
            self._desired_double_support_gate()
            * self._forward_progress_gate()
            * self._sagittal_torque_headroom()
            * self._body_angular_headroom()
            * support
        )

    def _reward_handoff_body_twist(self):
        """Directly penalize angular motion in the two load-transfer windows."""
        normalized = torch.stack((
            self.base_ang_vel[:, 0] / 1.5,
            self.base_ang_vel[:, 1] / 1.2,
            self.base_ang_vel[:, 2] / 2.2,
        ), dim=1)
        return (
            torch.sum(torch.square(normalized), dim=1).clip(max=4.0)
            * self._desired_double_support_gate()
        )

    def _reward_sagittal_motor_saturation(self):
        """Target the thigh/calf saturation found in phase-resolved analysis."""
        indices = []
        for leg in ("FL", "FR", "RL", "RR"):
            indices.extend((
                self.leg_dof_indices[leg]["thigh"],
                self.leg_dof_indices[leg]["calf"],
            ))
        ratio = (
            torch.abs(self.raw_torques[:, indices])
            / self._active_episode_torque_limits()[:, indices]
        )
        excess = (ratio - 0.70).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * (1.0 - self._stand_command_gate())
        )

    def _hip_joint_delta(self):
        """Measured hip excursion about the supplied zero-angle stand."""
        return (
            self.dof_pos[:, self.hip_dof_indices]
            - self.default_dof_pos[:, self.hip_dof_indices]
        )

    def _reward_hip_joint_excursion(self):
        """Penalize visible hip ab/adduction outside a small balance band.

        A dead band preserves the lateral authority required to catch the
        heavy trunk.  Only excess motion is penalized, so PPO is not rewarded
        for making the hip artificially rigid at the expense of a fall.
        """
        soft_limit = max(float(getattr(
            self.cfg.rewards, "hip_excursion_soft_limit_rad", 0.070
        )), 1.0e-4)
        width = max(float(getattr(
            self.cfg.rewards, "hip_excursion_penalty_width_rad", 0.090
        )), 1.0e-4)
        excess = (
            (torch.abs(self._hip_joint_delta()) - soft_limit) / width
        ).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * self._straight_motion_gate()
        )

    def _reward_hip_target_excursion(self):
        """Discourage using large hip targets before they become body sway."""
        target_delta = (
            self.target_dof_pos_rl[:, self.hip_dof_indices]
            - self.default_dof_pos[:, self.hip_dof_indices]
        )
        soft_limit = max(float(getattr(
            self.cfg.rewards, "hip_target_soft_limit_rad", 0.080
        )), 1.0e-4)
        width = max(float(getattr(
            self.cfg.rewards, "hip_target_penalty_width_rad", 0.080
        )), 1.0e-4)
        excess = (
            (torch.abs(target_delta) - soft_limit) / width
        ).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * self._straight_motion_gate()
        )

    def _reward_hip_policy_action_rate(self):
        """Selective smoothing for the four direct hip policy outputs."""
        delta = (
            self.policy_actions[:, self.hip_dof_indices]
            - self.last_policy_actions[:, self.hip_dof_indices]
        )
        return (
            torch.mean(torch.square(delta), dim=1)
            * self._straight_motion_gate()
        )

    def _reward_motor_torque_usage(self):
        """Dense all-motor torque cost below hard saturation.

        The existing clip rewards activate mostly at the peak limit.  This
        term begins at 35% of each episode's real RS01 limit and therefore
        teaches the policy to unload thigh/calf motors before clipping.
        """
        ratio = (
            torch.abs(self.raw_torques)
            / self._active_episode_torque_limits()
        )
        excess = (ratio - 0.35).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * (1.0 - self._stand_command_gate())
        )

    def _compact_hip_headroom(self):
        """Smooth headroom used by the positive forward objective."""
        limit = max(float(getattr(
            self.cfg.rewards, "compact_hip_headroom_limit_rad", 0.16
        )), 1.0e-4)
        usage = (
            torch.abs(self._hip_joint_delta()) / limit
        ).clip(0.0, 1.0)
        return (1.0 - torch.mean(usage, dim=1)).clip(0.0, 1.0)

    def _reward_compact_hip_low_torque_forward(self):
        """Credit progress only when hip motion and motor demand stay low."""
        return (
            self._forward_progress_gate()
            * self._compact_hip_headroom()
            * self._sagittal_torque_headroom()
            * self._body_angular_headroom()
        )

    def _reward_hip_peak_excursion(self):
        """Prevent one large hip swing from hiding inside a four-joint mean."""
        soft_limit = max(float(getattr(
            self.cfg.rewards, "hip_peak_soft_limit_rad", 0.065
        )), 1.0e-4)
        width = max(float(getattr(
            self.cfg.rewards, "hip_peak_penalty_width_rad", 0.070
        )), 1.0e-4)
        excess = (
            (torch.abs(self._hip_joint_delta()) - soft_limit) / width
        ).clip(min=0.0, max=2.0)
        return (
            torch.max(torch.square(excess), dim=1).values
            * self._straight_motion_gate()
        )

    def _hip_diagonal_motion_error(self):
        """Normalized physical mismatch inside the two diagonal hip pairs."""
        q = self._hip_joint_delta()
        dq = self.dof_vel[:, self.hip_dof_indices]
        # hip_dof_indices order is FL, FR, RL, RR. All four joint axes point
        # along +x, so mirrored physical motion has opposite joint signs.
        q_error = torch.stack((
            q[:, 0] + q[:, 3],
            q[:, 1] + q[:, 2],
        ), dim=1) / 0.10
        dq_error = torch.stack((
            dq[:, 0] + dq[:, 3],
            dq[:, 1] + dq[:, 2],
        ), dim=1) / 2.5
        return (
            0.70 * torch.mean(torch.square(q_error), dim=1)
            + 0.30 * torch.mean(torch.square(dq_error), dim=1)
        ).clip(max=4.0)

    def _reward_hip_diagonal_motion_mismatch(self):
        """Coordinate actual hip angles and rates within each diagonal."""
        return (
            self._hip_diagonal_motion_error()
            * self._straight_motion_gate()
        )

    def _reward_hip_trunk_twist_coupling(self):
        """Penalize rapid hip sweep specifically while the trunk is twisting."""
        hip_speed = (
            torch.mean(
                torch.abs(self.dof_vel[:, self.hip_dof_indices]), dim=1
            ) / 2.5
        ).clip(0.0, 1.5)
        trunk_twist = (
            0.55 * torch.abs(self.base_ang_vel[:, 2]) / 1.5
            + 0.45 * torch.abs(self.base_ang_vel[:, 0]) / 1.0
        ).clip(0.0, 1.5)
        return (
            hip_speed
            * trunk_twist
            * self._straight_motion_gate()
        )

    def _reward_compact_symmetric_forward(self):
        """Reward useful motion only with compact, coordinated physical hips."""
        symmetry_headroom = (
            1.0 - self._hip_diagonal_motion_error() / 2.0
        ).clip(0.0, 1.0)
        return (
            self._forward_progress_gate()
            * self._compact_hip_headroom()
            * symmetry_headroom
            * self._body_angular_headroom()
            * self._sagittal_torque_headroom()
        )

    def _reward_motor_continuous_usage(self):
        """Dense motor-output cost referenced to the 6 Nm continuous rating."""
        ratio = (
            torch.abs(self.motor_torques)
            / self._continuous_torque_ratings()
        )
        excess = (ratio - 0.65).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * (1.0 - self._stand_command_gate())
            * self._continuous_torque_penalty_blend()
        )

    def _reward_motor_continuous_overload(self):
        """Penalize instantaneous use above the continuous motor rating."""
        ratio = (
            torch.abs(self.motor_torques)
            / self._continuous_torque_ratings()
        )
        excess = (ratio - 1.0).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * (1.0 - self._stand_command_gate())
            * self._continuous_torque_penalty_blend()
        )

    def _reward_motor_thermal_overload(self):
        """Penalize sustained RMS heating before thermal derating is complete."""
        thermal_ratio = torch.sqrt(
            self.thermal_torque_sq_ema.clip(min=0.0)
        )
        excess = (thermal_ratio - 0.80).clip(min=0.0, max=1.5)
        return (
            torch.mean(torch.square(excess), dim=1)
            * (1.0 - self._stand_command_gate())
            * self._continuous_torque_penalty_blend()
        )

    def _reward_motor_thermal_peak(self):
        """Prevent one motor from carrying the whole sustained load."""
        thermal_ratio = torch.sqrt(
            self.thermal_torque_sq_ema.clip(min=0.0)
        )
        peak = torch.max(thermal_ratio, dim=1).values
        return (
            torch.square((peak - 0.90).clip(min=0.0, max=1.5))
            * (1.0 - self._stand_command_gate())
            * self._continuous_torque_penalty_blend()
        )

    def _continuous_torque_penalty_blend(self):
        """Ramp thermal costs independently from physical motor derating.

        A resumed high-torque policy needs a useful forward/contact gradient
        while it redistributes support.  Applying every thermal cost at full
        strength on the first PPO rollout caused the v1 continuation to
        collapse into short, repeatedly terminated episodes.
        """
        if getattr(self.cfg.env, "test", False):
            return 1.0
        curriculum_iterations = float(getattr(
            self.cfg.rewards,
            "continuous_torque_penalty_curriculum_iterations",
            0.0,
        ))
        if curriculum_iterations <= 0.0:
            return 1.0
        initial_blend = float(getattr(
            self.cfg.rewards,
            "continuous_torque_penalty_initial_blend",
            0.0,
        ))
        progress = min(
            max(
                self._get_torque_curriculum_iteration()
                / curriculum_iterations,
                0.0,
            ),
            1.0,
        )
        return initial_blend + (1.0 - initial_blend) * progress

    def _reward_safe_torque_straight_progress(self):
        """Credit progress only with continuous-torque and path headroom."""
        _, lateral_velocity, heading_error = self._straight_path_state()
        thermal_ratio = torch.sqrt(
            self.thermal_torque_sq_ema.clip(min=0.0)
        )
        torque_headroom = (
            1.0 - torch.mean(
                ((thermal_ratio - 0.55) / 0.55).clip(0.0, 1.0),
                dim=1,
            )
        ).clip(0.0, 1.0)
        lateral_headroom = (
            1.0 - torch.abs(lateral_velocity) / 0.22
        ).clip(0.0, 1.0)
        heading_headroom = (
            1.0 - torch.abs(heading_error) / 0.65
        ).clip(0.0, 1.0)
        return (
            self._forward_progress_gate()
            * torque_headroom
            * lateral_headroom
            * heading_headroom
            * self._body_angular_headroom()
            * self._compact_hip_headroom()
        )

    def _straight_path_state(self):
        """Return lateral displacement/velocity and heading error.

        Position and velocity are resolved in the commanded path frame rather
        than the rotating trunk frame.  This prevents a policy from hiding
        side drift by yawing the body toward the drift direction.
        """
        heading = self.commands[:, 3]
        sin_heading = torch.sin(heading)
        cos_heading = torch.cos(heading)
        displacement = self.root_states[:, :2] - self.episode_start_xy
        lateral_displacement = (
            -sin_heading * displacement[:, 0]
            + cos_heading * displacement[:, 1]
        )
        world_velocity = self.root_states[:, 7:9]
        lateral_velocity = (
            -sin_heading * world_velocity[:, 0]
            + cos_heading * world_velocity[:, 1]
        )
        heading_error = torch.atan2(
            torch.sin(heading - self.rpy[:, 2]),
            torch.cos(heading - self.rpy[:, 2]),
        )
        return lateral_displacement, lateral_velocity, heading_error

    def _reward_straight_path_lateral_displacement(self):
        """Penalize accumulated departure from the commanded straight line."""
        lateral_displacement, _, _ = self._straight_path_state()
        deadband = max(float(getattr(
            self.cfg.rewards,
            "straight_path_lateral_deadband_m",
            0.020,
        )), 0.0)
        width = max(float(getattr(
            self.cfg.rewards,
            "straight_path_lateral_penalty_width_m",
            0.180,
        )), 1.0e-4)
        excess = (
            (torch.abs(lateral_displacement) - deadband) / width
        ).clip(min=0.0, max=2.0)
        return (
            torch.square(excess)
            * self._straight_motion_gate()
        )

    def _reward_straight_path_recovery_velocity(self):
        """Track a lateral velocity that actively returns to the path.

        Penalizing displacement and lateral speed independently creates a
        conflict: once displaced, the policy is punished for the corrective
        side velocity needed to return. This closed-loop target supplies the
        missing direction while remaining an instantaneous reward.
        """
        lateral_displacement, lateral_velocity, _ = (
            self._straight_path_state()
        )
        gain = float(getattr(
            self.cfg.rewards, "straight_path_recovery_gain_s", 1.5
        ))
        max_velocity = max(float(getattr(
            self.cfg.rewards,
            "straight_path_recovery_max_velocity_m_s",
            0.12,
        )), 1.0e-4)
        desired_velocity = (
            -gain * lateral_displacement
        ).clip(-max_velocity, max_velocity)
        tracking_error = (
            (lateral_velocity - desired_velocity) / max_velocity
        ).clip(-2.0, 2.0)
        return (
            torch.square(tracking_error)
            * self._straight_motion_gate()
        )

    def _reward_straight_heading_recovery_rate(self):
        """Track the yaw rate required to remove current heading error.

        A plain yaw-rate penalty resists both unwanted oscillation and useful
        correction. The target below asks for zero yaw rate only at zero
        heading error and otherwise permits a bounded restoring rotation.
        """
        _, _, heading_error = self._straight_path_state()
        gain = float(getattr(
            self.cfg.rewards, "straight_heading_recovery_gain_s", 2.0
        ))
        max_rate = max(float(getattr(
            self.cfg.rewards,
            "straight_heading_recovery_max_rate_rad_s",
            0.80,
        )), 1.0e-4)
        desired_world_yaw_rate = (
            gain * heading_error
        ).clip(-max_rate, max_rate)
        tracking_error = (
            (self.root_states[:, 12] - desired_world_yaw_rate) / max_rate
        ).clip(-2.0, 2.0)
        return (
            torch.square(tracking_error)
            * self._straight_motion_gate()
        )

    def _reward_straight_path_lateral_acceleration(self):
        """Suppress rapid left/right side-slip reversals."""
        heading = self.commands[:, 3]
        previous_world_velocity = self.last_root_vel[:, :2]
        current_world_velocity = self.root_states[:, 7:9]
        world_acceleration = (
            current_world_velocity - previous_world_velocity
        ) / self.dt
        lateral_acceleration = (
            -torch.sin(heading) * world_acceleration[:, 0]
            + torch.cos(heading) * world_acceleration[:, 1]
        )
        normalized = (lateral_acceleration / 4.0).clip(-2.0, 2.0)
        valid = (self.episode_length_buf > 1).float()
        return (
            torch.square(normalized)
            * self._straight_motion_gate()
            * valid
        )

    def _reward_straight_balanced_progress(self):
        """Reward forward motion only when path and trunk stay controlled."""
        _, lateral_velocity, heading_error = self._straight_path_state()
        lateral_headroom = (
            1.0 - torch.abs(lateral_velocity) / 0.25
        ).clip(0.0, 1.0)
        heading_headroom = (
            1.0 - torch.abs(heading_error) / 0.75
        ).clip(0.0, 1.0)
        attitude_usage = torch.stack((
            torch.abs(self.rpy[:, 0]) / 0.12,
            torch.abs(self.rpy[:, 1]) / 0.20,
        ), dim=1).clip(0.0, 1.0)
        attitude_headroom = (
            1.0 - torch.mean(attitude_usage, dim=1)
        ).clip(0.0, 1.0)
        return (
            self._forward_progress_gate()
            * lateral_headroom
            * heading_headroom
            * attitude_headroom
            * self._body_angular_headroom()
        )

    def _reward_commanded_smooth_straight_progress(self):
        """Reward motion only near the requested speed and a quiet straight path."""
        _, lateral_velocity, heading_error = self._straight_path_state()
        speed_error = torch.abs(
            self.base_lin_vel[:, 0] - self.commands[:, 0]
        )
        speed_headroom = (1.0 - speed_error / 0.12).clip(0.0, 1.0)
        lateral_headroom = (
            1.0 - torch.abs(lateral_velocity) / 0.18
        ).clip(0.0, 1.0)
        heading_headroom = (
            1.0 - torch.abs(heading_error) / 0.50
        ).clip(0.0, 1.0)
        diagonal_valid = (~self._get_non_diagonal_swing_mask()).float()
        command_active = (
            1.0 - torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        )
        load_transfer_score, _ = self._get_diagonal_load_transfer()
        transfer_floor = float(getattr(
            self.cfg.rewards,
            "smooth_progress_load_transfer_floor",
            1.0,
        ))
        load_transfer_headroom = (
            transfer_floor
            + (1.0 - transfer_floor) * load_transfer_score
        )
        return (
            speed_headroom
            * lateral_headroom
            * heading_headroom
            * diagonal_valid
            * command_active
            * self._body_angular_headroom()
            * self._body_angular_acceleration_headroom()
            * load_transfer_headroom
        )

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
