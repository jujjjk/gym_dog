"""Independent Go2-style environment using the measured RS01 actuator."""

import torch

from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot
from .rs01_actuator import (
    compute_rs01_joint_torques,
    limit_position_target,
    step_identified_position_response,
)


class Rs01Go2StraightRobot(LeggedRobot):
    """50-observation, 12-action task with phase-conditioned foot loading."""

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(
            self.num_envs, -1, 13
        )
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]

    def __init__(self, *args, **kwargs):
        self._rs01_actuator_ready = False
        self._anti_stand_ready = False
        self._straight_path_ready = False
        super().__init__(*args, **kwargs)
        if self.num_dof != 12 or self.num_actions != 12:
            raise ValueError(
                "rs01_go2_straight requires exactly 12 URDF joints and 12 actions"
            )
        if len(self.feet_indices) != 4:
            raise ValueError(
                "rs01_go2_straight requires four retained foot rigid bodies"
            )
        if abs(self.dt - float(self.cfg.rs01_actuator.control_dt_s)) > 1.0e-9:
            raise ValueError(
                "Policy/control dt must match the measured RS01 50 Hz contract"
            )
        self.calf_dof_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self.dof_names)
                if "calf" in name.lower()
            ],
            device=self.device,
            dtype=torch.long,
        )
        if self.calf_dof_indices.numel() != 4:
            raise ValueError(
                "rs01_go2_straight requires exactly four calf joints"
            )
        self.policy_actions_unclipped = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=torch.float,
        )
        self._initialize_rs01_actuator()
        self.all_feet_contact_time_s = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self._initialize_contact_pattern_masks()
        self._anti_stand_ready = True
        self.straight_heading_target_rad = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self._straight_path_ready = True

    def _initialize_contact_pattern_masks(self):
        actor_body_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0]
        )
        foot_names = [
            actor_body_names[int(index)]
            for index in self.feet_indices.detach().cpu().tolist()
        ]
        column_by_leg = {}
        for column, name in enumerate(foot_names):
            matches = [
                leg for leg in ("FR", "FL", "RR", "RL")
                if name.startswith(leg)
            ]
            if len(matches) != 1:
                raise ValueError(f"Cannot map RS01 foot name to leg: {name}")
            column_by_leg[matches[0]] = column
        if set(column_by_leg) != {"FR", "FL", "RR", "RL"}:
            raise ValueError(
                "RS01 contact rewards require FR/FL/RR/RL foot bodies"
            )
        # Keep the URDF-derived mapping public so playback/telemetry never
        # assumes that Isaac Gym preserved a particular rigid-body order.
        self.foot_slot_by_leg = column_by_leg.copy()

        self.diagonal_a_contact_mask = self._make_contact_mask(
            column_by_leg, ("FL", "RR")
        )
        self.diagonal_b_contact_mask = self._make_contact_mask(
            column_by_leg, ("FR", "RL")
        )

    def _make_contact_mask(self, column_by_leg, contacting_legs):
        mask = torch.zeros(4, device=self.device, dtype=torch.bool)
        for leg in contacting_legs:
            mask[column_by_leg[leg]] = True
        return mask

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Indexing rigid bodies with feet_indices creates a tensor copy rather
        # than a live view.  Re-slice after every refresh or clearance rewards
        # and playback would keep seeing the all-zero initialization frame.
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        super()._post_physics_step_callback()
        if not self._anti_stand_ready:
            return
        all_feet_contact = torch.all(self.get_foot_contact_mask(), dim=1)
        self.all_feet_contact_time_s = torch.where(
            all_feet_contact,
            self.all_feet_contact_time_s + self.dt,
            torch.zeros_like(self.all_feet_contact_time_s),
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._anti_stand_ready:
            self.all_feet_contact_time_s[env_ids] = 0.0
        if self._straight_path_ready:
            self.straight_heading_target_rad[env_ids] = 0.0

    def _walking_command_gate(self):
        return (self.commands[:, 0] > 0.1).to(dtype=torch.float)

    def _reward_yaw_rate(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _straight_heading_error(self):
        raw_error = self.straight_heading_target_rad - self.rpy[:, 2]
        return torch.atan2(torch.sin(raw_error), torch.cos(raw_error))

    def _reward_heading_recovery(self):
        heading_error = self._straight_heading_error()
        gain = float(self.cfg.rewards.heading_recovery_gain_rad_s_per_rad)
        max_rate = float(self.cfg.rewards.heading_recovery_max_rate_rad_s)
        target_yaw_rate = torch.clamp(
            gain * heading_error,
            min=-max_rate,
            max=max_rate,
        )
        normalized_error = (
            self.base_ang_vel[:, 2] - target_yaw_rate
        ) / max(max_rate, 1.0e-6)
        return (
            torch.square(normalized_error)
            * self._walking_command_gate()
        )

    def _reward_prolonged_all_feet_contact(self):
        grace_s = float(self.cfg.rewards.all_feet_contact_grace_s)
        duration_excess = torch.clamp(
            (self.all_feet_contact_time_s - grace_s)
            / max(grace_s, self.dt),
            min=0.0,
            max=1.0,
        )
        return duration_excess * self._walking_command_gate()

    @staticmethod
    def _desired_contact_mask_from_phase(
        phase,
        diagonal_a_mask,
        diagonal_b_mask,
        stance_ratio,
    ):
        diagonal_a_stance = phase < stance_ratio
        diagonal_b_phase = torch.remainder(phase + 0.5, 1.0)
        diagonal_b_stance = diagonal_b_phase < stance_ratio
        return (
            diagonal_a_stance.unsqueeze(1) & diagonal_a_mask.unsqueeze(0)
        ) | (
            diagonal_b_stance.unsqueeze(1) & diagonal_b_mask.unsqueeze(0)
        )

    def _gait_phase(self):
        period_s = float(self.cfg.rewards.gait_period_s)
        return torch.remainder(
            self.episode_length_buf.to(dtype=torch.float) * self.dt,
            period_s,
        ) / period_s

    def _desired_contact_mask(self):
        return self._desired_contact_mask_from_phase(
            self._gait_phase(),
            self.diagonal_a_contact_mask,
            self.diagonal_b_contact_mask,
            float(self.cfg.rewards.gait_stance_ratio),
        )

    @staticmethod
    def _foot_load_distribution_error(vertical_force_n, desired_contact):
        force = torch.clamp(vertical_force_n, min=0.0)
        force_share = force / torch.clamp(
            torch.sum(force, dim=1, keepdim=True), min=1.0e-6
        )
        desired = desired_contact.to(dtype=force.dtype)
        desired_share = desired / torch.clamp(
            torch.sum(desired, dim=1, keepdim=True), min=1.0
        )
        return torch.sum(torch.square(force_share - desired_share), dim=1)

    def _phase_support_error(self):
        desired_contact = self._desired_contact_mask()
        vertical_force = self.contact_forces[:, self.feet_indices, 2]
        load_error = self._foot_load_distribution_error(
            vertical_force,
            desired_contact,
        )
        contact_error = torch.mean(
            (
                self.get_foot_contact_mask()
                != desired_contact
            ).to(dtype=torch.float),
            dim=1,
        )
        return load_error + contact_error

    def _reward_phase_support_tracking(self):
        tracking = torch.exp(
            -self._phase_support_error()
            / float(self.cfg.rewards.phase_support_sigma)
        )
        return tracking * self._walking_command_gate()

    @staticmethod
    def _swing_height_target_from_phase(
        phase,
        diagonal_a_mask,
        diagonal_b_mask,
        stance_ratio,
        foot_radius_m,
        swing_clearance_m,
    ):
        phase_a = phase
        phase_b = torch.remainder(phase + 0.5, 1.0)
        denominator = max(1.0 - stance_ratio, 1.0e-6)
        progress_a = torch.clamp(
            (phase_a - stance_ratio) / denominator, min=0.0, max=1.0
        )
        progress_b = torch.clamp(
            (phase_b - stance_ratio) / denominator, min=0.0, max=1.0
        )
        swing_a = phase_a >= stance_ratio
        swing_b = phase_b >= stance_ratio
        progress = (
            progress_a.unsqueeze(1)
            * diagonal_a_mask.unsqueeze(0).to(dtype=phase.dtype)
            + progress_b.unsqueeze(1)
            * diagonal_b_mask.unsqueeze(0).to(dtype=phase.dtype)
        )
        swing = (
            swing_a.unsqueeze(1) & diagonal_a_mask.unsqueeze(0)
        ) | (
            swing_b.unsqueeze(1) & diagonal_b_mask.unsqueeze(0)
        )
        target = (
            foot_radius_m
            + swing_clearance_m * torch.sin(torch.pi * progress)
        )
        return target, swing

    def _reward_phase_swing_clearance(self):
        return self._phase_swing_clearance_error() * self._walking_command_gate()

    def _phase_swing_clearance_error(self, selected_feet=None):
        clearance_m = float(self.cfg.rewards.swing_clearance_m)
        target_height, desired_swing = self._swing_height_target_from_phase(
            self._gait_phase(),
            self.diagonal_a_contact_mask,
            self.diagonal_b_contact_mask,
            float(self.cfg.rewards.gait_stance_ratio),
            float(self.cfg.rewards.foot_collision_radius_m),
            clearance_m,
        )
        if selected_feet is not None:
            desired_swing = (
                desired_swing
                & selected_feet.unsqueeze(0).to(dtype=torch.bool)
            )
        normalized_shortfall = torch.clamp(
            target_height - self.feet_pos[:, :, 2],
            min=0.0,
        ) / max(clearance_m, 1.0e-6)
        error = torch.sum(
            torch.square(normalized_shortfall)
            * desired_swing.to(dtype=torch.float),
            dim=1,
        ) / torch.clamp(
            torch.sum(desired_swing.to(dtype=torch.float), dim=1),
            min=1.0,
        )
        return error

    def _reward_rear_swing_clearance(self):
        rear_feet = torch.zeros(
            len(self.feet_indices),
            device=self.device,
            dtype=torch.bool,
        )
        rear_feet[self.foot_slot_by_leg["RL"]] = True
        rear_feet[self.foot_slot_by_leg["RR"]] = True
        return (
            self._phase_swing_clearance_error(rear_feet)
            * self._walking_command_gate()
        )

    def _reward_diagonal_contact_sync(self):
        contact = self.get_foot_contact_mask()
        fl = self.foot_slot_by_leg["FL"]
        fr = self.foot_slot_by_leg["FR"]
        rl = self.foot_slot_by_leg["RL"]
        rr = self.foot_slot_by_leg["RR"]
        mismatch_a = torch.logical_xor(contact[:, fl], contact[:, rr])
        mismatch_b = torch.logical_xor(contact[:, fr], contact[:, rl])
        mismatch = 0.5 * (
            mismatch_a.to(dtype=torch.float)
            + mismatch_b.to(dtype=torch.float)
        )
        return mismatch * self._walking_command_gate()

    def _reward_same_axle_flight(self):
        contact = self.get_foot_contact_mask()
        fl = self.foot_slot_by_leg["FL"]
        fr = self.foot_slot_by_leg["FR"]
        rl = self.foot_slot_by_leg["RL"]
        rr = self.foot_slot_by_leg["RR"]
        front_flight = ~contact[:, fl] & ~contact[:, fr]
        rear_flight = ~contact[:, rl] & ~contact[:, rr]
        return (
            front_flight.to(dtype=torch.float)
            + rear_flight.to(dtype=torch.float)
        ) * self._walking_command_gate()

    def _reward_flight(self):
        flight = ~torch.any(self.get_foot_contact_mask(), dim=1)
        return flight.to(dtype=torch.float) * self._walking_command_gate()

    def compute_observations(self):
        phase_angle = 2.0 * torch.pi * self._gait_phase()
        phase_observation = torch.stack(
            (torch.sin(phase_angle), torch.cos(phase_angle)), dim=1
        )
        observations = [
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos)
            * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            phase_observation,
        ]
        if getattr(
            self.cfg.commands, "observe_straight_heading_error", False
        ):
            scale = float(
                self.cfg.commands.straight_heading_observation_scale
            )
            observations.append(
                torch.clamp(
                    self._straight_heading_error() * scale,
                    min=-1.0,
                    max=1.0,
                ).unsqueeze(1)
            )
        self.obs_buf = torch.cat(
            observations,
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self.noise_scale_vec

    def _reward_raw_torque_over_peak(self):
        excess = torch.clamp(
            torch.abs(self.raw_pd_torques) - self.peak_torque_limit_nm,
            min=0.0,
        )
        normalized = excess / self.peak_torque_limit_nm
        return torch.mean(torch.square(normalized), dim=1)

    def _reward_motor_saturation(self):
        saturated = (
            torch.abs(self.motor_electromagnetic_torques)
            >= self.peak_torque_limit_nm - 1.0e-4
        )
        return torch.mean(saturated.to(dtype=torch.float), dim=1)

    def _reward_calf_velocity_excess(self):
        """Penalize the measured swing snap before the URDF velocity ceiling."""
        soft_limit = float(
            self.cfg.rewards.calf_velocity_soft_limit_rad_s
        )
        if soft_limit <= 0.0:
            raise ValueError(
                "calf_velocity_soft_limit_rad_s must be positive"
            )
        calf_speed = torch.abs(
            self.dof_vel[:, self.calf_dof_indices]
        )
        normalized_excess = torch.clamp(
            calf_speed - soft_limit,
            min=0.0,
        ) / soft_limit
        return torch.mean(torch.square(normalized_excess), dim=1)

    def _reward_action_saturation(self):
        """Charge actor means outside the executable action interval."""
        soft_limit = float(
            self.cfg.rewards.action_saturation_soft_limit
        )
        clip_limit = float(self.cfg.normalization.clip_actions)
        if soft_limit < 0.0 or soft_limit >= clip_limit:
            raise ValueError(
                "action_saturation_soft_limit must satisfy "
                f"0 <= soft limit < clip limit, got {soft_limit} and "
                f"{clip_limit}"
            )
        excess = torch.clamp(
            torch.abs(self.policy_actions_unclipped) - soft_limit,
            min=0.0,
        )
        return torch.mean(torch.square(excess), dim=1)

    def _joint_type_value(self, values):
        result = []
        for name in self.dof_names:
            matches = [value for key, value in values.items() if key in name]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one RS01 joint-type value for {name}, got {matches}"
                )
            result.append(matches[0])
        return torch.tensor(result, device=self.device, dtype=torch.float)

    def _motor_value(self, values):
        motor_cfg = self.cfg.rs01_actuator
        result = []
        for name in self.dof_names:
            motor_id = motor_cfg.joint_to_motor_id[name]
            result.append(values[motor_id])
        return torch.tensor(result, device=self.device, dtype=torch.float)

    def _initialize_rs01_actuator(self):
        motor_cfg = self.cfg.rs01_actuator
        shape = (self.num_envs, self.num_dof)

        action_scale_by_joint = getattr(
            self.cfg.control,
            "action_scale_by_joint",
            None,
        )
        if action_scale_by_joint is None:
            self.rs01_action_scale_rad = torch.full(
                (self.num_dof,),
                float(self.cfg.control.action_scale),
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.rs01_action_scale_rad = self._joint_type_value(
                action_scale_by_joint
            )
        self.rs01_target_rate_limit_rad_s = self._joint_type_value(
            motor_cfg.target_rate_limit_rad_s
        )
        self.rs01_target_acceleration_limit_rad_s2 = self._joint_type_value(
            motor_cfg.target_acceleration_limit_rad_s2
        )
        self.rs01_nominal_response_gain = self._motor_value(
            motor_cfg.response_gain
        )
        self.rs01_nominal_time_constant_s = self._motor_value(
            motor_cfg.time_constant_s
        )
        self.rs01_nominal_coulomb_friction_nm = self._motor_value(
            motor_cfg.coulomb_friction_nm
        )
        observed_delay_s = self._motor_value(
            motor_cfg.observed_closed_loop_delay_s
        )
        self.rs01_nominal_delay_steps = torch.round(
            observed_delay_s / float(self.sim_params.dt)
        ).to(dtype=torch.long)
        randomize_actuator = bool(getattr(
            self.cfg.domain_rand,
            "randomize_rs01_actuator",
            False,
        ))
        if randomize_actuator:
            self.rs01_response_gain = (
                self.rs01_nominal_response_gain.unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            self.rs01_time_constant_s = (
                self.rs01_nominal_time_constant_s.unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            self.rs01_coulomb_friction_nm = (
                self.rs01_nominal_coulomb_friction_nm.unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            self.rs01_delay_steps = (
                self.rs01_nominal_delay_steps.unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            delay_range = getattr(
                self.cfg.domain_rand,
                "rs01_delay_step_offset_range",
                [0, 0],
            )
            maximum_delay_offset = max(0, int(delay_range[1]))
        else:
            self.rs01_response_gain = self.rs01_nominal_response_gain.clone()
            self.rs01_time_constant_s = (
                self.rs01_nominal_time_constant_s.clone()
            )
            self.rs01_coulomb_friction_nm = (
                self.rs01_nominal_coulomb_friction_nm.clone()
            )
            self.rs01_delay_steps = self.rs01_nominal_delay_steps.clone()
            maximum_delay_offset = 0
        max_delay_steps = (
            int(torch.max(self.rs01_nominal_delay_steps).item())
            + maximum_delay_offset
        )

        default = self.default_dof_pos.repeat(self.num_envs, 1)
        self.rs01_limited_position_target_rad = default.clone()
        self.rs01_target_rate_rad_s = torch.zeros(
            shape, device=self.device, dtype=torch.float
        )
        self.rs01_response_target_rad = default.clone()
        self.rs01_target_delay_buffer = default.unsqueeze(0).repeat(
            max_delay_steps + 1, 1, 1
        )

        self.peak_torque_limit_nm = torch.full(
            shape,
            float(motor_cfg.peak_torque_limit_nm),
            device=self.device,
            dtype=torch.float,
        )
        self.continuous_torque_nm = torch.full(
            shape,
            float(motor_cfg.continuous_torque_nm),
            device=self.device,
            dtype=torch.float,
        )
        self.raw_pd_torques = torch.zeros(
            shape, device=self.device, dtype=torch.float
        )
        self.motor_electromagnetic_torques = torch.zeros_like(
            self.raw_pd_torques
        )
        self.applied_joint_torques = torch.zeros_like(self.raw_pd_torques)
        if randomize_actuator:
            all_env_ids = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
            self._randomize_rs01_actuator(all_env_ids)
        self._rs01_actuator_ready = True

    def _randomize_rs01_actuator(self, env_ids):
        if len(env_ids) == 0 or self.rs01_delay_steps.ndim != 2:
            return
        domain_cfg = self.cfg.domain_rand
        if self.cfg.env.test:
            response_range = [1.0, 1.0]
            time_range = [1.0, 1.0]
            friction_range = [1.0, 1.0]
            delay_range = [0, 0]
        else:
            response_range = domain_cfg.rs01_response_gain_scale_range
            time_range = domain_cfg.rs01_time_constant_scale_range
            friction_range = domain_cfg.rs01_friction_scale_range
            delay_range = domain_cfg.rs01_delay_step_offset_range

        count = len(env_ids)
        # Most continuation tasks use one scale per robot.  The matched
        # Sim2Sim transfer task deliberately samples each identified motor
        # independently: the nominal centre remains the measured RS01 motor,
        # while the actor must recover from small left/right response and
        # delay differences instead of relying on perfectly paired dynamics.
        motor_columns = (
            self.num_dof
            if bool(getattr(
                domain_cfg,
                "rs01_independent_motor_randomization",
                False,
            ))
            else 1
        )
        delay_columns = (
            self.num_dof
            if bool(getattr(
                domain_cfg,
                "rs01_independent_delay_randomization",
                False,
            ))
            else 1
        )
        response_scale = torch.empty(
            count, motor_columns, device=self.device
        ).uniform_(float(response_range[0]), float(response_range[1]))
        time_scale = torch.empty(
            count, motor_columns, device=self.device
        ).uniform_(float(time_range[0]), float(time_range[1]))
        friction_scale = torch.empty(
            count, motor_columns, device=self.device
        ).uniform_(float(friction_range[0]), float(friction_range[1]))
        delay_offset = torch.randint(
            int(delay_range[0]),
            int(delay_range[1]) + 1,
            (count, delay_columns),
            device=self.device,
        )

        self.rs01_response_gain[env_ids] = (
            self.rs01_nominal_response_gain.unsqueeze(0) * response_scale
        )
        self.rs01_time_constant_s[env_ids] = (
            self.rs01_nominal_time_constant_s.unsqueeze(0) * time_scale
        )
        self.rs01_coulomb_friction_nm[env_ids] = (
            self.rs01_nominal_coulomb_friction_nm.unsqueeze(0)
            * friction_scale
        )
        self.rs01_delay_steps[env_ids] = torch.clamp(
            self.rs01_nominal_delay_steps.unsqueeze(0) + delay_offset,
            min=0,
            max=self.rs01_target_delay_buffer.shape[0] - 1,
        )

    def step(self, actions):
        if self._rs01_actuator_ready:
            self.policy_actions_unclipped.copy_(
                actions.to(self.device)
            )
            clipped_actions = torch.clamp(
                actions.to(self.device),
                -float(self.cfg.normalization.clip_actions),
                float(self.cfg.normalization.clip_actions),
            )
            desired_target = (
                self.default_dof_pos
                + self.rs01_action_scale_rad * clipped_actions
            )
            (
                self.rs01_limited_position_target_rad,
                self.rs01_target_rate_rad_s,
            ) = limit_position_target(
                desired_target,
                self.rs01_limited_position_target_rad,
                self.rs01_target_rate_rad_s,
                self.rs01_target_rate_limit_rad_s,
                self.rs01_target_acceleration_limit_rad_s2,
                self.cfg.rs01_actuator.control_dt_s,
            )
        return super().step(actions)

    def _compute_torques(self, actions):
        if not self._rs01_actuator_ready:
            return super()._compute_torques(actions)

        self.rs01_target_delay_buffer = torch.roll(
            self.rs01_target_delay_buffer, shifts=1, dims=0
        )
        self.rs01_target_delay_buffer[0].copy_(
            self.rs01_limited_position_target_rad
        )
        if self.rs01_delay_steps.ndim == 1:
            delayed_columns = [
                self.rs01_target_delay_buffer[
                    int(self.rs01_delay_steps[j]), :, j
                ]
                for j in range(self.num_dof)
            ]
            delayed_target = torch.stack(delayed_columns, dim=1)
        else:
            delay_history = self.rs01_target_delay_buffer.permute(1, 2, 0)
            delayed_target = torch.gather(
                delay_history,
                dim=2,
                index=self.rs01_delay_steps.unsqueeze(-1),
            ).squeeze(-1)

        self.rs01_response_target_rad = step_identified_position_response(
            self.rs01_response_target_rad,
            delayed_target,
            self.default_dof_pos,
            self.rs01_response_gain,
            self.rs01_time_constant_s,
            self.sim_params.dt,
        )
        (
            self.raw_pd_torques,
            self.motor_electromagnetic_torques,
            self.applied_joint_torques,
        ) = compute_rs01_joint_torques(
            self.rs01_response_target_rad,
            self.dof_pos,
            self.dof_vel,
            self.p_gains,
            self.d_gains,
            self.peak_torque_limit_nm,
            self.rs01_coulomb_friction_nm,
            self.cfg.rs01_actuator.friction_smoothing_rad_s,
        )
        return self.applied_joint_torques

    def _reset_dofs(self, env_ids):
        noise = float(self.cfg.init_state.reset_dof_position_noise_rad)
        position_noise = (2.0 * torch.rand(
            len(env_ids), self.num_dof, device=self.device
        ) - 1.0) * noise
        self.dof_pos[env_ids] = self.default_dof_pos + position_noise
        self.dof_vel[env_ids] = 0.0

        if self._rs01_actuator_ready:
            self.rs01_limited_position_target_rad[env_ids] = self.default_dof_pos
            self.rs01_target_rate_rad_s[env_ids] = 0.0
            self.rs01_response_target_rad[env_ids] = self.default_dof_pos
            self.rs01_target_delay_buffer[:, env_ids, :] = self.default_dof_pos
            self.raw_pd_torques[env_ids] = 0.0
            self.motor_electromagnetic_torques[env_ids] = 0.0
            self.applied_joint_torques[env_ids] = 0.0
            self._randomize_rs01_actuator(env_ids)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        heading_noise = float(getattr(
            self.cfg.init_state,
            "reset_heading_noise_rad",
            0.0,
        ))
        if heading_noise > 0.0 and not self.cfg.env.test:
            yaw = (
                2.0 * torch.rand(
                    len(env_ids), device=self.device
                ) - 1.0
            ) * heading_noise
            self.root_states[env_ids, 3] = 0.0
            self.root_states[env_ids, 4] = 0.0
            self.root_states[env_ids, 5] = torch.sin(0.5 * yaw)
            self.root_states[env_ids, 6] = torch.cos(0.5 * yaw)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
