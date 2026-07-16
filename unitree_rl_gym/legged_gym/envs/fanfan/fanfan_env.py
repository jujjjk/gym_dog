from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
import numpy as np
import torch


class FanfanRobot(LeggedRobot):
    def step(self, actions):
        # A smooth bound preserves control resolution when the Gaussian policy
        # produces values outside [-1, 1]. Hard clipping made the legs bang
        # between their limits and destroyed the diagonal timing.
        bounded_actions = torch.tanh(actions)
        bounded_actions = self._apply_straight_action_bias(bounded_actions)
        bounded_actions = self._project_straight_diagonal_actions(
            bounded_actions
        )
        if getattr(self.cfg.control, "filter_policy_actions", False):
            self.last_policy_actions[:] = self.policy_actions
            self.policy_actions[:] = bounded_actions
            alpha = self.cfg.control.policy_action_filter_alpha
            desired = self.filtered_actions + alpha * (
                bounded_actions - self.filtered_actions
            )
            desired_velocity = (desired - self.filtered_actions) / self.dt
            desired_velocity = torch.clamp(
                desired_velocity,
                -self.policy_action_rate_limits,
                self.policy_action_rate_limits,
            )
            velocity_delta = desired_velocity - self.filtered_action_velocity
            velocity_delta = torch.clamp(
                velocity_delta,
                -self.policy_action_accel_limits * self.dt,
                self.policy_action_accel_limits * self.dt,
            )
            self.filtered_action_velocity += velocity_delta
            next_actions = self.filtered_actions + self.filtered_action_velocity * self.dt
            crossed = (desired - self.filtered_actions) * (desired - next_actions) < 0.0
            next_actions = torch.where(crossed, desired, next_actions)
            next_actions = self._project_straight_diagonal_actions(
                next_actions
            )
            self.filtered_action_velocity = (
                next_actions - self.filtered_actions
            ) / self.dt
            self.filtered_actions[:] = next_actions
            self.policy_filter_gap[:] = bounded_actions - next_actions
            return super().step(next_actions)
        self.last_policy_actions[:] = self.policy_actions
        self.policy_actions[:] = bounded_actions
        self.policy_filter_gap.zero_()
        return super().step(bounded_actions)

    def _apply_straight_action_bias(self, actions):
        """Apply a calibrated, command-gated normalized action correction."""
        bias_cfg = getattr(
            self.cfg.control, "straight_action_bias_by_joint", None
        )
        if bias_cfg is None:
            return actions
        bias = torch.tensor(
            [float(bias_cfg.get(name, 0.0)) for name in self.dof_names],
            dtype=actions.dtype,
            device=self.device,
        ).unsqueeze(0)
        straight = (
            (torch.abs(self.commands[:, 0]) > 0.03)
            & (torch.abs(self.commands[:, 1]) < 0.02)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        ).unsqueeze(1)
        corrected = (actions + bias).clip(-1.0, 1.0)
        return torch.where(straight, corrected, actions)

    def _project_straight_diagonal_actions(self, actions):
        """Project straight-motion actions onto physical diagonal symmetry."""
        if not getattr(
            self.cfg.control, "project_straight_diagonal_actions", False
        ):
            return actions

        projected = actions.clone()
        physical = actions.clone()
        physical[:, self.hip_dof_indices] *= self.cfg.control.hip_action_scale
        physical[:, self.front_sagittal_dof_indices] *= (
            self.cfg.control.action_scale
        )
        physical[:, self.rear_sagittal_dof_indices] *= (
            self.cfg.control.rear_action_scale
        )

        symmetric = physical.clone()
        for first, second in (("FL", "RR"), ("FR", "RL")):
            for joint in ("hip", "thigh", "calf"):
                a = self.leg_dof_indices[first][joint]
                b = self.leg_dof_indices[second][joint]
                if joint == "hip":
                    mean = 0.5 * (physical[:, a] - physical[:, b])
                    symmetric[:, a] = mean
                    symmetric[:, b] = -mean
                else:
                    mean = 0.5 * (physical[:, a] + physical[:, b])
                    symmetric[:, a] = mean
                    symmetric[:, b] = mean

        symmetric[:, self.hip_dof_indices] /= self.cfg.control.hip_action_scale
        symmetric[:, self.front_sagittal_dof_indices] /= (
            self.cfg.control.action_scale
        )
        symmetric[:, self.rear_sagittal_dof_indices] /= (
            self.cfg.control.rear_action_scale
        )
        symmetric = symmetric.clip(-1.0, 1.0)

        straight = (
            (torch.abs(self.commands[:, 0]) > 0.03)
            & (torch.abs(self.commands[:, 1]) < 0.02)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        ).unsqueeze(1)
        return torch.where(straight, symmetric, projected)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = super()._get_noise_scale_vec(cfg)
        noise_vec[-4:] = 0.0
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        phase_offsets = []
        self.foot_slot_by_leg = {}
        for foot_slot, body_index in enumerate(self.feet_indices.cpu().tolist()):
            name = body_names[body_index]
            self.foot_slot_by_leg[name.split("_", 1)[0]] = foot_slot
            phase_offsets.append(
                0.0 if name.startswith("FL_") or name.startswith("RR_") else 0.5
            )
        self.gait_phase_offsets = torch.tensor(
            phase_offsets, dtype=torch.float, device=self.device
        )
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.command_transition_age = torch.full(
            (self.num_envs,), 10.0, dtype=torch.float, device=self.device
        )
        self.command_transition_magnitude = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.leg_dof_indices = {}
        for leg in ("FL", "FR", "RL", "RR"):
            self.leg_dof_indices[leg] = {
                "hip": self.dof_names.index(f"{leg}_hip_joint"),
                "thigh": self.dof_names.index(f"{leg}_thigh_joint"),
                "calf": self.dof_names.index(f"{leg}_calf_joint"),
            }
        self.hip_dof_indices = torch.tensor(
            [self.leg_dof_indices[leg]["hip"] for leg in ("FL", "FR", "RL", "RR")],
            dtype=torch.long,
            device=self.device,
        )
        self.rear_sagittal_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("RL", "RR")
                for joint in ("thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.front_sagittal_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("FL", "FR")
                for joint in ("thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.sagittal_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("FL", "FR", "RL", "RR")
                for joint in ("thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.left_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("FL", "RL")
                for joint in ("hip", "thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.right_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("FR", "RR")
                for joint in ("hip", "thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.raw_torques = torch.zeros_like(self.torques)
        self.motor_strength = torch.ones_like(self.torques)
        self.target_dof_pos_rl = self.default_dof_pos.repeat(self.num_envs, 1)
        self.torque_clip_error = torch.zeros_like(self.torques)
        self.torque_ema = torch.zeros_like(self.torques)
        self.torque_metric_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.torque_metric_sums = {
            "mean_abs_raw_torque": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "torque_saturation_ratio": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "torque_over_13_ratio": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "torque_over_15_ratio": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "torque_over_17_ratio": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
        }
        self.max_abs_raw_torque = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.policy_actions = torch.zeros_like(self.actions)
        self.last_policy_actions = torch.zeros_like(self.actions)
        self.filtered_actions = torch.zeros_like(self.actions)
        self.filtered_action_velocity = torch.zeros_like(self.actions)
        self.policy_filter_gap = torch.zeros_like(self.actions)
        rate_cfg = getattr(self.cfg.control, "policy_action_rate_limits", {})
        accel_cfg = getattr(self.cfg.control, "policy_action_accel_limits", {})
        self.policy_action_rate_limits = torch.tensor(
            [next((value for key, value in rate_cfg.items() if key in name), 1.0e6)
             for name in self.dof_names], dtype=torch.float, device=self.device
        )
        self.policy_action_accel_limits = torch.tensor(
            [next((value for key, value in accel_cfg.items() if key in name), 1.0e6)
             for name in self.dof_names], dtype=torch.float, device=self.device
        )
        torque_limits_by_joint = getattr(
            self.cfg.control, "torque_limits_by_joint", None
        )
        if torque_limits_by_joint is not None:
            for dof_index, dof_name in enumerate(self.dof_names):
                matches = [
                    float(limit)
                    for joint_type, limit in torque_limits_by_joint.items()
                    if joint_type in dof_name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Expected exactly one torque limit match for "
                        f"{dof_name}, got {matches} from {torque_limits_by_joint}"
                    )
                self.torque_limits[dof_index] = matches[0]
        else:
            torque_limit = getattr(self.cfg.control, "torque_limit_override", None)
            if torque_limit is not None:
                self.torque_limits[:] = float(torque_limit)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.rear_sagittal_dof_indices] = (
            actions[:, self.rear_sagittal_dof_indices]
            * self.cfg.control.rear_action_scale
        )
        actions_scaled[:, self.hip_dof_indices] = (
            actions[:, self.hip_dof_indices] * self.cfg.control.hip_action_scale
        )
        phase = (
            self.gait_phase.unsqueeze(1) + self.gait_phase_offsets.unsqueeze(0)
        ) % 1.0
        stance_ratio = self.cfg.rewards.gait_stance_ratio
        swing_progress = ((phase - stance_ratio) / (1.0 - stance_ratio)).clip(0.0, 1.0)
        smooth_swing = swing_progress * swing_progress * (3.0 - 2.0 * swing_progress)
        swing_profile = torch.sin(torch.pi * smooth_swing) * (phase >= stance_ratio)
        stance_progress = (phase / stance_ratio).clip(0.0, 1.0)
        thigh_profile = torch.where(
            phase < stance_ratio,
            -1.0 + 2.0 * stance_progress,
            1.0 - 2.0 * smooth_swing,
        )

        gait_offset = torch.zeros_like(actions_scaled)
        foot_names = ("FL", "FR", "RL", "RR")
        for foot_slot, leg in enumerate(foot_names):
            gait_offset[:, self.leg_dof_indices[leg]["thigh"]] = (
                self.cfg.rewards.gait_thigh_amplitude * thigh_profile[:, foot_slot]
            )
            gait_offset[:, self.leg_dof_indices[leg]["calf"]] = (
                self.cfg.rewards.gait_calf_amplitude * swing_profile[:, foot_slot]
            )
        if getattr(self.cfg.control, "gate_gait_with_command", False):
            command_energy = (
                torch.sum(torch.square(self.commands[:, :2]), dim=1)
                + 0.04 * torch.square(self.commands[:, 2])
            )
            sigma = self.cfg.control.gait_command_gate_sigma
            gait_gate = 1.0 - torch.exp(-command_energy / sigma)
            gait_offset *= gait_gate.unsqueeze(1)
        target_dof_pos = actions_scaled + gait_offset + self.default_dof_pos
        backward_rear_calf_target_min = getattr(
            self.cfg.control, "backward_rear_calf_target_min", None
        )
        if backward_rear_calf_target_min is not None:
            backward = (self.commands[:, 0] < -0.03).unsqueeze(1)
            rear_targets = target_dof_pos[:, self._get_rear_calf_indices()]
            guarded_targets = torch.clamp(
                rear_targets, min=float(backward_rear_calf_target_min)
            )
            target_dof_pos[:, self._get_rear_calf_indices()] = torch.where(
                backward, guarded_targets, rear_targets
            )
        raw_torques = self.motor_strength * (self.p_gains * (
            target_dof_pos - self.dof_pos
        ) - self.d_gains * self.dof_vel)
        clipped_torques = torch.clip(
            raw_torques, -self.torque_limits, self.torque_limits
        )

        self.raw_torques = raw_torques
        self.target_dof_pos_rl = target_dof_pos
        self.torque_clip_error = raw_torques - clipped_torques
        self.torque_ema = 0.98 * self.torque_ema + 0.02 * torch.abs(raw_torques)
        self._update_torque_metrics(raw_torques)
        return clipped_torques

    def _update_torque_metrics(self, raw_torques):
        abs_raw = torch.abs(raw_torques)
        torque_limits = self.torque_limits.unsqueeze(0)

        self.torque_metric_count += 1.0
        self.max_abs_raw_torque = torch.maximum(
            self.max_abs_raw_torque, torch.max(abs_raw, dim=1).values
        )
        self.torque_metric_sums["mean_abs_raw_torque"] += torch.mean(abs_raw, dim=1)
        self.torque_metric_sums["torque_saturation_ratio"] += torch.mean(
            (abs_raw >= torque_limits).float(), dim=1
        )
        self.torque_metric_sums["torque_over_13_ratio"] += torch.mean(
            (abs_raw > 13.0).float(), dim=1
        )
        self.torque_metric_sums["torque_over_15_ratio"] += torch.mean(
            (abs_raw > 15.0).float(), dim=1
        )
        self.torque_metric_sums["torque_over_17_ratio"] += torch.mean(
            (abs_raw > 17.0).float(), dim=1
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)
        if getattr(self.cfg.domain_rand, "randomize_motor_strength", False):
            low, high = self.cfg.domain_rand.motor_strength_range
            strength = torch_rand_float(
                low, high, (len(env_ids), self.num_actions), device=self.device
            )
            if getattr(
                self.cfg.domain_rand, "pair_diagonal_motor_strength", False
            ):
                for first, second in (("FL", "RR"), ("FR", "RL")):
                    for joint in ("hip", "thigh", "calf"):
                        a = self.leg_dof_indices[first][joint]
                        b = self.leg_dof_indices[second][joint]
                        mean = 0.5 * (strength[:, a] + strength[:, b])
                        strength[:, a] = mean
                        strength[:, b] = mean
            self.motor_strength[env_ids] = strength
        metric_count = self.torque_metric_count[env_ids].clip(min=1.0)
        self.extras["episode"]["max_abs_raw_torque"] = torch.mean(
            self.max_abs_raw_torque[env_ids]
        )
        for name, values in self.torque_metric_sums.items():
            self.extras["episode"][name] = torch.mean(values[env_ids] / metric_count)
        self.extras["episode"]["torque_curriculum_iteration"] = (
            self._get_torque_curriculum_iteration()
        )
        self.extras["episode"]["torque_curriculum_stage"] = (
            self._get_torque_curriculum_stage()
        )

        self.torque_ema[env_ids] = 0.0
        self.torque_clip_error[env_ids] = 0.0
        self.raw_torques[env_ids] = 0.0
        self.target_dof_pos_rl[env_ids] = self.default_dof_pos
        self.policy_actions[env_ids] = 0.0
        self.last_policy_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.filtered_action_velocity[env_ids] = 0.0
        self.policy_filter_gap[env_ids] = 0.0
        self.max_abs_raw_torque[env_ids] = 0.0
        self.torque_metric_count[env_ids] = 0.0
        for values in self.torque_metric_sums.values():
            values[env_ids] = 0.0
        self.command_transition_age[env_ids] = 0.0
        self.command_transition_magnitude[env_ids] = 0.0

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if getattr(self.cfg.domain_rand, "randomize_base_com", False):
            x_range = self.cfg.domain_rand.base_com_x_range
            y_range = self.cfg.domain_rand.base_com_y_range
            props[0].com.x += np.random.uniform(x_range[0], x_range[1])
            props[0].com.y += np.random.uniform(y_range[0], y_range[1])
        return props

    def _post_physics_step_callback(self):
        self.command_transition_age += self.dt
        super()._post_physics_step_callback()
        if (not self.cfg.commands.heading_command
                and getattr(self.cfg.commands, "observe_heading_error", False)):
            target = self.commands[:, 3] + self.commands[:, 2] * self.dt
            self.commands[:, 3] = torch.atan2(torch.sin(target), torch.cos(target))
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]

        period = self.cfg.rewards.gait_period
        self.gait_phase = (self.episode_length_buf * self.dt) % period / period

    def compute_observations(self):
        phase_angle = 2.0 * torch.pi * self.gait_phase
        phase_obs = torch.stack((torch.sin(phase_angle), torch.cos(phase_angle)), dim=1)
        if getattr(self.cfg.commands, "observe_heading_error", False):
            heading_error = torch.atan2(
                torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
                torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
            )
            heading_obs = torch.stack(
                (torch.sin(heading_error), torch.cos(heading_error)), dim=1
            )
        else:
            heading_obs = torch.zeros(self.num_envs, 2, device=self.device)
            heading_obs[:, 1] = 1.0
        observed_commands = self.commands[:, :3].clone()
        longitudinal_gain = getattr(
            self.cfg.control, "command_feedback_longitudinal_gain", 0.0
        )
        lateral_gain = getattr(
            self.cfg.control, "command_feedback_lateral_gain", 0.0
        )
        yaw_gain = getattr(
            self.cfg.control, "command_feedback_yaw_gain", 0.0
        )
        heading_gain = getattr(
            self.cfg.control, "command_feedback_heading_gain", 0.0
        )
        heading_damping = getattr(
            self.cfg.control, "command_feedback_heading_damping", 0.0
        )
        diagonal_x_scale = getattr(
            self.cfg.control,
            "command_feedback_diagonal_longitudinal_scale",
            1.0,
        )
        if longitudinal_gain != 0.0:
            observed_commands[:, 0] += longitudinal_gain * (
                self.commands[:, 0] - self.base_lin_vel[:, 0]
            )
        if lateral_gain != 0.0:
            observed_commands[:, 1] += lateral_gain * (
                self.commands[:, 1] - self.base_lin_vel[:, 1]
            )
        if yaw_gain != 0.0:
            observed_commands[:, 2] += yaw_gain * (
                self.commands[:, 2] - self.base_ang_vel[:, 2]
            )
        if diagonal_x_scale != 1.0:
            diagonal = (
                (torch.abs(self.commands[:, 0]) > 0.05)
                & (torch.abs(self.commands[:, 1]) > 0.05)
                & (torch.abs(self.commands[:, 2]) < 0.05)
            )
            observed_commands[:, 0] = torch.where(
                diagonal,
                diagonal_x_scale * observed_commands[:, 0],
                observed_commands[:, 0],
            )
        if heading_gain != 0.0:
            heading_error = torch.atan2(heading_obs[:, 0], heading_obs[:, 1])
            heading_hold = (
                torch.linalg.norm(self.commands[:, :2], dim=1) > 0.04
            ) & (torch.abs(self.commands[:, 2]) < 0.05)
            heading_correction = (
                heading_gain * heading_error
                - heading_damping * self.base_ang_vel[:, 2]
            )
            observed_commands[:, 2] = torch.where(
                heading_hold, heading_correction, observed_commands[:, 2]
            )
        ranges = self.cfg.commands.ranges
        observed_commands[:, 0].clip_(
            ranges.lin_vel_x[0], ranges.lin_vel_x[1]
        )
        observed_commands[:, 1].clip_(
            ranges.lin_vel_y[0], ranges.lin_vel_y[1]
        )
        observed_commands[:, 2].clip_(
            ranges.ang_vel_yaw[0], ranges.ang_vel_yaw[1]
        )
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            observed_commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.filtered_actions if getattr(self.cfg.control, "filter_policy_actions", False) else self.actions,
            phase_obs,
            heading_obs,
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reset_dofs(self, env_ids):
        # Fanfan is small enough that the base task's 0.5-1.5 multiplier can
        # spawn a foot through the floor or put a calf directly on its limit.
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_calf_indices(self):
        if not hasattr(self, "calf_dof_indices"):
            indices = [i for i, name in enumerate(self.dof_names) if "calf" in name]
            self.calf_dof_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.calf_dof_indices

    def _get_front_feet_indices(self):
        if not hasattr(self, "front_feet_indices"):
            body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            indices = [
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                for name in body_names
                if name.startswith("FL_foot") or name.startswith("FR_foot")
            ]
            self.front_feet_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.front_feet_indices

    def _get_rear_feet_indices(self):
        if not hasattr(self, "rear_feet_indices"):
            body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            indices = [
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                for name in body_names
                if name.startswith("RL_foot") or name.startswith("RR_foot")
            ]
            self.rear_feet_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.rear_feet_indices

    def _get_rear_calf_indices(self):
        if not hasattr(self, "rear_calf_dof_indices"):
            indices = [
                i for i, name in enumerate(self.dof_names)
                if name.startswith("RL_calf") or name.startswith("RR_calf")
            ]
            self.rear_calf_dof_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.rear_calf_dof_indices

    def _get_rear_leg_indices(self):
        if not hasattr(self, "rear_leg_dof_indices"):
            indices = [
                i for i, name in enumerate(self.dof_names)
                if name.startswith("RL_thigh")
                or name.startswith("RR_thigh")
                or name.startswith("RL_calf")
                or name.startswith("RR_calf")
            ]
            self.rear_leg_dof_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.rear_leg_dof_indices

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        previous = self.commands[env_ids, :3].clone()
        ranges = self._active_command_ranges()
        self.commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0], ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0], ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                ranges["heading"][0], ranges["heading"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                ranges["ang_vel_yaw"][0], ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
            if getattr(self.cfg.commands, "observe_heading_error", False):
                self.commands[env_ids, 3] = self.rpy[env_ids, 2]
            sample = torch.rand(len(env_ids), device=self.device)
            p_yaw = getattr(self.cfg.commands, "pure_yaw_probability", 0.0)
            p_stand = getattr(self.cfg.commands, "stand_probability", 0.0)
            p_lat = getattr(self.cfg.commands, "pure_lateral_probability", 0.0)
            p_sag = getattr(self.cfg.commands, "pure_sagittal_probability", 0.0)
            pure_yaw = sample < p_yaw
            stand = (sample >= p_yaw) & (sample < p_yaw + p_stand)
            pure_lateral = (sample >= p_yaw + p_stand) & (sample < p_yaw + p_stand + p_lat)
            pure_sagittal = ((sample >= p_yaw + p_stand + p_lat)
                             & (sample < p_yaw + p_stand + p_lat + p_sag))
            self.commands[env_ids[pure_yaw], :2] = 0.0
            self.commands[env_ids[stand], :3] = 0.0
            self.commands[env_ids[pure_lateral], 0] = 0.0
            self.commands[env_ids[pure_lateral], 2] = 0.0
            self.commands[env_ids[pure_sagittal], 1:3] = 0.0

            hard_probability = getattr(
                self.cfg.commands, "hard_transition_probability", 0.0
            )
            if hard_probability > 0.0:
                hard = (
                    torch.rand(len(env_ids), device=self.device)
                    < hard_probability
                ) & (torch.linalg.norm(previous, dim=1) > 0.05)
                hard_ids = env_ids[hard]
                if len(hard_ids) > 0:
                    factor = torch_rand_float(
                        0.75, 1.10, (len(hard_ids), 1), device=self.device
                    )
                    flipped = -previous[hard] * factor
                    for axis, key in enumerate(
                        ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
                    ):
                        flipped[:, axis] = flipped[:, axis].clip(
                            ranges[key][0], ranges[key][1]
                        )
                    self.commands[hard_ids, :3] = flipped

        delta = self.commands[env_ids, :3] - previous
        normalized_delta = torch.stack((
            delta[:, 0] / 0.40,
            delta[:, 1] / 0.18,
            delta[:, 2] / 0.90,
        ), dim=1)
        self.command_transition_magnitude[env_ids] = torch.linalg.norm(
            normalized_delta, dim=1
        )
        self.command_transition_age[env_ids] = 0.0

    def _active_command_ranges(self):
        if not getattr(self.cfg.commands, "omni_curriculum", False):
            return self.command_ranges
        iteration = self._get_torque_curriculum_iteration()
        for stage in self.cfg.commands.omni_curriculum_stages:
            if iteration < stage["until_iteration"]:
                return stage
        return self.cfg.commands.omni_curriculum_stages[-1]

    def check_termination(self):
        super().check_termination()
        max_straight_heading_error = getattr(
            self.cfg.rewards, "terminate_straight_heading_error", None
        )
        if max_straight_heading_error is not None:
            heading_error = torch.atan2(
                torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
                torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
            )
            straight = (
                (torch.abs(self.commands[:, 0]) > 0.03)
                & (torch.abs(self.commands[:, 1]) < 0.02)
                & (torch.abs(self.commands[:, 2]) < 0.05)
            )
            self.reset_buf |= straight & (
                torch.abs(heading_error) > max_straight_heading_error
            )
        max_translation_heading_error = getattr(
            self.cfg.rewards, "terminate_translation_heading_error", None
        )
        if max_translation_heading_error is not None:
            heading_error = torch.atan2(
                torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
                torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
            )
            translation = (
                torch.linalg.norm(self.commands[:, :2], dim=1) > 0.05
            ) & (torch.abs(self.commands[:, 2]) < 0.05)
            self.reset_buf |= translation & (
                torch.abs(heading_error) > max_translation_heading_error
            )
        min_base_height = getattr(self.cfg.rewards, "min_base_height", None)
        if min_base_height is not None:
            self.reset_buf |= self.root_states[:, 2] < min_base_height

        terminate_rear_sit_pitch = getattr(self.cfg.rewards, "terminate_rear_sit_pitch", None)
        if terminate_rear_sit_pitch is not None:
            self.reset_buf |= self.rpy[:, 1] < terminate_rear_sit_pitch

        calf_angle_limits = getattr(self.cfg.rewards, "calf_angle_limits", None)
        terminate_on_calf_angle = getattr(self.cfg.rewards, "terminate_on_calf_angle", False)
        if terminate_on_calf_angle and calf_angle_limits is not None:
            calf_pos = self.dof_pos[:, self._get_calf_indices()]
            lower, upper = calf_angle_limits
            self.reset_buf |= torch.any((calf_pos < lower) | (calf_pos > upper), dim=1)

    def _reward_calf_angle_limits(self):
        calf_angle_limits = getattr(self.cfg.rewards, "calf_angle_limits", None)
        if calf_angle_limits is None:
            return torch.zeros(self.num_envs, device=self.device)
        calf_pos = self.dof_pos[:, self._get_calf_indices()]
        lower, upper = calf_angle_limits
        lower_violation = (lower - calf_pos).clip(min=0.0)
        upper_violation = (calf_pos - upper).clip(min=0.0)
        return torch.sum(lower_violation + upper_violation, dim=1)

    def _reward_rear_sit(self):
        max_rear_sit_pitch = getattr(self.cfg.rewards, "max_rear_sit_pitch", None)
        if max_rear_sit_pitch is None:
            return torch.zeros(self.num_envs, device=self.device)
        return (-self.rpy[:, 1] - max_rear_sit_pitch).clip(min=0.0)

    def _reward_backward_velocity(self):
        return (-self.base_lin_vel[:, 0]).clip(min=0.0)

    def _reward_yaw_rate(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_heading_tracking(self):
        heading_error = torch.atan2(
            torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
            torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
        )
        return torch.exp(-torch.square(heading_error) / 0.25)

    def _reward_tracking_lateral_vel(self):
        error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-error / self.cfg.rewards.lateral_tracking_sigma)

    def _reward_tracking_longitudinal_vel(self):
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.cfg.rewards.longitudinal_tracking_sigma)

    def _lateral_command_activity(self):
        return 1.0 - torch.exp(-torch.square(self.commands[:, 1])
                               / self.cfg.rewards.hip_symmetry_lateral_sigma)

    def _reward_lateral_hip_common_mode(self):
        hip_action = self.actions[:, self.hip_dof_indices]
        front_common = 0.5 * (hip_action[:, 0] + hip_action[:, 1])
        rear_common = 0.5 * (hip_action[:, 2] + hip_action[:, 3])
        return torch.square(front_common - rear_common) * self._lateral_command_activity()

    def _reward_lateral_yaw_error(self):
        yaw_rate_error = self.base_ang_vel[:, 2] - self.commands[:, 2]
        return torch.square(yaw_rate_error) * self._lateral_command_activity()

    def _reward_translation_yaw_error(self):
        planar_command_sq = torch.sum(torch.square(self.commands[:, :2]), dim=1)
        activity = 1.0 - torch.exp(-planar_command_sq / 0.0025)
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2]) * activity

    def _reward_lateral_forward_drift(self):
        """Penalize unintended forward/backward velocity during pure lateral commands."""
        lateral_sq = torch.square(self.commands[:, 1])
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        sagittal_gate = torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        lateral_activity = 1.0 - torch.exp(-lateral_sq / 0.0016)
        return torch.square(self.base_lin_vel[:, 0]) * lateral_activity * yaw_gate * sagittal_gate

    def _reward_diagonal_yaw_error(self):
        """Penalize heading drift while translating diagonally without a yaw command."""
        x_activity = 1.0 - torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        y_activity = 1.0 - torch.exp(-torch.square(self.commands[:, 1]) / 0.0016)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        return torch.square(self.base_ang_vel[:, 2]) * x_activity * y_activity * yaw_gate

    def _reward_straight_yaw_error(self):
        """Penalize yaw drift during straight forward/backward commands only."""
        sagittal_activity = 1.0 - torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        lateral_gate = torch.exp(-torch.square(self.commands[:, 1]) / 0.0016)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        return torch.square(self.base_ang_vel[:, 2]) * sagittal_activity * lateral_gate * yaw_gate
    def _reward_straight_lateral_drift(self):
        """Penalize lateral velocity drift during straight forward/backward commands."""
        sagittal_activity = 1.0 - torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        lateral_gate = torch.exp(-torch.square(self.commands[:, 1]) / 0.0016)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)

        return torch.square(self.base_lin_vel[:, 1]) * sagittal_activity * lateral_gate * yaw_gate

    def _reward_left_lateral_yaw_error(self):
        """Extra yaw drift penalty for left lateral commands.

        Your play CSV shows left lateral has much larger yaw drift than right lateral,
        so this targets cmd_vy > 0 without changing right lateral too much.
        """
        left_lateral_activity = torch.clamp(self.commands[:, 1], min=0.0)
        left_lateral_activity = 1.0 - torch.exp(-torch.square(left_lateral_activity) / 0.0016)

        sagittal_gate = torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)

        return torch.square(self.base_ang_vel[:, 2]) * left_lateral_activity * sagittal_gate * yaw_gate
    

    def _reward_lateral_velocity(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_hip_velocity(self):
        return torch.sum(torch.square(self.dof_vel[:, self.hip_dof_indices]), dim=1)

    def _reward_hip_symmetry(self):
        hip_pos = self.dof_pos[:, self.hip_dof_indices]
        front_mirror_error = torch.square(hip_pos[:, 0] + hip_pos[:, 1])
        rear_mirror_error = torch.square(hip_pos[:, 2] + hip_pos[:, 3])
        gate = torch.exp(-torch.square(self.commands[:, 1])
                         / self.cfg.rewards.hip_symmetry_lateral_sigma)
        return (front_mirror_error + rear_mirror_error) * gate

    def _reward_diagonal_joint_sync(self):
        error = torch.zeros(self.num_envs, device=self.device)
        for joint in ("thigh", "calf"):
            fl = self.leg_dof_indices["FL"][joint]
            fr = self.leg_dof_indices["FR"][joint]
            rl = self.leg_dof_indices["RL"][joint]
            rr = self.leg_dof_indices["RR"][joint]
            error += torch.square(self.dof_pos[:, fl] - self.dof_pos[:, rr])
            error += torch.square(self.dof_pos[:, fr] - self.dof_pos[:, rl])
        return error

    def _straight_motion_gate(self):
        """Only activate symmetry penalties for straight translation."""
        vx_activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, 0]) / 0.01
        )
        vy_gate = torch.exp(
            -torch.square(self.commands[:, 1]) / 0.0016
        )
        yaw_gate = torch.exp(
            -torch.square(self.commands[:, 2]) / 0.01
        )
        return vx_activity * vy_gate * yaw_gate

    def _reward_straight_policy_side_balance(self):
        """Balance left/right policy action energy during straight motion."""
        actions = self.policy_actions
        left_energy = torch.mean(
            torch.square(actions[:, self.left_dof_indices]), dim=1
        )
        right_energy = torch.mean(
            torch.square(actions[:, self.right_dof_indices]), dim=1
        )
        normalized_error = (
            (left_energy - right_energy)
            / (left_energy + right_energy + 1.0e-4)
        )
        return torch.square(normalized_error) * self._straight_motion_gate()

    def _reward_straight_torque_side_balance(self):
        """Balance normalized left/right PD torque demand."""
        torque_ratio = (
            torch.abs(self.raw_torques) / self.torque_limits.unsqueeze(0)
        )
        left_energy = torch.mean(
            torch.square(torque_ratio[:, self.left_dof_indices]), dim=1
        )
        right_energy = torch.mean(
            torch.square(torque_ratio[:, self.right_dof_indices]), dim=1
        )
        normalized_error = (
            (left_energy - right_energy)
            / (left_energy + right_energy + 1.0e-4)
        )
        return torch.square(normalized_error) * self._straight_motion_gate()

    def _reward_straight_diagonal_target_sync(self):
        """Synchronize physical targets for same-phase diagonal legs."""
        target_delta = self.target_dof_pos_rl - self.default_dof_pos
        error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        for joint in ("thigh", "calf"):
            fl = self.leg_dof_indices["FL"][joint]
            fr = self.leg_dof_indices["FR"][joint]
            rl = self.leg_dof_indices["RL"][joint]
            rr = self.leg_dof_indices["RR"][joint]
            error += torch.square(target_delta[:, fl] - target_delta[:, rr])
            error += torch.square(target_delta[:, fr] - target_delta[:, rl])
        return error * self._straight_motion_gate()

    def _reward_straight_path_lateral_velocity(self):
        """Penalize lateral velocity relative to the commanded path heading.

        Body-frame ``vy`` alone does not measure visible path drift: a small
        heading error rotates forward velocity into the world's lateral axis.
        For straight commands, ``commands[:, 3]`` is the fixed path heading.
        Transform the measured body velocity into that heading frame so the
        reward directly targets lateral displacement seen in play/MuJoCo.
        """
        heading_error = torch.atan2(
            torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
            torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
        )
        path_lateral_velocity = (
            -torch.sin(heading_error) * self.base_lin_vel[:, 0]
            + torch.cos(heading_error) * self.base_lin_vel[:, 1]
        )
        return (
            torch.square(path_lateral_velocity)
            * self._straight_motion_gate()
        )

    def _diagonal_mirror_error(self, values):
        """Return physical FL-RR / FR-RL mirror error for 12-DOF values."""
        error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        for joint in ("hip", "thigh", "calf"):
            fl = self.leg_dof_indices["FL"][joint]
            fr = self.leg_dof_indices["FR"][joint]
            rl = self.leg_dof_indices["RL"][joint]
            rr = self.leg_dof_indices["RR"][joint]
            if joint == "hip":
                # All hip axes point along +x. Mirrored left/right physical
                # motion therefore has the opposite joint-coordinate sign.
                error += torch.square(values[:, fl] + values[:, rr])
                error += torch.square(values[:, fr] + values[:, rl])
            else:
                error += torch.square(values[:, fl] - values[:, rr])
                error += torch.square(values[:, fr] - values[:, rl])
        return error

    def _reward_straight_diagonal_target_mirror(self):
        """Mirror same-phase diagonal physical joint targets."""
        target_delta = self.target_dof_pos_rl - self.default_dof_pos
        return (
            self._diagonal_mirror_error(target_delta)
            * self._straight_motion_gate()
        )

    def _reward_straight_diagonal_joint_mirror(self):
        """Mirror achieved same-phase diagonal joint motion."""
        joint_delta = self.dof_pos - self.default_dof_pos
        return (
            self._diagonal_mirror_error(joint_delta)
            * self._straight_motion_gate()
        )

    def _reward_straight_diagonal_torque_mirror(self):
        """Mirror normalized same-phase diagonal raw PD torque demand."""
        normalized_torque = self.raw_torques / self.torque_limits.unsqueeze(0)
        return (
            self._diagonal_mirror_error(normalized_torque)
            * self._straight_motion_gate()
        )

    def _reward_planar_direction_error(self):
        """Penalize velocity pointing away from the commanded xy direction."""
        command = self.commands[:, :2]
        velocity = self.base_lin_vel[:, :2]
        command_norm = torch.linalg.norm(command, dim=1)
        velocity_norm = torch.linalg.norm(velocity, dim=1)
        cosine = torch.sum(command * velocity, dim=1) / (
            command_norm * velocity_norm + 1.0e-5
        )
        activity = 1.0 - torch.exp(-torch.square(command_norm) / 0.01)
        return (1.0 - cosine.clip(-1.0, 1.0)) * activity

    def _reward_absolute_longitudinal_tracking_error(self):
        """Keep useful gradient when high-speed vx error is large."""
        return torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])

    def _reward_absolute_lateral_tracking_error(self):
        """Keep useful gradient across the full left/right speed envelope."""
        return torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])

    def _reward_absolute_yaw_tracking_error(self):
        """Keep useful gradient for fast turns and mixed-command tracking."""
        return torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])

    def _reward_translation_heading_error_abs(self):
        """Lock heading for forward, lateral, and diagonal translation."""
        heading_error = torch.atan2(
            torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
            torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
        )
        planar_activity = 1.0 - torch.exp(
            -torch.sum(torch.square(self.commands[:, :2]), dim=1) / 0.01
        )
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        return torch.abs(heading_error) * planar_activity * yaw_gate

    def _reward_pure_lateral_forward_speed(self):
        """Prevent lateral commands from leaking into forward motion."""
        lateral_activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, 1]) / 0.0036
        )
        sagittal_gate = torch.exp(-torch.square(self.commands[:, 0]) / 0.0036)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        return (
            torch.abs(self.base_lin_vel[:, 0])
            * lateral_activity
            * sagittal_gate
            * yaw_gate
        )

    def _reward_command_transition_tracking(self):
        """Reward rapid tracking after command changes and direction reversals."""
        duration = getattr(
            self.cfg.rewards, "command_transition_duration", 1.2
        )
        time_gate = (self.command_transition_age < duration).float()
        change_gate = self.command_transition_magnitude.clip(0.0, 1.0)
        vx_error = (
            self.commands[:, 0] - self.base_lin_vel[:, 0]
        ) / (torch.abs(self.commands[:, 0]) + 0.12)
        vy_error = (
            self.commands[:, 1] - self.base_lin_vel[:, 1]
        ) / (torch.abs(self.commands[:, 1]) + 0.08)
        yaw_error = (
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        ) / (torch.abs(self.commands[:, 2]) + 0.30)
        normalized_error = (
            torch.square(vx_error)
            + torch.square(vy_error)
            + torch.square(yaw_error)
        ) / 3.0
        return (
            torch.exp(-normalized_error / 0.35)
            * time_gate
            * change_gate
        )

    def _reward_diagonal_contact_sync_all(self):
        """Keep trot contact timing coordinated for every command mixture."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        mismatch = torch.zeros(self.num_envs, device=self.device)
        for first, second in (("FL", "RR"), ("FR", "RL")):
            mismatch += (
                contact[:, self.foot_slot_by_leg[first]]
                != contact[:, self.foot_slot_by_leg[second]]
            ).float()
        return mismatch

    def _reward_diagonal_foot_height_sync_all(self):
        """Synchronize swing height and vertical speed of diagonal leg pairs."""
        error = torch.zeros(self.num_envs, device=self.device)
        foot_vertical_velocity = self.feet_state[:, :, 9]
        for first, second in (("FL", "RR"), ("FR", "RL")):
            first_slot = self.foot_slot_by_leg[first]
            second_slot = self.foot_slot_by_leg[second]
            height_error = (
                self.feet_pos[:, first_slot, 2]
                - self.feet_pos[:, second_slot, 2]
            )
            velocity_error = (
                foot_vertical_velocity[:, first_slot]
                - foot_vertical_velocity[:, second_slot]
            )
            error += torch.square(height_error)
            error += 0.01 * torch.square(velocity_error)
        return error

    def _reward_translation_roll(self):
        """Keep the trunk level during forward, backward and lateral motion."""
        planar_activity = 1.0 - torch.exp(
            -torch.sum(torch.square(self.commands[:, :2]), dim=1) / 0.0025
        )
        return torch.square(self.rpy[:, 0]) * planar_activity

    def _reward_lateral_roll(self):
        """Apply an extra roll guard to the real-world-risky lateral gait."""
        lateral_activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, 1]) / 0.0016
        )
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        return torch.square(self.rpy[:, 0]) * lateral_activity * yaw_gate

    def _reward_lateral_action_magnitude(self):
        """Prefer small lateral steps instead of abrupt whole-body throws."""
        lateral_activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, 1]) / 0.0016
        )
        sagittal_gate = torch.exp(-torch.square(self.commands[:, 0]) / 0.01)
        yaw_gate = torch.exp(-torch.square(self.commands[:, 2]) / 0.01)
        hip_energy = torch.mean(
            torch.square(self.policy_actions[:, self.hip_dof_indices]), dim=1
        )
        sagittal_energy = torch.mean(
            torch.square(self.policy_actions[:, self.sagittal_dof_indices]), dim=1
        )
        return (
            2.0 * hip_energy + sagittal_energy
        ) * lateral_activity * sagittal_gate * yaw_gate

    def _reward_backward_rear_calf_fold(self):
        """Stop backward gait from folding the rear knees into a deep squat."""
        soft_limit = getattr(
            self.cfg.rewards, "backward_rear_calf_soft_limit", -1.60
        )
        rear_calf_pos = self.dof_pos[:, self._get_rear_calf_indices()]
        fold = (soft_limit - rear_calf_pos).clip(min=0.0)
        backward_activity = 1.0 - torch.exp(
            -torch.square(torch.clamp(self.commands[:, 0], max=0.0)) / 0.0025
        )
        lateral_gate = torch.exp(-torch.square(self.commands[:, 1]) / 0.0025)
        return (
            # Linear excess keeps a useful gradient close to the safety
            # boundary; the previous squared form was too weak around 0.05 rad.
            torch.mean(fold, dim=1)
            * backward_activity
            * lateral_gate
        )

    def _reward_backward_rear_target_fold(self):
        """Keep commanded rear-knee targets recoverable on weaker hardware."""
        soft_limit = getattr(
            self.cfg.rewards, "backward_rear_target_soft_limit", -1.38
        )
        rear_targets = self.target_dof_pos_rl[
            :, self._get_rear_calf_indices()
        ]
        fold = (soft_limit - rear_targets).clip(min=0.0)
        backward_activity = 1.0 - torch.exp(
            -torch.square(torch.clamp(self.commands[:, 0], max=0.0)) / 0.0025
        )
        lateral_gate = torch.exp(-torch.square(self.commands[:, 1]) / 0.0025)
        return (
            torch.mean(fold, dim=1)
            * backward_activity
            * lateral_gate
        )

    def _reward_backward_rear_action(self):
        """Keep rear-leg recovery authority during backward stepping."""
        rear_energy = torch.mean(
            torch.square(self.policy_actions[:, self.rear_sagittal_dof_indices]),
            dim=1,
        )
        backward_activity = 1.0 - torch.exp(
            -torch.square(torch.clamp(self.commands[:, 0], max=0.0)) / 0.0025
        )
        return rear_energy * backward_activity

    def _reward_transition_action_rate(self):
        """Make direction changes controlled during their first few steps."""
        duration = getattr(
            self.cfg.rewards, "transition_smooth_duration", 0.60
        )
        time_gate = (
            1.0 - self.command_transition_age / duration
        ).clip(min=0.0, max=1.0)
        change_gate = self.command_transition_magnitude.clip(0.0, 1.0)
        action_delta = self.policy_actions - self.last_policy_actions
        return (
            torch.mean(torch.square(action_delta), dim=1)
            * time_gate
            * change_gate
        )

    def _straight_contact_load(self):
        """Return foot contact data and a gate that excludes flight phases."""
        forces = self.contact_forces[:, self.feet_indices, :]
        vertical = forces[:, :, 2].clip(min=0.0)
        total_vertical = torch.sum(vertical, dim=1)
        load_gate = (total_vertical > 10.0).float()
        return forces, vertical, total_vertical, load_gate

    def _reward_straight_contact_lateral_force(self):
        """Suppress net side force instead of compensating it with body yaw."""
        forces, _, total_vertical, load_gate = self._straight_contact_load()
        normalized = torch.sum(forces[:, :, 1], dim=1) / (
            total_vertical + 1.0
        )
        return (
            torch.square(normalized)
            * load_gate
            * self._straight_motion_gate()
        )

    def _reward_straight_contact_yaw_moment(self):
        """Suppress the ground-reaction yaw moment during straight motion."""
        forces, _, total_vertical, load_gate = self._straight_contact_load()
        relative = self.feet_pos - self.root_states[:, None, :3]
        yaw_moment = torch.sum(
            relative[:, :, 0] * forces[:, :, 1]
            - relative[:, :, 1] * forces[:, :, 0],
            dim=1,
        )
        normalized = yaw_moment / (0.30 * total_vertical + 1.0)
        return (
            torch.square(normalized)
            * load_gate
            * self._straight_motion_gate()
        )

    def _reward_straight_contact_side_load_balance(self):
        """Balance total left/right support while allowing front/rear differences."""
        _, vertical, total_vertical, load_gate = self._straight_contact_load()
        left = vertical[:, [
            self.foot_slot_by_leg["FL"], self.foot_slot_by_leg["RL"]
        ]].sum(dim=1)
        right = vertical[:, [
            self.foot_slot_by_leg["FR"], self.foot_slot_by_leg["RR"]
        ]].sum(dim=1)
        normalized = (left - right) / (total_vertical + 1.0)
        return (
            torch.square(normalized)
            * load_gate
            * self._straight_motion_gate()
        )

    def _reward_straight_diagonal_contact_sync(self):
        """Keep same-phase diagonal feet in contact at the same time."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        mismatch = torch.zeros(self.num_envs, device=self.device)
        for first, second in (("FL", "RR"), ("FR", "RL")):
            mismatch += (
                contact[:, self.foot_slot_by_leg[first]]
                != contact[:, self.foot_slot_by_leg[second]]
            ).float()
        return mismatch * self._straight_motion_gate()

    def _reward_straight_lateral_speed(self):
        """Constant-gradient penalty for residual straight-line side slip."""
        return (
            torch.abs(self.base_lin_vel[:, 1])
            * self._straight_motion_gate()
        )

    def _reward_straight_heading_error(self):
        """Prevent yaw angle from compensating for lateral body velocity."""
        heading_error = torch.atan2(
            torch.sin(self.commands[:, 3] - self.rpy[:, 2]),
            torch.cos(self.commands[:, 3] - self.rpy[:, 2]),
        )
        return torch.abs(heading_error) * self._straight_motion_gate()

    def _reward_action_magnitude(self):
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_action_saturation(self):
        threshold = getattr(self.cfg.rewards, "action_saturation_threshold", 0.8)
        return torch.sum((torch.abs(self.actions) - threshold).clip(min=0.0) ** 2, dim=1)

    def _reward_policy_action_magnitude(self):
        return torch.sum(torch.square(self.policy_actions), dim=1)

    def _reward_policy_action_rate(self):
        return torch.sum(torch.square(self.policy_actions - self.last_policy_actions), dim=1)

    def _reward_policy_action_saturation(self):
        threshold = getattr(self.cfg.rewards, "action_saturation_threshold", 0.75)
        excess = (torch.abs(self.policy_actions) - threshold).clip(min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_policy_filter_gap(self):
        return torch.sum(torch.square(self.policy_filter_gap), dim=1)

    def _stand_command_gate(self):
        command_sq = torch.sum(torch.square(self.commands[:, :2]), dim=1)
        command_sq += 0.04 * torch.square(self.commands[:, 2])
        sigma = getattr(self.cfg.rewards, "stand_command_sigma", 0.0004)
        return torch.exp(-command_sq / sigma)

    def _reward_stand_action(self):
        actions = self.policy_actions if getattr(
            self.cfg.control, "filter_policy_actions", False
        ) else self.actions
        return torch.sum(torch.square(actions), dim=1) * self._stand_command_gate()

    def _reward_stand_dof_velocity(self):
        return torch.sum(torch.square(self.dof_vel), dim=1) * self._stand_command_gate()

    def _reward_torques(self):
        return torch.sum(torch.square(self.raw_torques), dim=1)

    def _reward_torque_clip(self):
        ratio = torch.abs(self.torque_clip_error) / self.torque_limits.unsqueeze(0)
        ratio = ratio.clip(max=2.0)
        return torch.mean(torch.square(ratio), dim=1) * self._torque_curriculum_multiplier(
            "torque_clip"
        )

    def _reward_torque_near_limit(self):
        ratio = torch.abs(self.raw_torques) / self.torque_limits.unsqueeze(0)
        excess = (
            ratio - self.cfg.rewards.torque_near_limit_ratio
        ).clip(min=0.0)
        return torch.mean(torch.square(excess), dim=1) * self._torque_curriculum_multiplier(
            "torque_near_limit"
        )

    def _reward_peak_torque(self):
        ratio = torch.abs(self.raw_torques) / self.torque_limits.unsqueeze(0)
        peak_ratio = torch.max(ratio, dim=1).values
        excess = (
            peak_ratio - self.cfg.rewards.peak_torque_soft_ratio
        ).clip(min=0.0)
        return torch.square(excess) * self._torque_curriculum_multiplier("peak_torque")

    def _reward_sustained_torque(self):
        ema_ratio = self.torque_ema / self.torque_limits.unsqueeze(0)
        excess = (
            ema_ratio - self.cfg.rewards.sustained_torque_ratio
        ).clip(min=0.0)
        return torch.mean(torch.square(excess), dim=1) * self._torque_curriculum_multiplier(
            "sustained_torque"
        )

    def _reward_mechanical_power(self):
        return torch.mean(torch.abs(self.raw_torques * self.dof_vel), dim=1)

    def _reward_pd_position_error_over_limit(self):
        soft_limit = self.cfg.rewards.pd_pos_err_soft_limit
        position_error = torch.abs(self.target_dof_pos_rl - self.dof_pos)
        excess = (position_error - soft_limit).clip(min=0.0)
        return torch.mean(torch.square(excess / soft_limit), dim=1)

    def _get_torque_curriculum_iteration(self):
        steps_per_iteration = getattr(
            self.cfg.rewards, "torque_curriculum_steps_per_iteration", 24
        )
        return float(self.common_step_counter) / float(steps_per_iteration)

    def _get_torque_curriculum_stage(self):
        iteration = self._get_torque_curriculum_iteration()
        if iteration >= self.cfg.rewards.torque_curriculum_stage4_iteration:
            return 4.0
        if iteration >= self.cfg.rewards.torque_curriculum_stage3_iteration:
            return 3.0
        if iteration >= self.cfg.rewards.torque_curriculum_stage2_iteration:
            return 2.0
        return 1.0

    def _torque_curriculum_multiplier(self, reward_name):
        if not getattr(self.cfg.rewards, "torque_curriculum", False):
            return 1.0

        base_scale = abs(getattr(self.cfg.rewards.scales, reward_name))
        target_scale = self._torque_curriculum_target_scale(reward_name)
        if base_scale <= 0.0:
            return 1.0
        return target_scale / base_scale

    def _torque_curriculum_target_scale(self, reward_name):
        iteration = self._get_torque_curriculum_iteration()
        base_scale = abs(getattr(self.cfg.rewards.scales, reward_name))
        stage2 = abs(self.cfg.rewards.torque_curriculum_stage2[reward_name])
        stage3 = abs(self.cfg.rewards.torque_curriculum_stage3[reward_name])
        stage4 = abs(self.cfg.rewards.torque_curriculum_stage4[reward_name])

        scale = base_scale
        scale = self._blend_torque_scale(
            scale,
            stage2,
            iteration,
            self.cfg.rewards.torque_curriculum_stage2_iteration,
        )
        scale = self._blend_torque_scale(
            scale,
            stage3,
            iteration,
            self.cfg.rewards.torque_curriculum_stage3_iteration,
        )
        scale = self._blend_torque_scale(
            scale,
            stage4,
            iteration,
            self.cfg.rewards.torque_curriculum_stage4_iteration,
        )
        return scale

    def _blend_torque_scale(self, current, target, iteration, start_iteration):
        blend_iterations = self.cfg.rewards.torque_curriculum_blend_iterations
        progress = (iteration - start_iteration) / blend_iterations
        progress = min(max(progress, 0.0), 1.0)
        return current + (target - current) * progress

    def _get_desired_foot_contacts(self):
        stance_ratio = self.cfg.rewards.gait_stance_ratio
        desired = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        desired[:] = (
            (self.gait_phase.unsqueeze(1) + self.gait_phase_offsets.unsqueeze(0)) % 1.0
        ) < stance_ratio
        return desired

    def _reward_diagonal_gait(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        desired_contact = self._get_desired_foot_contacts()
        mismatch_count = torch.sum(contact != desired_contact, dim=1)
        return torch.exp(-1.5 * mismatch_count.float())

    def _reward_swing_height(self):
        desired_swing = ~self._get_desired_foot_contacts()
        height_error = torch.square(
            self.feet_pos[:, :, 2] - self.cfg.rewards.swing_height_target
        )
        swing_score = torch.exp(-height_error / self.cfg.rewards.swing_height_sigma)
        return torch.sum(swing_score * desired_swing.float(), dim=1) / (
            torch.sum(desired_swing.float(), dim=1) + 1.0e-6
        )

    def _reward_flight(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return torch.sum(contact, dim=1) == 0

    def _reward_low_base_height(self):
        min_base_height_soft = getattr(self.cfg.rewards, "min_base_height_soft", None)
        if min_base_height_soft is None:
            return torch.zeros(self.num_envs, device=self.device)
        return (min_base_height_soft - self.root_states[:, 2]).clip(min=0.0)

    def _reward_stand_height(self):
        stand_height_sigma = getattr(self.cfg.rewards, "stand_height_sigma", None)
        if stand_height_sigma is None:
            return torch.zeros(self.num_envs, device=self.device)
        height_error = torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        return torch.exp(-height_error / stand_height_sigma)

    def _reward_stand_posture(self):
        stand_posture_sigma = getattr(self.cfg.rewards, "stand_posture_sigma", None)
        if stand_posture_sigma is None:
            return torch.zeros(self.num_envs, device=self.device)
        posture_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-posture_error / stand_posture_sigma)

    def _reward_front_feet_contact(self):
        front_feet_contact_height = getattr(self.cfg.rewards, "front_feet_contact_height", None)
        max_rear_sit_pitch = getattr(self.cfg.rewards, "max_rear_sit_pitch", 0.0)
        contact = self.contact_forces[:, self._get_front_feet_indices(), 2] > 1.0
        missing_front_feet = torch.sum((~contact).float(), dim=1)
        if front_feet_contact_height is None:
            return missing_front_feet
        low_or_sitting = torch.logical_or(
            self.root_states[:, 2] < front_feet_contact_height,
            self.rpy[:, 1] < -max_rear_sit_pitch,
        )
        return missing_front_feet * low_or_sitting.float()

    def _reward_rear_calf_fold(self):
        rear_calf_fold_limit = getattr(self.cfg.rewards, "rear_calf_fold_limit", None)
        if rear_calf_fold_limit is None:
            return torch.zeros(self.num_envs, device=self.device)
        rear_calf_pos = self.dof_pos[:, self._get_rear_calf_indices()]
        return torch.sum((rear_calf_fold_limit - rear_calf_pos).clip(min=0.0), dim=1)

    def _reward_rear_load_bias(self):
        rear_load_bias_force = getattr(self.cfg.rewards, "rear_load_bias_force", None)
        if rear_load_bias_force is None:
            return torch.zeros(self.num_envs, device=self.device)
        front_force = torch.sum(self.contact_forces[:, self._get_front_feet_indices(), 2].clip(min=0.0), dim=1)
        rear_force = torch.sum(self.contact_forces[:, self._get_rear_feet_indices(), 2].clip(min=0.0), dim=1)
        low_body = self.root_states[:, 2] < getattr(self.cfg.rewards, "front_feet_contact_height", 0.25)
        rear_bias = (rear_force - front_force - rear_load_bias_force).clip(min=0.0) / rear_load_bias_force
        return rear_bias * low_body.float()

    def _reward_rear_leg_posture(self):
        rear_leg_posture_height = getattr(self.cfg.rewards, "rear_leg_posture_height", None)
        if rear_leg_posture_height is None:
            return torch.zeros(self.num_envs, device=self.device)
        rear_leg_indices = self._get_rear_leg_indices()
        posture_error = torch.sum(torch.square(self.dof_pos[:, rear_leg_indices] - self.default_dof_pos[:, rear_leg_indices]), dim=1)
        return posture_error * (self.root_states[:, 2] < rear_leg_posture_height).float()
