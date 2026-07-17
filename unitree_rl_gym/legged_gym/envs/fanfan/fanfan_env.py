from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_from_euler_xyz,
    quat_mul,
    torch_rand_float,
)
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
            alpha = getattr(
                self,
                "episode_action_filter_alpha",
                self.cfg.control.policy_action_filter_alpha,
            )
            if torch.is_tensor(alpha) and alpha.ndim == 1:
                alpha = alpha.unsqueeze(1)
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

        # Optional real-data actuator/estimator model. All tensors are present
        # only for the new task, so legacy Fanfan tasks retain their exact path.
        self.use_real_actuator_model = bool(getattr(
            self.cfg.control, "use_real_actuator_model", False
        ))
        self.use_continuous_gait_scaling = bool(getattr(
            self.cfg.control, "use_continuous_gait_scaling", False
        ))
        self.use_real_observation_model = bool(getattr(
            self.cfg.noise, "use_real_observation_model", False
        ))
        if self.use_real_actuator_model:
            default = self.default_dof_pos.repeat(self.num_envs, 1)
            self.raw_target_dof_pos = default.clone()
            self.limited_target_dof_pos = default.clone()
            self.delayed_target_dof_pos = default.clone()
            self.motor_target_dof_pos = default.clone()
            self.final_target_velocity = torch.zeros_like(default)
            self.final_target_acceleration = torch.zeros_like(default)
            self.last_final_target_velocity = torch.zeros_like(default)
            self._last_target_control_step = -1

            max_delay_s = float(getattr(
                self.cfg.control, "command_delay_max_s", 0.030
            ))
            self.target_delay_history_len = max(
                3, int(np.ceil(max_delay_s / self.sim_params.dt)) + 3
            )
            self.target_delay_history = default.unsqueeze(0).repeat(
                self.target_delay_history_len, 1, 1
            )
            self.target_delay_write_index = 0
            self.command_delay_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

            self.actuator_tau = torch.full_like(default, 0.060)
            self.actuator_backlash = torch.zeros_like(default)
            self.kp_multiplier = torch.ones_like(default)
            self.kd_multiplier = torch.ones_like(default)
            self.joint_zero_offset = torch.zeros_like(default)
            self.episode_torque_limits = self.torque_limits.unsqueeze(0).repeat(
                self.num_envs, 1
            )
            self.episode_action_filter_alpha = torch.full(
                (self.num_envs,),
                float(self.cfg.control.policy_action_filter_alpha),
                dtype=torch.float,
                device=self.device,
            )

            self.target_rate_limit_initial = self._joint_type_tensor(
                self.cfg.control.final_target_rate_limits_initial
            )
            self.target_rate_limit_final = self._joint_type_tensor(
                self.cfg.control.final_target_rate_limits_final
            )
            self.target_accel_limit_initial = self._joint_type_tensor(
                self.cfg.control.final_target_accel_limits_initial
            )
            self.target_accel_limit_final = self._joint_type_tensor(
                self.cfg.control.final_target_accel_limits_final
            )

            self.gait_calf_amplitude_max = torch.full(
                (self.num_envs,), 0.20, dtype=torch.float, device=self.device
            )
            self.gait_stance_ratio = torch.full(
                (self.num_envs,),
                float(self.cfg.rewards.gait_stance_ratio),
                dtype=torch.float,
                device=self.device,
            )
            self.gait_period_low_speed = torch.full(
                (self.num_envs,), 0.75, dtype=torch.float, device=self.device
            )
            self.gait_period_high_speed = torch.full(
                (self.num_envs,), 0.54, dtype=torch.float, device=self.device
            )
            self.gait_backward_scale = torch.full(
                (self.num_envs,), 0.82, dtype=torch.float, device=self.device
            )

            self.raw_torque_over_counter = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.calf_error_over_counter = torch.zeros_like(
                self.raw_torque_over_counter
            )
            saturation_window = int(getattr(
                self.cfg.rewards, "torque_saturation_window_steps", 25
            ))
            self.torque_saturation_history = torch.zeros(
                self.num_envs,
                saturation_window,
                dtype=torch.float,
                device=self.device,
            )
            self.torque_saturation_history_index = 0

        if self.use_real_observation_model:
            max_delay_s = float(getattr(
                self.cfg.noise, "observation_delay_max_s", 0.060
            ))
            self.observation_history_len = max(
                4, int(np.ceil(max_delay_s / self.dt)) + 3
            )
            state_shapes = {
                "lin": (self.num_envs, 3),
                "ang": (self.num_envs, 3),
                "gravity": (self.num_envs, 3),
                "rpy": (self.num_envs, 3),
                "q": (self.num_envs, self.num_dof),
                "dq": (self.num_envs, self.num_dof),
            }
            self.observation_history = {
                name: torch.zeros(
                    self.observation_history_len,
                    *shape,
                    dtype=torch.float,
                    device=self.device,
                )
                for name, shape in state_shapes.items()
            }
            self.observation_history_write_index = 0
            self.observation_delay_steps = torch.ones(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.velocity_estimate_bias = torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device
            )
            self.velocity_estimate_walk = torch.zeros_like(
                self.velocity_estimate_bias
            )
            self.estimated_base_lin_vel = torch.zeros_like(
                self.velocity_estimate_bias
            )
            self.velocity_slip_offset = torch.zeros_like(
                self.velocity_estimate_bias
            )
            self.velocity_slip_remaining = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.velocity_hold_remaining = torch.zeros_like(
                self.velocity_slip_remaining
            )
            self.velocity_hold_zero = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.imu_install_bias = torch.zeros(
                self.num_envs, 2, dtype=torch.float, device=self.device
            )
            self.observed_contact_state = torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                dtype=torch.bool,
                device=self.device,
            )

    def _joint_type_tensor(self, values):
        """Expand a hip/thigh/calf mapping into policy joint order."""
        return torch.tensor(
            [
                next(float(value) for key, value in values.items() if key in name)
                for name in self.dof_names
            ],
            dtype=torch.float,
            device=self.device,
        )

    def _compute_torques(self, actions):
        target_dof_pos = self._compute_raw_joint_target(actions)

        if self.use_real_actuator_model:
            target_dof_pos = self._apply_real_actuator_chain(target_dof_pos)

        measured_dof_pos = self.dof_pos
        kp = self.p_gains
        kd = self.d_gains
        torque_limits = self.torque_limits.unsqueeze(0)
        if self.use_real_actuator_model:
            measured_dof_pos = self.dof_pos + self.joint_zero_offset
            kp = self.p_gains * self.kp_multiplier
            kd = self.d_gains * self.kd_multiplier
            torque_limits = self.episode_torque_limits

        raw_torques = self.motor_strength * (
            kp * (target_dof_pos - measured_dof_pos) - kd * self.dof_vel
        )
        clipped_torques = torch.maximum(
            torch.minimum(raw_torques, torque_limits), -torque_limits
        )

        self.raw_torques = raw_torques
        self.target_dof_pos_rl = target_dof_pos
        self.torque_clip_error = raw_torques - clipped_torques
        ema_alpha = float(getattr(
            self.cfg.rewards, "torque_ema_alpha", 0.98
        ))
        self.torque_ema = (
            ema_alpha * self.torque_ema
            + (1.0 - ema_alpha) * torch.abs(raw_torques)
        )
        self._update_torque_metrics(raw_torques)
        return clipped_torques

    def _compute_raw_joint_target(self, actions):
        """Build default + policy + continuously-scaled reference gait."""
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.rear_sagittal_dof_indices] = (
            actions[:, self.rear_sagittal_dof_indices]
            * self.cfg.control.rear_action_scale
        )
        actions_scaled[:, self.hip_dof_indices] = (
            actions[:, self.hip_dof_indices] * self.cfg.control.hip_action_scale
        )
        phase = (
            self.gait_phase.unsqueeze(1)
            + float(getattr(
                self.cfg.control, "gait_target_phase_lead", 0.0
            ))
            + self.gait_phase_offsets.unsqueeze(0)
        ) % 1.0
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if torch.is_tensor(stance_ratio):
            stance_ratio = stance_ratio.unsqueeze(1)
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
        dynamic_calf_amplitude = None
        dynamic_thigh_amplitude = None
        gait_amplitude_fraction = None
        if self.use_continuous_gait_scaling:
            dynamic_calf_amplitude = -self._continuous_gait_amplitude()
            # Scale the thigh reference with the same continuous gait envelope.
            # Without this, a non-zero thigh reference would keep sweeping even
            # for a stand command while the calf reference correctly becomes 0.
            amplitude_fraction = (
                torch.abs(dynamic_calf_amplitude)
                / self.gait_calf_amplitude_max.clip(min=1.0e-6)
            )
            gait_amplitude_fraction = amplitude_fraction
            sagittal_direction = torch.where(
                self.commands[:, 0] < -0.03,
                -torch.ones_like(amplitude_fraction),
                torch.ones_like(amplitude_fraction),
            )
            dynamic_thigh_amplitude = (
                float(self.cfg.rewards.gait_thigh_amplitude)
                * amplitude_fraction
                * sagittal_direction
            )
            pure_lateral = (
                (torch.abs(self.commands[:, 0]) < 0.03)
                & (torch.abs(self.commands[:, 1]) > 0.02)
            )
            lateral_thigh_scale = float(getattr(
                self.cfg.rewards, "gait_thigh_lateral_scale", 1.0
            ))
            dynamic_thigh_amplitude *= torch.where(
                pure_lateral,
                torch.full_like(amplitude_fraction, lateral_thigh_scale),
                torch.ones_like(amplitude_fraction),
            )
        foot_names = ("FL", "FR", "RL", "RR")
        for foot_slot, leg in enumerate(foot_names):
            thigh_amplitude = (
                dynamic_thigh_amplitude
                if dynamic_thigh_amplitude is not None
                else self.cfg.rewards.gait_thigh_amplitude
            )
            gait_offset[:, self.leg_dof_indices[leg]["thigh"]] = (
                thigh_amplitude * thigh_profile[:, foot_slot]
            )
            calf_amplitude = (
                dynamic_calf_amplitude
                if dynamic_calf_amplitude is not None
                else self.cfg.rewards.gait_calf_amplitude
            )
            gait_offset[:, self.leg_dof_indices[leg]["calf"]] = (
                calf_amplitude * swing_profile[:, foot_slot]
            )
        lateral_hip_amplitude = float(getattr(
            self.cfg.rewards, "gait_lateral_hip_amplitude", 0.0
        ))
        if abs(lateral_hip_amplitude) > 1.0e-8:
            lateral_command_scale = max(float(getattr(
                self.cfg.rewards, "gait_lateral_command_scale", 0.08
            )), 1.0e-6)
            lateral_fraction = torch.clamp(
                self.commands[:, 1] / lateral_command_scale, -1.0, 1.0
            )
            diagonal_scale = float(getattr(
                self.cfg.rewards, "gait_lateral_diagonal_scale", 1.0
            ))
            lateral_fraction *= torch.where(
                torch.abs(self.commands[:, 0]) > 0.05,
                torch.full_like(lateral_fraction, diagonal_scale),
                torch.ones_like(lateral_fraction),
            )
            if gait_amplitude_fraction is not None:
                lateral_fraction *= gait_amplitude_fraction
            for foot_slot, leg in enumerate(foot_names):
                hip = self.leg_dof_indices[leg]["hip"]
                gait_offset[:, hip] += (
                    lateral_hip_amplitude
                    * lateral_fraction
                    * thigh_profile[:, foot_slot]
                )
        if (
            dynamic_calf_amplitude is None
            and getattr(self.cfg.control, "gate_gait_with_command", False)
        ):
            command_energy = (
                torch.sum(torch.square(self.commands[:, :2]), dim=1)
                + 0.04 * torch.square(self.commands[:, 2])
            )
            sigma = self.cfg.control.gait_command_gate_sigma
            gait_gate = 1.0 - torch.exp(-command_energy / sigma)
            gait_offset *= gait_gate.unsqueeze(1)
        target_dof_pos = actions_scaled + gait_offset + self.default_dof_pos
        if getattr(
            self.cfg.control, "enforce_swing_calf_reference", False
        ):
            # The learned residual must not cancel the coordinated swing-lift
            # reference.  This is a one-sided minimum-flexion envelope, not a
            # scripted joint trajectory: the policy remains free to lift more
            # or to shape every other joint, while final rate/acceleration and
            # torque limits still govern the resulting target.
            reference_scale = float(getattr(
                self.cfg.control, "swing_calf_reference_scale", 1.0
            ))
            for leg in foot_names:
                calf = self.leg_dof_indices[leg]["calf"]
                leg_reference_scale = float(getattr(
                    self.cfg.control,
                    "front_swing_calf_reference_scale"
                    if leg.startswith("F")
                    else "rear_swing_calf_reference_scale",
                    reference_scale,
                ))
                reference_scale_tensor = torch.full_like(
                    gait_offset[:, calf], leg_reference_scale
                )
                if getattr(
                    self.cfg.control, "preserve_forward_gait", False
                ):
                    clean_forward = (
                        (self.commands[:, 0] > 0.03)
                        & (torch.abs(self.commands[:, 1]) < 0.02)
                        & (torch.abs(self.commands[:, 2]) < 0.10)
                    )
                    reference_scale_tensor = torch.where(
                        clean_forward,
                        torch.full_like(
                            reference_scale_tensor, reference_scale
                        ),
                        reference_scale_tensor,
                    )
                reference = (
                    self.default_dof_pos[:, calf]
                    + reference_scale_tensor * gait_offset[:, calf]
                )
                active = gait_offset[:, calf] < -1.0e-5
                target_dof_pos[:, calf] = torch.where(
                    active,
                    torch.minimum(target_dof_pos[:, calf], reference),
                    target_dof_pos[:, calf],
                )
        if (
            gait_amplitude_fraction is not None
            and getattr(
                self.cfg.control, "enforce_stance_leg_extension", False
            )
        ):
            calf_extension = float(getattr(
                self.cfg.control, "stance_calf_extension", 0.0
            )) * gait_amplitude_fraction
            thigh_extension = float(getattr(
                self.cfg.control, "stance_thigh_extension", 0.0
            )) * gait_amplitude_fraction
            if getattr(
                self.cfg.control, "preserve_forward_gait", False
            ):
                clean_forward = (
                    (self.commands[:, 0] > 0.03)
                    & (torch.abs(self.commands[:, 1]) < 0.02)
                    & (torch.abs(self.commands[:, 2]) < 0.10)
                )
                extension_gate = (~clean_forward).float()
                calf_extension *= extension_gate
                thigh_extension *= extension_gate
            else:
                extension_gate = torch.ones_like(
                    gait_amplitude_fraction
                )
            for foot_slot, leg in enumerate(foot_names):
                stance = (
                    (phase[:, foot_slot] < stance_ratio[:, 0])
                    & (extension_gate > 0.5)
                )
                calf = self.leg_dof_indices[leg]["calf"]
                thigh = self.leg_dof_indices[leg]["thigh"]
                calf_reference = (
                    self.default_dof_pos[:, calf] + calf_extension
                )
                thigh_reference = (
                    self.default_dof_pos[:, thigh] + thigh_extension
                )
                target_dof_pos[:, calf] = torch.where(
                    stance,
                    torch.maximum(target_dof_pos[:, calf], calf_reference),
                    target_dof_pos[:, calf],
                )
                target_dof_pos[:, thigh] = torch.where(
                    stance,
                    torch.minimum(target_dof_pos[:, thigh], thigh_reference),
                    target_dof_pos[:, thigh],
                )
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
        return target_dof_pos

    def _command_equivalent_speed(self):
        weights = getattr(
            self.cfg.control,
            "gait_equivalent_speed_weights",
            [1.0, 1.5, 0.18],
        )
        return torch.sqrt(
            torch.square(self.commands[:, 0] * float(weights[0]))
            + torch.square(self.commands[:, 1] * float(weights[1]))
            + torch.square(self.commands[:, 2] * float(weights[2]))
        )

    def _continuous_gait_amplitude(self):
        """Piecewise-linear gait amplitude fitted to the real CSV envelope."""
        speed = self._command_equivalent_speed()
        speed_knots = list(self.cfg.control.gait_speed_knots)
        amplitude_knots = list(self.cfg.control.gait_calf_amplitude_knots)
        amplitude = torch.zeros_like(speed)
        for index in range(len(speed_knots) - 1):
            low_speed = float(speed_knots[index])
            high_speed = float(speed_knots[index + 1])
            low_amp = float(amplitude_knots[index])
            high_amp = float(amplitude_knots[index + 1])
            ratio = ((speed - low_speed) / (high_speed - low_speed)).clip(
                0.0, 1.0
            )
            segment = low_amp + ratio * (high_amp - low_amp)
            in_segment = (speed >= low_speed) & (speed < high_speed)
            amplitude = torch.where(in_segment, segment, amplitude)
        amplitude = torch.where(
            speed >= float(speed_knots[-1]),
            torch.full_like(amplitude, float(amplitude_knots[-1])),
            amplitude,
        )
        reference_max = max(float(amplitude_knots[-1]), 1.0e-6)
        amplitude *= self.gait_calf_amplitude_max / reference_max
        amplitude = torch.where(
            self.commands[:, 0] < -0.03,
            amplitude * self.gait_backward_scale,
            amplitude,
        )
        return amplitude

    def _target_limit_progress(self):
        start = float(getattr(
            self.cfg.control, "final_target_limit_open_start_iteration", 250
        ))
        end = float(getattr(
            self.cfg.control, "final_target_limit_open_end_iteration", 950
        ))
        iteration = self._get_torque_curriculum_iteration()
        return min(max((iteration - start) / max(end - start, 1.0), 0.0), 1.0)

    def _current_final_target_limits(self):
        progress = self._target_limit_progress()
        rate = self.target_rate_limit_initial + progress * (
            self.target_rate_limit_final - self.target_rate_limit_initial
        )
        accel = self.target_accel_limit_initial + progress * (
            self.target_accel_limit_final - self.target_accel_limit_initial
        )
        rate = rate.unsqueeze(0).repeat(self.num_envs, 1)
        accel = accel.unsqueeze(0).repeat(self.num_envs, 1)
        rear_calf_scale = float(getattr(
            self.cfg.control, "rear_calf_target_rate_scale", 0.90
        ))
        rear_calf = self._get_rear_calf_indices()
        rate[:, rear_calf] *= rear_calf_scale
        accel[:, rear_calf] *= rear_calf_scale
        return rate, accel

    def _apply_final_target_limits(self, raw_target):
        rate_limit, accel_limit = self._current_final_target_limits()
        desired_velocity = (
            raw_target - self.limited_target_dof_pos
        ) / self.dt
        desired_velocity = torch.maximum(
            torch.minimum(desired_velocity, rate_limit), -rate_limit
        )
        velocity_delta = desired_velocity - self.final_target_velocity
        max_velocity_delta = accel_limit * self.dt
        velocity_delta = torch.maximum(
            torch.minimum(velocity_delta, max_velocity_delta),
            -max_velocity_delta,
        )
        next_velocity = self.final_target_velocity + velocity_delta
        next_target = self.limited_target_dof_pos + next_velocity * self.dt
        crossed = (
            (raw_target - self.limited_target_dof_pos)
            * (raw_target - next_target)
            < 0.0
        )
        next_target = torch.where(crossed, raw_target, next_target)
        next_velocity = (
            next_target - self.limited_target_dof_pos
        ) / self.dt
        self.last_final_target_velocity[:] = self.final_target_velocity
        self.final_target_velocity[:] = next_velocity
        self.final_target_acceleration[:] = (
            self.final_target_velocity - self.last_final_target_velocity
        ) / self.dt
        self.limited_target_dof_pos[:] = next_target

    def _apply_real_actuator_chain(self, raw_target):
        # The policy and gait produce one target at exactly 50 Hz. The final
        # physical target limiter is applied after their sum, once per cycle.
        if self._last_target_control_step != self.common_step_counter:
            self.raw_target_dof_pos[:] = raw_target
            self._apply_final_target_limits(raw_target)
            self._last_target_control_step = self.common_step_counter

        write_index = self.target_delay_write_index
        self.target_delay_history[write_index] = self.limited_target_dof_pos
        env_index = torch.arange(self.num_envs, device=self.device)
        read_index = (
            write_index - self.command_delay_steps
        ) % self.target_delay_history_len
        delayed_target = self.target_delay_history[read_index, env_index]
        self.delayed_target_dof_pos[:] = delayed_target
        self.target_delay_write_index = (
            write_index + 1
        ) % self.target_delay_history_len

        target_error = delayed_target - self.motor_target_dof_pos
        effective_error = torch.sign(target_error) * (
            torch.abs(target_error) - self.actuator_backlash
        ).clip(min=0.0)
        alpha = self.sim_params.dt / (self.actuator_tau + self.sim_params.dt)
        self.motor_target_dof_pos += alpha * effective_error
        return self.motor_target_dof_pos

    def _update_torque_metrics(self, raw_torques):
        abs_raw = torch.abs(raw_torques)
        torque_limits = self._active_episode_torque_limits()

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

    def _active_episode_torque_limits(self):
        if self.use_real_actuator_model:
            return self.episode_torque_limits
        return self.torque_limits.unsqueeze(0)

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
        if self.use_real_actuator_model or self.use_real_observation_model:
            self._randomize_real_hardware_episode(env_ids)
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
        self.gait_phase[env_ids] = 0.0

    def _sample_range(self, value_range, shape):
        # Isaac Gym's scripted helper only accepts a two-dimensional shape;
        # episode randomization also needs scalar and one-dimensional draws.
        return torch.empty(shape, device=self.device).uniform_(
            float(value_range[0]), float(value_range[1])
        )

    def _sample_joint_type_ranges(self, ranges, count):
        result = torch.empty(
            count, self.num_dof, dtype=torch.float, device=self.device
        )
        for dof_index, name in enumerate(self.dof_names):
            selected = next(
                value for key, value in ranges.items() if key in name
            )
            result[:, dof_index] = self._sample_range(selected, (count,))
        return result

    def _randomize_real_hardware_episode(self, env_ids):
        count = len(env_ids)
        if count == 0:
            return

        if self.use_real_actuator_model:
            domain = self.cfg.domain_rand
            control = self.cfg.control
            # Every motor is independent; the rear calves deliberately use the
            # weaker data-driven range seen on hardware.
            strength = self._sample_range(
                domain.motor_strength_range, (count, self.num_dof)
            )
            rear_calf = self._get_rear_calf_indices()
            strength[:, rear_calf] = self._sample_range(
                domain.rear_calf_strength_range,
                (count, len(rear_calf)),
            )
            self.motor_strength[env_ids] = strength
            self.kp_multiplier[env_ids] = self._sample_range(
                domain.kp_multiplier_range, (count, self.num_dof)
            )
            self.kd_multiplier[env_ids] = self._sample_range(
                domain.kd_multiplier_range, (count, self.num_dof)
            )
            self.actuator_tau[env_ids] = self._sample_joint_type_ranges(
                control.actuator_time_constant_ranges, count
            )
            self.actuator_backlash[env_ids] = self._sample_joint_type_ranges(
                domain.joint_backlash_ranges, count
            )
            self.joint_zero_offset[env_ids] = self._sample_joint_type_ranges(
                domain.joint_zero_offset_ranges, count
            )

            strict_limits = self.torque_limits.unsqueeze(0).repeat(count, 1)
            randomized_limits = self._sample_joint_type_ranges(
                control.training_torque_limit_ranges, count
            )
            self.episode_torque_limits[env_ids] = torch.minimum(
                strict_limits, randomized_limits
            )
            alpha_range = control.policy_action_filter_alpha_range
            self.episode_action_filter_alpha[env_ids] = self._sample_range(
                alpha_range, (count,)
            )

            normal_delay = self._sample_range(
                control.command_delay_range_s, (count,)
            )
            slow = torch.rand(count, device=self.device) < float(
                control.command_delay_slow_probability
            )
            slow_delay = self._sample_range(
                control.command_delay_slow_range_s, (count,)
            )
            delay = torch.where(slow, slow_delay, normal_delay)
            self.command_delay_steps[env_ids] = torch.round(
                delay / self.sim_params.dt
            ).long().clip(0, self.target_delay_history_len - 2)

            gait = self.cfg.domain_rand
            self.gait_calf_amplitude_max[env_ids] = self._sample_range(
                gait.gait_calf_amplitude_max_range, (count,)
            )
            self.gait_stance_ratio[env_ids] = self._sample_range(
                gait.gait_stance_ratio_range, (count,)
            )
            self.gait_period_low_speed[env_ids] = self._sample_range(
                gait.gait_low_speed_period_range, (count,)
            )
            self.gait_period_high_speed[env_ids] = self._sample_range(
                gait.gait_high_speed_period_range, (count,)
            )
            self.gait_backward_scale[env_ids] = self._sample_range(
                gait.gait_backward_scale_range, (count,)
            )

            default = self.default_dof_pos.repeat(count, 1)
            self.raw_target_dof_pos[env_ids] = default
            self.limited_target_dof_pos[env_ids] = default
            self.delayed_target_dof_pos[env_ids] = default
            self.motor_target_dof_pos[env_ids] = default
            self.final_target_velocity[env_ids] = 0.0
            self.final_target_acceleration[env_ids] = 0.0
            self.last_final_target_velocity[env_ids] = 0.0
            self.target_delay_history[:, env_ids] = default.unsqueeze(0)
            self.raw_torque_over_counter[env_ids] = 0
            self.calf_error_over_counter[env_ids] = 0
            self.torque_saturation_history[env_ids] = 0.0

        if self.use_real_observation_model:
            noise = self.cfg.noise
            normal_delay = self._sample_range(
                noise.observation_delay_range_s, (count,)
            )
            slow = torch.rand(count, device=self.device) < float(
                noise.observation_delay_slow_probability
            )
            slow_delay = self._sample_range(
                noise.observation_delay_slow_range_s, (count,)
            )
            rare = torch.rand(count, device=self.device) < float(
                noise.observation_delay_rare_probability
            )
            rare_delay = self._sample_range(
                noise.observation_delay_rare_range_s, (count,)
            )
            delay = torch.where(slow, slow_delay, normal_delay)
            delay = torch.where(rare, rare_delay, delay)
            self.observation_delay_steps[env_ids] = torch.ceil(
                delay / self.dt
            ).long().clip(0, self.observation_history_len - 2)
            bias_limit = float(noise.lin_vel_episode_bias_max)
            self.velocity_estimate_bias[env_ids] = self._sample_range(
                [-bias_limit, bias_limit], (count, 3)
            )
            self.velocity_estimate_bias[env_ids, 2] *= 0.5
            self.velocity_estimate_walk[env_ids] = 0.0
            self.estimated_base_lin_vel[env_ids] = 0.0
            self.velocity_slip_offset[env_ids] = 0.0
            self.velocity_slip_remaining[env_ids] = 0
            self.velocity_hold_remaining[env_ids] = 0
            self.velocity_hold_zero[env_ids] = False
            install_limit = float(noise.imu_install_bias_max_rad)
            self.imu_install_bias[env_ids] = self._sample_range(
                [-install_limit, install_limit], (count, 2)
            )
            self.observed_contact_state[env_ids] = False
            self._fill_observation_history(env_ids)

    def _process_rigid_shape_props(self, props, env_id):
        if not getattr(
            self.cfg.domain_rand, "randomize_foot_friction_independent", False
        ):
            return super()._process_rigid_shape_props(props, env_id)
        # PhysX supports at most 64K materials.  A unique floating-point
        # friction value for every shape of every one of 4096 environments
        # exceeds that cap, so use a data-rich but finite 64 x 5 table.
        if env_id == 0 or not hasattr(self, "_shape_friction_base_ids"):
            num_base_buckets = 64
            num_low_buckets = max(1, int(round(
                num_base_buckets
                * float(self.cfg.domain_rand.low_friction_probability)
            )))
            low_range = self.cfg.domain_rand.low_friction_range
            normal_range = self.cfg.domain_rand.friction_range
            self._shape_friction_bases = np.concatenate((
                np.linspace(low_range[0], low_range[1], num_low_buckets),
                np.linspace(
                    normal_range[0], normal_range[1],
                    num_base_buckets - num_low_buckets,
                ),
            ))
            low_env = np.random.random(self.num_envs) < float(
                self.cfg.domain_rand.low_friction_probability
            )
            self._shape_friction_base_ids = np.empty(
                self.num_envs, dtype=np.int64
            )
            self._shape_friction_base_ids[low_env] = np.random.randint(
                0, num_low_buckets, size=np.count_nonzero(low_env)
            )
            self._shape_friction_base_ids[~low_env] = np.random.randint(
                num_low_buckets, num_base_buckets,
                size=np.count_nonzero(~low_env),
            )
            self._shape_friction_multiplier_ids = np.random.randint(
                0, 5, size=(self.num_envs, len(props))
            )
        independent = float(
            self.cfg.domain_rand.independent_shape_friction_fraction
        )
        multipliers = np.linspace(1.0 - independent, 1.0 + independent, 5)
        base_friction = self._shape_friction_bases[
            self._shape_friction_base_ids[env_id]
        ]
        for index, prop in enumerate(props):
            multiplier = multipliers[
                self._shape_friction_multiplier_ids[env_id, index]
            ]
            prop.friction = float(base_friction * multiplier)
        return props

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if not getattr(
            self.cfg.domain_rand, "randomize_joint_friction_damping", False
        ):
            return props
        if env_id == 0 or not hasattr(self, "_nominal_dof_friction"):
            self._nominal_dof_friction = props["friction"].copy()
            self._nominal_dof_damping = props["damping"].copy()
        friction_range = self.cfg.domain_rand.joint_friction_multiplier_range
        damping_range = self.cfg.domain_rand.joint_damping_multiplier_range
        props["friction"][:] = self._nominal_dof_friction * np.random.uniform(
            friction_range[0], friction_range[1], self.num_dof
        )
        props["damping"][:] = self._nominal_dof_damping * np.random.uniform(
            damping_range[0], damping_range[1], self.num_dof
        )
        return props

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        mass_fraction_range = getattr(
            self.cfg.domain_rand, "base_mass_fraction_range", None
        )
        if mass_fraction_range is not None:
            props[0].mass *= np.random.uniform(
                mass_fraction_range[0], mass_fraction_range[1]
            )
        if getattr(self.cfg.domain_rand, "randomize_base_com", False):
            x_range = self.cfg.domain_rand.base_com_x_range
            y_range = self.cfg.domain_rand.base_com_y_range
            z_range = getattr(
                self.cfg.domain_rand, "base_com_z_range", [0.0, 0.0]
            )
            props[0].com.x += np.random.uniform(x_range[0], x_range[1])
            props[0].com.y += np.random.uniform(y_range[0], y_range[1])
            props[0].com.z += np.random.uniform(z_range[0], z_range[1])
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

        if self.use_continuous_gait_scaling:
            speed = self._command_equivalent_speed()
            blend = ((speed - 0.01) / 0.29).clip(0.0, 1.0)
            period = self.gait_period_low_speed + blend * (
                self.gait_period_high_speed - self.gait_period_low_speed
            )
            self.gait_phase = (
                self.gait_phase + self.dt / period.clip(min=0.20)
            ) % 1.0
        else:
            period = self.cfg.rewards.gait_period
            self.gait_phase = (
                (self.episode_length_buf * self.dt) % period / period
            )

    def _fill_observation_history(self, env_ids):
        if not self.use_real_observation_model or len(env_ids) == 0:
            return
        values = {
            "lin": self.base_lin_vel[env_ids],
            "ang": self.base_ang_vel[env_ids],
            "gravity": self.projected_gravity[env_ids],
            "rpy": self.rpy[env_ids],
            "q": self.dof_pos[env_ids],
            "dq": self.dof_vel[env_ids],
        }
        for name, value in values.items():
            self.observation_history[name][:, env_ids] = value.unsqueeze(0)

    def _real_observation_state(self):
        write_index = self.observation_history_write_index
        current = {
            "lin": self.base_lin_vel,
            "ang": self.base_ang_vel,
            "gravity": self.projected_gravity,
            "rpy": self.rpy,
            "q": self.dof_pos,
            "dq": self.dof_vel,
        }
        for name, value in current.items():
            self.observation_history[name][write_index] = value
        env_index = torch.arange(self.num_envs, device=self.device)
        read_index = (
            write_index - self.observation_delay_steps
        ) % self.observation_history_len
        delayed = {
            name: history[read_index, env_index].clone()
            for name, history in self.observation_history.items()
        }
        self.observation_history_write_index = (
            write_index + 1
        ) % self.observation_history_len

        noise = self.cfg.noise
        walk_sigma = float(noise.lin_vel_random_walk_sigma_per_s)
        self.velocity_estimate_walk += (
            torch.randn_like(self.velocity_estimate_walk)
            * walk_sigma
            * np.sqrt(self.dt)
        )
        walk_limit = float(noise.lin_vel_random_walk_clip)
        self.velocity_estimate_walk.clip_(-walk_limit, walk_limit)

        inactive_slip = self.velocity_slip_remaining <= 0
        start_slip = inactive_slip & (
            torch.rand(self.num_envs, device=self.device)
            < float(noise.velocity_slip_event_probability_per_step)
        )
        if torch.any(start_slip):
            slip_ids = start_slip.nonzero(as_tuple=False).flatten()
            duration = noise.velocity_slip_duration_steps
            self.velocity_slip_remaining[slip_ids] = torch.randint(
                int(duration[0]), int(duration[1]) + 1,
                (len(slip_ids),), device=self.device
            )
            magnitude = self._sample_range(
                noise.velocity_slip_error_range,
                (len(slip_ids), 2),
            )
            signs = torch.where(
                torch.rand_like(magnitude) < 0.5,
                -torch.ones_like(magnitude),
                torch.ones_like(magnitude),
            )
            self.velocity_slip_offset[slip_ids, :2] = magnitude * signs
            self.velocity_slip_offset[slip_ids, 2] = 0.0
        slip_active = self.velocity_slip_remaining > 0
        self.velocity_slip_remaining[slip_active] -= 1
        self.velocity_slip_offset[~slip_active] = 0.0

        inactive_hold = self.velocity_hold_remaining <= 0
        start_hold = inactive_hold & (
            torch.rand(self.num_envs, device=self.device)
            < float(noise.velocity_hold_event_probability_per_step)
        )
        if torch.any(start_hold):
            hold_ids = start_hold.nonzero(as_tuple=False).flatten()
            duration = noise.velocity_hold_duration_steps
            self.velocity_hold_remaining[hold_ids] = torch.randint(
                int(duration[0]), int(duration[1]) + 1,
                (len(hold_ids),), device=self.device
            )
            self.velocity_hold_zero[hold_ids] = (
                torch.rand(len(hold_ids), device=self.device)
                < float(noise.velocity_zero_fraction)
            )

        true_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        flip = torch.rand_like(true_contact.float()) < float(
            noise.contact_misclassification_probability
        )
        observed_contact = torch.logical_xor(true_contact, flip)
        contact_error = torch.mean(
            (observed_contact != true_contact).float(), dim=1
        ).unsqueeze(1)
        self.observed_contact_state[:] = observed_contact

        velocity_noise = torch.randn_like(delayed["lin"]) * float(
            noise.lin_vel_white_noise_sigma
        )
        candidate_velocity = (
            delayed["lin"]
            + self.velocity_estimate_bias
            + self.velocity_estimate_walk
            + self.velocity_slip_offset
            + velocity_noise
        )
        contact_velocity_error = torch.randn_like(candidate_velocity)
        contact_velocity_error[:, 2] *= 0.25
        candidate_velocity += (
            contact_velocity_error
            * contact_error
            * float(noise.contact_velocity_error_scale)
        )
        hold_active = self.velocity_hold_remaining > 0
        hold_zero = hold_active & self.velocity_hold_zero
        hold_old = hold_active & ~self.velocity_hold_zero
        candidate_velocity[hold_zero] = 0.0
        candidate_velocity[hold_old] = self.estimated_base_lin_vel[hold_old]
        self.velocity_hold_remaining[hold_active] -= 1
        finished_hold = self.velocity_hold_remaining <= 0
        self.velocity_hold_zero[finished_hold] = False
        self.estimated_base_lin_vel[:] = candidate_velocity

        ang_vel = delayed["ang"] + torch.randn_like(delayed["ang"]) * float(
            noise.imu_ang_vel_noise_sigma
        )
        gravity = delayed["gravity"].clone()
        gravity[:, 0] += self.imu_install_bias[:, 1]
        gravity[:, 1] -= self.imu_install_bias[:, 0]
        gravity[:, :2] += torch.randn_like(gravity[:, :2]) * float(
            noise.gravity_xy_noise_sigma
        )
        gravity /= torch.linalg.norm(gravity, dim=1, keepdim=True).clip(
            min=1.0e-6
        )
        joint_position = delayed["q"] + getattr(
            self, "joint_zero_offset", 0.0
        )
        joint_position += torch.randn_like(joint_position) * float(
            noise.joint_position_noise_sigma
        )
        velocity_error = torch.randn_like(delayed["dq"]) * float(
            noise.joint_velocity_noise_sigma
        )
        velocity_error.clip_(
            -float(noise.joint_velocity_noise_clip),
            float(noise.joint_velocity_noise_clip),
        )
        joint_velocity = delayed["dq"] + velocity_error
        return {
            "lin": candidate_velocity,
            "ang": ang_vel,
            "gravity": gravity,
            "rpy": delayed["rpy"],
            "q": joint_position,
            "dq": joint_velocity,
        }

    def compute_observations(self):
        phase_angle = 2.0 * torch.pi * self.gait_phase
        phase_obs = torch.stack((torch.sin(phase_angle), torch.cos(phase_angle)), dim=1)
        if self.use_real_observation_model:
            observed_state = self._real_observation_state()
            observed_lin_vel = observed_state["lin"]
            observed_ang_vel = observed_state["ang"]
            observed_gravity = observed_state["gravity"]
            observed_rpy = observed_state["rpy"]
            observed_dof_pos = observed_state["q"]
            observed_dof_vel = observed_state["dq"]
        else:
            observed_lin_vel = self.base_lin_vel
            observed_ang_vel = self.base_ang_vel
            observed_gravity = self.projected_gravity
            observed_rpy = self.rpy
            observed_dof_pos = self.dof_pos
            observed_dof_vel = self.dof_vel
        if getattr(self.cfg.commands, "observe_heading_error", False):
            heading_error = torch.atan2(
                torch.sin(self.commands[:, 3] - observed_rpy[:, 2]),
                torch.cos(self.commands[:, 3] - observed_rpy[:, 2]),
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
                self.commands[:, 0] - observed_lin_vel[:, 0]
            )
        if lateral_gain != 0.0:
            observed_commands[:, 1] += lateral_gain * (
                self.commands[:, 1] - observed_lin_vel[:, 1]
            )
        if yaw_gain != 0.0:
            observed_commands[:, 2] += yaw_gain * (
                self.commands[:, 2] - observed_ang_vel[:, 2]
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
                - heading_damping * observed_ang_vel[:, 2]
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
            observed_lin_vel * self.obs_scales.lin_vel,
            observed_ang_vel * self.obs_scales.ang_vel,
            observed_gravity,
            observed_commands * self.commands_scale,
            (observed_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            observed_dof_vel * self.obs_scales.dof_vel,
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
        if getattr(
            self.cfg.domain_rand, "randomize_initial_tilt", False
        ):
            tilt = self.cfg.domain_rand.initial_tilt_range_rad
            roll = self._sample_range(tilt, (len(env_ids),))
            pitch = self._sample_range(tilt, (len(env_ids),))
            yaw = torch.zeros_like(roll)
            tilt_quat = quat_from_euler_xyz(roll, pitch, yaw)
            self.root_states[env_ids, 3:7] = quat_mul(
                self.root_states[env_ids, 3:7], tilt_quat
            )
            velocity = float(getattr(
                self.cfg.domain_rand, "initial_velocity_max", 0.0
            ))
            if velocity > 0.0:
                self.root_states[env_ids, 7:9] = self._sample_range(
                    [-velocity, velocity], (len(env_ids), 2)
                )
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
        stage = self._active_command_stage()
        ranges = stage
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
            p_yaw = stage.get(
                "pure_yaw_probability",
                getattr(self.cfg.commands, "pure_yaw_probability", 0.0),
            )
            p_stand = stage.get(
                "stand_probability",
                getattr(self.cfg.commands, "stand_probability", 0.0),
            )
            p_lat = stage.get(
                "pure_lateral_probability",
                getattr(self.cfg.commands, "pure_lateral_probability", 0.0),
            )
            p_sag = stage.get(
                "pure_sagittal_probability",
                getattr(self.cfg.commands, "pure_sagittal_probability", 0.0),
            )
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

            hard_probability = stage.get(
                "hard_transition_probability",
                getattr(self.cfg.commands, "hard_transition_probability", 0.0),
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
        return self._active_command_stage()

    def _active_command_stage(self):
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

        if (
            self.use_real_actuator_model
            and getattr(
                self.cfg.rewards,
                "enable_actuator_safety_termination",
                False,
            )
        ):
            active_limits = self._active_episode_torque_limits()
            raw_ratio = torch.abs(self.raw_torques) / active_limits
            raw_over = torch.any(
                raw_ratio
                > float(self.cfg.rewards.raw_torque_termination_ratio),
                dim=1,
            )
            self.raw_torque_over_counter = torch.where(
                raw_over,
                self.raw_torque_over_counter + 1,
                torch.zeros_like(self.raw_torque_over_counter),
            )

            saturation_ratio = torch.mean(
                (torch.abs(self.raw_torques) >= active_limits).float(),
                dim=1,
            )
            history_index = self.torque_saturation_history_index
            self.torque_saturation_history[:, history_index] = saturation_ratio
            self.torque_saturation_history_index = (
                history_index + 1
            ) % self.torque_saturation_history.shape[1]
            window_saturation = torch.mean(
                self.torque_saturation_history, dim=1
            )

            calf = self._get_calf_indices()
            calf_error = torch.max(
                torch.abs(
                    self.motor_target_dof_pos[:, calf]
                    - (self.dof_pos[:, calf] + self.joint_zero_offset[:, calf])
                ),
                dim=1,
            ).values
            calf_over = calf_error > float(
                self.cfg.rewards.calf_error_termination_rad
            )
            self.calf_error_over_counter = torch.where(
                calf_over,
                self.calf_error_over_counter + 1,
                torch.zeros_like(self.calf_error_over_counter),
            )
            grace = self.episode_length_buf >= int(
                self.cfg.rewards.actuator_safety_grace_steps
            )
            actuator_failure = (
                self.raw_torque_over_counter
                >= int(self.cfg.rewards.raw_torque_termination_steps)
            )
            actuator_failure |= (
                window_saturation
                > float(self.cfg.rewards.torque_saturation_window_ratio)
            )
            actuator_failure |= (
                self.calf_error_over_counter
                >= int(self.cfg.rewards.calf_error_termination_steps)
            )
            self.reset_buf |= grace & actuator_failure

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
            torch.abs(self.raw_torques)
            / self._active_episode_torque_limits()
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
        normalized_torque = (
            self.raw_torques / self._active_episode_torque_limits()
        )
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
        ratio = (
            torch.abs(self.torque_clip_error)
            / self._active_episode_torque_limits()
        )
        ratio = ratio.clip(max=2.0)
        return torch.mean(torch.square(ratio), dim=1) * self._torque_curriculum_multiplier(
            "torque_clip"
        )

    def _reward_torque_near_limit(self):
        ratio = (
            torch.abs(self.raw_torques)
            / self._active_episode_torque_limits()
        )
        excess = (
            ratio - self.cfg.rewards.torque_near_limit_ratio
        ).clip(min=0.0)
        return torch.mean(torch.square(excess), dim=1) * self._torque_curriculum_multiplier(
            "torque_near_limit"
        )

    def _reward_peak_torque(self):
        ratio = (
            torch.abs(self.raw_torques)
            / self._active_episode_torque_limits()
        )
        peak_ratio = torch.max(ratio, dim=1).values
        excess = (
            peak_ratio - self.cfg.rewards.peak_torque_soft_ratio
        ).clip(min=0.0)
        return torch.square(excess) * self._torque_curriculum_multiplier("peak_torque")

    def _reward_sustained_torque(self):
        ema_ratio = self.torque_ema / self._active_episode_torque_limits()
        excess = (
            ema_ratio - self.cfg.rewards.sustained_torque_ratio
        ).clip(min=0.0)
        return torch.mean(torch.square(excess), dim=1) * self._torque_curriculum_multiplier(
            "sustained_torque"
        )

    def _reward_sustained_torque_max(self):
        ema_ratio = self.torque_ema / self._active_episode_torque_limits()
        peak_ratio = torch.max(ema_ratio, dim=1).values
        excess = (
            peak_ratio - self.cfg.rewards.sustained_torque_ratio
        ).clip(min=0.0)
        return torch.square(excess)

    def _reward_mechanical_power(self):
        return torch.mean(torch.abs(self.raw_torques * self.dof_vel), dim=1)

    def _reward_pd_position_error_over_limit(self):
        soft_limit_cfg = self.cfg.rewards.pd_pos_err_soft_limit
        if isinstance(soft_limit_cfg, dict):
            soft_limit = self._joint_type_tensor(soft_limit_cfg).unsqueeze(0)
        else:
            soft_limit = float(soft_limit_cfg)
        position_error = torch.abs(self.target_dof_pos_rl - self.dof_pos)
        excess = (position_error - soft_limit).clip(min=0.0)
        return torch.mean(torch.square(excess / soft_limit), dim=1)

    def _reward_final_target_velocity(self):
        if not self.use_real_actuator_model:
            return torch.zeros(self.num_envs, device=self.device)
        limits, _ = self._current_final_target_limits()
        normalized = self.final_target_velocity / limits.clip(min=1.0e-6)
        return torch.mean(torch.square(normalized), dim=1)

    def _reward_final_target_acceleration(self):
        if not self.use_real_actuator_model:
            return torch.zeros(self.num_envs, device=self.device)
        _, limits = self._current_final_target_limits()
        normalized = self.final_target_acceleration / limits.clip(min=1.0e-6)
        return torch.mean(torch.square(normalized), dim=1)

    def _reward_motor_target_tracking_error(self):
        if not self.use_real_actuator_model:
            return torch.zeros(self.num_envs, device=self.device)
        error = self.motor_target_dof_pos - (
            self.dof_pos + self.joint_zero_offset
        )
        soft_limit = self._joint_type_tensor(
            self.cfg.rewards.pd_pos_err_soft_limit
        ).unsqueeze(0)
        return torch.mean(
            torch.square(error / soft_limit.clip(min=1.0e-6)), dim=1
        )

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
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if torch.is_tensor(stance_ratio):
            stance_ratio = stance_ratio.unsqueeze(1)
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

    def _reward_swing_clearance_shortfall(self):
        """Penalize toe drag in the useful middle portion of swing.

        The old Gaussian swing-height reward was easy to ignore: a policy that
        barely moved received almost no gradient-like reward contrast.  This
        hinge term remains dense until every diagonal swing clears the target.
        """
        target = getattr(
            self.cfg.rewards, "swing_clearance_minimum", None
        )
        if target is None:
            return torch.zeros(self.num_envs, device=self.device)
        phase = (
            self.gait_phase.unsqueeze(1)
            + self.gait_phase_offsets.unsqueeze(0)
        ) % 1.0
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if torch.is_tensor(stance_ratio):
            stance_ratio = stance_ratio.unsqueeze(1)
        swing_progress = (
            (phase - stance_ratio) / (1.0 - stance_ratio)
        ).clip(0.0, 1.0)
        # Ignore lift-off and touch-down, where a low foot is physically right.
        mid_swing = (
            (phase >= stance_ratio)
            & (swing_progress >= 0.20)
            & (swing_progress <= 0.80)
        )
        shortfall = (
            float(target) - self.feet_pos[:, :, 2]
        ).clip(min=0.0)
        return torch.sum(shortfall * mid_swing.float(), dim=1) / (
            torch.sum(mid_swing.float(), dim=1) + 1.0e-6
        )

    def _reward_swing_contact(self):
        """Reject the all-feet-down shuffle during commanded locomotion."""
        desired_swing = ~self._get_desired_foot_contacts()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        command_active = (
            torch.linalg.norm(self.commands[:, :2], dim=1) > 0.015
        ) | (torch.abs(self.commands[:, 2]) > 0.08)
        swing_contact_ratio = torch.sum(
            (contact & desired_swing).float(), dim=1
        ) / (torch.sum(desired_swing.float(), dim=1) + 1.0e-6)
        return swing_contact_ratio * command_active.float()

    def _reward_command_velocity_progress(self):
        """Dense signed progress toward every commanded planar/yaw velocity."""
        floors = self.commands.new_tensor((0.10, 0.055, 0.30))
        actual = torch.stack(
            (self.base_lin_vel[:, 0], self.base_lin_vel[:, 1],
             self.base_ang_vel[:, 2]), dim=1
        )
        activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, :3] / floors)
        )
        component_progress = (
            actual * self.commands[:, :3]
            / (torch.square(self.commands[:, :3]) + torch.square(floors))
        ).clip(-1.0, 1.25)
        return torch.sum(component_progress * activity, dim=1) / (
            torch.sum(activity, dim=1) + 1.0e-6
        )

    def _reward_normalized_command_tracking(self):
        """Broad tracking basin that does not vanish when the robot is slow."""
        floors = self.commands.new_tensor((0.10, 0.055, 0.30))
        actual = torch.stack(
            (self.base_lin_vel[:, 0], self.base_lin_vel[:, 1],
             self.base_ang_vel[:, 2]), dim=1
        )
        activity = 1.0 - torch.exp(
            -torch.square(self.commands[:, :3] / floors)
        )
        normalized_error = torch.abs(
            actual - self.commands[:, :3]
        ) / (torch.abs(self.commands[:, :3]) + floors)
        score = (1.0 - normalized_error).clip(-1.0, 1.0)
        return torch.sum(score * activity, dim=1) / (
            torch.sum(activity, dim=1) + 1.0e-6
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
