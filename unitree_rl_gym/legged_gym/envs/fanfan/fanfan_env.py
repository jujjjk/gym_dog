from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.actuator_torque import (
    apply_coulomb_friction,
    limit_electromagnetic_torque,
)
from legged_gym.envs.base.contact_state import update_consecutive_true_count
from legged_gym.envs.base.terminal_snapshot import (
    RESET_REASON_BITS,
    TerminalSnapshot,
)
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    torch_rand_float,
)
import numpy as np
import torch


class FanfanRobot(LeggedRobot):
    def _recovery_curriculum_progress(self):
        end = float(getattr(
            self.cfg.domain_rand, "recovery_curriculum_end_iteration", 0.0
        ))
        if end <= 0.0:
            return 1.0
        steps_per_iteration = float(getattr(
            self.cfg.rewards, "torque_curriculum_steps_per_iteration", 24.0
        ))
        iteration = float(self.common_step_counter) / steps_per_iteration
        return min(max(iteration / end, 0.0), 1.0)

    @staticmethod
    def _curriculum_value(initial, final, progress):
        return float(initial) + progress * (float(final) - float(initial))

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

        # Lock the gross diagonal phase while leaving a small per-motor
        # residual for compensation of measured RS01 gain/tau differences.
        blend = float(getattr(
            self.cfg.control, "straight_diagonal_projection_blend", 1.0
        ))
        blend = min(max(blend, 0.0), 1.0)
        symmetric = physical + blend * (symmetric - physical)

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
        # Continuous contact duration for each physical foot.  This is
        # separate from ``feet_air_time`` because the locomotion task needs
        # to reject a foot that remains planted for multiple gait cycles.
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.all_feet_contact_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.non_diagonal_swing_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
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
        self.gait_phase_reset_offset = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.gait_transfer_wait_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        initial_transition_age = (
            0.0
            if float(getattr(
                self.cfg.control, "gait_transition_ramp_s", 0.0
            )) > 0.0
            else 10.0
        )
        self.command_transition_age = torch.full(
            (self.num_envs,),
            initial_transition_age,
            dtype=torch.float,
            device=self.device,
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
        self.raw_pd_torques = torch.zeros_like(self.torques)
        self.last_raw_pd_torques = torch.zeros_like(self.torques)
        self.motor_electromagnetic_torques = torch.zeros_like(self.torques)
        self.applied_joint_torques = torch.zeros_like(self.torques)
        # Actual per-step motor command ceiling after peak clipping and
        # continuous-torque thermal derating.  Keep this separate from the
        # episode peak limit so the policy can observe the headroom it really
        # has on the current control step.
        self.active_motor_torque_limits = torch.zeros_like(self.torques)
        self.reset_reason_bits = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_snapshot = TerminalSnapshot(
            self.num_envs,
            len(self.feet_indices),
            self.num_actions,
            self.device,
        )
        self.motor_strength = torch.ones_like(self.torques)
        self.target_dof_pos_rl = self.default_dof_pos.repeat(self.num_envs, 1)
        self.torque_clip_error = torch.zeros_like(self.torques)
        self.torque_ema = torch.zeros_like(self.torques)
        self.motor_torque_ema = torch.zeros_like(self.torques)
        self.thermal_torque_sq_ema = torch.zeros_like(self.torques)
        initial_motor_temperature = float(getattr(
            self.cfg.control, "motor_temperature_initial_c", 30.0
        ))
        self.motor_temperature_c = torch.full_like(
            self.torques, initial_motor_temperature
        )
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
            "mean_abs_motor_torque": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "motor_over_continuous_ratio": torch.zeros(
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
        self.max_thermal_torque_ratio = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.policy_actions = torch.zeros_like(self.actions)
        self.last_policy_actions = torch.zeros_like(self.actions)
        self.filtered_actions = torch.zeros_like(self.actions)
        self.filtered_action_velocity = torch.zeros_like(self.actions)
        self.policy_filter_gap = torch.zeros_like(self.actions)
        self.recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.recovery_upright_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.recovery_completion_pulse = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.post_recovery_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
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
            # Optional identified FOPDT gain. It acts on commanded motion
            # about the calibrated standing pose, rather than scaling torque
            # or absolute joint angle. Legacy tasks keep the ideal gain 1.
            self.actuator_position_gain = torch.ones_like(default)
            self.actuator_backlash = torch.zeros_like(default)
            self.actuator_coulomb_friction = torch.zeros_like(default)
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

        if self._needs_continuous_gait_buffers():
            self._init_continuous_gait_buffers()

        # Apply the identified hardware distribution before the very first
        # episode as well as after resets.  Without this, playback could run
        # indefinitely with placeholder tau/gain/friction values and would
        # not actually represent the measured RS01 motors.
        if (
            self.use_real_actuator_model
            or self.use_real_observation_model
            or self._needs_continuous_gait_buffers()
        ):
            self._randomize_real_hardware_episode(
                torch.arange(self.num_envs, device=self.device)
            )
        self.active_motor_torque_limits[:] = (
            self._active_episode_torque_limits()
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

    def post_physics_step(self):
        """Retain the previous 50 Hz raw motor request for rate penalties."""
        super().post_physics_step()
        self.last_raw_pd_torques[:] = self.raw_pd_torques

    def _add_reset_reason(self, condition, name):
        self.reset_reason_bits |= (
            condition.to(torch.long) * RESET_REASON_BITS[name]
        )

    def _capture_terminal_snapshot(self):
        peak_limits = self._active_episode_torque_limits()
        self.terminal_snapshot.capture(
            self.reset_buf,
            self.reset_reason_bits,
            self.non_diagonal_swing_counter,
            self.get_foot_contact_mask(),
            self._get_desired_foot_contacts(),
            self.gait_phase,
            self.rpy,
            self.base_ang_vel[:, 2],
            self.raw_pd_torques,
            self.motor_electromagnetic_torques,
            self.applied_joint_torques,
            peak_limits,
            self.active_motor_torque_limits,
        )

    def _compute_torques(self, actions):
        target_dof_pos = self._compute_raw_joint_target(actions)

        target_limits = getattr(
            self.cfg.control, "target_position_limits_by_joint", None
        )
        if target_limits is not None:
            lower = self._joint_type_tensor({
                key: value[0] for key, value in target_limits.items()
            }).unsqueeze(0)
            upper = self._joint_type_tensor({
                key: value[1] for key, value in target_limits.items()
            }).unsqueeze(0)
            target_dof_pos = torch.maximum(
                torch.minimum(target_dof_pos, upper), lower
            )

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
        active_limits = torque_limits
        if getattr(
            self.cfg.control, "apply_continuous_torque_derating", False
        ):
            rated = self._continuous_torque_ratings()
            thermal_ratio = torch.sqrt(
                self.thermal_torque_sq_ema.clip(min=0.0)
            )
            start_ratio = float(getattr(
                self.cfg.control,
                "continuous_derating_start_ratio",
                0.80,
            ))
            full_ratio = max(float(getattr(
                self.cfg.control,
                "continuous_derating_full_ratio",
                1.05,
            )), start_ratio + 1.0e-4)
            thermal_headroom = (
                1.0
                - (thermal_ratio - start_ratio)
                / (full_ratio - start_ratio)
            ).clip(0.0, 1.0)
            derated_limits = (
                rated + thermal_headroom * (torque_limits - rated)
            )

            # Let a resumed policy adapt before applying full thermal
            # derating. Deployment/test always uses the complete contract.
            curriculum_iterations = float(getattr(
                self.cfg.control,
                "continuous_derating_curriculum_iterations",
                0.0,
            ))
            if getattr(self.cfg.env, "test", False):
                derating_blend = 1.0
            elif curriculum_iterations > 0.0:
                steps_per_iteration = float(getattr(
                    self.cfg.rewards,
                    "torque_curriculum_steps_per_iteration",
                    24.0,
                ))
                current_iteration = (
                    float(self.common_step_counter)
                    / max(steps_per_iteration, 1.0)
                )
                derating_blend = min(
                    max(current_iteration / curriculum_iterations, 0.0),
                    1.0,
                )
            else:
                derating_blend = 1.0
            active_limits = (
                torque_limits
                + derating_blend * (derated_limits - torque_limits)
            )
        # The electromagnetic output is the current-proportional motor
        # torque after both the peak ceiling and the active thermal derating.
        # It is intentionally independent from the net joint torque below.
        active_limits = torch.minimum(active_limits, torque_limits)
        motor_electromagnetic_torques = limit_electromagnetic_torque(
            raw_torques, active_limits
        )

        torque_clip_error = raw_torques - motor_electromagnetic_torques
        self.motor_electromagnetic_torques.copy_(
            motor_electromagnetic_torques
        )
        self.active_motor_torque_limits[:] = active_limits

        # Thermal load is driven by electromagnetic torque: winding copper
        # loss scales with motor current squared. Coulomb friction is a
        # mechanical output loss and therefore must not enter this I^2 state.
        rated = self._continuous_torque_ratings()
        thermal_time_constant = max(float(getattr(
            self.cfg.control,
            "continuous_torque_thermal_time_constant_s",
            2.0,
        )), self.sim_params.dt)
        thermal_alpha = float(np.exp(
            -self.sim_params.dt / thermal_time_constant
        ))
        normalized_motor_sq = torch.square(
            torch.abs(motor_electromagnetic_torques) / rated
        )
        self.thermal_torque_sq_ema = (
            thermal_alpha * self.thermal_torque_sq_ema
            + (1.0 - thermal_alpha) * normalized_motor_sq
        )
        # RS01 feedback type 2 reports motor temperature directly. Isaac Gym
        # has no winding-temperature state, so use a slow first-order I^2
        # thermal plant for the corresponding simulated feedback channel.
        # This state is observation-only; the conservative fast RMS model
        # above remains the authoritative continuous-torque safety limiter.
        ambient_temperature = float(getattr(
            self.cfg.control, "motor_temperature_ambient_c", 25.0
        ))
        temperature_rise_at_rated = float(getattr(
            self.cfg.control, "motor_temperature_rise_at_rated_c", 55.0
        ))
        temperature_time_constant = max(float(getattr(
            self.cfg.control, "motor_temperature_time_constant_s", 180.0
        )), self.sim_params.dt)
        temperature_alpha = float(np.exp(
            -self.sim_params.dt / temperature_time_constant
        ))
        temperature_target = (
            ambient_temperature
            + temperature_rise_at_rated * normalized_motor_sq
        ).clip(max=float(getattr(
            self.cfg.control, "motor_temperature_protection_c", 103.0
        )))
        self.motor_temperature_c = (
            temperature_alpha * self.motor_temperature_c
            + (1.0 - temperature_alpha) * temperature_target
        )
        motor_ema_alpha = float(getattr(
            self.cfg.rewards, "motor_torque_ema_alpha", 0.99
        ))
        self.motor_torque_ema = (
            motor_ema_alpha * self.motor_torque_ema
            + (1.0 - motor_ema_alpha)
            * torch.abs(motor_electromagnetic_torques)
        )

        applied_joint_torques = motor_electromagnetic_torques.clone()
        if self.use_real_actuator_model:
            # Limit the commanded motor output first. Internal Coulomb
            # friction then reduces the torque delivered to the joint; it
            # must not create hidden command headroom above the safety cap.
            smoothing = float(getattr(
                self.cfg.control,
                "coulomb_friction_velocity_smoothing_rad_s",
                0.05,
            ))
            applied_joint_torques = apply_coulomb_friction(
                motor_electromagnetic_torques,
                self.dof_vel,
                self.actuator_coulomb_friction,
                smoothing,
            )

        self.raw_pd_torques.copy_(raw_torques)
        self.applied_joint_torques.copy_(applied_joint_torques)
        self.target_dof_pos_rl = target_dof_pos
        self.torque_clip_error = torque_clip_error
        ema_alpha = float(getattr(
            self.cfg.rewards, "torque_ema_alpha", 0.98
        ))
        self.torque_ema = (
            ema_alpha * self.torque_ema
            + (1.0 - ema_alpha) * torch.abs(raw_torques)
        )
        self._update_torque_metrics(raw_torques)
        return applied_joint_torques

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
        # Optional per-joint-type authority is useful when a direct policy
        # saturates only the sagittal motors. It remains a pure 12-output
        # mapping; no trajectory or cross-joint compensation is introduced.
        thigh_action_scale = getattr(
            self.cfg.control, "thigh_action_scale", None
        )
        if thigh_action_scale is not None:
            thigh_indices = [
                self.leg_dof_indices[leg]["thigh"]
                for leg in ("FL", "FR", "RL", "RR")
            ]
            actions_scaled[:, thigh_indices] = (
                actions[:, thigh_indices] * float(thigh_action_scale)
            )
        calf_action_scale = getattr(
            self.cfg.control, "calf_action_scale", None
        )
        if calf_action_scale is not None:
            calf_indices = self._get_calf_indices()
            actions_scaled[:, calf_indices] = (
                actions[:, calf_indices] * float(calf_action_scale)
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
        if getattr(self.cfg.control, "use_fast_swing_profile", False):
            shape = getattr(
                self.cfg.control, "fast_swing_profile_shape", "sine"
            )
            if shape == "sine":
                # Fanfan-rate arch: faster off/on the floor than
                # sin(pi*smoothstep(s)), without a long airborne plateau.
                swing_profile = torch.sin(
                    torch.pi * swing_progress
                ) * (phase >= stance_ratio)
            elif shape == "plateau":
                lift_fraction = float(getattr(
                    self.cfg.control, "swing_lift_fraction", 0.30
                ))
                lower_start = float(getattr(
                    self.cfg.control, "swing_lower_start_fraction", 0.65
                ))
                lift_fraction = min(max(lift_fraction, 0.10), 0.45)
                lower_start = min(
                    max(lower_start, lift_fraction + 0.10), 0.90
                )
                rise_progress = (
                    swing_progress / lift_fraction
                ).clip(0.0, 1.0)
                rise = rise_progress * rise_progress * (
                    3.0 - 2.0 * rise_progress
                )
                fall_progress = (
                    (swing_progress - lower_start) / (1.0 - lower_start)
                ).clip(0.0, 1.0)
                fall = 1.0 - (
                    fall_progress * fall_progress
                    * (3.0 - 2.0 * fall_progress)
                )
                swing_profile = torch.minimum(rise, fall) * (
                    phase >= stance_ratio
                )
            else:
                raise ValueError(
                    f"Unknown fast_swing_profile_shape: {shape}"
                )
        else:
            swing_profile = torch.sin(torch.pi * smooth_swing) * (
                phase >= stance_ratio
            )
        stance_progress = (phase / stance_ratio).clip(0.0, 1.0)
        thigh_profile = torch.where(
            phase < stance_ratio,
            -1.0 + 2.0 * stance_progress,
            1.0 - 2.0 * smooth_swing,
        )

        gait_offset = torch.zeros_like(actions_scaled)
        dynamic_calf_amplitude = None
        dynamic_thigh_amplitude = None
        dynamic_swing_thigh_lift = 0.0
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
            # Flexing this dog's calf also moves the toe forward.  Add the
            # corresponding positive thigh motion at mid-swing so the toe
            # rises approximately vertically instead of dragging along the
            # floor.  The calibrated default pose Jacobian gives about
            # +0.17 rad thigh for -0.30 rad calf at a 4 cm lift.
            dynamic_swing_thigh_lift = (
                float(getattr(
                    self.cfg.rewards,
                    "gait_swing_thigh_lift_amplitude",
                    0.0,
                ))
                * amplitude_fraction
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
        for leg in foot_names:
            # ``phase`` follows Isaac Gym's rigid-body/foot order, which is
            # not guaranteed to be FL, FR, RL, RR across assets. Looking it
            # up by leg name keeps the physical swing reference aligned with
            # the contact-phase reward.
            foot_slot = self.foot_slot_by_leg[leg]
            thigh_amplitude = (
                dynamic_thigh_amplitude
                if dynamic_thigh_amplitude is not None
                else self.cfg.rewards.gait_thigh_amplitude
            )
            gait_offset[:, self.leg_dof_indices[leg]["thigh"]] = (
                thigh_amplitude * thigh_profile[:, foot_slot]
                + dynamic_swing_thigh_lift * swing_profile[:, foot_slot]
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
        # Robot-specific controllers may replace the generic joint-space
        # reference while retaining every downstream residual, actuator and
        # safety guard.  The RS01 dog uses this hook for foot-space CPG + IK.
        specialized_gait = getattr(
            self, "_compute_specialized_gait_offset", None
        )
        if callable(specialized_gait):
            replacement = specialized_gait(
                phase=phase,
                stance_ratio=stance_ratio,
                gait_amplitude_fraction=gait_amplitude_fraction,
            )
            if replacement is not None:
                if replacement.shape != gait_offset.shape:
                    raise ValueError(
                        "specialized gait offset must have shape "
                        f"{tuple(gait_offset.shape)}, got "
                        f"{tuple(replacement.shape)}"
                    )
                gait_offset = replacement
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
        if (
            self.use_real_actuator_model
            and getattr(
                self.cfg.control,
                "compensate_identified_position_gain_in_gait",
                False,
            )
        ):
            # Keep the measured FOPDT gain downstream, but pre-scale the
            # deterministic reference so unequal RS01 gains do not turn one
            # shared diagonal trajectory into unequal physical toe strides.
            # Learned residuals are intentionally left unmodified.
            gait_offset = gait_offset / self.actuator_position_gain.clip(
                min=0.75, max=1.25
            )
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
            for leg in foot_names:
                foot_slot = self.foot_slot_by_leg[leg]
                stance = (
                    (phase[:, foot_slot] < stance_ratio[:, 0])
                    & (extension_gate > 0.5)
                )
                calf = self.leg_dof_indices[leg]["calf"]
                thigh = self.leg_dof_indices[leg]["thigh"]
                calf_reference = (
                    self.default_dof_pos[:, calf]
                    + (
                        gait_offset[:, calf]
                        if getattr(
                            self.cfg.control,
                            "stance_guard_preserve_gait_reference",
                            False,
                        )
                        else 0.0
                    )
                    + calf_extension
                )
                thigh_reference = (
                    self.default_dof_pos[:, thigh]
                    + (
                        gait_offset[:, thigh]
                        if getattr(
                            self.cfg.control,
                            "stance_guard_preserve_gait_reference",
                            False,
                        )
                        else 0.0
                    )
                    + thigh_extension
                )
                target_dof_pos[:, calf] = torch.where(
                    stance,
                    torch.maximum(target_dof_pos[:, calf], calf_reference),
                    target_dof_pos[:, calf],
                )
                if getattr(
                    self.cfg.control,
                    "enforce_stance_thigh_reference",
                    True,
                ):
                    target_dof_pos[:, thigh] = torch.where(
                        stance,
                        torch.minimum(target_dof_pos[:, thigh], thigh_reference),
                        target_dof_pos[:, thigh],
                    )
        if getattr(
            self.cfg.control, "use_active_diagonal_load_transfer", False
        ):
            # During double support, extend only the newly landed diagonal and
            # only by its measured load deficit. This makes it take over body
            # weight before the old diagonal is released, without a fixed
            # high-torque push when it is already sufficiently loaded.
            overlap = (stance_ratio[:, 0] - 0.5).clip(
                min=1.0e-4, max=0.49
            )
            clock = self.gait_phase
            fl_rr_window = clock < overlap
            fr_rl_window = (
                (clock >= 0.5) & (clock < 0.5 + overlap)
            )
            progress = torch.where(
                fl_rr_window,
                clock / overlap,
                (clock - 0.5) / overlap,
            ).clip(0.0, 1.0)
            progress = progress * progress * (3.0 - 2.0 * progress)

            force = self.contact_forces[:, self.feet_indices, 2].clip(min=0.0)
            fl_rr_force = (
                force[:, self.foot_slot_by_leg["FL"]]
                + force[:, self.foot_slot_by_leg["RR"]]
            )
            fr_rl_force = (
                force[:, self.foot_slot_by_leg["FR"]]
                + force[:, self.foot_slot_by_leg["RL"]]
            )
            new_pair_force = torch.where(
                fl_rr_window, fl_rr_force, fr_rl_force
            )
            nominal_weight = float(getattr(
                self.cfg.rewards, "transition_nominal_weight_n", 115.1
            ))
            final_fraction = float(getattr(
                self.cfg.control,
                "active_transfer_target_weight_fraction",
                0.55,
            ))
            desired_force = progress * final_fraction * nominal_weight
            deficit = (
                (desired_force - new_pair_force)
                / max(final_fraction * nominal_weight, 1.0)
            ).clip(0.0, 1.0)
            extension = float(getattr(
                self.cfg.control,
                "active_transfer_max_calf_extension_rad",
                0.035,
            )) * deficit
            if gait_amplitude_fraction is not None:
                extension *= gait_amplitude_fraction
            for pair, pair_window in (
                (("FL", "RR"), fl_rr_window),
                (("FR", "RL"), fr_rl_window),
            ):
                for leg in pair:
                    calf = self.leg_dof_indices[leg]["calf"]
                    transfer_reference = (
                        self.default_dof_pos[:, calf] + extension
                    )
                    target_dof_pos[:, calf] = torch.where(
                        pair_window,
                        torch.maximum(
                            target_dof_pos[:, calf], transfer_reference
                        ),
                        target_dof_pos[:, calf],
                    )
        if getattr(
            self.cfg.control,
            "gate_swing_on_opposite_diagonal_support",
            False,
        ):
            # Joint-target symmetry does not by itself guarantee a walking
            # contact sequence: a landing impulse can unload both rear (or
            # both front) feet. Do not flex a scheduled swing diagonal until
            # the opposite physical diagonal is carrying a useful load.
            fl_rr_support, fr_rl_support = self._get_diagonal_support_scores()
            support_floor = float(getattr(
                self.cfg.control,
                "opposite_diagonal_support_floor_score",
                0.25,
            ))
            support_full = max(float(getattr(
                self.cfg.control,
                "opposite_diagonal_support_full_score",
                0.75,
            )), support_floor + 1.0e-4)
            locomotion = self._stand_command_gate() < 0.5
            hold_thigh = bool(getattr(
                self.cfg.control,
                "hold_blocked_swing_thigh",
                True,
            ))
            for swing_pair, support_score in (
                (("FL", "RR"), fr_rl_support),
                (("FR", "RL"), fl_rr_support),
            ):
                support_gate = (
                    (support_score - support_floor)
                    / (support_full - support_floor)
                ).clip(0.0, 1.0)
                support_gate = support_gate * support_gate * (
                    3.0 - 2.0 * support_gate
                )
                for leg in swing_pair:
                    foot_slot = self.foot_slot_by_leg[leg]
                    scheduled_swing = phase[:, foot_slot] >= stance_ratio[:, 0]
                    gated = scheduled_swing & locomotion
                    calf = self.leg_dof_indices[leg]["calf"]
                    calf_guard = torch.maximum(
                        target_dof_pos[:, calf],
                        self.default_dof_pos[:, calf],
                    )
                    target_dof_pos[:, calf] = torch.where(
                        gated,
                        calf_guard + support_gate * (
                            target_dof_pos[:, calf] - calf_guard
                        ),
                        target_dof_pos[:, calf],
                    )
                    if hold_thigh:
                        thigh = self.leg_dof_indices[leg]["thigh"]
                        thigh_guard = torch.minimum(
                            target_dof_pos[:, thigh],
                            self.default_dof_pos[:, thigh],
                        )
                        target_dof_pos[:, thigh] = torch.where(
                            gated,
                            thigh_guard + support_gate * (
                                target_dof_pos[:, thigh] - thigh_guard
                            ),
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
        ramp_duration = float(getattr(
            self.cfg.control, "gait_transition_ramp_s", 0.0
        ))
        if ramp_duration > 0.0:
            ramp = (self.command_transition_age / ramp_duration).clip(0.0, 1.0)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            amplitude *= ramp
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

        identified_target = self.default_dof_pos + (
            self.actuator_position_gain
            * (delayed_target - self.default_dof_pos)
        )
        target_error = identified_target - self.motor_target_dof_pos
        effective_error = torch.sign(target_error) * (
            torch.abs(target_error) - self.actuator_backlash
        ).clip(min=0.0)
        alpha = self.sim_params.dt / (self.actuator_tau + self.sim_params.dt)
        self.motor_target_dof_pos += alpha * effective_error
        return self.motor_target_dof_pos

    def _update_torque_metrics(self, raw_torques):
        abs_raw = torch.abs(raw_torques)
        abs_motor = torch.abs(self.motor_electromagnetic_torques)
        torque_limits = self._active_episode_torque_limits()
        continuous_ratings = self._continuous_torque_ratings()

        self.torque_metric_count += 1.0
        self.max_abs_raw_torque = torch.maximum(
            self.max_abs_raw_torque, torch.max(abs_raw, dim=1).values
        )
        self.torque_metric_sums["mean_abs_raw_torque"] += torch.mean(abs_raw, dim=1)
        self.torque_metric_sums["torque_saturation_ratio"] += torch.mean(
            (abs_raw >= torque_limits).float(), dim=1
        )
        self.torque_metric_sums["mean_abs_motor_torque"] += torch.mean(
            abs_motor, dim=1
        )
        self.torque_metric_sums["motor_over_continuous_ratio"] += torch.mean(
            (abs_motor > continuous_ratings).float(), dim=1
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
        self.max_thermal_torque_ratio = torch.maximum(
            self.max_thermal_torque_ratio,
            torch.max(
                torch.sqrt(self.thermal_torque_sq_ema.clip(min=0.0)),
                dim=1,
            ).values,
        )

    def _active_episode_torque_limits(self):
        if self.use_real_actuator_model:
            return self.episode_torque_limits
        return self.torque_limits.unsqueeze(0)

    def _continuous_torque_ratings(self):
        ratings = getattr(
            self.cfg.control, "continuous_torque_limits_by_joint", None
        )
        if ratings is None:
            ratings = getattr(
                self.cfg.rewards, "continuous_torque_limits_by_joint", None
            )
        if ratings is None:
            return self._active_episode_torque_limits()
        return self._joint_type_tensor(ratings).unsqueeze(0)

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
        if (
            self.use_real_actuator_model
            or self.use_real_observation_model
            or self._needs_continuous_gait_buffers()
        ):
            self._randomize_real_hardware_episode(env_ids)
        metric_count = self.torque_metric_count[env_ids].clip(min=1.0)
        self.extras["episode"]["max_abs_raw_torque"] = torch.mean(
            self.max_abs_raw_torque[env_ids]
        )
        self.extras["episode"]["max_thermal_torque_ratio"] = torch.mean(
            self.max_thermal_torque_ratio[env_ids]
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
        self.motor_torque_ema[env_ids] = 0.0
        initial_thermal_ratio = float(getattr(
            self.cfg.control,
            "continuous_torque_initial_thermal_ratio",
            0.0,
        ))
        self.thermal_torque_sq_ema[env_ids] = (
            initial_thermal_ratio * initial_thermal_ratio
        )
        self.motor_temperature_c[env_ids] = float(getattr(
            self.cfg.control, "motor_temperature_initial_c", 30.0
        ))
        self.torque_clip_error[env_ids] = 0.0
        self.raw_pd_torques[env_ids] = 0.0
        self.last_raw_pd_torques[env_ids] = 0.0
        self.motor_electromagnetic_torques[env_ids] = 0.0
        self.applied_joint_torques[env_ids] = 0.0
        self.active_motor_torque_limits[env_ids] = (
            self._active_episode_torque_limits()[env_ids]
            if self.use_real_actuator_model
            else self._active_episode_torque_limits()
        )
        self.target_dof_pos_rl[env_ids] = self.default_dof_pos
        self.policy_actions[env_ids] = 0.0
        self.last_policy_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.filtered_action_velocity[env_ids] = 0.0
        self.policy_filter_gap[env_ids] = 0.0
        self.feet_contact_time[env_ids] = 0.0
        self.all_feet_contact_time[env_ids] = 0.0
        self.non_diagonal_swing_counter[env_ids] = 0
        self.max_abs_raw_torque[env_ids] = 0.0
        self.max_thermal_torque_ratio[env_ids] = initial_thermal_ratio
        self.torque_metric_count[env_ids] = 0.0
        for values in self.torque_metric_sums.values():
            values[env_ids] = 0.0
        self.command_transition_age[env_ids] = 0.0
        self.command_transition_magnitude[env_ids] = 0.0
        self.gait_phase[env_ids] = 0.0
        self.gait_phase_reset_offset[env_ids] = 0.0
        self.gait_transfer_wait_steps[env_ids] = 0
        self.recovery_active[env_ids] = False
        self.recovery_upright_steps[env_ids] = 0
        self.recovery_completion_pulse[env_ids] = 0.0
        self.post_recovery_steps[env_ids] = 0

        if getattr(
            self.cfg.domain_rand, "randomize_gait_phase_on_reset", False
        ):
            self.gait_phase_reset_offset[env_ids] = torch.rand(
                len(env_ids), device=self.device
            )
            self.gait_phase[env_ids] = self.gait_phase_reset_offset[env_ids]

        if getattr(
            self.cfg.domain_rand, "randomize_previous_action", False
        ):
            progress = self._recovery_curriculum_progress()
            probability = self._curriculum_value(
                getattr(
                    self.cfg.domain_rand,
                    "abnormal_action_state_probability_initial",
                    0.10,
                ),
                getattr(
                    self.cfg.domain_rand,
                    "abnormal_action_state_probability",
                    0.35,
                ),
                progress,
            )
            selected = torch.rand(len(env_ids), device=self.device) < probability
            if torch.any(selected):
                selected_ids = env_ids[selected]
                magnitude = float(getattr(
                    self.cfg.domain_rand, "abnormal_action_magnitude", 0.70
                ))
                previous = torch.empty(
                    len(selected_ids), self.num_actions, device=self.device
                ).uniform_(-magnitude, magnitude)
                self.policy_actions[selected_ids] = previous
                self.last_policy_actions[selected_ids] = previous
                self.filtered_actions[selected_ids] = previous
                self.actions[selected_ids] = previous
                self.last_actions[selected_ids] = previous
                self.recovery_active[selected_ids] = True

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

    def _sample_joint_name_ranges(self, ranges, count):
        """Sample ranges keyed by the exact URDF joint name."""
        missing = [name for name in self.dof_names if name not in ranges]
        if missing:
            raise ValueError(
                "Missing identified actuator ranges for joints: "
                + ", ".join(missing)
            )
        result = torch.empty(
            count, self.num_dof, dtype=torch.float, device=self.device
        )
        for dof_index, name in enumerate(self.dof_names):
            result[:, dof_index] = self._sample_range(
                ranges[name], (count,)
            )
        return result

    def _needs_continuous_gait_buffers(self):
        return bool(
            self.use_continuous_gait_scaling
            or getattr(self.cfg.control, "use_rs01_diagonal_cpg", False)
        )

    def _init_continuous_gait_buffers(self):
        gait_cfg = self.cfg.domain_rand
        calf_range = gait_cfg.gait_calf_amplitude_max_range
        self.gait_calf_amplitude_max = torch.full(
            (self.num_envs,),
            0.5 * (float(calf_range[0]) + float(calf_range[1])),
            dtype=torch.float,
            device=self.device,
        )
        self.gait_stance_ratio = torch.full(
            (self.num_envs,),
            float(self.cfg.rewards.gait_stance_ratio),
            dtype=torch.float,
            device=self.device,
        )
        low_period_range = gait_cfg.gait_low_speed_period_range
        high_period_range = gait_cfg.gait_high_speed_period_range
        self.gait_period_low_speed = torch.full(
            (self.num_envs,),
            0.5 * (float(low_period_range[0]) + float(low_period_range[1])),
            dtype=torch.float,
            device=self.device,
        )
        self.gait_period_high_speed = torch.full(
            (self.num_envs,),
            0.5 * (float(high_period_range[0]) + float(high_period_range[1])),
            dtype=torch.float,
            device=self.device,
        )
        self.gait_backward_scale = torch.full(
            (self.num_envs,), 0.82, dtype=torch.float, device=self.device
        )

    def _randomize_gait_episode(self, env_ids):
        count = len(env_ids)
        if count == 0 or not self._needs_continuous_gait_buffers():
            return
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
            per_joint_tau = getattr(
                control, "actuator_time_constant_ranges_by_joint", None
            )
            if per_joint_tau is not None:
                self.actuator_tau[env_ids] = self._sample_joint_name_ranges(
                    per_joint_tau, count
                )

            per_joint_gain = getattr(
                control, "actuator_position_gain_ranges_by_joint", None
            )
            if per_joint_gain is not None:
                self.actuator_position_gain[env_ids] = (
                    self._sample_joint_name_ranges(per_joint_gain, count)
                )
            else:
                self.actuator_position_gain[env_ids] = 1.0

            reversal_ranges = getattr(
                domain, "effective_reversal_gap_ranges_by_joint", None
            )
            if reversal_ranges is not None:
                # The bench test cannot separate backlash, compliance,
                # filtering and friction. Model it only as an effective
                # reversal deadband, not as a claimed gear-lash value.
                self.actuator_backlash[env_ids] = (
                    self._sample_joint_name_ranges(reversal_ranges, count)
                )
            else:
                self.actuator_backlash[env_ids] = (
                    self._sample_joint_type_ranges(
                        domain.joint_backlash_ranges, count
                    )
                )

            friction_ranges = getattr(
                domain, "coulomb_friction_ranges_by_joint", None
            )
            if friction_ranges is not None:
                self.actuator_coulomb_friction[env_ids] = (
                    self._sample_joint_name_ranges(friction_ranges, count)
                )
            else:
                self.actuator_coulomb_friction[env_ids] = 0.0
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

        self._randomize_gait_episode(env_ids)

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
        foot_contact = self.get_foot_contact_mask()
        self.feet_contact_time[:] = torch.where(
            foot_contact,
            self.feet_contact_time + self.dt,
            torch.zeros_like(self.feet_contact_time),
        )
        all_contact = torch.all(foot_contact, dim=1)
        self.all_feet_contact_time[:] = torch.where(
            all_contact,
            self.all_feet_contact_time + self.dt,
            torch.zeros_like(self.all_feet_contact_time),
        )

        if self.use_continuous_gait_scaling:
            speed = self._command_equivalent_speed()
            blend = ((speed - 0.01) / 0.29).clip(0.0, 1.0)
            period = self.gait_period_low_speed + blend * (
                self.gait_period_high_speed - self.gait_period_low_speed
            )
            proposed_phase = (
                self.gait_phase + self.dt / period.clip(min=0.20)
            ) % 1.0
            self.gait_phase = self._contact_aware_gait_phase(proposed_phase)
        else:
            period = self.cfg.rewards.gait_period
            self.gait_phase = (
                self.episode_length_buf * self.dt / period
                + self.gait_phase_reset_offset
            ) % 1.0
        self._update_recovery_state()

    def _contact_aware_gait_phase(self, proposed_phase):
        """Hold each diagonal transfer until the landing pair takes load.

        The oscillator is allowed to wait only at the end of the double-
        support window. This preserves the commanded period everywhere else
        while enforcing the walking order: touchdown, load acceptance, then
        release of the previous diagonal. A bounded wait prevents permanent
        deadlock during early exploration.
        """
        if not getattr(
            self.cfg.control, "use_contact_aware_phase_transfer", False
        ):
            return proposed_phase

        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if not torch.is_tensor(stance_ratio):
            stance_ratio = torch.full(
                (self.num_envs,), float(stance_ratio), device=self.device
            )
        overlap = (stance_ratio - 0.5).clip(min=0.0, max=0.49)
        force = self.contact_forces[:, self.feet_indices, 2].clip(min=0.0)
        fl_rr_force = (
            force[:, self.foot_slot_by_leg["FL"]]
            + force[:, self.foot_slot_by_leg["RR"]]
        )
        fr_rl_force = (
            force[:, self.foot_slot_by_leg["FR"]]
            + force[:, self.foot_slot_by_leg["RL"]]
        )
        total_force = (fl_rr_force + fr_rl_force).clip(min=1.0)
        nominal_weight = float(getattr(
            self.cfg.rewards, "transition_nominal_weight_n", 115.1
        ))
        min_weight_fraction = float(getattr(
            self.cfg.control,
            "phase_transfer_min_pair_weight_fraction",
            0.42,
        ))
        min_load_fraction = float(getattr(
            self.cfg.control,
            "phase_transfer_min_load_fraction",
            0.58,
        ))
        fl_rr_ready = (
            (fl_rr_force >= min_weight_fraction * nominal_weight)
            & (fl_rr_force / total_force >= min_load_fraction)
        )
        fr_rl_ready = (
            (fr_rl_force >= min_weight_fraction * nominal_weight)
            & (fr_rl_force / total_force >= min_load_fraction)
        )
        # A large impulse on one toe must not release the previous diagonal.
        # Require both toes of the arriving physical diagonal to be in
        # contact and the pair to carry a useful minimum load.
        contact = self.get_foot_contact_mask()
        fl_rr_ready &= (
            contact[:, self.foot_slot_by_leg["FL"]]
            & contact[:, self.foot_slot_by_leg["RR"]]
        )
        fr_rl_ready &= (
            contact[:, self.foot_slot_by_leg["FR"]]
            & contact[:, self.foot_slot_by_leg["RL"]]
        )
        fl_rr_support, fr_rl_support = self._get_diagonal_support_scores()
        fl_rr_ready &= fl_rr_support >= 1.0
        fr_rl_ready &= fr_rl_support >= 1.0
        max_wait = int(getattr(
            self.cfg.control, "phase_transfer_max_wait_steps", 10
        ))
        locomotion = self._stand_command_gate() < 0.5

        # The joint target is intentionally advanced to compensate the
        # identified actuator delay.  Therefore the *commanded* release of
        # the old support pair happens ``phase_lead`` before the nominal
        # contact boundary.  Holding at the unshifted boundary is too late:
        # the old pair has already received a swing command.  Work on the
        # unit circle so this also handles a release boundary near phase 1.
        phase_lead = float(getattr(
            self.cfg.control, "gait_target_phase_lead", 0.0
        )) % 1.0
        first_boundary = torch.remainder(overlap - phase_lead, 1.0)
        second_boundary = torch.remainder(
            0.5 + overlap - phase_lead, 1.0
        )
        phase_increment = torch.remainder(
            proposed_phase - self.gait_phase, 1.0
        )
        first_distance = torch.remainder(
            first_boundary - self.gait_phase, 1.0
        )
        second_distance = torch.remainder(
            second_boundary - self.gait_phase, 1.0
        )
        crossing_first = (
            (phase_increment > 1.0e-7)
            & (first_distance <= phase_increment + 1.0e-7)
        )
        crossing_second = (
            (phase_increment > 1.0e-7)
            & (second_distance <= phase_increment + 1.0e-7)
        )
        waiting = locomotion & (
            (crossing_first & ~fl_rr_ready)
            | (crossing_second & ~fr_rl_ready)
        )
        force_release = self.gait_transfer_wait_steps >= max_wait
        hold = waiting & ~force_release
        held_phase = torch.where(
            crossing_first, first_boundary, second_boundary
        )
        proposed_phase = torch.where(hold, held_phase, proposed_phase)
        self.gait_transfer_wait_steps = torch.where(
            hold,
            self.gait_transfer_wait_steps + 1,
            torch.zeros_like(self.gait_transfer_wait_steps),
        )
        return proposed_phase

    def _update_recovery_state(self):
        """Track a disturbance until level, low-rate walking is restored."""
        self.recovery_completion_pulse.zero_()
        trigger_deg = float(getattr(
            self.cfg.rewards, "recovery_trigger_tilt_deg", 3.0
        ))
        upright_deg = float(getattr(
            self.cfg.rewards, "recovery_upright_tilt_deg", 2.0
        ))
        tilt = torch.linalg.norm(self.projected_gravity[:, :2], dim=1)
        self.recovery_active |= tilt > np.sin(np.deg2rad(trigger_deg))
        upright = (
            (tilt < np.sin(np.deg2rad(upright_deg)))
            & (torch.linalg.norm(self.base_ang_vel[:, :2], dim=1) < 0.45)
        )
        stable = self.recovery_active & upright
        self.recovery_upright_steps = torch.where(
            stable,
            self.recovery_upright_steps + 1,
            torch.zeros_like(self.recovery_upright_steps),
        )
        required = int(getattr(
            self.cfg.rewards, "recovery_stable_steps", 10
        ))
        completed = self.recovery_active & (
            self.recovery_upright_steps >= required
        )
        self.recovery_completion_pulse[completed] = 1.0
        self.recovery_active[completed] = False
        self.recovery_upright_steps[completed] = 0
        hold_steps = int(round(float(getattr(
            self.cfg.rewards, "post_recovery_hold_s", 2.0
        )) / self.dt))
        self.post_recovery_steps[completed] = hold_steps
        self.post_recovery_steps = torch.clamp(
            self.post_recovery_steps - 1, min=0
        )

    def _push_robots(self):
        """Push scheduled environments only, with balanced left/right draws."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        interval = max(1, int(self.cfg.domain_rand.push_interval))
        push_ids = env_ids[self.episode_length_buf % interval == 0]
        if len(push_ids) == 0:
            return
        progress = self._recovery_curriculum_progress()
        max_vel = self._curriculum_value(
            getattr(self.cfg.domain_rand, "max_push_vel_xy_initial", 0.08),
            self.cfg.domain_rand.max_push_vel_xy,
            progress,
        )
        if getattr(self.cfg.domain_rand, "lateral_push_only", False):
            self.root_states[push_ids, 7] = 0.0
            self.root_states[push_ids, 8] = torch.empty(
                len(push_ids), device=self.device
            ).uniform_(-max_vel, max_vel)
        else:
            self.root_states[push_ids, 7:9] = torch.empty(
                len(push_ids), 2, device=self.device
            ).uniform_(-max_vel, max_vel)
        max_roll_rate = float(getattr(
            self.cfg.domain_rand, "max_push_roll_rate", 0.0
        )) * progress
        if max_roll_rate > 0.0:
            self.root_states[push_ids, 10] = torch.empty(
                len(push_ids), device=self.device
            ).uniform_(-max_roll_rate, max_roll_rate)
        self.recovery_active[push_ids] = True
        env_ids_int32 = push_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
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

        true_contact = self.get_foot_contact_mask()
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
        if getattr(
            self.cfg.commands,
            "inject_straight_path_recovery_velocity",
            False,
        ):
            # Keep the observation width/checkpoint contract unchanged. For a
            # pure-straight command, reuse the existing observed vy command as
            # a bounded closed-loop return-to-path request. The physical
            # command remains vy=0; only the policy input receives this target.
            lateral_displacement, _, _ = self._straight_path_state()
            recovery_gain = float(getattr(
                self.cfg.commands,
                "straight_path_observation_gain_s",
                1.25,
            ))
            recovery_limit = float(getattr(
                self.cfg.commands,
                "straight_path_observation_max_velocity_m_s",
                0.08,
            ))
            observed_commands[:, 1] = (
                -recovery_gain * lateral_displacement
            ).clip(-recovery_limit, recovery_limit)
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
        if getattr(self.cfg.env, "observe_actuator_state", False):
            continuous = self._continuous_torque_ratings()
            motor_torque_obs = (
                self.motor_electromagnetic_torques / continuous
            ).clip(-2.5, 2.5)
            # Match the two fields available in the real RS01 type-2 feedback
            # packet: reported torque and motor temperature. Thermal RMS and
            # active derating limits remain internal safety-controller state.
            motor_temperature_obs = (
                self.motor_temperature_c / 100.0
            ).clip(-0.5, 1.5)
            self.obs_buf = torch.cat((
                self.obs_buf,
                motor_torque_obs,
                motor_temperature_obs,
            ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reset_dofs(self, env_ids):
        # Fanfan is small enough that the base task's 0.5-1.5 multiplier can
        # spawn a foot through the floor or put a calf directly on its limit.
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        if getattr(
            self.cfg.domain_rand, "randomize_asymmetric_joint_state", False
        ):
            progress = self._recovery_curriculum_progress()
            probability = self._curriculum_value(
                getattr(
                    self.cfg.domain_rand,
                    "asymmetric_joint_state_probability_initial",
                    0.10,
                ),
                getattr(
                    self.cfg.domain_rand,
                    "asymmetric_joint_state_probability",
                    0.35,
                ),
                progress,
            )
            selected = torch.rand(len(env_ids), device=self.device) < probability
            if torch.any(selected):
                selected_ids = env_ids[selected]
                ranges = getattr(
                    self.cfg.domain_rand,
                    "asymmetric_joint_offset_ranges",
                    {"hip": [-0.03, 0.03], "thigh": [-0.05, 0.05],
                     "calf": [-0.07, 0.07]},
                )
                self.dof_pos[selected_ids] += self._sample_joint_type_ranges(
                    ranges, len(selected_ids)
                )
                velocity = float(getattr(
                    self.cfg.domain_rand,
                    "asymmetric_joint_velocity_max",
                    0.12,
                ))
                self.dof_vel[selected_ids] = torch.empty(
                    len(selected_ids), self.num_dof, device=self.device
                ).uniform_(-velocity, velocity)

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
            levels = getattr(
                self.cfg.domain_rand, "initial_roll_levels_deg", None
            )
            if levels is not None:
                progress = self._recovery_curriculum_progress()
                unlocked = max(1, min(
                    len(levels), 1 + int(progress * len(levels))
                ))
                indices = torch.randint(
                    0, unlocked, (len(env_ids),), device=self.device
                )
                level_tensor = torch.tensor(
                    levels, dtype=torch.float, device=self.device
                )
                sign = torch.where(
                    torch.rand(len(env_ids), device=self.device) < 0.5,
                    -torch.ones(len(env_ids), device=self.device),
                    torch.ones(len(env_ids), device=self.device),
                )
                roll = torch.deg2rad(level_tensor[indices]) * sign
                zero_probability = self._curriculum_value(
                    getattr(
                        self.cfg.domain_rand,
                        "initial_roll_zero_probability_initial",
                        0.70,
                    ),
                    getattr(
                        self.cfg.domain_rand,
                        "initial_roll_zero_probability",
                        0.35,
                    ),
                    progress,
                )
                roll[torch.rand(len(env_ids), device=self.device)
                     < zero_probability] = 0.0
                pitch_range = getattr(
                    self.cfg.domain_rand,
                    "initial_pitch_range_rad",
                    [-0.035, 0.035],
                )
                pitch = self._sample_range(pitch_range, (len(env_ids),))
            else:
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
        self.reset_reason_bits.zero_()
        trunk_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            ) >= self.foot_contact_enter_force_n,
            dim=1,
        )
        orientation = torch.logical_or(
            torch.abs(self.rpy[:, 1]) > 1.0,
            torch.abs(self.rpy[:, 0]) > 0.8,
        )
        self._add_reset_reason(trunk_contact, "trunk_contact")
        self._add_reset_reason(orientation, "orientation")
        self._add_reset_reason(self.time_out_buf, "timeout")
        if getattr(
            self.cfg.rewards,
            "enable_flight_termination",
            False,
        ):
            locomotion = self._stand_command_gate() < 0.5
            grace = self.episode_length_buf >= int(getattr(
                self.cfg.rewards,
                "flight_termination_grace_steps",
                0,
            ))
            # Unlike other invalid multi-foot patterns, complete flight has
            # no contact ambiguity worth tolerating: one 50 Hz sample is
            # enough to reject a hopping cycle.
            flight_failure = grace & locomotion & self._get_flight_mask()
            self.reset_buf |= flight_failure
            self._add_reset_reason(flight_failure, "flight")
        if getattr(
            self.cfg.rewards,
            "enable_non_diagonal_swing_termination",
            False,
        ):
            locomotion = self._stand_command_gate() < 0.5
            invalid_swing = self._get_non_diagonal_swing_mask() & locomotion
            self.non_diagonal_swing_counter = update_consecutive_true_count(
                invalid_swing, self.non_diagonal_swing_counter
            )
            grace = self.episode_length_buf >= int(getattr(
                self.cfg.rewards,
                "non_diagonal_swing_grace_steps",
                0,
            ))
            termination_steps = int(getattr(
                self.cfg.rewards,
                "non_diagonal_swing_termination_steps",
                3,
            ))
            if getattr(self.cfg.env, "test", False):
                termination_steps = int(getattr(
                    self.cfg.rewards,
                    "non_diagonal_swing_termination_steps_test",
                    termination_steps,
                ))
            curriculum = getattr(
                self.cfg.rewards,
                "non_diagonal_termination_curriculum",
                None,
            )
            # Evaluation remains strict. During PPO continuation, first allow
            # enough contact samples to observe a complete gait cycle, then
            # tighten to the same one-frame contract used by play/deployment.
            if curriculum and not getattr(self.cfg.env, "test", False):
                iteration = self._get_torque_curriculum_iteration()
                for stage in curriculum:
                    if iteration < float(stage["until_iteration"]):
                        termination_steps = int(stage["steps"])
                        break
            sustained_invalid_swing = (
                self.non_diagonal_swing_counter >= termination_steps
            )
            illegal_contact = grace & sustained_invalid_swing
            self.reset_buf |= illegal_contact
            self._add_reset_reason(illegal_contact, "illegal_contact")
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
            straight_heading = straight & (
                torch.abs(heading_error) > max_straight_heading_error
            )
            self.reset_buf |= straight_heading
            self._add_reset_reason(straight_heading, "straight_heading")
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
            translation_heading = translation & (
                torch.abs(heading_error) > max_translation_heading_error
            )
            self.reset_buf |= translation_heading
            self._add_reset_reason(
                translation_heading, "translation_heading"
            )
        min_base_height = getattr(self.cfg.rewards, "min_base_height", None)
        if min_base_height is not None:
            low_height = self.root_states[:, 2] < min_base_height
            self.reset_buf |= low_height
            self._add_reset_reason(low_height, "low_height")

        terminate_rear_sit_pitch = getattr(self.cfg.rewards, "terminate_rear_sit_pitch", None)
        if terminate_rear_sit_pitch is not None:
            rear_sit = self.rpy[:, 1] < terminate_rear_sit_pitch
            self.reset_buf |= rear_sit
            self._add_reset_reason(rear_sit, "rear_sit")

        calf_angle_limits = getattr(self.cfg.rewards, "calf_angle_limits", None)
        terminate_on_calf_angle = getattr(self.cfg.rewards, "terminate_on_calf_angle", False)
        if terminate_on_calf_angle and calf_angle_limits is not None:
            calf_pos = self.dof_pos[:, self._get_calf_indices()]
            lower, upper = calf_angle_limits
            calf_angle = torch.any(
                (calf_pos < lower) | (calf_pos > upper), dim=1
            )
            self.reset_buf |= calf_angle
            self._add_reset_reason(calf_angle, "calf_angle")

        if (
            self.use_real_actuator_model
            and getattr(
                self.cfg.rewards,
                "enable_actuator_safety_termination",
                False,
            )
        ):
            active_limits = self._active_episode_torque_limits()
            raw_ratio = torch.abs(self.raw_pd_torques) / active_limits
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
                (torch.abs(self.raw_pd_torques) >= active_limits).float(),
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
            actuator_safety = grace & actuator_failure
            self.reset_buf |= actuator_safety
            self._add_reset_reason(actuator_safety, "actuator_safety")

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
            torch.abs(self.raw_pd_torques)
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
            self.raw_pd_torques / self._active_episode_torque_limits()
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
        contact = self.get_foot_contact_mask()
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

    def _get_diagonal_stride_sync_score(self):
        """Measure actual foot-trajectory agreement inside each diagonal.

        Equal joint targets are insufficient with the independently measured
        RS01 motor gains and delays. Compare body-frame toe displacement and
        velocity instead, so PPO can use its small per-motor residual to make
        the *physical* FL+RR and FR+RL strides agree.
        """
        foot_count = len(self.feet_indices)
        base_quat = self.base_quat[:, None, :].expand(
            -1, foot_count, -1
        ).reshape(-1, 4)
        relative_position = (
            self.feet_pos - self.root_states[:, None, :3]
        )
        relative_velocity = (
            self.feet_state[:, :, 7:10] - self.root_states[:, None, 7:10]
        )
        body_position = quat_rotate_inverse(
            base_quat, relative_position.reshape(-1, 3)
        ).reshape(self.num_envs, foot_count, 3)
        body_velocity = quat_rotate_inverse(
            base_quat, relative_velocity.reshape(-1, 3)
        ).reshape(self.num_envs, foot_count, 3)

        nominal_x = getattr(
            self.cfg.rewards,
            "nominal_foot_x_by_leg_m",
            {"FL": 0.216, "FR": 0.216, "RL": -0.216, "RR": -0.216},
        )
        x_sigma = max(float(getattr(
            self.cfg.rewards, "diagonal_stride_position_sigma_m", 0.018
        )), 1.0e-4)
        z_sigma = max(float(getattr(
            self.cfg.rewards, "diagonal_stride_height_sigma_m", 0.012
        )), 1.0e-4)
        vx_sigma = max(float(getattr(
            self.cfg.rewards, "diagonal_stride_velocity_sigma_m_s", 0.30
        )), 1.0e-4)
        vz_sigma = max(float(getattr(
            self.cfg.rewards, "diagonal_vertical_velocity_sigma_m_s", 0.25
        )), 1.0e-4)

        score = torch.zeros(self.num_envs, device=self.device)
        for first, second in (("FL", "RR"), ("FR", "RL")):
            first_slot = self.foot_slot_by_leg[first]
            second_slot = self.foot_slot_by_leg[second]
            expected_x_separation = float(
                nominal_x[first] - nominal_x[second]
            )
            x_error = (
                body_position[:, first_slot, 0]
                - body_position[:, second_slot, 0]
                - expected_x_separation
            )
            z_error = (
                body_position[:, first_slot, 2]
                - body_position[:, second_slot, 2]
            )
            vx_error = (
                body_velocity[:, first_slot, 0]
                - body_velocity[:, second_slot, 0]
            )
            vz_error = (
                body_velocity[:, first_slot, 2]
                - body_velocity[:, second_slot, 2]
            )
            normalized_error = (
                torch.square(x_error / x_sigma)
                + torch.square(z_error / z_sigma)
                + 0.25 * torch.square(vx_error / vx_sigma)
                + 0.25 * torch.square(vz_error / vz_sigma)
            )
            score += torch.exp(-normalized_error)
        return 0.5 * score

    def _reward_diagonal_stride_sync_all(self):
        return (
            self._get_diagonal_stride_sync_score()
            * (1.0 - self._stand_command_gate())
        )

    def _reward_diagonal_stride_sync_shortfall(self):
        return (
            (1.0 - self._get_diagonal_stride_sync_score())
            * (1.0 - self._stand_command_gate())
        )

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
        contact = self.get_foot_contact_mask()
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

    def _recovery_or_settle_gate(self):
        return (
            self.recovery_active | (self.post_recovery_steps > 0)
        ).float()

    def _reward_recovery_upright(self):
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        roll_pitch_rate = torch.sum(
            torch.square(self.base_ang_vel[:, :2]), dim=1
        )
        return (tilt + 0.08 * roll_pitch_rate) * self.recovery_active.float()

    def _reward_recovery_command_tracking(self):
        linear_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]
        ), dim=1)
        yaw_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return (linear_error + 0.15 * yaw_error) * self._recovery_or_settle_gate()

    def _reward_recovery_action_settle(self):
        action_rate = self.policy_actions - self.last_policy_actions
        return torch.sum(torch.square(action_rate), dim=1) * (
            (self.post_recovery_steps > 0).float()
        )

    def _reward_recovery_diagonal_symmetry(self):
        error = torch.zeros(self.num_envs, device=self.device)
        for first, second in (("FL", "RR"), ("FR", "RL")):
            for joint in ("hip", "thigh", "calf"):
                first_index = self.leg_dof_indices[first][joint]
                second_index = self.leg_dof_indices[second][joint]
                if joint == "hip":
                    pair_error = (
                        self.policy_actions[:, first_index]
                        + self.policy_actions[:, second_index]
                    )
                else:
                    pair_error = (
                        self.policy_actions[:, first_index]
                        - self.policy_actions[:, second_index]
                    )
                error += torch.square(pair_error)
        return error * (self.post_recovery_steps > 0).float()

    def _reward_recovery_completion(self):
        return self.recovery_completion_pulse

    def _reward_post_recovery_stability(self):
        gate = (self.post_recovery_steps > 0).float()
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        velocity_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]
        ), dim=1)
        yaw_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return (
            torch.exp(-tilt / 0.0025)
            * torch.exp(-velocity_error / 0.08)
            * torch.exp(-yaw_error / 0.25)
            * gate
        )

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
        return torch.sum(torch.square(self.raw_pd_torques), dim=1)

    def _reward_torque_clip(self):
        ratio = (
            torch.abs(self.torque_clip_error)
            / self._active_episode_torque_limits()
        )
        ratio = ratio.clip(max=2.0)
        return torch.mean(torch.square(ratio), dim=1) * self._torque_curriculum_multiplier(
            "torque_clip"
        )

    def _reward_raw_torque_rate(self):
        """Penalize rapid changes in the unclipped 50 Hz PD request.

        Action-rate cost alone misses torque chatter caused by delayed motor
        state error. Normalize by each joint's active peak limit so hips,
        thighs and calves contribute on the same scale.
        """
        normalized_delta = (
            (self.raw_pd_torques - self.last_raw_pd_torques)
            / self._active_episode_torque_limits()
        ).clip(-3.0, 3.0)
        valid = (self.episode_length_buf > 1).float()
        return torch.mean(torch.square(normalized_delta), dim=1) * valid

    def _reward_torque_near_limit(self):
        ratio = (
            torch.abs(self.raw_pd_torques)
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
            torch.abs(self.raw_pd_torques)
            / self._active_episode_torque_limits()
        )
        peak_ratio = torch.max(ratio, dim=1).values
        excess = (
            peak_ratio - self.cfg.rewards.peak_torque_soft_ratio
        ).clip(min=0.0)
        return torch.square(excess) * self._torque_curriculum_multiplier("peak_torque")

    def _reward_sustained_torque(self):
        continuous_limits = getattr(
            self.cfg.rewards, "continuous_torque_limits_by_joint", None
        )
        if continuous_limits is None:
            ema_ratio = self.torque_ema / self._active_episode_torque_limits()
            excess = (
                ema_ratio - self.cfg.rewards.sustained_torque_ratio
            ).clip(min=0.0)
        else:
            rated = self._joint_type_tensor(continuous_limits).unsqueeze(0)
            excess = (self.torque_ema / rated - 1.0).clip(min=0.0)
        return torch.mean(torch.square(excess), dim=1) * self._torque_curriculum_multiplier(
            "sustained_torque"
        )

    def _reward_sustained_torque_max(self):
        continuous_limits = getattr(
            self.cfg.rewards, "continuous_torque_limits_by_joint", None
        )
        if continuous_limits is None:
            ema_ratio = self.torque_ema / self._active_episode_torque_limits()
            peak_ratio = torch.max(ema_ratio, dim=1).values
            excess = (
                peak_ratio - self.cfg.rewards.sustained_torque_ratio
            ).clip(min=0.0)
        else:
            rated = self._joint_type_tensor(continuous_limits).unsqueeze(0)
            peak_ratio = torch.max(self.torque_ema / rated, dim=1).values
            excess = (peak_ratio - 1.0).clip(min=0.0)
        return torch.square(excess)

    def _reward_mechanical_power(self):
        return torch.mean(
            torch.abs(self.raw_pd_torques * self.dof_vel), dim=1
        )

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

    def _get_diagonal_swing_masks(self):
        """Return airborne feet and the two legal exact diagonal patterns."""
        air = self.get_foot_air_mask()
        fl = air[:, self.foot_slot_by_leg["FL"]]
        fr = air[:, self.foot_slot_by_leg["FR"]]
        rl = air[:, self.foot_slot_by_leg["RL"]]
        rr = air[:, self.foot_slot_by_leg["RR"]]
        fl_rr = fl & rr & ~fr & ~rl
        fr_rl = fr & rl & ~fl & ~rr
        return air, fl_rr, fr_rl

    def _get_non_diagonal_swing_mask(self):
        """Flag pace, bound, triple-leg and full-flight patterns.

        Zero or one airborne foot is intentionally tolerated because real
        diagonal partners need not cross the force threshold on the same
        controller sample.  Once at least two feet are airborne, only one of
        the two exact physical diagonals is legal.
        """
        air, fl_rr, fr_rl = self._get_diagonal_swing_masks()
        multiple_air = torch.sum(air, dim=1) >= 2
        return multiple_air & ~(fl_rr | fr_rl)

    def _get_flight_mask(self):
        return torch.all(self.get_foot_air_mask(), dim=1)

    def _get_diagonal_support_scores(self):
        """Return load-bearing scores for FL+RR and FR+RL.

        Both toes must carry load; one heavily loaded front or rear toe cannot
        masquerade as a valid diagonal support pair.
        """
        force = self.contact_forces[:, self.feet_indices, 2].clip(min=0.0)
        nominal_weight = float(getattr(
            self.cfg.rewards, "transition_nominal_weight_n", 115.1
        ))
        min_foot_force = max(float(getattr(
            self.cfg.rewards,
            "diagonal_support_min_foot_force_n",
            5.0,
        )), 1.0)
        min_pair_force = max(
            float(getattr(
                self.cfg.rewards,
                "diagonal_support_min_pair_weight_fraction",
                0.30,
            )) * nominal_weight,
            2.0 * min_foot_force,
        )

        def pair_score(first, second):
            first_force = force[:, self.foot_slot_by_leg[first]]
            second_force = force[:, self.foot_slot_by_leg[second]]
            foot_score = torch.minimum(
                first_force / min_foot_force,
                second_force / min_foot_force,
            )
            pair_force_score = (
                (first_force + second_force) / min_pair_force
            )
            return torch.minimum(foot_score, pair_force_score).clip(0.0, 1.0)

        return pair_score("FL", "RR"), pair_score("FR", "RL")

    def _reward_diagonal_gait(self):
        contact = self.get_foot_contact_mask()
        desired_contact = self._get_desired_foot_contacts()
        mismatch_count = torch.sum(contact != desired_contact, dim=1)
        trot_score = torch.exp(-1.5 * mismatch_count.float())
        if not getattr(
            self.cfg.rewards, "gate_phase_rewards_with_command", False
        ):
            return trot_score
        command_energy = (
            torch.sum(torch.square(self.commands[:, :2]), dim=1)
            + 0.04 * torch.square(self.commands[:, 2])
        )
        sigma = float(getattr(
            self.cfg.rewards, "phase_command_gate_sigma", 0.0004
        ))
        gait_gate = 1.0 - torch.exp(-command_energy / sigma)
        stand_score = torch.exp(
            -1.5 * torch.sum(~contact, dim=1).float()
        )
        return gait_gate * trot_score + (1.0 - gait_gate) * stand_score

    def _reward_exact_diagonal_swing(self):
        """Reward only the phase-scheduled, exact two-foot diagonal swing."""
        _, actual_fl_rr, actual_fr_rl = self._get_diagonal_swing_masks()
        desired_air = ~self._get_desired_foot_contacts()
        fl = desired_air[:, self.foot_slot_by_leg["FL"]]
        fr = desired_air[:, self.foot_slot_by_leg["FR"]]
        rl = desired_air[:, self.foot_slot_by_leg["RL"]]
        rr = desired_air[:, self.foot_slot_by_leg["RR"]]
        desired_fl_rr = fl & rr & ~fr & ~rl
        desired_fr_rl = fr & rl & ~fl & ~rr
        correct = (
            (actual_fl_rr & desired_fl_rr)
            | (actual_fr_rl & desired_fr_rl)
        )
        return correct.float() * (1.0 - self._stand_command_gate())

    def _reward_scheduled_diagonal_pair_lift(self):
        """Reward the weaker member of the scheduled diagonal pair.

        Averaging four independent foot rewards lets a policy collect reward
        by lifting only one foot.  Here the pair score is limited by the lower
        toe and the more heavily loaded toe, so FL+RR or FR+RL must unload and
        rise together.  The exact-contact reward remains the final binary
        confirmation after both toes have actually left the floor.
        """
        desired_air = ~self._get_desired_foot_contacts()
        fl = self.foot_slot_by_leg["FL"]
        fr = self.foot_slot_by_leg["FR"]
        rl = self.foot_slot_by_leg["RL"]
        rr = self.foot_slot_by_leg["RR"]
        desired_fl_rr = desired_air[:, fl] & desired_air[:, rr]
        desired_fr_rl = desired_air[:, fr] & desired_air[:, rl]

        lift_start = float(getattr(
            self.cfg.rewards, "diagonal_pair_lift_start_height", 0.018
        ))
        lift_target = max(float(getattr(
            self.cfg.rewards,
            "diagonal_pair_lift_target_height",
            self.cfg.rewards.swing_height_target,
        )), lift_start + 1.0e-4)
        lift = ((self.feet_pos[:, :, 2] - lift_start)
                / (lift_target - lift_start)).clip(0.0, 1.0)
        fl_rr_lift = torch.minimum(lift[:, fl], lift[:, rr])
        fr_rl_lift = torch.minimum(lift[:, fr], lift[:, rl])

        vertical_force = self.contact_forces[:, self.feet_indices, 2].clip(
            min=0.0
        )
        total_force = torch.sum(vertical_force, dim=1, keepdim=True)
        load_fraction = torch.where(
            total_force > 1.0,
            vertical_force / total_force.clip(min=1.0),
            torch.zeros_like(vertical_force),
        )
        fl_rr_load = torch.maximum(load_fraction[:, fl], load_fraction[:, rr])
        fr_rl_load = torch.maximum(load_fraction[:, fr], load_fraction[:, rl])
        fl_rr_unload = (1.0 - fl_rr_load / 0.25).clip(0.0, 1.0)
        fr_rl_unload = (1.0 - fr_rl_load / 0.25).clip(0.0, 1.0)

        fl_rr_score = 0.5 * (fl_rr_lift + fl_rr_unload)
        fr_rl_score = 0.5 * (fr_rl_lift + fr_rl_unload)
        score = torch.where(
            desired_fl_rr,
            fl_rr_score,
            torch.where(desired_fr_rl, fr_rl_score, torch.zeros_like(fl_rr_score)),
        )
        return score * (1.0 - self._stand_command_gate())

    def _get_touchdown_pair_support(self):
        """Measure load transfer to the newly landed diagonal.

        With diagonal phase offsets 0 and 0.5, a duty factor above 0.5
        creates two overlap windows. FL+RR is the newly landed pair around
        phase zero; FR+RL is newly landed around phase 0.5. Its required load
        ramps through the overlap so contact occurs before the old support
        pair is allowed to leave at the end of that window.
        """
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if not torch.is_tensor(stance_ratio):
            stance_ratio = torch.full(
                (self.num_envs,), float(stance_ratio), device=self.device
            )
        overlap = (stance_ratio - 0.5).clip(min=0.0, max=0.49)
        phase = self.gait_phase
        fl_rr_window = phase < overlap
        fr_rl_window = (phase >= 0.5) & (phase < 0.5 + overlap)
        active = fl_rr_window | fr_rl_window
        safe_overlap = overlap.clip(min=1.0e-4)
        progress = torch.where(
            fl_rr_window,
            phase / safe_overlap,
            (phase - 0.5) / safe_overlap,
        ).clip(0.0, 1.0)

        force = self.contact_forces[:, self.feet_indices, 2].clip(min=0.0)
        fl_rr_force = (
            force[:, self.foot_slot_by_leg["FL"]]
            + force[:, self.foot_slot_by_leg["RR"]]
        )
        fr_rl_force = (
            force[:, self.foot_slot_by_leg["FR"]]
            + force[:, self.foot_slot_by_leg["RL"]]
        )
        new_pair_force = torch.where(
            fl_rr_window, fl_rr_force, fr_rl_force
        )
        total_force = torch.sum(force, dim=1)

        nominal_weight = float(getattr(
            self.cfg.rewards, "transition_nominal_weight_n", 115.1
        ))
        end_pair_fraction = float(getattr(
            self.cfg.rewards,
            "transition_new_pair_weight_fraction",
            0.45,
        ))
        total_fraction = float(getattr(
            self.cfg.rewards,
            "transition_total_weight_fraction",
            0.70,
        ))
        desired_pair_force = (
            progress * end_pair_fraction * nominal_weight
        )
        pair_score = torch.where(
            desired_pair_force > 1.0,
            (new_pair_force / desired_pair_force.clip(min=1.0)).clip(0.0, 1.0),
            torch.ones_like(new_pair_force),
        )
        total_score = (
            total_force / max(total_fraction * nominal_weight, 1.0)
        ).clip(0.0, 1.0)
        return active, torch.minimum(pair_score, total_score)

    def _reward_touchdown_pair_support(self):
        active, score = self._get_touchdown_pair_support()
        return (
            active.float() * score * (1.0 - self._stand_command_gate())
        )

    def _reward_touchdown_pair_support_shortfall(self):
        active, score = self._get_touchdown_pair_support()
        return (
            active.float() * (1.0 - score)
            * (1.0 - self._stand_command_gate())
        )

    def _get_diagonal_load_transfer(self):
        """Track a complete old-pair to new-pair load handoff.

        Touchdown force alone allowed both diagonals to remain planted. This
        target ramps the newly landed pair from zero to full normalized load
        through double support, then keeps it as the sole load-bearing pair
        until the next transfer window.
        """
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if not torch.is_tensor(stance_ratio):
            stance_ratio = torch.full(
                (self.num_envs,), float(stance_ratio), device=self.device
            )
        overlap = (stance_ratio - 0.5).clip(min=1.0e-4, max=0.49)
        phase = self.gait_phase
        first_progress = (phase / overlap).clip(0.0, 1.0)
        second_progress = ((phase - 0.5) / overlap).clip(0.0, 1.0)
        first_progress = first_progress * first_progress * (
            3.0 - 2.0 * first_progress
        )
        second_progress = second_progress * second_progress * (
            3.0 - 2.0 * second_progress
        )
        desired_fl_rr_fraction = torch.where(
            phase < overlap,
            first_progress,
            torch.where(
                phase < 0.5,
                torch.ones_like(phase),
                torch.where(
                    phase < 0.5 + overlap,
                    1.0 - second_progress,
                    torch.zeros_like(phase),
                ),
            ),
        )

        force = self.contact_forces[:, self.feet_indices, 2].clip(min=0.0)
        fl_rr_force = (
            force[:, self.foot_slot_by_leg["FL"]]
            + force[:, self.foot_slot_by_leg["RR"]]
        )
        fr_rl_force = (
            force[:, self.foot_slot_by_leg["FR"]]
            + force[:, self.foot_slot_by_leg["RL"]]
        )
        total_force = fl_rr_force + fr_rl_force
        measured_fl_rr_fraction = torch.where(
            total_force > 1.0,
            fl_rr_force / total_force.clip(min=1.0),
            torch.full_like(total_force, 0.5),
        )
        load_error = torch.abs(
            measured_fl_rr_fraction - desired_fl_rr_fraction
        )
        sigma = max(float(getattr(
            self.cfg.rewards, "diagonal_load_transfer_sigma", 0.12
        )), 1.0e-4)
        nominal_weight = float(getattr(
            self.cfg.rewards, "transition_nominal_weight_n", 115.1
        ))
        total_support = (
            total_force / max(0.70 * nominal_weight, 1.0)
        ).clip(0.0, 1.0)
        score = torch.exp(-torch.square(load_error / sigma)) * total_support
        return score, load_error

    def _reward_diagonal_load_transfer(self):
        score, _ = self._get_diagonal_load_transfer()
        return score * (1.0 - self._stand_command_gate())

    def _reward_diagonal_load_transfer_error(self):
        _, error = self._get_diagonal_load_transfer()
        return error * (1.0 - self._stand_command_gate())

    def _reward_diagonal_support(self):
        """Reward having at least one genuinely load-bearing diagonal."""
        fl_rr_score, fr_rl_score = self._get_diagonal_support_scores()
        return (
            torch.maximum(fl_rr_score, fr_rl_score)
            * (1.0 - self._stand_command_gate())
        )

    def _reward_diagonal_support_shortfall(self):
        """Penalize front/rear bounds, pace patterns and unsupported hops."""
        fl_rr_score, fr_rl_score = self._get_diagonal_support_scores()
        return (
            (1.0 - torch.maximum(fl_rr_score, fr_rl_score))
            * (1.0 - self._stand_command_gate())
        )

    def _reward_non_diagonal_swing(self):
        """Penalize every multi-foot swing that is not a physical diagonal."""
        return (
            self._get_non_diagonal_swing_mask().float()
            * (1.0 - self._stand_command_gate())
        )

    def _reward_single_foot_swing(self):
        """Reject tentative one-toe stepping during a scheduled pair swing."""
        actual_air = self.get_foot_air_mask()
        desired_air = ~self._get_desired_foot_contacts()
        tentative = (
            (torch.sum(desired_air, dim=1) == 2)
            & (torch.sum(actual_air, dim=1) == 1)
        )
        return tentative.float() * (1.0 - self._stand_command_gate())

    def _reward_phase_contact_mismatch(self):
        """Penalize every foot whose contact disagrees with the trot clock.

        Pair-synchrony terms alone cannot distinguish a diagonal trot from a
        four-leg hop: all four feet are still perfectly pair-synchronized in
        flight.  This term compares each physical foot with the Fanfan phase
        schedule, while stand commands explicitly ask for all four contacts.
        """
        contact = self.get_foot_contact_mask()
        desired_contact = self._get_desired_foot_contacts()
        gait_error = torch.mean(
            (contact != desired_contact).float(), dim=1
        )
        stand_error = torch.mean((~contact).float(), dim=1)
        stand_gate = self._stand_command_gate()
        return (1.0 - stand_gate) * gait_error + stand_gate * stand_error

    def _reward_phase_foot_force_tracking(self):
        """Track the phase-scheduled vertical load distribution.

        Binary contact changes only after lift-off, so it gives a stationary
        four-foot policy no useful intermediate signal.  Normalized vertical
        force already changes while the swing diagonal is unloading.  This
        mass-independent error therefore guides load transfer without asking
        for more motor torque or rewarding flight.
        """
        vertical_force = self.contact_forces[:, self.feet_indices, 2].clip(
            min=0.0
        )
        total_force = torch.sum(vertical_force, dim=1, keepdim=True)
        measured_load = torch.where(
            total_force > 1.0,
            vertical_force / total_force.clip(min=1.0),
            torch.zeros_like(vertical_force),
        )

        desired_contact = self._get_desired_foot_contacts().float()
        desired_load = desired_contact / torch.sum(
            desired_contact, dim=1, keepdim=True
        ).clip(min=1.0)
        gait_error = torch.mean(
            torch.square(measured_load - desired_load), dim=1
        )

        stand_load = torch.full_like(measured_load, 0.25)
        stand_error = torch.mean(
            torch.square(measured_load - stand_load), dim=1
        )
        stand_gate = self._stand_command_gate()
        return (1.0 - stand_gate) * gait_error + stand_gate * stand_error

    def _reward_phase_foot_velocity_tracking(self):
        """Give the phase clock an explicit forward stride direction.

        Contact timing alone also rewards stepping in place.  In a forward
        trot, a stance toe moves backward relative to the trunk while a swing
        toe advances.  Tracking those body-frame velocities supplies a dense,
        phase-resolved learning signal before the base has begun to translate.
        """
        foot_world_velocity = self.feet_state[:, :, 7:10]
        relative_world_velocity = (
            foot_world_velocity - self.root_states[:, None, 7:10]
        )
        foot_count = len(self.feet_indices)
        base_quat = self.base_quat[:, None, :].expand(
            -1, foot_count, -1
        ).reshape(-1, 4)
        relative_body_velocity = quat_rotate_inverse(
            base_quat, relative_world_velocity.reshape(-1, 3)
        ).reshape(self.num_envs, foot_count, 3)

        desired_contact = self._get_desired_foot_contacts()
        stance_ratio = getattr(
            self, "gait_stance_ratio", self.cfg.rewards.gait_stance_ratio
        )
        if not torch.is_tensor(stance_ratio):
            stance_ratio = torch.full(
                (self.num_envs,), float(stance_ratio),
                device=self.device,
            )
        swing_to_stance = (
            stance_ratio / (1.0 - stance_ratio).clip(min=0.20)
        ).clip(max=2.0)
        command_x = self.commands[:, 0]
        stance_velocity = -command_x[:, None]
        swing_velocity = (
            command_x * swing_to_stance
        )[:, None]
        desired_velocity = torch.where(
            desired_contact, stance_velocity, swing_velocity
        )
        error = torch.square(
            relative_body_velocity[:, :, 0] - desired_velocity
        )
        sigma = max(float(getattr(
            self.cfg.rewards, "phase_foot_velocity_sigma", 0.04
        )), 1.0e-6)
        score = torch.mean(torch.exp(-error / sigma), dim=1)
        return score * (1.0 - self._stand_command_gate())

    def _reward_swing_height(self):
        desired_swing = ~self._get_desired_foot_contacts()
        height_error = torch.square(
            self.feet_pos[:, :, 2] - self.cfg.rewards.swing_height_target
        )
        swing_score = torch.exp(-height_error / self.cfg.rewards.swing_height_sigma)
        reward = torch.sum(swing_score * desired_swing.float(), dim=1) / (
            torch.sum(desired_swing.float(), dim=1) + 1.0e-6
        )
        if getattr(
            self.cfg.rewards, "gate_phase_rewards_with_command", False
        ):
            command_energy = (
                torch.sum(torch.square(self.commands[:, :2]), dim=1)
                + 0.04 * torch.square(self.commands[:, 2])
            )
            sigma = float(getattr(
                self.cfg.rewards, "phase_command_gate_sigma", 0.0004
            ))
            reward *= 1.0 - torch.exp(-command_energy / sigma)
        return reward

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
        contact = self.get_foot_contact_mask()
        command_active = (
            torch.linalg.norm(self.commands[:, :2], dim=1) > 0.015
        ) | (torch.abs(self.commands[:, 2]) > 0.08)
        swing_contact_ratio = torch.sum(
            (contact & desired_swing).float(), dim=1
        ) / (torch.sum(desired_swing.float(), dim=1) + 1.0e-6)
        return swing_contact_ratio * command_active.float()

    def _reward_all_feet_contact(self):
        """Penalize prolonged four-foot support during locomotion."""
        allowed = float(getattr(
            self.cfg.rewards, "max_all_feet_contact_time_s", 0.08
        ))
        saturation = max(float(getattr(
            self.cfg.rewards,
            "all_feet_contact_penalty_saturation_s",
            0.12,
        )), 1.0e-6)
        excess = (
            (self.all_feet_contact_time - allowed) / saturation
        ).clip(min=0.0, max=1.0)
        locomotion_gate = 1.0 - self._stand_command_gate()
        return excess * locomotion_gate

    def _reward_excessive_foot_contact_time(self):
        """Penalize each foot that stays planted longer than one stance."""
        allowed = float(getattr(
            self.cfg.rewards, "max_foot_contact_time_s", 0.40
        ))
        saturation = max(float(getattr(
            self.cfg.rewards, "foot_contact_time_penalty_saturation_s", 0.20
        )), 1.0e-6)
        excess = ((self.feet_contact_time - allowed) / saturation).clip(
            min=0.0, max=1.0
        )
        locomotion_gate = 1.0 - self._stand_command_gate()
        # Sum rather than average so one stuck foot is penalized and four
        # continuously planted feet receive four times the cost.
        return torch.sum(excess, dim=1) * locomotion_gate

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
        return self._get_flight_mask()

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
        reward = torch.exp(-posture_error / stand_posture_sigma)
        if getattr(
            self.cfg.rewards, "gate_stand_posture_with_command", False
        ):
            reward *= self._stand_command_gate()
        return reward

    def _reward_front_feet_contact(self):
        front_feet_contact_height = getattr(self.cfg.rewards, "front_feet_contact_height", None)
        max_rear_sit_pitch = getattr(self.cfg.rewards, "max_rear_sit_pitch", 0.0)
        contact = self.get_foot_contact_mask()
        contact = contact[:, [
            self.foot_slot_by_leg["FL"], self.foot_slot_by_leg["FR"]
        ]]
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
