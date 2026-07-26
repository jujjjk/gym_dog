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
    """84-observation, 12-action task with no prescribed gait generator."""

    def __init__(self, *args, **kwargs):
        self._rs01_actuator_ready = False
        self._action_history_ready = False
        super().__init__(*args, **kwargs)
        if self.num_dof != 12 or self.num_actions != 12:
            raise ValueError(
                "rs01_go2_straight requires exactly 12 URDF joints and 12 actions"
            )
        expected_observations = 48 + (
            int(self.cfg.env.action_history_frames) - 1
        ) * self.num_actions
        if self.num_obs != expected_observations:
            raise ValueError(
                "Observation size does not match the configured action history: "
                f"expected {expected_observations}, got {self.num_obs}"
            )
        if len(self.feet_indices) != 4:
            raise ValueError(
                "rs01_go2_straight requires four retained foot rigid bodies"
            )
        if abs(self.dt - float(self.cfg.rs01_actuator.control_dt_s)) > 1.0e-9:
            raise ValueError(
                "Policy/control dt must match the measured RS01 50 Hz contract"
            )
        self._initialize_action_history()
        self._initialize_rs01_actuator()

    def _initialize_action_history(self):
        older_frames = int(self.cfg.env.action_history_frames) - 1
        if older_frames < 0:
            raise ValueError("action_history_frames must be at least one")
        self.action_history = torch.zeros(
            self.num_envs,
            older_frames,
            self.num_actions,
            device=self.device,
            dtype=torch.float,
        )
        self._action_history_ready = True

    def compute_observations(self):
        """Append three older executed actions to the standard 48-D input."""
        base_observation = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        older_actions = self.action_history.reshape(self.num_envs, -1)
        self.obs_buf = torch.cat((base_observation, older_actions), dim=-1)
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        # The observation above contains [a_t, a_t-1, a_t-2, a_t-3].
        # Advance history only after constructing it for the next policy call.
        if self.action_history.shape[1] > 1:
            self.action_history[:, 1:] = self.action_history[:, :-1].clone()
        if self.action_history.shape[1] > 0:
            self.action_history[:, 0] = self.actions

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._action_history_ready:
            self.action_history[env_ids] = 0.0

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

        self.rs01_target_rate_limit_rad_s = self._joint_type_value(
            motor_cfg.target_rate_limit_rad_s
        )
        self.rs01_target_acceleration_limit_rad_s2 = self._joint_type_value(
            motor_cfg.target_acceleration_limit_rad_s2
        )
        self.rs01_response_gain = self._motor_value(motor_cfg.response_gain)
        self.rs01_time_constant_s = self._motor_value(motor_cfg.time_constant_s)
        self.rs01_coulomb_friction_nm = self._motor_value(
            motor_cfg.coulomb_friction_nm
        )
        observed_delay_s = self._motor_value(
            motor_cfg.observed_closed_loop_delay_s
        )
        self.rs01_delay_steps = torch.round(
            observed_delay_s / float(self.sim_params.dt)
        ).to(dtype=torch.long)
        max_delay_steps = int(torch.max(self.rs01_delay_steps).item())

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
        self._rs01_actuator_ready = True

    def step(self, actions):
        if self._rs01_actuator_ready:
            clipped_actions = torch.clamp(
                actions.to(self.device),
                -float(self.cfg.normalization.clip_actions),
                float(self.cfg.normalization.clip_actions),
            )
            desired_target = (
                self.default_dof_pos
                + float(self.cfg.control.action_scale) * clipped_actions
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
        delayed_columns = [
            self.rs01_target_delay_buffer[int(self.rs01_delay_steps[j]), :, j]
            for j in range(self.num_dof)
        ]
        delayed_target = torch.stack(delayed_columns, dim=1)

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

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
