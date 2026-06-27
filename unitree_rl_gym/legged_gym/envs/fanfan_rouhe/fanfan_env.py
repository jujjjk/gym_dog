from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
import torch


class FanfanRouheRobot(LeggedRobot):
    def step(self, actions):
        # A smooth bound preserves control resolution when the Gaussian policy
        # produces values outside [-1, 1]. Hard clipping made the legs bang
        # between their limits and destroyed the diagonal timing.
        return super().step(torch.tanh(actions))

    def _get_noise_scale_vec(self, cfg):
        noise_vec = super()._get_noise_scale_vec(cfg)
        noise_vec[-2:] = 0.0
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
        for body_index in self.feet_indices.cpu().tolist():
            name = body_names[body_index]
            phase_offsets.append(
                0.0 if name.startswith("FL_") or name.startswith("RR_") else 0.5
            )
        self.gait_phase_offsets = torch.tensor(
            phase_offsets, dtype=torch.float, device=self.device
        )
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
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
        self.sagittal_dof_indices = torch.tensor(
            [
                self.leg_dof_indices[leg][joint]
                for leg in ("FL", "FR", "RL", "RR")
                for joint in ("thigh", "calf")
            ],
            dtype=torch.long,
            device=self.device,
        )

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
        torques = self.p_gains * (
            actions_scaled + gait_offset + self.default_dof_pos - self.dof_pos
        ) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]

        period = self.cfg.rewards.gait_period
        self.gait_phase = (self.episode_length_buf * self.dt) % period / period

    def compute_observations(self):
        phase_angle = 2.0 * torch.pi * self.gait_phase
        phase_obs = torch.stack((torch.sin(phase_angle), torch.cos(phase_angle)), dim=1)
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            phase_obs,
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
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def check_termination(self):
        super().check_termination()
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

    def _reward_hip_velocity(self):
        return torch.sum(torch.square(self.dof_vel[:, self.hip_dof_indices]), dim=1)

    def _reward_hip_symmetry(self):
        hip_pos = self.dof_pos[:, self.hip_dof_indices]
        front_mirror_error = torch.square(hip_pos[:, 0] + hip_pos[:, 1])
        rear_mirror_error = torch.square(hip_pos[:, 2] + hip_pos[:, 3])
        return front_mirror_error + rear_mirror_error

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

    def _reward_action_magnitude(self):
        return torch.sum(torch.square(self.actions), dim=1)

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
