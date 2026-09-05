"""Hold at zero command, track full pose, and gate task payoff by contacts."""

import torch

from .rs01_omni_v5_env import Rs01OmniV5Robot


class Rs01OmniV10Robot(Rs01OmniV5Robot):
    def _reset_command_reference(self, env_ids):
        if len(env_ids) == 0 or not getattr(self, "_omni_reference_ready", False):
            return
        # A new command starts here, avoiding a catch-up sprint toward a
        # previous command's accumulated position error.
        self.omni_desired_position_xy[env_ids] = self.root_states[env_ids, :2]
        self.omni_estimated_position_xy[env_ids] = self.root_states[env_ids, :2]
        self.omni_desired_heading_rad[env_ids] = self.rpy[env_ids, 2]
        self.straight_heading_target_rad[env_ids] = self.rpy[env_ids, 2]

    def _set_command_modes(self, env_ids, modes):
        super()._set_command_modes(env_ids, modes)
        self._reset_command_reference(env_ids)

    def set_evaluation_command(self, command, gait_enable):
        changed = torch.any(self.commands[:, :3] != command, dim=1)
        changed |= self.gait_enable != gait_enable
        self._reset_command_reference(changed.nonzero(as_tuple=False).flatten())
        self.commands[:, :3] = command
        self.gait_enable[:] = gait_enable

    def _legal_task_contact_gate(self):
        contact = self.get_foot_contact_mask()
        desired = self._desired_contact_mask()
        exact = torch.all(contact == desired, dim=1)
        count = contact.sum(dim=1)
        # Desired four-foot handoff lasts 0.09 s at the unchanged duty factor.
        handoff = (count == 4) & (
            self.all_feet_contact_time_s <= self.cfg.rewards.all_feet_contact_grace_s
        )
        walking_legal = exact & ((count == 2) | handoff)
        standing_legal = count == 4
        return torch.where(self._walking_command_gate() > 0.5,
                           walking_legal, standing_legal).float()

    def _reward_tracking_command_velocity(self):
        # Symmetric tracking includes zero commands. Contact legality prevents
        # a stationary four-foot policy from collecting the march objective.
        planar_error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1)
        yaw_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]).square()
        accuracy = torch.exp(
            -planar_error / self.cfg.rewards.command_planar_tracking_sigma
            -yaw_error / self.cfg.rewards.command_yaw_tracking_sigma
        )
        return accuracy * self._legal_task_contact_gate()

    def _reward_phase_support_tracking(self):
        return super()._reward_phase_support_tracking() * self._legal_task_contact_gate()

    @staticmethod
    def _huber_unit(value):
        magnitude = torch.abs(value)
        return torch.where(magnitude < 1.0, 0.5 * magnitude.square(), magnitude - 0.5)

    def _reward_pose_error(self):
        # Both stand and march retain this cost. Large errors remain
        # distinguishable; the linear tail avoids a quadratic runaway cost.
        position_error = self.root_states[:, :2] - self.omni_desired_position_xy
        distance = torch.linalg.vector_norm(position_error, dim=1)
        return (
            self._huber_unit(distance / self.cfg.rewards.pose_position_scale_m)
            + self._huber_unit(self._straight_heading_error() / self.cfg.rewards.pose_heading_scale_rad)
        )

    def compute_observations(self):
        noisy = self.add_noise
        self.add_noise = False
        try:
            super().compute_observations()
        finally:
            self.add_noise = noisy
        extra = torch.zeros(self.num_envs, 2, device=self.device)
        if getattr(self, "_omni_reference_ready", False):
            heading = self.omni_desired_heading_rad
            axis = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
            estimated = self._rs01_observation_estimator_ready
            position = self.omni_estimated_position_xy if estimated else self.root_states[:, :2]
            velocity = self.estimated_base_lin_vel if estimated else self.base_lin_vel
            world_velocity = self._body_to_world_xy(velocity[:, :2], self.rpy[:, 2])
            lateral_error, _ = self._straight_path_observation_state()
            self.obs_buf[:, 52] = (lateral_error * 2.0).clamp(-10.0, 10.0)
            extra[:, 0] = ((position - self.omni_desired_position_xy) * axis).sum(dim=1) * 2.0
            extra[:, 1] = ((world_velocity * axis).sum(dim=1) - self.commands[:, 0]) * 2.0
        # Linear observation range extends beyond the former 0.5 m clip.
        self.obs_buf = torch.cat((self.obs_buf, extra.clamp(-10.0, 10.0)), dim=1)
        if noisy:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec
