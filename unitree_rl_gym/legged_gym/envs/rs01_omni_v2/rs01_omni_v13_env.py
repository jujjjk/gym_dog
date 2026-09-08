"""Shared bounded direction target for actor and reward; raw intent stays intact."""

import torch

from .rs01_omni_v12_env import Rs01OmniV12Robot


class Rs01OmniV13Robot(Rs01OmniV12Robot):
    def __init__(self, *args, **kwargs):
        self._direction_ready = False
        super().__init__(*args, **kwargs)
        self.direction_turning = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._direction_ready = True
        self._start_direction_segment(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    def _reset_command_reference(self, env_ids):
        # Parent command samplers call this every 4 s. Preserve the reference
        # during ordinary speed/direction changes; episode and turn events
        # use _start_direction_segment explicitly.
        if not getattr(self, "_direction_ready", False):
            super()._reset_command_reference(env_ids)

    def _start_direction_segment(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_command_reference(env_ids)
        self.direction_turning[env_ids] = (
            self.commands[env_ids, 2].abs() >= self.cfg.commands.direction_turn_enter_rad_s
        )

    def _update_turn_mode(self, env_ids):
        if not getattr(self, "_direction_ready", False) or len(env_ids) == 0:
            return
        old = self.direction_turning[env_ids]
        yaw = self.commands[env_ids, 2].abs()
        new = torch.where(old, yaw > self.cfg.commands.direction_turn_exit_rad_s,
                          yaw >= self.cfg.commands.direction_turn_enter_rad_s)
        changed = env_ids[old != new]
        # Entering a user-requested turn, or stopping it, starts a new heading.
        # Never restore the heading from before an intentional turn.
        self._start_direction_segment(changed)
        self.direction_turning[env_ids] = new

    def _set_command_modes(self, env_ids, modes):
        super()._set_command_modes(env_ids, modes)
        self._update_turn_mode(env_ids)

    def set_evaluation_command(self, command, gait_enable, reset_reference=False):
        # Same semantics for training and playback: do not reanchor on every
        # tiny command update. Explicit new-path requests can still reanchor.
        self.commands[:, :3] = command
        self.gait_enable[:] = gait_enable
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_turn_mode(env_ids)
        if reset_reference:
            self._start_direction_segment(env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if getattr(self, "_direction_ready", False):
            self._start_direction_segment(env_ids)

    def _direction_velocity_target(self):
        target = self.commands[:, :3].clone()
        if not getattr(self, "_direction_ready", False):
            return target
        cfg = self.cfg.commands
        error = self._straight_heading_error()
        blend = (1.0 - self.commands[:, 2].abs() / cfg.direction_blend_yaw_rad_s).clamp(0., 1.)
        blend *= (~self.direction_turning).float()
        # Raw planar commands describe the held-heading frame while not
        # intentionally turning. Convert to body coordinates before tracking.
        angle = error.clamp(-cfg.direction_rotation_limit_rad, cfg.direction_rotation_limit_rad)
        rotated = self._body_to_world_xy(target[:, :2], angle)
        correction = (rotated - target[:, :2]) * blend.unsqueeze(1)
        norm = torch.linalg.vector_norm(correction, dim=1, keepdim=True)
        correction *= (cfg.direction_planar_correction_limit_m_s / norm.clamp(min=1e-6)).clamp(max=1.)
        target[:, :2] += correction
        limit = cfg.direction_yaw_correction_limit_rad_s
        target[:, 2] += blend * limit * torch.tanh(cfg.direction_heading_gain * error / limit)
        target[:, 0].clamp_(*cfg.ranges.lin_vel_x)
        target[:, 1].clamp_(*cfg.ranges.lin_vel_y)
        target[:, 2].clamp_(*cfg.ranges.ang_vel_yaw)
        return target

    def _reward_tracking_command_velocity(self):
        target = self._direction_velocity_target()
        error = (
            (target[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1)
            / self.cfg.rewards.command_planar_tracking_sigma
            + (target[:, 2] - self.base_ang_vel[:, 2]).square()
            / self.cfg.rewards.command_yaw_tracking_sigma
        )
        accuracy = torch.reciprocal(1.0 + error)
        self.direction_reward_target = target.detach()
        self.v11_tracking_accuracy = accuracy.detach()
        self.v11_tracking_reward = accuracy.detach()
        self.v11_legal_contact_gate = self._legal_task_contact_gate().detach()
        return accuracy

    def compute_observations(self):
        noisy = self.add_noise
        self.add_noise = False
        try:
            super().compute_observations()
        finally:
            self.add_noise = noisy
        target = self._direction_velocity_target()
        estimated = getattr(self, "_rs01_observation_estimator_ready", False)
        velocity = self.estimated_base_lin_vel if estimated else self.base_lin_vel
        confidence = (self.estimated_odom_confidence.clamp(0., 1.) if estimated
                      else torch.ones(self.num_envs, device=self.device))
        # Keep the old command slots executable, and expose raw user intent
        # separately. The same target function is used by the reward above.
        self.obs_buf[:, 9:12] = target * self.commands_scale
        self.obs_buf[:, 52] = 0.  # Remove unreliable accumulated lateral position.
        self.obs_buf[:, 55] = 0.  # Remove accumulated longitudinal position.
        self.obs_buf[:, 53] = ((velocity[:, 1] - target[:, 1]) * 2.).clamp(-10., 10.)
        self.obs_buf[:, 56] = ((velocity[:, 0] - target[:, 0]) * 2.).clamp(-10., 10.)
        self.obs_buf = torch.cat((self.obs_buf, self.commands[:, :3] * self.commands_scale,
                                  confidence.unsqueeze(1)), dim=1)
        if noisy:
            self.obs_buf += (2. * torch.rand_like(self.obs_buf) - 1.) * self.noise_scale_vec
