"""Stratified wide commands, heavy-tail tracking, continuous variable gait clock."""

import torch

from .rs01_omni_v11_env import Rs01OmniV11Robot


class Rs01OmniV12Robot(Rs01OmniV11Robot):
    def __init__(self, *args, **kwargs):
        self._wide_phase_ready = False
        super().__init__(*args, **kwargs)
        self.wide_gait_phase = super()._gait_phase().clone()
        self._wide_phase_ready = True
        self.compute_observations()

    def _sample_magnitude(self, count, value_range):
        edges = torch.tensor(self.cfg.commands.speed_band_edges, device=self.device)
        probabilities = torch.tensor(self.cfg.commands.speed_band_probabilities, device=self.device)
        band = torch.multinomial(probabilities, count, replacement=True)
        fraction = edges[band] + torch.rand(count, device=self.device) * (edges[band + 1] - edges[band])
        low, high = value_range
        return low + (high - low) * fraction

    def _set_command_modes(self, env_ids, modes):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, :3] = 0.0
        self.command_mode[env_ids] = modes
        self.gait_enable[env_ids] = (modes != self.COMMAND_STAND).float()
        cfg = self.cfg.commands
        specs = (
            (self.COMMAND_FORWARD, 0, cfg.forward_velocity_range_m_s, 1),
            (self.COMMAND_BACKWARD, 0, cfg.backward_speed_range_m_s, -1),
            (self.COMMAND_LATERAL, 1, cfg.lateral_speed_range_m_s, 0),
            (self.COMMAND_YAW, 2, cfg.yaw_speed_range_rad_s, 0),
            (self.COMMAND_COMBINED, 0, cfg.combined_forward_speed_range_m_s, 0),
            (self.COMMAND_COMBINED, 1, cfg.combined_lateral_speed_range_m_s, 0),
            (self.COMMAND_COMBINED, 2, cfg.combined_yaw_speed_range_rad_s, 0),
        )
        for mode, axis, bounds, sign in specs:
            ids = env_ids[modes == mode]
            if len(ids):
                signs = sign if sign else self._random_sign(len(ids), self.device)
                self.commands[ids, axis] = self._sample_magnitude(len(ids), bounds) * signs
        combined = env_ids[modes == self.COMMAND_COMBINED]
        if len(combined):
            # Explore all octants without requesting every axis maximum at
            # once. This is a sampling envelope, not a proven hardware bound.
            caps = torch.tensor(cfg.combined_axis_caps, device=self.device)
            command = self.commands[combined, :3]
            norm = torch.linalg.vector_norm(command / caps, dim=1, keepdim=True)
            self.commands[combined, :3] = command / norm.clamp(min=1.0)
        self._reset_command_reference(env_ids)

    def _command_gait_frequency(self):
        cfg = self.cfg.commands
        x_cap = torch.where(self.commands[:, 0] >= 0,
                            cfg.forward_velocity_range_m_s[1], cfg.backward_speed_range_m_s[1])
        intensity = torch.stack((
            self.commands[:, 0].abs() / x_cap,
            self.commands[:, 1].abs() / cfg.lateral_speed_range_m_s[1],
            self.commands[:, 2].abs() / cfg.yaw_speed_range_rad_s[1],
        ), dim=1).amax(dim=1)
        deadband = self.cfg.rewards.wide_gait_speed_deadband
        blend = ((intensity - deadband) / (1.0 - deadband)).clamp(0.0, 1.0)
        low = 1.0 / self.cfg.rewards.gait_period_s
        return low + (self.cfg.rewards.wide_gait_max_frequency_hz - low) * blend

    def _advance_wide_gait_phase(self):
        self.wide_gait_phase.add_(self.dt * self._command_gait_frequency()).remainder_(1.0)

    def _post_physics_step_callback(self):
        if self._wide_phase_ready:
            # Integrate the command that was active during this step, before
            # the parent callback can resample. Never use t * new_frequency.
            self._advance_wide_gait_phase()
        super()._post_physics_step_callback()

    def _gait_phase(self):
        if getattr(self, "_wide_phase_ready", False):
            return self.wide_gait_phase
        return super()._gait_phase()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if getattr(self, "_wide_phase_ready", False):
            self.wide_gait_phase[env_ids] = self.gait_phase_offset[env_ids]

    def _reward_tracking_command_velocity(self):
        planar_error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1)
        yaw_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]).square()
        normalized_error = (
            planar_error / self.cfg.rewards.command_planar_tracking_sigma
            + yaw_error / self.cfg.rewards.command_yaw_tracking_sigma
        )
        # Replace exp(-E), not add a second reward. Both have 1-E behavior
        # near zero; 1/(1+E) keeps useful differences far from the target.
        accuracy = torch.reciprocal(1.0 + normalized_error)
        self.v11_tracking_accuracy = accuracy.detach()
        self.v11_legal_contact_gate = self._legal_task_contact_gate().detach()
        self.v11_tracking_reward = accuracy.detach()
        return accuracy
