"""Anti-static RS01 omni environment shared by the contact-penalty A/B."""

import torch

from .rs01_omni_v2_env import Rs01OmniV2Robot


class Rs01OmniV3Robot(Rs01OmniV2Robot):
    """Make commanded progress and actual diagonal switching unavoidable."""

    def __init__(self, *args, **kwargs):
        self._v3_ready = False
        super().__init__(*args, **kwargs)
        self.gait_phase_offset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._v3_ready = True
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_reset_commands_and_phase(env_ids)
        self.compute_observations()

    def _sample_initial_modes(self, count):
        return torch.multinomial(
            self.command_mode_probabilities, count, replacement=True
        )

    def _sample_reset_commands_and_phase(self, env_ids):
        if len(env_ids) == 0:
            return
        self.gait_phase_offset[env_ids] = torch.rand(
            len(env_ids), device=self.device
        )
        self._set_command_modes(
            env_ids, self._sample_initial_modes(len(env_ids))
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if getattr(self, "_v3_ready", False):
            self._sample_reset_commands_and_phase(env_ids)

    def _gait_phase(self):
        phase = super()._gait_phase()
        if getattr(self, "_v3_ready", False):
            phase = torch.remainder(phase + self.gait_phase_offset, 1.0)
        return phase

    def _moving_command_gate(self):
        planar_requested = torch.linalg.vector_norm(
            self.commands[:, :2], dim=1
        ) > 1.0e-4
        yaw_requested = torch.abs(self.commands[:, 2]) > 1.0e-4
        return (planar_requested | yaw_requested).to(dtype=torch.float)

    def _reward_tracking_command_velocity(self):
        """Reward accurate commanded motion, with zero payoff at zero progress."""
        command_planar = self.commands[:, :2]
        actual_planar = self.base_lin_vel[:, :2]
        command_yaw = self.commands[:, 2]
        actual_yaw = self.base_ang_vel[:, 2]

        planar_error = torch.sum(
            torch.square(command_planar - actual_planar), dim=1
        )
        yaw_error = torch.square(command_yaw - actual_yaw)
        accuracy = torch.exp(
            -planar_error
            / float(self.cfg.rewards.command_planar_tracking_sigma)
            -yaw_error / float(self.cfg.rewards.command_yaw_tracking_sigma)
        )

        planar_norm_squared = torch.sum(
            torch.square(command_planar), dim=1
        )
        planar_active = planar_norm_squared > 1.0e-8
        planar_progress = torch.clamp(
            torch.sum(command_planar * actual_planar, dim=1)
            / torch.clamp(planar_norm_squared, min=1.0e-8),
            min=0.0,
            max=1.0,
        )

        yaw_norm_squared = torch.square(command_yaw)
        yaw_active = yaw_norm_squared > 1.0e-8
        yaw_progress = torch.clamp(
            command_yaw * actual_yaw
            / torch.clamp(yaw_norm_squared, min=1.0e-8),
            min=0.0,
            max=1.0,
        )

        active_count = (
            planar_active.to(dtype=torch.float)
            + yaw_active.to(dtype=torch.float)
        )
        progress = (
            planar_progress * planar_active.to(dtype=torch.float)
            + yaw_progress * yaw_active.to(dtype=torch.float)
        ) / torch.clamp(active_count, min=1.0)
        return accuracy * progress * self._moving_command_gate()

    def _reward_phase_contact_error(self):
        return self._phase_support_error() * self._walking_command_gate()
