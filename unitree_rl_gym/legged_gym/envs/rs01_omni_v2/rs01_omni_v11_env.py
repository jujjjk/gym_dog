"""Isolate velocity-feedback gating, without adding a reward or symmetry lock."""

import torch

from .rs01_omni_v10_env import Rs01OmniV10Robot


class Rs01OmniV11Robot(Rs01OmniV10Robot):
    def _reward_tracking_command_velocity(self):
        planar_error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1)
        yaw_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]).square()
        accuracy = torch.exp(
            -planar_error / self.cfg.rewards.command_planar_tracking_sigma
            -yaw_error / self.cfg.rewards.command_yaw_tracking_sigma
        )
        legal = self._legal_task_contact_gate()
        reward = accuracy * legal if self.cfg.rewards.tracking_contact_gate else accuracy
        # Capture the actual reward-time values (before automatic resets), not
        # a second evaluation of rewards against post-reset state.
        self.v11_tracking_accuracy = accuracy.detach()
        self.v11_legal_contact_gate = legal.detach()
        self.v11_tracking_reward = reward.detach()
        return reward

    def _reward_phase_two_contact_quality(self):
        reward = super()._reward_phase_two_contact_quality()
        self.v11_contact_quality_reward = reward.detach()
        return reward
