"""Positive diagonal discovery signals for RS01 omni scratch training."""

import torch

from .rs01_omni_v3_env import Rs01OmniV3Robot


class Rs01OmniV4Robot(Rs01OmniV3Robot):
    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_phase_two_contact_quality(self):
        desired = self._desired_contact_mask()
        actual = self.get_foot_contact_mask()
        desired_two_contacts = torch.sum(desired, dim=1) == 2
        exact_contact = torch.all(actual == desired, dim=1)
        return (
            desired_two_contacts & exact_contact
        ).to(dtype=torch.float) * self._walking_command_gate()
