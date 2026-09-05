"""Dense diagonal-support rewards for RS01 omni scratch training."""

import torch

from .rs01_omni_v4_env import Rs01OmniV4Robot


class Rs01OmniV5Robot(Rs01OmniV4Robot):
    def _reward_phase_support_tracking(self):
        """Reward smooth load transfer only during two-foot target phases."""
        desired = self._desired_contact_mask()
        desired_two_contacts = torch.sum(desired, dim=1) == 2
        tracking = torch.exp(
            -self._phase_support_error()
            / float(self.cfg.rewards.phase_support_sigma)
        )
        return (
            tracking
            * desired_two_contacts.to(dtype=torch.float)
            * self._walking_command_gate()
        )

    def _reward_odd_feet_contact(self):
        """Block the one/three-foot loophole while leaving both diagonals free."""
        contact_count = torch.sum(self.get_foot_contact_mask(), dim=1)
        odd_support = torch.remainder(contact_count, 2) == 1
        return odd_support.to(dtype=torch.float) * self._walking_command_gate()
