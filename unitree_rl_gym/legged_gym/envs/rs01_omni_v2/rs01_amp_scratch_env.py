import torch
from .rs01_amp_env import Rs01AMPRobot


def legal_diagonal_or_handoff(contact, diagonal_a, diagonal_b):
    return (contact==diagonal_a[None,:]).all(1)|(contact==diagonal_b[None,:]).all(1)|contact.all(1)


class Rs01AMPScratchRobot(Rs01AMPRobot):
    def _legal_task_contact_gate(self):
        # AMP learns relative timing, not mandatory alignment with an external clock.
        # Four-foot transfer is allowed; its prolonged duration has one penalty.
        return legal_diagonal_or_handoff(self.get_foot_contact_mask(),
            self.diagonal_a_contact_mask,self.diagonal_b_contact_mask).float()

    def _reward_illegal_support(self):
        return 1.-self._legal_task_contact_gate()
