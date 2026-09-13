"""Replace V17 lean cost; preserve the identified RS01 execution chain."""
import torch
from .rs01_omni_v16_env import Rs01OmniV16Robot


def envelope_roll_cost(roll, raw_commands, cfg):
    omni = (raw_commands[:, 1].abs() / cfg.balance_lateral_release_m_s
            + raw_commands[:, 2].abs() / cfg.balance_yaw_release_rad_s).clamp(0., 1.)
    tolerance = cfg.balance_roll_deadband_rad + omni * (
        cfg.balance_omni_deadband_rad - cfg.balance_roll_deadband_rad)
    excess = ((roll.abs() - tolerance) / cfg.balance_roll_scale_rad).clamp(min=0.)
    return torch.where(excess < 1., .5 * excess.square(), excess - .5)


class Rs01OmniV18Robot(Rs01OmniV16Robot):
    def _reward_phase_two_contact_quality(self):
        # Deliberately inherit V16: never subtract BOTH V17 and V18 costs.
        support_quality = super()._reward_phase_two_contact_quality()
        cost = envelope_roll_cost(self.rpy[:, 0], self.commands[:, :3], self.cfg.rewards)
        cost *= self._walking_command_gate()
        self.v17_balance_cost = cost.detach()  # shared diagnostic interface
        reward = support_quality - self.cfg.rewards.balance_prior_weight * cost
        self.v11_contact_quality_reward = reward.detach()
        return reward
