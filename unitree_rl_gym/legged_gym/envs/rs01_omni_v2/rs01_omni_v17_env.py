"""One conditional posture prior inside the existing gait-quality budget."""
import torch
from .rs01_omni_v16_env import Rs01OmniV16Robot


def conditional_roll_cost(roll, raw_commands, cfg):
    # Commands are exogenous raw intent, not the heading controller's
    # corrective velocity. Do not penalize corrective vy or yaw rate.
    gate = (1.0 - raw_commands[:, 1].abs() / cfg.balance_lateral_release_m_s
            - raw_commands[:, 2].abs() / cfg.balance_yaw_release_rad_s).clamp(0., 1.)
    excess = ((roll.abs() - cfg.balance_roll_deadband_rad)
              / cfg.balance_roll_scale_rad).clamp(min=0.)
    huber = torch.where(excess < 1., .5 * excess.square(), excess - .5)
    return gate * huber


class Rs01OmniV17Robot(Rs01OmniV16Robot):
    def _reward_phase_two_contact_quality(self):
        support_quality = super()._reward_phase_two_contact_quality()
        cost = conditional_roll_cost(self.rpy[:, 0], self.commands[:, :3], self.cfg.rewards)
        cost = cost * self._walking_command_gate()
        # Subtract rather than multiply: a tilted robot still receives the
        # same marginal incentive for correct support. No tracking gate.
        self.v17_balance_cost = cost.detach()
        reward = support_quality - self.cfg.rewards.balance_prior_weight * cost
        self.v11_contact_quality_reward = reward.detach()
        return reward
