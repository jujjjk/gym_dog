"""Command-conditioned inward target compression, upstream of real RS01 plant.

Deployment contract: apply map_hip_targets after default + scale * clip(action),
before the existing rate/acceleration limiter. Never export B as a bare V19 actor.
Previous-action observations and saturation reward retain original semantics.
"""
import torch
from .rs01_omni_v19_env import Rs01OmniV19Robot


def map_hip_targets(target, commands, hip_ids, sides, full_range, inward_limit):
    straight = (1. - commands[:, 1].abs() / .08
                - commands[:, 2].abs() / .20).clamp(0., 1.)
    factor = 1. - straight[:, None] * (1. - inward_limit / full_range)
    hip = target[:, hip_ids]
    # URDF: left inward negative; right inward positive. Outward unchanged.
    result = target.clone()
    result[:, hip_ids] = torch.where(hip * sides < 0., hip * factor, hip)
    return result


class Rs01OmniV20GeometryRobot(Rs01OmniV19Robot):
    def _placement_reward_weight(self):
        straight = (1. - self.commands[:, 1].abs() / .08
                    - self.commands[:, 2].abs() / .20).clamp(0., 1.)
        return .5 + straight * (self.cfg.rewards.placement_weight - .5)


class Rs01OmniV20BoundedRobot(Rs01OmniV20GeometryRobot):
    def _project_rs01_policy_target(self, target):
        if not hasattr(self, 'v20_hip_ids'):
            legs = ('FL', 'FR', 'RL', 'RR')
            self.v20_hip_ids = [self.dof_names.index(x + '_hip_joint') for x in legs]
            self.v20_sides = target.new_tensor([1., -1., 1., -1.])
            if not torch.allclose(self.default_dof_pos[:, self.v20_hip_ids],
                                  torch.zeros_like(self.default_dof_pos[:, self.v20_hip_ids])):
                raise ValueError('V20 mapping requires zero default hip angles')
        ranges = self.rs01_action_scale_rad[self.v20_hip_ids] * self.cfg.normalization.clip_actions
        return map_hip_targets(target, self.commands, self.v20_hip_ids,
                               self.v20_sides, ranges,
                               self.cfg.control.straight_inward_target_rad)
