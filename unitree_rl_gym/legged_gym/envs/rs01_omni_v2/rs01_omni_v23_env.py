"""Replace the existing placement weighting, never add a duplicate reward.

V21 still owns foot geometry + conditional half-cycle coordination; V18 owns
the sole envelope roll cost. V22 sensor-derived command/host guard is unchanged.
No action mirroring, new hard hip clamp, phase reset, or synthetic sensor change.
"""
from .rs01_omni_v22_env import Rs01OmniV22Robot


class Rs01OmniV23Robot(Rs01OmniV22Robot):
    def _placement_reward_weight(self):
        straight = (1. - self.commands[:, 1].abs()/.08
                    - self.commands[:, 2].abs()/.20).clamp(0., 1.)
        cfg = self.cfg.rewards
        return cfg.omni_placement_weight + straight*(cfg.placement_weight-cfg.omni_placement_weight)
