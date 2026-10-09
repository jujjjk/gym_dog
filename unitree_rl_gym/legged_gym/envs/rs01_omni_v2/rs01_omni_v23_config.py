"""Reward-only omni balance A/B, warm-started from V22 B23500."""
import math
from .rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CfgPPO


class Rs01OmniV23Cfg(Rs01OmniV22Cfg):
    class rewards(Rs01OmniV22Cfg.rewards):
        balance_omni_deadband_rad = math.radians(3.)
        footprint_omni_min_width_m = .18
        omni_placement_weight = 1.0


class Rs01OmniV23WideCfg(Rs01OmniV23Cfg):
    class rewards(Rs01OmniV23Cfg.rewards):
        footprint_omni_min_width_m = .20


class Rs01OmniV23CfgPPO(Rs01OmniV22CfgPPO):
    class runner(Rs01OmniV22CfgPPO.runner):
        experiment_name = 'rs01_omni_v23_balance18'
        load_optimizer = False


class Rs01OmniV23WideCfgPPO(Rs01OmniV23CfgPPO):
    class runner(Rs01OmniV23CfgPPO.runner):
        experiment_name = 'rs01_omni_v23_balance20'
