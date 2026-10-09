"""Isolate the omni hip-target loophole; no further reward/cadence changes."""
import math
from .rs01_omni_v25_config import Rs01OmniV25Cfg, Rs01OmniV25CfgPPO


class Rs01OmniV26Cfg(Rs01OmniV25Cfg):
    class control(Rs01OmniV25Cfg.control):
        omni_inward_target_rad = math.radians(3.)


class Rs01OmniV26SoftCfg(Rs01OmniV26Cfg):
    class control(Rs01OmniV26Cfg.control):
        omni_inward_target_rad = math.radians(5.)


class Rs01OmniV26CfgPPO(Rs01OmniV25CfgPPO):
    class runner(Rs01OmniV25CfgPPO.runner):
        experiment_name = 'rs01_omni_v26_inward3'


class Rs01OmniV26SoftCfgPPO(Rs01OmniV26CfgPPO):
    class runner(Rs01OmniV26CfgPPO.runner):
        experiment_name = 'rs01_omni_v26_inward5'
