"""Controlled clearance A/B with one replacement posture prior."""
from .rs01_omni_v17_config import Rs01OmniV17Cfg, Rs01OmniV17CfgPPO


class Rs01OmniV18Cfg(Rs01OmniV17Cfg):
    class asset(Rs01OmniV17Cfg.asset):
        name = 'rs01_omni_v18_balance18'

    class rewards(Rs01OmniV17Cfg.rewards):
        # Straight/march: 2 deg compensation; omni: widen to 5 deg,
        # never turn posture supervision off. No joint/force mirroring.
        balance_omni_deadband_rad = 0.08726646259971647
        balance_roll_scale_rad = 0.05235987755982989
        balance_prior_weight = 3.0
        swing_clearance_m = 0.018


class Rs01OmniV18LowCfg(Rs01OmniV18Cfg):
    class asset(Rs01OmniV18Cfg.asset):
        name = 'rs01_omni_v18_balance14'

    class rewards(Rs01OmniV18Cfg.rewards):
        swing_clearance_m = 0.014


class Rs01OmniV18CfgPPO(Rs01OmniV17CfgPPO):
    class runner(Rs01OmniV17CfgPPO.runner):
        experiment_name = 'rs01_omni_v18_balance18'


class Rs01OmniV18LowCfgPPO(Rs01OmniV18CfgPPO):
    class runner(Rs01OmniV18CfgPPO.runner):
        experiment_name = 'rs01_omni_v18_balance14'


class Rs01OmniV18SoftCfg(Rs01OmniV18Cfg):
    class asset(Rs01OmniV18Cfg.asset):
        name = 'rs01_omni_v18_balance_soft'

    class rewards(Rs01OmniV18Cfg.rewards):
        balance_prior_weight = 1.5


class Rs01OmniV18SoftCfgPPO(Rs01OmniV18CfgPPO):
    class runner(Rs01OmniV18CfgPPO.runner):
        experiment_name = 'rs01_omni_v18_balance_soft'
