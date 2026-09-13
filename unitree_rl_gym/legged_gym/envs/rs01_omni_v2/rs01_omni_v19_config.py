"""Replace footprint shaping; keep the identified RS01 control contract."""
from .rs01_omni_v18_config import Rs01OmniV18Cfg, Rs01OmniV18CfgPPO


class Rs01OmniV19Cfg(Rs01OmniV18Cfg):
    class asset(Rs01OmniV18Cfg.asset):
        name = 'rs01_omni_v19_placement'

    class rewards(Rs01OmniV18Cfg.rewards):
        placement_filter_tau_s = 0.6
        placement_deadband_m = 0.015
        placement_scale_m = 0.03
        placement_weight = 1.0
        footprint_straight_min_width_m = 0.22

        class scales(Rs01OmniV18Cfg.rewards.scales):
            action_saturation = -0.5


class Rs01OmniV19SoftCfg(Rs01OmniV19Cfg):
    class asset(Rs01OmniV19Cfg.asset):
        name = 'rs01_omni_v19_placement_soft'

    class rewards(Rs01OmniV19Cfg.rewards):
        placement_weight = 0.5


class Rs01OmniV19CfgPPO(Rs01OmniV18CfgPPO):
    class runner(Rs01OmniV18CfgPPO.runner):
        experiment_name = 'rs01_omni_v19_placement'


class Rs01OmniV19SoftCfgPPO(Rs01OmniV19CfgPPO):
    class runner(Rs01OmniV19CfgPPO.runner):
        experiment_name = 'rs01_omni_v19_placement_soft'
