"""Modest directional clearance increase; unchanged V29 clock and RS01 plant."""
from .rs01_omni_v29_config import Rs01OmniV29Cfg, Rs01OmniV29CfgPPO


class Rs01OmniV30Cfg(Rs01OmniV29Cfg):
    class rewards(Rs01OmniV29Cfg.rewards):
        straight_clearance_m = .035
        swing_clearance_m = .028


class Rs01OmniV30CfgPPO(Rs01OmniV29CfgPPO):
    class runner(Rs01OmniV29CfgPPO.runner):
        experiment_name = 'rs01_omni_v30_clearance'
