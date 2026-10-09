"""Stage1: support/landing geometry only. Perturbations are explicit audit inputs."""
from .rs01_omni_v23_config import Rs01OmniV23Cfg, Rs01OmniV23CfgPPO


class Rs01OmniV24Cfg(Rs01OmniV23Cfg):
    class rewards(Rs01OmniV23Cfg.rewards):
        support_inner_m = .10
        support_outer_m = .22
        support_region_scale_m = .03
        touchdown_region_weight = 1.0

    class disturbance:
        # Each entry: {mass_delta_kg, com_offset_m}; empty means untouched URDF.
        # Creation-time ONLY. Not sampled or changed within an episode.
        audit_profiles = []


class Rs01OmniV24WideCfg(Rs01OmniV24Cfg):
    class rewards(Rs01OmniV24Cfg.rewards):
        support_inner_m = .11


class Rs01OmniV24CfgPPO(Rs01OmniV23CfgPPO):
    class runner(Rs01OmniV23CfgPPO.runner):
        experiment_name = 'rs01_omni_v24_support10'


class Rs01OmniV24WideCfgPPO(Rs01OmniV24CfgPPO):
    class runner(Rs01OmniV24CfgPPO.runner):
        experiment_name = 'rs01_omni_v24_support11'
