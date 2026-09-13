"""Controlled A/B: stronger existing geometry versus bounded hip adduction."""
from .rs01_omni_v19_config import Rs01OmniV19SoftCfg, Rs01OmniV19SoftCfgPPO


class Rs01OmniV20ACfg(Rs01OmniV19SoftCfg):
    class asset(Rs01OmniV19SoftCfg.asset):
        name = 'rs01_omni_v20_geometry'

    class rewards(Rs01OmniV19SoftCfg.rewards):
        placement_weight = 1.0


class Rs01OmniV20BCfg(Rs01OmniV20ACfg):
    class asset(Rs01OmniV20ACfg.asset):
        name = 'rs01_omni_v20_bounded_hip'

    class control(Rs01OmniV20ACfg.control):
        # Desired target only, NOT a clamp of physical joint state.
        straight_inward_target_rad = 0.13962634015954636  # 8 degrees


class Rs01OmniV20ACfgPPO(Rs01OmniV19SoftCfgPPO):
    class runner(Rs01OmniV19SoftCfgPPO.runner):
        experiment_name = 'rs01_omni_v20_geometry'


class Rs01OmniV20BCfgPPO(Rs01OmniV20ACfgPPO):
    class runner(Rs01OmniV20ACfgPPO.runner):
        experiment_name = 'rs01_omni_v20_bounded_hip'
