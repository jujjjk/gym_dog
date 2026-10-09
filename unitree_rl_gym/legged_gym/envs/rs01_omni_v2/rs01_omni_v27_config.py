"""Learn wider footholds without V26's new hip-target intervention."""
from .rs01_omni_v25_config import Rs01OmniV25Cfg,Rs01OmniV25CfgPPO


class Rs01OmniV27Cfg(Rs01OmniV25Cfg):
    class rewards(Rs01OmniV25Cfg.rewards):
        # Same landing/stance objective, shorter soft transition: no new scale entry.
        stance_inner_m = .10
        support_region_scale_m = .015


class Rs01OmniV27StrongCfg(Rs01OmniV27Cfg):
    class rewards(Rs01OmniV27Cfg.rewards):
        support_region_scale_m = .010


class Rs01OmniV27CfgPPO(Rs01OmniV25CfgPPO):
    class runner(Rs01OmniV25CfgPPO.runner):
        experiment_name = 'rs01_omni_v27_foot15'


class Rs01OmniV27StrongCfgPPO(Rs01OmniV27CfgPPO):
    class runner(Rs01OmniV27CfgPPO.runner):
        experiment_name = 'rs01_omni_v27_foot10'
