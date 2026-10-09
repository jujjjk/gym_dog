"""Stage 0: narrower lateral/reverse-combined sampling, no automatic expansion."""
from .rs01_omni_v27_config import Rs01OmniV27StrongCfg,Rs01OmniV27StrongCfgPPO


class Rs01OmniV28Cfg(Rs01OmniV27StrongCfg):
    class commands(Rs01OmniV27StrongCfg.commands):
        lateral_speed_range_m_s = [.08,.15]
        reverse_combined_axis_caps = [.18,.08,.25]
        # Sampling envelope must not silently change the physical gait clock.
        cadence_lateral_reference_m_s = .30


class Rs01OmniV28CfgPPO(Rs01OmniV27StrongCfgPPO):
    class runner(Rs01OmniV27StrongCfgPPO.runner):
        experiment_name = 'rs01_omni_v28_curriculum'
