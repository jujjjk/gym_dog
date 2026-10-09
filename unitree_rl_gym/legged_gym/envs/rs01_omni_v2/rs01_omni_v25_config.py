"""Directional cadence floor and landing-first geometry; unchanged RS01 plant."""
from .rs01_omni_v24_config import Rs01OmniV24WideCfg, Rs01OmniV24CfgPPO


class Rs01OmniV25Cfg(Rs01OmniV24WideCfg):
    class rewards(Rs01OmniV24WideCfg.rewards):
        lateral_frequency_floor_hz = 2.0
        cadence_slew_hz_s = 2.0
        cadence_lateral_full_m_s = .08
        cadence_yaw_full_rad_s = .20
        landing_inner_m = .11
        stance_inner_m = .08
        stance_region_weight = .25
        late_swing_start = .50
        straight_clearance_m = .030
        # Lateral/turning clearance remains .025 m.


class Rs01OmniV25FastCfg(Rs01OmniV25Cfg):
    class rewards(Rs01OmniV25Cfg.rewards):
        lateral_frequency_floor_hz = 2.2


class Rs01OmniV25CfgPPO(Rs01OmniV24CfgPPO):
    class runner(Rs01OmniV24CfgPPO.runner):
        experiment_name = 'rs01_omni_v25_cadence20'


class Rs01OmniV25FastCfgPPO(Rs01OmniV25CfgPPO):
    class runner(Rs01OmniV25CfgPPO.runner):
        experiment_name = 'rs01_omni_v25_cadence22'
