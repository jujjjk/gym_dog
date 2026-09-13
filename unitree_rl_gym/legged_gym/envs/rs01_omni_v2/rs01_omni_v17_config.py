"""Correct measured static lean without imposing motor/load mirroring."""
from .rs01_omni_v16_config import Rs01OmniV16Cfg, Rs01OmniV16CfgPPO


class Rs01OmniV17Cfg(Rs01OmniV16Cfg):
    class asset(Rs01OmniV16Cfg.asset):
        name = 'rs01_omni_v17_balance'

    class rewards(Rs01OmniV16Cfg.rewards):
        # Evidence: B6000 signed roll -5 to -10 degrees on straight/march.
        # Allow a 2-degree compensation band; Huber tail is not a hard lock.
        balance_roll_deadband_rad = 0.03490658503988659
        balance_roll_scale_rad = 0.10471975511965977
        balance_prior_weight = 1.0
        balance_lateral_release_m_s = 0.08
        balance_yaw_release_rad_s = 0.20
        # Same existing one-sided swing-clearance term, not another reward.
        swing_clearance_m = 0.018


class Rs01OmniV17CfgPPO(Rs01OmniV16CfgPPO):
    class runner(Rs01OmniV16CfgPPO.runner):
        experiment_name = 'rs01_omni_v17_balance'
