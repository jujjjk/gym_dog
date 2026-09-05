"""Random-policy training with a short, decaying diagonal exploration seed."""

from .rs01_omni_v5_config import (
    Rs01OmniV5Odd10Cfg,
    Rs01OmniV5Odd10CfgPPO,
)


class Rs01OmniV6Seed08Cfg(Rs01OmniV5Odd10Cfg):
    """Use a conservative correlated seed for the first 200 PPO updates."""

    class asset(Rs01OmniV5Odd10Cfg.asset):
        name = "rs01_omni_v6_seed08"

    class control(Rs01OmniV5Odd10Cfg.control):
        structured_exploration_enabled = True
        structured_exploration_amplitude = 0.80
        # 24 policy steps are collected by each PPO update. The final third
        # of the 300-update behavior pilot therefore runs with no seed.
        structured_exploration_decay_steps = 4800
        # The current dog_rs01 URDF/actuator probe requires both sagittal
        # joints to move negative to unload a diagonal pair. At amplitude 1.0
        # these correspond to -0.28 rad calf and -0.18 rad thigh targets.
        structured_exploration_calf_action = -2.0
        structured_exploration_swing_thigh_action = -1.0
        structured_exploration_stride_thigh_action = 0.55
        structured_exploration_full_stride_speed_m_s = 0.20
        structured_exploration_profile = "sine"
        structured_exploration_lift_fraction = 0.18
        structured_exploration_lower_start_fraction = 0.70
        structured_exploration_phase_lead = 0.0


class Rs01OmniV6Seed11Cfg(Rs01OmniV6Seed08Cfg):
    """Same task with a 37.5% larger, still rate/torque-limited seed."""

    class asset(Rs01OmniV6Seed08Cfg.asset):
        name = "rs01_omni_v6_seed11"

    class control(Rs01OmniV6Seed08Cfg.control):
        structured_exploration_amplitude = 1.10


class Rs01OmniV6Seed08CfgPPO(Rs01OmniV5Odd10CfgPPO):
    class runner(Rs01OmniV5Odd10CfgPPO.runner):
        experiment_name = "rs01_omni_v6_seed08"


class Rs01OmniV6Seed11CfgPPO(Rs01OmniV5Odd10CfgPPO):
    class runner(Rs01OmniV5Odd10CfgPPO.runner):
        experiment_name = "rs01_omni_v6_seed11"
