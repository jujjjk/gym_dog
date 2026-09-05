"""Speed-priority continuation after scratch diagonal-gait discovery."""

from .rs01_omni_v8_config import (
    Rs01OmniV8Clearance2Cfg,
    Rs01OmniV8Clearance2CfgPPO,
)


class Rs01OmniV9Speed10Cfg(Rs01OmniV8Clearance2Cfg):
    """Retain gait legality while making commanded velocity dominant."""

    class asset(Rs01OmniV8Clearance2Cfg.asset):
        name = "rs01_omni_v9_speed10"

    class rewards(Rs01OmniV8Clearance2Cfg.rewards):
        class scales(Rs01OmniV8Clearance2Cfg.rewards.scales):
            tracking_command_velocity = 10.0
            phase_support_tracking = 1.0
            phase_two_contact_quality = 0.75
            phase_swing_clearance = -1.0


class Rs01OmniV9Speed14Cfg(Rs01OmniV9Speed10Cfg):
    """Same continuation, changing only command-velocity reward strength."""

    class asset(Rs01OmniV9Speed10Cfg.asset):
        name = "rs01_omni_v9_speed14"

    class rewards(Rs01OmniV9Speed10Cfg.rewards):
        class scales(Rs01OmniV9Speed10Cfg.rewards.scales):
            tracking_command_velocity = 14.0


class Rs01OmniV9Speed10CfgPPO(Rs01OmniV8Clearance2CfgPPO):
    class policy(Rs01OmniV8Clearance2CfgPPO.policy):
        init_noise_std = 0.30

    class runner(Rs01OmniV8Clearance2CfgPPO.runner):
        experiment_name = "rs01_omni_v9_speed10"
        action_std_value = 0.30


class Rs01OmniV9Speed14CfgPPO(Rs01OmniV9Speed10CfgPPO):
    class runner(Rs01OmniV9Speed10CfgPPO.runner):
        experiment_name = "rs01_omni_v9_speed14"
