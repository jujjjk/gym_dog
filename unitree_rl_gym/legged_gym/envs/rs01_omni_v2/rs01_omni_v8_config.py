"""Pure scratch exploration with usable RS01 joint-target authority."""

from .rs01_omni_v5_config import (
    Rs01OmniV5Odd10Cfg,
    Rs01OmniV5Odd10CfgPPO,
)


class Rs01OmniV8Clearance2Cfg(Rs01OmniV5Odd10Cfg):
    """Keep physical caps, but let random actions reach a low swing pose."""

    class asset(Rs01OmniV5Odd10Cfg.asset):
        name = "rs01_omni_v8_clearance2"

    class control(Rs01OmniV5Odd10Cfg.control):
        action_scale_by_joint = {
            "hip": 0.22,
            "thigh": 0.22,
            "calf": 0.22,
        }

    class rs01_actuator(Rs01OmniV5Odd10Cfg.rs01_actuator):
        # Remove the extra calf restriction introduced only for the old
        # high-raw-torque Sim2Sim adaptation. These are the measured base
        # 50 Hz controller limits; the 17 N m electromagnetic cap remains.
        target_rate_limit_rad_s = {
            "hip": 2.0,
            "thigh": 2.6,
            "calf": 3.2,
        }
        target_acceleration_limit_rad_s2 = {
            "hip": 60.0,
            "thigh": 78.0,
            "calf": 96.0,
        }

    class rewards(Rs01OmniV5Odd10Cfg.rewards):
        class scales(Rs01OmniV5Odd10Cfg.rewards.scales):
            phase_swing_clearance = -2.0


class Rs01OmniV8Clearance4Cfg(Rs01OmniV8Clearance2Cfg):
    """Same scratch task, changing only paired swing-clearance pressure."""

    class asset(Rs01OmniV8Clearance2Cfg.asset):
        name = "rs01_omni_v8_clearance4"

    class rewards(Rs01OmniV8Clearance2Cfg.rewards):
        class scales(Rs01OmniV8Clearance2Cfg.rewards.scales):
            phase_swing_clearance = -4.0


class Rs01OmniV8Clearance2CfgPPO(Rs01OmniV5Odd10CfgPPO):
    class policy(Rs01OmniV5Odd10CfgPPO.policy):
        init_noise_std = 0.45

    class runner(Rs01OmniV5Odd10CfgPPO.runner):
        experiment_name = "rs01_omni_v8_clearance2"
        action_std_value = 0.45


class Rs01OmniV8Clearance4CfgPPO(Rs01OmniV8Clearance2CfgPPO):
    class runner(Rs01OmniV8Clearance2CfgPPO.runner):
        experiment_name = "rs01_omni_v8_clearance4"
