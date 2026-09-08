"""One direction-aware velocity objective; no accumulated-position objective."""

from .rs01_omni_v12_config import Rs01OmniV12WideCfg, Rs01OmniV12WideCfgPPO


class Rs01OmniV13DirectionCfg(Rs01OmniV12WideCfg):
    class env(Rs01OmniV12WideCfg.env):
        # Original 57 slots + raw command xyz + velocity-estimator confidence.
        num_observations = 61

    class asset(Rs01OmniV12WideCfg.asset):
        name = "rs01_omni_v13_direction"

    class commands(Rs01OmniV12WideCfg.commands):
        direction_heading_gain = 1.2
        direction_yaw_correction_limit_rad_s = 0.35
        direction_rotation_limit_rad = 0.1745329252
        direction_planar_correction_limit_m_s = 0.08
        direction_turn_enter_rad_s = 0.06
        direction_turn_exit_rad_s = 0.02
        direction_blend_yaw_rad_s = 0.06

    class rewards(Rs01OmniV12WideCfg.rewards):
        class scales(Rs01OmniV12WideCfg.rewards.scales):
            # Velocity tracking is the ONLY task objective. Old position/yaw
            # costs must not fight the command's permitted recovery motion.
            pose_error = 0.0
            trajectory_lateral_error = 0.0
            omni_heading_error = 0.0


class Rs01OmniV13DirectionCfgPPO(Rs01OmniV12WideCfgPPO):
    class runner(Rs01OmniV12WideCfgPPO.runner):
        experiment_name = "rs01_omni_v13_direction"
