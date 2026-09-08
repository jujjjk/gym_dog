"""Wide-speed exploration on the unchanged RS01 physical/control model."""

from .rs01_omni_v11_config import Rs01OmniV11ContinuousCfg, Rs01OmniV11ContinuousCfgPPO


class Rs01OmniV12WideCfg(Rs01OmniV11ContinuousCfg):
    class asset(Rs01OmniV11ContinuousCfg.asset):
        name = "rs01_omni_v12_wide"

    class commands(Rs01OmniV11ContinuousCfg.commands):
        forward_velocity_range_m_s = [0.10, 0.60]
        backward_speed_range_m_s = [0.10, 0.35]
        lateral_speed_range_m_s = [0.08, 0.30]
        yaw_speed_range_rad_s = [0.20, 1.00]
        combined_forward_speed_range_m_s = [0.06, 0.40]
        combined_lateral_speed_range_m_s = [0.03, 0.20]
        combined_yaw_speed_range_rad_s = [0.10, 0.70]
        # Fractional edges within each magnitude range. All bands are active
        # from update zero; this is not a low-speed-only curriculum.
        speed_band_edges = [0.0, 0.2, 0.6, 1.0]
        speed_band_probabilities = [0.30, 0.40, 0.30]
        combined_axis_caps = [0.40, 0.20, 0.70]

        class ranges(Rs01OmniV11ContinuousCfg.commands.ranges):
            # Combined vx can be negative, down to -0.40 m/s.
            lin_vel_x = [-0.40, 0.60]
            lin_vel_y = [-0.30, 0.30]
            ang_vel_yaw = [-1.00, 1.00]

    class rewards(Rs01OmniV11ContinuousCfg.rewards):
        # The unchanged 0.60 s period is the low-speed/march period.
        wide_gait_max_frequency_hz = 2.50
        wide_gait_speed_deadband = 1.0 / 3.0


class Rs01OmniV12WideCfgPPO(Rs01OmniV11ContinuousCfgPPO):
    class runner(Rs01OmniV11ContinuousCfgPPO.runner):
        experiment_name = "rs01_omni_v12_wide"
