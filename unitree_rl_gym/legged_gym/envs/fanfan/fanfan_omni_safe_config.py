"""Single-policy omnidirectional continuation from the sim2real forward gait."""
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO


class FanfanOmniSafeCfg(FanfanRoughCfg):
    class commands(FanfanRoughCfg.commands):
        heading_command = False
        observe_heading_error = True
        resampling_time = 8.0
        pure_yaw_probability = 0.18
        stand_probability = 0.08
        pure_lateral_probability = 0.20
        pure_sagittal_probability = 0.28
        omni_curriculum = True
        # Add one capability at a time while continuously replaying forward gait.
        omni_curriculum_stages = [
            {"until_iteration": 200, "lin_vel_x": [0.05, 0.28],
             "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [-0.35, 0.35]},
            {"until_iteration": 500, "lin_vel_x": [-0.05, 0.28],
             "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [-0.50, 0.50]},
            {"until_iteration": 900, "lin_vel_x": [-0.08, 0.28],
             "lin_vel_y": [-0.04, 0.04], "ang_vel_yaw": [-0.50, 0.50]},
            {"until_iteration": 1300, "lin_vel_x": [-0.10, 0.30],
             "lin_vel_y": [-0.06, 0.06], "ang_vel_yaw": [-0.60, 0.60]},
            {"until_iteration": 1.0e12, "lin_vel_x": [-0.12, 0.30],
             "lin_vel_y": [-0.08, 0.08], "ang_vel_yaw": [-0.70, 0.70]},
        ]

        class ranges(FanfanRoughCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.30]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.70, 0.70]

    class rewards(FanfanRoughCfg.rewards):
        lateral_tracking_sigma = 0.0004
        longitudinal_tracking_sigma = 0.002

        class scales(FanfanRoughCfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_lateral_vel = 4.0
            tracking_longitudinal_vel = 4.0
            tracking_ang_vel = 4.0
            heading_tracking = 3.0
            backward_velocity = 0.0
            lateral_velocity = 0.0
            yaw_rate = -0.20
            lateral_hip_common_mode = -1.0
            lateral_yaw_error = -8.0
            translation_yaw_error = -6.0


class FanfanOmniSafeCfgPPO(FanfanRoughCfgPPO):
    class algorithm(FanfanRoughCfgPPO.algorithm):
        entropy_coef = 0.003

    class runner(FanfanRoughCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_safe"
        run_name = "omni_safe"


class FanfanOmniSafe900Cfg(FanfanOmniSafeCfg):
    """Deployment snapshot: command envelope actually learned by model_900."""
    class commands(FanfanOmniSafeCfg.commands):
        omni_curriculum = False

        class ranges(FanfanOmniSafeCfg.commands.ranges):
            lin_vel_x = [-0.08, 0.28]
            lin_vel_y = [-0.04, 0.04]
            ang_vel_yaw = [-0.50, 0.50]


class FanfanOmniFastCfg(FanfanOmniSafeCfg):
    """Higher-speed continuation with a gradually enlarged action envelope."""
    class control(FanfanOmniSafeCfg.control):
        # Increase stride authority gradually through retraining, while keeping
        # the sim2real-proven Kp/Kd and torque limits unchanged.
        action_scale = 0.20
        rear_action_scale = 0.22
        hip_action_scale = 0.10

    class commands(FanfanOmniSafeCfg.commands):
        resampling_time = 7.0
        omni_curriculum = True
        omni_curriculum_stages = [
            {"until_iteration": 250, "lin_vel_x": [-0.08, 0.32],
             "lin_vel_y": [-0.05, 0.05], "ang_vel_yaw": [-0.55, 0.55]},
            {"until_iteration": 600, "lin_vel_x": [-0.10, 0.36],
             "lin_vel_y": [-0.06, 0.06], "ang_vel_yaw": [-0.65, 0.65]},
            {"until_iteration": 1.0e12, "lin_vel_x": [-0.12, 0.40],
             "lin_vel_y": [-0.08, 0.08], "ang_vel_yaw": [-0.80, 0.80]},
        ]

        class ranges(FanfanOmniSafeCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.40]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.80, 0.80]

    class rewards(FanfanOmniSafeCfg.rewards):
        class scales(FanfanOmniSafeCfg.rewards.scales):
            tracking_lin_vel = 7.0
            tracking_lateral_vel = 5.0
            tracking_longitudinal_vel = 5.0
            tracking_ang_vel = 5.0


class FanfanOmniFastCfgPPO(FanfanOmniSafeCfgPPO):
    class runner(FanfanOmniSafeCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_fast"
        run_name = "omni_fast"
