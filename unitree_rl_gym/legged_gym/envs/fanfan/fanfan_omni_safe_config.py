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


class FanfanOmniSmoothRealCfg(FanfanOmniFastCfg):
    """Real-data correction: smooth outputs and match the deployed actuator envelope."""
    class control(FanfanOmniFastCfg.control):
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        torque_limit_override = 10.0
        action_scale = 0.19
        rear_action_scale = 0.21
        hip_action_scale = 0.09
        gate_gait_with_command = True
        gait_command_gate_sigma = 0.0004

    class commands(FanfanOmniFastCfg.commands):
        stand_probability = 0.25
        pure_yaw_probability = 0.15
        pure_lateral_probability = 0.15
        pure_sagittal_probability = 0.25
        omni_curriculum_stages = [
            {"until_iteration": 250, "lin_vel_x": [-0.10, 0.32],
             "lin_vel_y": [-0.05, 0.05], "ang_vel_yaw": [-0.55, 0.55]},
            {"until_iteration": 600, "lin_vel_x": [-0.12, 0.36],
             "lin_vel_y": [-0.06, 0.06], "ang_vel_yaw": [-0.65, 0.65]},
            {"until_iteration": 1.0e12, "lin_vel_x": [-0.12, 0.40],
             "lin_vel_y": [-0.08, 0.08], "ang_vel_yaw": [-0.80, 0.80]},
        ]

    class rewards(FanfanOmniFastCfg.rewards):
        action_saturation_threshold = 0.75
        stand_command_sigma = 0.0004

        class scales(FanfanOmniFastCfg.rewards.scales):
            action_magnitude = -0.04
            action_rate = -0.18
            action_saturation = -0.30
            stand_action = -0.30
            stand_dof_velocity = -0.003
            dof_acc = -3.0e-7
            torques = -5.0e-6


class FanfanOmniSmoothRealCfgPPO(FanfanOmniFastCfgPPO):
    class algorithm(FanfanOmniFastCfgPPO.algorithm):
        entropy_coef = 0.001

    class runner(FanfanOmniFastCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_smooth_real"
        run_name = "omni_smooth_real"


class FanfanOmniFilteredCfg(FanfanOmniSmoothRealCfg):
    """Train through the same action smoothing envelope used on hardware."""
    class control(FanfanOmniSmoothRealCfg.control):
        filter_policy_actions = True
        policy_action_filter_alpha = 0.25
        # Normalized action rates. With the configured action scales these map
        # approximately to 0.8/1.2/2.0 rad/s joint-target limits.
        policy_action_rate_limits = {
            "hip": 0.8 / 0.09,
            "thigh": 1.2 / 0.19,
            "calf": 2.0 / 0.19,
        }
        policy_action_accel_limits = {
            "hip": 25.0 / 0.09,
            "thigh": 40.0 / 0.19,
            "calf": 70.0 / 0.19,
        }

    class commands(FanfanOmniSmoothRealCfg.commands):
        stand_probability = 0.35
        pure_yaw_probability = 0.12
        pure_lateral_probability = 0.13
        pure_sagittal_probability = 0.22

    class rewards(FanfanOmniSmoothRealCfg.rewards):
        only_positive_rewards = False

        class scales(FanfanOmniSmoothRealCfg.rewards.scales):
            policy_action_magnitude = -0.12
            policy_action_rate = -0.30
            policy_action_saturation = -0.60
            policy_filter_gap = -0.40
            stand_action = -0.60


class FanfanOmniFilteredCfgPPO(FanfanOmniSmoothRealCfgPPO):
    class algorithm(FanfanOmniSmoothRealCfgPPO.algorithm):
        entropy_coef = 0.0005

    class runner(FanfanOmniSmoothRealCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_filtered"
        run_name = "omni_filtered"
