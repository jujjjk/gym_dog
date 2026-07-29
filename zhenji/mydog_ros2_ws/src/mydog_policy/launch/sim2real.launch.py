from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    motor_url = LaunchConfiguration("motor_base_url")
    onnx_path = LaunchConfiguration("onnx_path")

    state_estimator = Node(
        package="mydog_policy",
        executable="mydog_state_estimator_node",
        name="mydog_state_estimator_node",
        output="screen",
        parameters=[{
            "motor_base_url": motor_url,
            "estimator_hz": 50.0,
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
        }],
    )

    policy = Node(
        package="mydog_policy",
        executable="mydog_policy_node",
        name="mydog_policy_node",
        output="screen",
        parameters=[{
            "onnx_path": onnx_path,
            "motor_base_url": motor_url,
            "base_lin_vel_source": "estimator",
            "policy_hz": 50.0,
            "action_mode": "pure_rl",
            "enable_cmd_smoothing": LaunchConfiguration("enable_cmd_smoothing"),
            "max_cmd_x_rate_mps2": LaunchConfiguration("max_cmd_x_rate_mps2"),
            "max_cmd_y_rate_mps2": LaunchConfiguration("max_cmd_y_rate_mps2"),
            "max_cmd_yaw_rate_rad_s2": LaunchConfiguration("max_cmd_yaw_rate_rad_s2"),
            "deployment_gait_phase_period_scale": LaunchConfiguration(
                "deployment_gait_phase_period_scale"
            ),
            "deployment_command_scale_x_mul": LaunchConfiguration(
                "deployment_command_scale_x_mul"
            ),
            "deployment_command_scale_y_mul": LaunchConfiguration(
                "deployment_command_scale_y_mul"
            ),
            "deployment_command_scale_yaw_mul": LaunchConfiguration(
                "deployment_command_scale_yaw_mul"
            ),
            "enable_cmd_limits": LaunchConfiguration("enable_cmd_limits"),
            "cmd_min_x": LaunchConfiguration("cmd_min_x"),
            "cmd_max_x": LaunchConfiguration("cmd_max_x"),
            "cmd_min_y": LaunchConfiguration("cmd_min_y"),
            "cmd_max_y": LaunchConfiguration("cmd_max_y"),
            "cmd_min_yaw": LaunchConfiguration("cmd_min_yaw"),
            "cmd_max_yaw": LaunchConfiguration("cmd_max_yaw"),
            "clip_action": False,
            "enable_send": LaunchConfiguration("enable_send"),
            "print_only": LaunchConfiguration("print_only"),
            "startup_stand_first": LaunchConfiguration("startup_stand_first"),
            "startup_stand_kp": LaunchConfiguration("startup_stand_kp"),
            "startup_stand_kd": LaunchConfiguration("startup_stand_kd"),
            "startup_stand_hold_current_sec": LaunchConfiguration("startup_stand_hold_current_sec"),
            "startup_stand_ramp_sec": LaunchConfiguration("startup_stand_ramp_sec"),
            "startup_stand_timeout_sec": LaunchConfiguration("startup_stand_timeout_sec"),
            "startup_stand_max_rate_hip": LaunchConfiguration("startup_stand_max_rate_hip"),
            "startup_stand_max_rate_thigh": LaunchConfiguration("startup_stand_max_rate_thigh"),
            "startup_stand_max_rate_calf": LaunchConfiguration("startup_stand_max_rate_calf"),
            "startup_stand_max_step_rad": LaunchConfiguration("startup_stand_max_step_rad"),
            "startup_stand_ready_error_rad": LaunchConfiguration("startup_stand_ready_error_rad"),
            "startup_stand_stop_error_rad": LaunchConfiguration("startup_stand_stop_error_rad"),
            "startup_stand_stop_torque_nm": LaunchConfiguration("startup_stand_stop_torque_nm"),
            "require_cmd_vel": True,
            "cmd_vel_timeout_sec": LaunchConfiguration("cmd_vel_timeout_sec"),
            "zero_cmd_inhibits_policy": LaunchConfiguration("zero_cmd_inhibits_policy"),
            "enable_zero_cmd_stand_protection": LaunchConfiguration(
                "enable_zero_cmd_stand_protection"
            ),
            "zero_cmd_stand_x_threshold": LaunchConfiguration(
                "zero_cmd_stand_x_threshold"
            ),
            "zero_cmd_stand_y_threshold": LaunchConfiguration(
                "zero_cmd_stand_y_threshold"
            ),
            "zero_cmd_stand_yaw_threshold": LaunchConfiguration(
                "zero_cmd_stand_yaw_threshold"
            ),
            "enable_policy_action_cmd_gate": LaunchConfiguration(
                "enable_policy_action_cmd_gate"
            ),
            "policy_action_cmd_gate_start_ratio": LaunchConfiguration(
                "policy_action_cmd_gate_start_ratio"
            ),
            "policy_action_cmd_gate_full_ratio": LaunchConfiguration(
                "policy_action_cmd_gate_full_ratio"
            ),
            "policy_action_cmd_gate_max_scale": LaunchConfiguration(
                "policy_action_cmd_gate_max_scale"
            ),
            "reset_gait_phase_on_command_start": LaunchConfiguration(
                "reset_gait_phase_on_command_start"
            ),
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
            "motor_state_async": LaunchConfiguration("motor_state_async"),
            "motor_state_poll_hz": LaunchConfiguration("motor_state_poll_hz"),
            "require_online": LaunchConfiguration("require_online"),
            "send_kp": LaunchConfiguration("send_kp"),
            "send_kd": LaunchConfiguration("send_kd"),
            "use_model_pd_gains": LaunchConfiguration("use_model_pd_gains"),
            "model_kp_scale": LaunchConfiguration("model_kp_scale"),
            "model_kd_scale": LaunchConfiguration("model_kd_scale"),
            "motor_torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_limit_nm": LaunchConfiguration("torque_limit_nm"),
            "torque_safety_budget_nm": LaunchConfiguration("torque_safety_budget_nm"),
            "expected_active_torque_budget_nm": LaunchConfiguration(
                "expected_active_torque_budget_nm"
            ),
            "max_target_rate_rad_s": LaunchConfiguration("max_target_rate_rad_s"),
            "max_target_accel_rad_s2": LaunchConfiguration("max_target_accel_rad_s2"),
            "calf_target_rate_mul": LaunchConfiguration("calf_target_rate_mul"),
            "calf_target_accel_mul": LaunchConfiguration("calf_target_accel_mul"),
            "calf_err_limit_mul": LaunchConfiguration("calf_err_limit_mul"),
            "debug_print_arrays": LaunchConfiguration("debug_print_arrays"),
            "debug_csv_path": LaunchConfiguration("debug_csv_path"),
            "debug_csv_period_sec": LaunchConfiguration("debug_csv_period_sec"),
            "debug_csv_async": LaunchConfiguration("debug_csv_async"),
            "debug_csv_queue_size": LaunchConfiguration("debug_csv_queue_size"),
            "debug_csv_flush_every_n": LaunchConfiguration("debug_csv_flush_every_n"),
            "expected_policy_task": LaunchConfiguration("expected_policy_task"),
            "expected_policy_sha256": LaunchConfiguration("expected_policy_sha256"),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value="/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/policy.onnx",
        ),
        DeclareLaunchArgument("motor_base_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("enable_cmd_smoothing", default_value="false"),
        DeclareLaunchArgument("max_cmd_x_rate_mps2", default_value="0.05"),
        DeclareLaunchArgument("max_cmd_y_rate_mps2", default_value="0.05"),
        DeclareLaunchArgument("max_cmd_yaw_rate_rad_s2", default_value="0.3"),
        DeclareLaunchArgument("deployment_gait_phase_period_scale", default_value="1.0"),
        DeclareLaunchArgument("deployment_command_scale_x_mul", default_value="1.0"),
        DeclareLaunchArgument("deployment_command_scale_y_mul", default_value="1.0"),
        DeclareLaunchArgument("deployment_command_scale_yaw_mul", default_value="1.0"),
        DeclareLaunchArgument("enable_cmd_limits", default_value="true"),
        DeclareLaunchArgument("cmd_min_x", default_value="0.10"),
        DeclareLaunchArgument("cmd_max_x", default_value="0.20"),
        DeclareLaunchArgument("cmd_min_y", default_value="0.0"),
        DeclareLaunchArgument("cmd_max_y", default_value="0.0"),
        DeclareLaunchArgument("cmd_min_yaw", default_value="0.0"),
        DeclareLaunchArgument("cmd_max_yaw", default_value="0.0"),
        DeclareLaunchArgument("startup_stand_first", default_value="true"),
        DeclareLaunchArgument("startup_stand_kp", default_value="12.0"),
        DeclareLaunchArgument("startup_stand_kd", default_value="2.0"),
        DeclareLaunchArgument("startup_stand_hold_current_sec", default_value="0.5"),
        DeclareLaunchArgument("startup_stand_ramp_sec", default_value="12.0"),
        DeclareLaunchArgument("startup_stand_timeout_sec", default_value="25.0"),
        DeclareLaunchArgument("startup_stand_max_rate_hip", default_value="0.12"),
        DeclareLaunchArgument("startup_stand_max_rate_thigh", default_value="0.15"),
        DeclareLaunchArgument("startup_stand_max_rate_calf", default_value="0.15"),
        DeclareLaunchArgument("startup_stand_max_step_rad", default_value="0.003"),
        DeclareLaunchArgument("startup_stand_ready_error_rad", default_value="0.08"),
        DeclareLaunchArgument("startup_stand_stop_error_rad", default_value="0.30"),
        DeclareLaunchArgument("startup_stand_stop_torque_nm", default_value="8.0"),
        DeclareLaunchArgument("cmd_vel_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("zero_cmd_inhibits_policy", default_value="true"),
        DeclareLaunchArgument("enable_zero_cmd_stand_protection", default_value="false"),
        DeclareLaunchArgument("zero_cmd_stand_x_threshold", default_value="0.01"),
        DeclareLaunchArgument("zero_cmd_stand_y_threshold", default_value="0.01"),
        DeclareLaunchArgument("zero_cmd_stand_yaw_threshold", default_value="0.03"),
        DeclareLaunchArgument("enable_policy_action_cmd_gate", default_value="false"),
        DeclareLaunchArgument("policy_action_cmd_gate_start_ratio", default_value="0.05"),
        DeclareLaunchArgument("policy_action_cmd_gate_full_ratio", default_value="1.0"),
        DeclareLaunchArgument("policy_action_cmd_gate_max_scale", default_value="1.0"),
        DeclareLaunchArgument("reset_gait_phase_on_command_start", default_value="false"),
        DeclareLaunchArgument("require_online", default_value="true"),
        DeclareLaunchArgument("max_motor_age_ms", default_value="100.0"),
        DeclareLaunchArgument("motor_state_async", default_value="true"),
        DeclareLaunchArgument("motor_state_poll_hz", default_value="50.0"),
        DeclareLaunchArgument("send_kp", default_value="40.0"),
        DeclareLaunchArgument("send_kd", default_value="1.2"),
        DeclareLaunchArgument("use_model_pd_gains", default_value="true"),
        DeclareLaunchArgument("model_kp_scale", default_value="1.0"),
        DeclareLaunchArgument("model_kd_scale", default_value="1.0"),
        DeclareLaunchArgument("motor_torque_limit_nm", default_value="6.0"),
        DeclareLaunchArgument("torque_limit_nm", default_value="-1.0"),
        DeclareLaunchArgument("torque_safety_budget_nm", default_value="-1.0"),
        DeclareLaunchArgument("expected_active_torque_budget_nm", default_value="-1.0"),
        DeclareLaunchArgument("max_target_rate_rad_s", default_value="2.0"),
        DeclareLaunchArgument("max_target_accel_rad_s2", default_value="60.0"),
        DeclareLaunchArgument("calf_target_rate_mul", default_value="3.0"),
        DeclareLaunchArgument("calf_target_accel_mul", default_value="3.0"),
        DeclareLaunchArgument("calf_err_limit_mul", default_value="1.6"),
        DeclareLaunchArgument("debug_print_arrays", default_value="false"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value="/home/jetson/mydog_ros2_ws/log/sim2real_debug.csv",
        ),
        DeclareLaunchArgument("debug_csv_period_sec", default_value="0.1"),
        DeclareLaunchArgument("debug_csv_async", default_value="true"),
        DeclareLaunchArgument("debug_csv_queue_size", default_value="128"),
        DeclareLaunchArgument("debug_csv_flush_every_n", default_value="20"),
        DeclareLaunchArgument("expected_policy_task", default_value="FanfanStraightPD8Cfg"),
        DeclareLaunchArgument(
            "expected_policy_sha256",
            default_value="e12d25d718e6afe12dc84368341f42dbb9eb156e1a23aea83bdc9982410450e3",
        ),
        state_estimator,
        policy,
    ])
