from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mydog_policy.force_coord_contract import MODEL_TASK


def generate_launch_description():
    motor_url = LaunchConfiguration("motor_base_url")

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
        executable="mydog_force_coord_node",
        name="mydog_force_coord_5280_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "motor_base_url": motor_url,
            "base_lin_vel_source": "estimator",
            "policy_hz": 50.0,
            "action_mode": "pure_rl",
            "clip_action": False,

            # Preserve the exported training contract.
            "enable_cmd_smoothing": False,
            "deployment_gait_phase_period_scale": LaunchConfiguration(
                "gait_period_scale"
            ),
            "gait_phase_lead_sec": LaunchConfiguration("gait_phase_lead_sec"),
            "deployment_command_scale_x_mul": 1.0,
            "deployment_command_scale_y_mul": 1.0,
            "deployment_command_scale_yaw_mul": 1.0,
            "enable_policy_action_cmd_gate": False,
            "policy_action_cmd_gate_max_scale": 1.0,
            "reset_gait_phase_on_command_start": False,

            # Conservative real-machine command envelope; model observation
            # scaling remains unchanged.
            "enable_cmd_limits": True,
            "cmd_min_x": LaunchConfiguration("cmd_min_x"),
            "cmd_max_x": LaunchConfiguration("cmd_max_x"),
            "cmd_min_y": LaunchConfiguration("cmd_min_y"),
            "cmd_max_y": LaunchConfiguration("cmd_max_y"),
            "cmd_min_yaw": LaunchConfiguration("cmd_min_yaw"),
            "cmd_max_yaw": LaunchConfiguration("cmd_max_yaw"),
            "require_cmd_vel": True,
            "cmd_vel_timeout_sec": 0.35,
            "zero_cmd_inhibits_policy": False,
            "enable_zero_cmd_stand_protection": True,
            "zero_cmd_stand_x_threshold": 0.01,
            "zero_cmd_stand_y_threshold": 0.01,
            "zero_cmd_stand_yaw_threshold": 0.03,

            "enable_send": LaunchConfiguration("enable_send"),
            "print_only": LaunchConfiguration("print_only"),
            "startup_stand_first": LaunchConfiguration("startup_stand_first"),
            "stand_pose_source": "policy_default",
            "startup_stand_stop_torque_nm": 8.0,
            "require_online": True,
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
            "motor_state_async": True,
            "motor_state_poll_hz": 50.0,

            "use_model_pd_gains": True,
            "model_kp_scale": 1.0,
            "model_kd_scale": 1.0,

            # Global cap is intersected with ONNX per-joint limits.
            # cap=13 -> hip/thigh/calf = 10/10/13 Nm.
            # cap=10 -> 10/10/10; cap=8 -> 8/8/8.
            "motor_torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_safety_budget_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),
            "expected_active_torque_budget_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),

            "enable_target_smoothing": False,
            "enable_torque_error_limit": False,
            "enable_velocity_ff": False,
            "send_speed": 0.0,
            "send_torque": 0.0,
            "enable_rear_leg_posture_bias": False,

            "debug_print_arrays": LaunchConfiguration("debug_print_arrays"),
            "debug_csv_path": LaunchConfiguration("debug_csv_path"),
            "debug_csv_period_sec": 0.0,
            "debug_csv_async": True,
            "debug_csv_queue_size": 256,
            "debug_csv_flush_every_n": 20,
            "expected_policy_task": MODEL_TASK,
            "expected_policy_sha256": LaunchConfiguration(
                "expected_policy_sha256"
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
                "resource/fanfan_force_coord_5280.onnx"
            ),
        ),
        DeclareLaunchArgument(
            "motor_base_url",
            default_value="http://127.0.0.1:8000",
        ),
        DeclareLaunchArgument("expected_policy_sha256", default_value=""),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("startup_stand_first", default_value="true"),
        DeclareLaunchArgument("gait_period_scale", default_value="1.00"),
        DeclareLaunchArgument("gait_phase_lead_sec", default_value="0.00"),
        DeclareLaunchArgument("motor_torque_limit_nm", default_value="13.0"),
        DeclareLaunchArgument("max_motor_age_ms", default_value="100.0"),
        DeclareLaunchArgument("cmd_min_x", default_value="-0.08"),
        DeclareLaunchArgument("cmd_max_x", default_value="0.12"),
        DeclareLaunchArgument("cmd_min_y", default_value="-0.025"),
        DeclareLaunchArgument("cmd_max_y", default_value="0.025"),
        DeclareLaunchArgument("cmd_min_yaw", default_value="-0.20"),
        DeclareLaunchArgument("cmd_max_yaw", default_value="0.20"),
        DeclareLaunchArgument("debug_print_arrays", default_value="false"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/log/"
                "fanfan_force_coord_5280_parity.csv"
            ),
        ),
        state_estimator,
        policy,
    ])
