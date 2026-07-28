import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mydog_policy.omni_fast_contract import MODEL_SHA256, MODEL_TASK


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
        executable="mydog_policy_parity_node",
        name="mydog_policy_parity_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "motor_base_url": motor_url,
            "base_lin_vel_source": "estimator",
            "policy_hz": 50.0,
            "action_mode": "pure_rl",
            "clip_action": False,

            # Match the exported command/phase contract. The parity node also
            # enforces these values internally so legacy launch overrides cannot
            # silently reintroduce the old closed-loop behavior.
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

            "enable_cmd_limits": True,
            "cmd_min_x": -0.12,
            "cmd_max_x": 0.12,
            "cmd_min_y": -0.025,
            "cmd_max_y": 0.025,
            "cmd_min_yaw": -0.25,
            "cmd_max_yaw": 0.25,
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
            "require_online": True,
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
            "motor_state_async": True,
            "motor_state_poll_hz": 50.0,

            "use_model_pd_gains": True,
            "model_kp_scale": 1.0,
            "model_kd_scale": 1.0,
            "motor_torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_safety_budget_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "expected_active_torque_budget_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),

            # The new node performs one signed PD-equivalent clipping stage.
            # Disable the two legacy target-position limiters and velocity feedforward.
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
            "debug_csv_queue_size": 128,
            "debug_csv_flush_every_n": 20,
            "expected_policy_task": MODEL_TASK,
            "expected_policy_sha256": MODEL_SHA256,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
                "resource/fanfan_yaw_clean_5100.onnx"
            ),
        ),
        DeclareLaunchArgument(
            "motor_base_url",
            default_value="http://127.0.0.1:8000",
        ),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("startup_stand_first", default_value="true"),
        # Keep the training gait period fixed for the phase-lead A/B test.
        DeclareLaunchArgument("gait_period_scale", default_value="1.00"),
        # Test 0.00, 0.04 and 0.06 seconds while keeping every other variable fixed.
        DeclareLaunchArgument("gait_phase_lead_sec", default_value="0.00"),
        DeclareLaunchArgument("motor_torque_limit_nm", default_value="10.0"),
        DeclareLaunchArgument("max_motor_age_ms", default_value="100.0"),
        DeclareLaunchArgument("debug_print_arrays", default_value="false"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/log/"
                "fanfan_yaw_clean_sim2real_parity.csv"
            ),
        ),
        state_estimator,
        policy,
    ])
