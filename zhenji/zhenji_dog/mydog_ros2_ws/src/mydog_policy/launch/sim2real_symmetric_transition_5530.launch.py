from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mydog_policy.symmetric_transition_contract import MODEL_TASK


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
            "robust_stance_height_margin": 0.035,
            "robust_vertical_speed_threshold": 0.22,
            "robust_velocity_residual_threshold": 0.30,
            "robust_max_stance_feet": 2,
            "robust_filter_alpha": 0.35,
            "robust_absolute_height_guard": 0.10,
            "robust_planar_velocity_clip": 1.0,
            "no_stance_velocity_decay": 0.85,
        }],
    )

    policy = Node(
        package="mydog_policy",
        executable="mydog_symmetric_transition_node",
        name="mydog_symmetric_transition_5530_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "motor_base_url": motor_url,
            "base_lin_vel_source": "estimator",
            "policy_hz": 50.0,
            "action_mode": "pure_rl",
            "clip_action": False,

            # The ONNX graph contains the feedback and strict symmetry.
            # Do not duplicate either transform in the ROS layer.
            "enable_cmd_smoothing": False,
            "deployment_gait_phase_period_scale": LaunchConfiguration(
                "gait_period_scale"
            ),
            "gait_phase_lead_sec": LaunchConfiguration(
                "gait_phase_lead_sec"
            ),
            "deployment_command_scale_x_mul": 1.0,
            "deployment_command_scale_y_mul": 1.0,
            "deployment_command_scale_yaw_mul": 1.0,
            "enable_policy_action_cmd_gate": False,
            "policy_action_cmd_gate_max_scale": 1.0,
            "reset_gait_phase_on_command_start": False,

            "enable_cmd_limits": True,
            "cmd_min_x": LaunchConfiguration("cmd_min_x"),
            "cmd_max_x": LaunchConfiguration("cmd_max_x"),
            "cmd_min_y": LaunchConfiguration("cmd_min_y"),
            "cmd_max_y": LaunchConfiguration("cmd_max_y"),
            "cmd_min_yaw": LaunchConfiguration("cmd_min_yaw"),
            "cmd_max_yaw": LaunchConfiguration("cmd_max_yaw"),
            "require_cmd_vel": True,
            "cmd_vel_timeout_sec": LaunchConfiguration(
                "cmd_vel_timeout_sec"
            ),
            "zero_cmd_inhibits_policy": False,
            "enable_zero_cmd_stand_protection": True,
            "zero_cmd_stand_x_threshold": 0.01,
            "zero_cmd_stand_y_threshold": 0.01,
            "zero_cmd_stand_yaw_threshold": 0.03,

            # Detect new command segments, reset only heading anchor, and
            # preserve phase/action-filter state across transitions.
            "transition_reset_vx_delta": 0.025,
            "transition_reset_vy_delta": 0.025,
            "transition_reset_yaw_delta": 0.08,

            # The model has strong velocity feedback. Do not inject zero velocity
            # for an isolated estimator/motor snapshot mismatch.
            "max_estimator_snapshot_lag": 3,
            "zero_lin_vel_on_estimator_mismatch": False,
            "hold_last_lin_vel_on_estimator_mismatch": True,
            "estimator_mismatch_velocity_decay": 0.98,

            "enable_send": LaunchConfiguration("enable_send"),
            "print_only": LaunchConfiguration("print_only"),
            "startup_stand_first": LaunchConfiguration(
                "startup_stand_first"
            ),
            "stand_pose_source": "policy_default",
            "startup_stand_stop_torque_nm": 8.0,
            "require_online": True,
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
            "motor_state_async": True,
            "motor_state_poll_hz": 50.0,
            "enable_tilt_protection": LaunchConfiguration(
                "enable_tilt_protection"
            ),
            "enable_command_timeout_stand_hold": LaunchConfiguration(
                "enable_command_timeout_stand_hold"
            ),
            "max_tilt_rad": LaunchConfiguration("max_tilt_rad"),

            "use_model_pd_gains": True,
            "model_kp_scale": 1.0,
            "model_kd_scale": 1.0,

            # Intersect global safety cap with the ONNX 10/10/13 Nm limits.
            "motor_torque_limit_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),
            "torque_limit_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),
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
            # RS04 motion-control torque feed-forward field range. The parity
            # node uses this field to saturate total PD torque without
            # rewriting the learned position target.
            "motion_torque_ff_limit_nm": LaunchConfiguration(
                "motion_torque_ff_limit_nm"
            ),
            "enable_rear_torque_boost": LaunchConfiguration(
                "enable_rear_torque_boost"
            ),
            "rear_torque_boost_nm": LaunchConfiguration(
                "rear_torque_boost_nm"
            ),
            "rear_torque_boost_duration_sec": LaunchConfiguration(
                "rear_torque_boost_duration_sec"
            ),
            "rear_torque_boost_tilt_threshold_rad": LaunchConfiguration(
                "rear_torque_boost_tilt_threshold_rad"
            ),
            "rear_torque_boost_q_error_rad": LaunchConfiguration(
                "rear_torque_boost_q_error_rad"
            ),
            "rear_torque_boost_overload_margin_nm": LaunchConfiguration(
                "rear_torque_boost_overload_margin_nm"
            ),
            "enable_rear_leg_posture_bias": False,

            "debug_print_arrays": LaunchConfiguration(
                "debug_print_arrays"
            ),
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
                "/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/"
                "fanfan_symmetric_transition_5530.onnx"
            ),
        ),
        DeclareLaunchArgument(
            "motor_base_url",
            default_value="http://127.0.0.1:8000",
        ),
        DeclareLaunchArgument("expected_policy_sha256", default_value=""),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument(
            "startup_stand_first",
            default_value="true",
        ),
        DeclareLaunchArgument("gait_period_scale", default_value="1.00"),
        DeclareLaunchArgument(
            "gait_phase_lead_sec",
            default_value="0.00",
        ),
        DeclareLaunchArgument(
            "motor_torque_limit_nm",
            default_value="13.0",
        ),
        DeclareLaunchArgument(
            "motion_torque_ff_limit_nm",
            default_value="17.0",
        ),
        DeclareLaunchArgument(
            "enable_rear_torque_boost",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "rear_torque_boost_nm",
            default_value="17.0",
        ),
        DeclareLaunchArgument(
            "rear_torque_boost_duration_sec",
            default_value="2.5",
        ),
        DeclareLaunchArgument(
            "rear_torque_boost_tilt_threshold_rad",
            default_value="0.10",
        ),
        DeclareLaunchArgument(
            "rear_torque_boost_q_error_rad",
            default_value="0.12",
        ),
        DeclareLaunchArgument(
            "rear_torque_boost_overload_margin_nm",
            default_value="1.0",
        ),
        DeclareLaunchArgument(
            "enable_tilt_protection",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "enable_command_timeout_stand_hold",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "max_tilt_rad",
            default_value="0.75",
        ),
        DeclareLaunchArgument("max_motor_age_ms", default_value="100.0"),
        DeclareLaunchArgument(
            "cmd_vel_timeout_sec",
            default_value="0.50",
        ),

        # Match the exported 5530 command contract. The low-rack procedure
        # reduces the values published to /cmd_vel; it does not silently
        # change the model's command range at the policy node.
        DeclareLaunchArgument("cmd_min_x", default_value="-0.25"),
        DeclareLaunchArgument("cmd_max_x", default_value="0.60"),
        DeclareLaunchArgument("cmd_min_y", default_value="-0.26"),
        DeclareLaunchArgument("cmd_max_y", default_value="0.26"),
        DeclareLaunchArgument("cmd_min_yaw", default_value="-1.30"),
        DeclareLaunchArgument("cmd_max_yaw", default_value="1.30"),

        DeclareLaunchArgument(
            "debug_print_arrays",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/log/"
                "fanfan_symmetric_transition_5530.csv"
            ),
        ),
        state_estimator,
        policy,
    ])
