from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from mydog_policy.clearance_robust_contract import MODEL_SHA256, MODEL_TASK


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
            "require_online": True,
            "max_imu_sample_age_sec": 0.10,
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
        executable="mydog_clearance_robust_node",
        name="mydog_clearance_robust_5730_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "motor_base_url": motor_url,
            "base_lin_vel_source": "estimator",
            "state_estimator_timeout_sec": 0.10,
            "policy_hz": 50.0,
            "action_mode": "pure_rl",
            "clip_action": False,

            # ONNX already contains velocity/heading feedback and strict symmetry.
            "enable_cmd_smoothing": False,
            "deployment_gait_phase_period_scale": 1.0,
            "gait_phase_lead_sec": 0.0,
            "deployment_command_scale_x_mul": 1.0,
            "deployment_command_scale_y_mul": 1.0,
            "deployment_command_scale_yaw_mul": 1.0,
            "enable_policy_action_cmd_gate": False,
            "policy_action_cmd_gate_max_scale": 1.0,
            "reset_gait_phase_on_command_start": False,

            "enable_cmd_limits": True,
            "cmd_min_x": -0.25,
            "cmd_max_x": 0.60,
            "cmd_min_y": -0.26,
            "cmd_max_y": 0.26,
            "cmd_min_yaw": -1.30,
            "cmd_max_yaw": 1.30,
            "require_cmd_vel": True,
            "zero_cmd_inhibits_policy": False,
            "cmd_vel_timeout_sec": LaunchConfiguration("cmd_vel_timeout_sec"),
            # Zero command still runs the learned stand policy, as in simulation.
            "enable_zero_cmd_stand_protection": False,
            "zero_cmd_stand_x_threshold": 0.01,
            "zero_cmd_stand_y_threshold": 0.01,
            "zero_cmd_stand_yaw_threshold": 0.03,

            "max_estimator_snapshot_lag": 3,
            "max_estimator_tick_lag_ms": 35.0,
            "zero_lin_vel_on_estimator_mismatch": False,
            "hold_last_lin_vel_on_estimator_mismatch": True,
            "estimator_mismatch_velocity_decay": 0.98,

            "enable_send": LaunchConfiguration("enable_send"),
            "print_only": LaunchConfiguration("print_only"),
            "startup_stand_first": True,
            "stand_pose_source": "policy_default",
            # User requested no 8 Nm rack profile: startup threshold is 13 Nm.
            "startup_stand_stop_torque_nm": 13.0,
            "require_online": True,
            "max_motor_age_ms": LaunchConfiguration("max_motor_age_ms"),
            "motor_state_async": True,
            "motor_state_poll_hz": 50.0,
            "enable_tilt_protection": LaunchConfiguration("enable_tilt_protection"),
            "enable_command_timeout_stand_hold": False,
            "max_tilt_rad": 0.75,

            "use_model_pd_gains": True,
            "model_kp_scale": 1.0,
            "model_kd_scale": 1.0,

            # Exact simulation contract: global cap 13 Nm intersects the model's
            # per-joint 10/10/13 Nm profile. No 8 Nm cap and no 17 Nm boost.
            "motor_torque_limit_nm": 13.0,
            "torque_limit_nm": 13.0,
            "torque_safety_budget_nm": 13.0,
            "expected_active_torque_budget_nm": 13.0,
            "motion_torque_ff_limit_nm": 13.0,
            "use_hardware_torque_limits": True,
            "require_verified_hardware_limits": True,
            # RS01: 7 Apk ~= 6 Nm. These ceilings allow 10/10/13 Nm while
            # 0x700B remains the authoritative torque clamp.
            "hip_current_limit_amp": 12.0,
            "thigh_current_limit_amp": 12.0,
            "calf_current_limit_amp": 16.0,
            "critical_state_failure_stop_cycles": 5,
            "critical_state_startup_grace_sec": 5.0,
            "fail_safe_stop_timeout_sec": 0.5,
            "enable_rear_torque_boost": False,
            "rear_torque_boost_nm": 13.0,
            "rear_torque_boost_duration_sec": 2.5,
            "rear_torque_boost_tilt_threshold_rad": 0.10,
            "rear_torque_boost_q_error_rad": 0.12,
            "rear_torque_boost_overload_margin_nm": 1.0,

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
            "expected_policy_sha256": LaunchConfiguration("expected_policy_sha256"),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("mydog_policy"),
                "models",
                "fanfan_clearance_robust_5730.onnx",
            ]),
        ),
        DeclareLaunchArgument(
            "motor_base_url",
            default_value="http://127.0.0.1:8000",
        ),
        DeclareLaunchArgument(
            "expected_policy_sha256",
            default_value=MODEL_SHA256,
        ),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("enable_tilt_protection", default_value="false"),
        DeclareLaunchArgument("max_motor_age_ms", default_value="100.0"),
        DeclareLaunchArgument("cmd_vel_timeout_sec", default_value="0.50"),
        DeclareLaunchArgument("debug_print_arrays", default_value="false"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/log/"
                "fanfan_clearance_robust_5730.csv"
            ),
        ),
        state_estimator,
        policy,
    ])
