import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from mydog_policy.omni_fast_contract import MODEL_SHA256, MODEL_TASK


def generate_launch_description():
    base_launch = os.path.join(
        get_package_share_directory("mydog_policy"), "launch", "sim2real.launch.py"
    )
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "onnx_path": LaunchConfiguration("onnx_path"),
            "motor_base_url": LaunchConfiguration("motor_base_url"),
            "enable_send": LaunchConfiguration("enable_send"),
            "print_only": LaunchConfiguration("print_only"),
            "startup_stand_first": LaunchConfiguration("startup_stand_first"),
            "enable_cmd_smoothing": "true",
            "max_cmd_x_rate_mps2": "0.20",
            "max_cmd_y_rate_mps2": "0.10",
            "max_cmd_yaw_rate_rad_s2": "0.60",
            "deployment_gait_phase_period_scale": "1.35",
            "deployment_command_scale_x_mul": LaunchConfiguration(
                "deployment_command_scale_x_mul"
            ),
            "deployment_command_scale_y_mul": LaunchConfiguration(
                "deployment_command_scale_y_mul"
            ),
            "deployment_command_scale_yaw_mul": LaunchConfiguration(
                "deployment_command_scale_yaw_mul"
            ),
            "enable_cmd_limits": "true",
            "cmd_min_x": "-0.12",
            "cmd_max_x": "0.12",
            "cmd_min_y": "-0.025",
            "cmd_max_y": "0.025",
            "cmd_min_yaw": "-0.25",
            "cmd_max_yaw": "0.25",
            "cmd_vel_timeout_sec": "0.35",
            "zero_cmd_inhibits_policy": "false",
            "enable_zero_cmd_stand_protection": "true",
            "zero_cmd_stand_x_threshold": "0.01",
            "zero_cmd_stand_y_threshold": "0.01",
            "zero_cmd_stand_yaw_threshold": "0.03",
            "enable_policy_action_cmd_gate": "true",
            "policy_action_cmd_gate_start_ratio": "0.05",
            "policy_action_cmd_gate_full_ratio": "0.5",
            "policy_action_cmd_gate_max_scale": "0.9",
            "reset_gait_phase_on_command_start": "true",
            "use_model_pd_gains": "true",
            "model_kp_scale": LaunchConfiguration("model_kp_scale"),
            "model_kd_scale": LaunchConfiguration("model_kd_scale"),
            "motor_torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            # Bind every legacy/current budget entry to one outer argument so a
            # nested launch default can never silently restore 6 Nm.
            "torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "torque_safety_budget_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "expected_active_torque_budget_nm": LaunchConfiguration(
                "motor_torque_limit_nm"
            ),
            "calf_target_rate_mul": "3.0",
            "calf_target_accel_mul": "3.0",
            "debug_csv_path": LaunchConfiguration("debug_csv_path"),
            # 0 means no logger-side decimation: one row for every 50 Hz
            # policy cycle (20 ms nominal period).
            "debug_csv_period_sec": "0.0",
            "expected_policy_task": MODEL_TASK,
            "expected_policy_sha256": MODEL_SHA256,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
                "resource/fanfan_yaw_clean_5100.onnx"
            ),
        ),
        DeclareLaunchArgument("motor_base_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("startup_stand_first", default_value="true"),
        DeclareLaunchArgument("model_kp_scale", default_value="1.0"),
        DeclareLaunchArgument("model_kd_scale", default_value="1.0"),
        DeclareLaunchArgument("motor_torque_limit_nm", default_value="8.0"),
        DeclareLaunchArgument("deployment_command_scale_x_mul", default_value="1.0"),
        DeclareLaunchArgument("deployment_command_scale_y_mul", default_value="1.0"),
        DeclareLaunchArgument("deployment_command_scale_yaw_mul", default_value="1.0"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value="/home/jetson/mydog_ros2_ws/log/fanfan_yaw_clean_sim2real.csv",
        ),
        include,
    ])
