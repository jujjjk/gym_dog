import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


OMNI_SHA256 = "95bd99ad12813a0b21220af70f8a36b75f22dca2c6ecd75d0b5f7fa5a7eebbff"


def generate_launch_description():
    base_launch = os.path.join(
        get_package_share_directory("mydog_policy"),
        "launch",
        "sim2real.launch.py",
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
            "max_cmd_x_rate_mps2": "0.50",
            "max_cmd_y_rate_mps2": "0.25",
            "max_cmd_yaw_rate_rad_s2": "1.50",
            "enable_cmd_limits": "true",
            "cmd_min_x": "-0.06",
            "cmd_max_x": "0.18",
            "cmd_min_y": "-0.04",
            "cmd_max_y": "0.04",
            "cmd_min_yaw": "-0.35",
            "cmd_max_yaw": "0.35",
            "cmd_vel_timeout_sec": "0.35",
            "use_model_pd_gains": "true",
            "model_kp_scale": LaunchConfiguration("model_kp_scale"),
            "model_kd_scale": LaunchConfiguration("model_kd_scale"),
            "motor_torque_limit_nm": LaunchConfiguration("motor_torque_limit_nm"),
            "calf_target_rate_mul": "3.0",
            "calf_target_accel_mul": "3.0",
            "debug_csv_path": LaunchConfiguration("debug_csv_path"),
            "debug_csv_period_sec": "0.1",
            "expected_policy_task": "FanfanOmniV4Cfg",
            "expected_policy_sha256": OMNI_SHA256,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=(
                "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
                "resource/policy.onnx"
            ),
        ),
        DeclareLaunchArgument("motor_base_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("print_only", default_value="false"),
        DeclareLaunchArgument("startup_stand_first", default_value="true"),
        DeclareLaunchArgument("model_kp_scale", default_value="1.0"),
        DeclareLaunchArgument("model_kd_scale", default_value="2.0"),
        DeclareLaunchArgument("motor_torque_limit_nm", default_value="6.0"),
        DeclareLaunchArgument(
            "debug_csv_path",
            default_value="/home/jetson/mydog_ros2_ws/log/omni_sim2real.csv",
        ),
        include,
    ])
