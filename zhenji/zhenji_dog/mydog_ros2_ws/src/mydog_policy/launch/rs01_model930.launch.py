from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


MODEL_SHA256 = (
    "8496a7bf39dd6729b978171c1276ec475512138258f593dfb44568170bf04bd6"
)


def generate_launch_description():
    policy = Node(
        package="mydog_policy",
        executable="mydog_rs01_model930_node",
        name="mydog_rs01_model930_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "expected_onnx_sha256": MODEL_SHA256,
            "motor_base_url": LaunchConfiguration("motor_base_url"),
            "imu_port": LaunchConfiguration("imu_port"),
            # Both interlocks default to their safest state. Enabling motor
            # output still commands only the soft-ramped stand until
            # stand_only is explicitly disabled on a later launch.
            "enable_send": LaunchConfiguration("enable_send"),
            "stand_only": LaunchConfiguration("stand_only"),
            "require_online": True,
            "max_motor_age_ms": 100.0,
            "max_imu_age_sec": 0.10,
            "max_temperature_c": 70.0,
            "max_abs_roll_rad": 0.60,
            "max_abs_pitch_rad": 0.60,
            "command_timeout_sec": 0.50,
            "command_min_vx_mps": 0.21,
            "command_max_vx_mps": 0.25,
            "hardware_torque_limit_nm": 14.0,
            "continuous_torque_nm": 6.0,
            "thermal_derate_full_rms_nm": 8.0,
            "thermal_rms_time_constant_sec": 2.0,
            "hip_current_limit_amp": 12.0,
            "thigh_current_limit_amp": 12.0,
            "calf_current_limit_amp": 16.0,
            "debug_csv_path": LaunchConfiguration("debug_csv_path"),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "onnx_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("mydog_policy"),
                "models",
                "model_930_rs01_heading52.onnx",
            ]),
        ),
        DeclareLaunchArgument(
            "motor_base_url",
            default_value="http://127.0.0.1:8000",
        ),
        DeclareLaunchArgument("imu_port", default_value="/dev/myimu"),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("stand_only", default_value="true"),
        DeclareLaunchArgument("debug_csv_path", default_value=""),
        policy,
    ])
