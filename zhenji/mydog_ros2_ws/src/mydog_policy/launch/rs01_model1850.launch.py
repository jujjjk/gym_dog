from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


MODEL_SHA256 = (
    "661415c63ebc75a5691a56782f281ea57b46d7dad13b2fd946a899dd42da1d06"
)


def generate_launch_description():
    policy = Node(
        package="mydog_policy",
        executable="mydog_rs01_model1850_node",
        name="mydog_rs01_model1850_node",
        output="screen",
        parameters=[{
            "onnx_path": LaunchConfiguration("onnx_path"),
            "expected_onnx_sha256": MODEL_SHA256,
            "motor_base_url": LaunchConfiguration("motor_base_url"),
            "imu_port": LaunchConfiguration("imu_port"),
            "enable_send": LaunchConfiguration("enable_send"),
            "stand_only": LaunchConfiguration("stand_only"),
            "require_online": True,
            "max_motor_age_ms": 100.0,
            "max_imu_age_sec": 0.10,
            "gyro_bias_calibration_sec": 5.0,
            "gyro_bias_max_abs_rad_s": 0.35,
            "gyro_calibration_max_std_rad_s": 0.05,
            "gyro_calibration_max_rpy_span_rad": 0.08,
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
                "model_1850_rs01_path54.onnx",
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
