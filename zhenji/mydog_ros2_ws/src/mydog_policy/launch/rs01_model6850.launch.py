"""Guarded A6850 hardware tests. Importing this file does not open devices."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_send', default_value='false'),
        DeclareLaunchArgument('stand_only', default_value='true'),
        DeclareLaunchArgument('motor_base_url', default_value='http://127.0.0.1:8000'),
        DeclareLaunchArgument('imu_port', default_value='/dev/myimu'),
        DeclareLaunchArgument('debug_csv_path', default_value=''),
        Node(package='mydog_policy', executable='mydog_rs01_model6850_node',
             name='mydog_rs01_model6850_node', output='screen', parameters=[{
                 'onnx_path': PathJoinSubstitution([
                     FindPackageShare('mydog_policy'), 'models', 'stand_only_6850.onnx']),
                 'enable_send': LaunchConfiguration('enable_send'),
                 'stand_only': LaunchConfiguration('stand_only'),
                 'motor_base_url': LaunchConfiguration('motor_base_url'),
                 'imu_port': LaunchConfiguration('imu_port'),
                 'debug_csv_path': LaunchConfiguration('debug_csv_path'),
                 'max_motor_age_ms': 250., 'max_imu_age_sec': .25,
                 'http_timeout_sec': .040,
                 'max_abs_roll_rad': .45, 'max_abs_pitch_rad': .45,
                 'startup_ready_error_rad': .12, 'startup_ready_hold_sec': 2.,
                 'hardware_torque_limit_nm': 14., 'continuous_torque_nm': 6.,
                 'gyro_bias_calibration_sec': 5., 'walk_start_stable_sec': 1.,
                 'command_min_vx_mps': .001, 'command_max_vx_mps': .30,
                 'command_timeout_sec': .35,
             }]),
    ])
