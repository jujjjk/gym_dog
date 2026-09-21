"""B18000: explicit opt-in to physical sending and walking."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_send',default_value='false'),
        DeclareLaunchArgument('stand_only',default_value='true'),
        DeclareLaunchArgument('motor_base_url',default_value='http://127.0.0.1:8000'),
        DeclareLaunchArgument('imu_port',default_value='/dev/myimu'),
        DeclareLaunchArgument('debug_csv_path',default_value=''),
        DeclareLaunchArgument('capture_dir'),
        Node(package='mydog_policy',executable='mydog_capture61_node',output='screen',parameters=[{
            'onnx_path':PathJoinSubstitution([FindPackageShare('mydog_policy'),'models','B18000.onnx']),
            **{k:LaunchConfiguration(k) for k in ('enable_send','stand_only','motor_base_url','imu_port','debug_csv_path','capture_dir')},
            'max_motor_age_ms':250.,'max_imu_age_sec':.25,'http_timeout_sec':.040,
            'max_abs_roll_rad':.45,'max_abs_pitch_rad':.45,
            'startup_ready_error_rad':.12,'startup_ready_hold_sec':2.,
            'hardware_torque_limit_nm':14.,'continuous_torque_nm':6.,
            'thermal_derate_full_rms_nm':8.,'thermal_rms_time_constant_sec':2.,
            'gyro_bias_calibration_sec':5.,'walk_start_stable_sec':1.,
            'command_min_vx_mps':.001,'command_max_vx_mps':.30,'command_timeout_sec':.35,
        }]),
    ])
