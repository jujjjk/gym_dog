from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('onnx_path',default_value=PathJoinSubstitution([
            FindPackageShare('mydog_policy'),'models','stand_only_6850.onnx'])),
        Node(package='mydog_policy',executable='mydog_rs01_model6850_shadow',output='screen',
             parameters=[{'onnx_path':LaunchConfiguration('onnx_path'),'enable_send':False}]),
    ])
