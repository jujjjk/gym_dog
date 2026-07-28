"""Safe launch wrapper for the selected adaptive-stride policy."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from mydog_policy.adaptive_stride_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_TASK,
)


def generate_launch_description():
    package_share = FindPackageShare("mydog_policy")
    shared_launch = PathJoinSubstitution([
        package_share,
        "launch",
        "sim2real_symmetric_transition_5530.launch.py",
    ])
    model_path = PathJoinSubstitution([
        package_share,
        "models",
        MODEL_FILENAME,
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(shared_launch),
            launch_arguments={
                "onnx_path": model_path,
                "expected_policy_sha256": MODEL_SHA256,
                "expected_policy_task": MODEL_TASK,
                "policy_executable": "mydog_adaptive_stride_node",
                "policy_node_name": "mydog_adaptive_stride_node",
                "debug_csv_path": (
                    "/home/jetson/mydog_ros2_ws/log/"
                    "fanfan_adaptive_stride_best.csv"
                ),
            }.items(),
        ),
    ])
