"""Safe deployment wrapper for the saturation-recovery policy."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from mydog_policy.saturation_recovery_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_TASK,
)


def generate_launch_description():
    share = FindPackageShare("mydog_policy")
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                share, "launch", "sim2real_symmetric_transition_5530.launch.py"
            ])),
            launch_arguments={
                "onnx_path": PathJoinSubstitution([
                    share, "models", MODEL_FILENAME
                ]),
                "expected_policy_sha256": MODEL_SHA256,
                "expected_policy_task": MODEL_TASK,
                "policy_executable": "mydog_saturation_recovery_node",
                "policy_node_name": "mydog_saturation_recovery_node",
                "debug_csv_path": (
                    "/home/jetson/mydog_ros2_ws/log/"
                    "fanfan_saturation_recovery_best.csv"
                ),
            }.items(),
        ),
    ])

