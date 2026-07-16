from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from mydog_policy.realdata_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_TASK,
)


def generate_launch_description():
    base_launch = PathJoinSubstitution([
        FindPackageShare("mydog_policy"),
        "launch",
        "sim2real_hardware_balance.launch.py",
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                "policy_executable": "mydog_realdata_node",
                "model_filename": MODEL_FILENAME,
                "expected_policy_task": MODEL_TASK,
                "expected_policy_sha256": MODEL_SHA256,
            }.items(),
        ),
    ])
