"""Conservative real-machine launch for selected recovery checkpoint 5650."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from mydog_policy.tilt_recovery_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_TASK,
)


def generate_launch_description():
    package = FindPackageShare("mydog_policy")
    base_launch = PathJoinSubstitution([
        package,
        "launch",
        "sim2real_symmetric_transition_5530.launch.py",
    ])
    model = PathJoinSubstitution([package, "models", MODEL_FILENAME])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                "onnx_path": model,
                "expected_policy_sha256": MODEL_SHA256,
                "expected_policy_task": MODEL_TASK,
                "policy_executable": "mydog_tilt_recovery_node",
                "policy_node_name": "mydog_tilt_recovery_5530_5650_node",
                # Conservative defaults for the first suspended/low-rack run.
                # Expand only after checking the generated 50 Hz CSV.
                "cmd_min_x": "-0.15",
                "cmd_max_x": "0.35",
                "cmd_min_y": "-0.12",
                "cmd_max_y": "0.12",
                "cmd_min_yaw": "-0.75",
                "cmd_max_yaw": "0.75",
                "debug_csv_path": (
                    "/home/jetson/mydog_ros2_ws/log/"
                    "fanfan_tilt_recovery_5530_5650.csv"
                ),
            }.items(),
        ),
    ])
