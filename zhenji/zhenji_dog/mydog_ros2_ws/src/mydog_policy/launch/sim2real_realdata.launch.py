from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Keep the established production entrypoint pinned to selected 5530."""
    base_launch = PathJoinSubstitution([
        FindPackageShare("mydog_policy"),
        "launch",
        "sim2real_symmetric_transition_5530.launch.py",
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
        ),
    ])
