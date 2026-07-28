from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("motor_base_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("enable_send", default_value="false"),
        DeclareLaunchArgument("debug_csv_path", default_value=""),
        DeclareLaunchArgument("debug_csv_period_sec", default_value="0.0"),
        DeclareLaunchArgument("debug_stale_recheck_ms", default_value="100.0"),
        DeclareLaunchArgument("gait_hz", default_value="30.0"),
        DeclareLaunchArgument("step_hz", default_value="0.35"),
        DeclareLaunchArgument("motion_mode", default_value="soft_crawl"),
        DeclareLaunchArgument("stand_sec", default_value="3.0"),
        DeclareLaunchArgument("warmup_sec", default_value="4.0"),
        DeclareLaunchArgument("stand_kp", default_value="45.0"),
        DeclareLaunchArgument("stand_kd", default_value="5.0"),
        DeclareLaunchArgument("send_kp", default_value="35.0"),
        DeclareLaunchArgument("send_kd", default_value="4.0"),
        DeclareLaunchArgument("send_speed", default_value="0.0"),
        DeclareLaunchArgument("send_torque", default_value="0.0"),
        DeclareLaunchArgument("http_timeout", default_value="0.08"),
        DeclareLaunchArgument("duty_factor", default_value="0.82"),
        DeclareLaunchArgument("max_delta", default_value="0.35"),
        DeclareLaunchArgument("rate_limit_rad_per_sec", default_value="0.80"),
        DeclareLaunchArgument("soft_stride_thigh_amp", default_value="0.055"),
        DeclareLaunchArgument("soft_swing_calf_amp", default_value="0.260"),
        DeclareLaunchArgument("soft_rear_extra_calf_amp", default_value="0.060"),
        DeclareLaunchArgument("soft_stance_calf_amp", default_value="0.015"),
        DeclareLaunchArgument("soft_hip_balance_amp", default_value="0.0"),
        DeclareLaunchArgument("soft_lift_only", default_value="false"),
    ]

    node = Node(
        package="mydog_policy",
        executable="mydog_soft_body_gait_node",
        name="mydog_soft_body_gait_node",
        output="screen",
        parameters=[
            {
                "motor_base_url": LaunchConfiguration("motor_base_url"),
                "enable_send": LaunchConfiguration("enable_send"),
                "debug_csv_path": LaunchConfiguration("debug_csv_path"),
                "debug_csv_period_sec": LaunchConfiguration("debug_csv_period_sec"),
                "debug_stale_recheck_ms": LaunchConfiguration("debug_stale_recheck_ms"),
                "gait_hz": LaunchConfiguration("gait_hz"),
                "step_hz": LaunchConfiguration("step_hz"),
                "motion_mode": LaunchConfiguration("motion_mode"),
                "stand_sec": LaunchConfiguration("stand_sec"),
                "warmup_sec": LaunchConfiguration("warmup_sec"),
                "stand_kp": LaunchConfiguration("stand_kp"),
                "stand_kd": LaunchConfiguration("stand_kd"),
                "send_kp": LaunchConfiguration("send_kp"),
                "send_kd": LaunchConfiguration("send_kd"),
                "send_speed": LaunchConfiguration("send_speed"),
                "send_torque": LaunchConfiguration("send_torque"),
                "http_timeout": LaunchConfiguration("http_timeout"),
                "duty_factor": LaunchConfiguration("duty_factor"),
                "max_delta": LaunchConfiguration("max_delta"),
                "rate_limit_rad_per_sec": LaunchConfiguration("rate_limit_rad_per_sec"),
                "soft_stride_thigh_amp": LaunchConfiguration("soft_stride_thigh_amp"),
                "soft_swing_calf_amp": LaunchConfiguration("soft_swing_calf_amp"),
                "soft_rear_extra_calf_amp": LaunchConfiguration("soft_rear_extra_calf_amp"),
                "soft_stance_calf_amp": LaunchConfiguration("soft_stance_calf_amp"),
                "soft_hip_balance_amp": LaunchConfiguration("soft_hip_balance_amp"),
                "soft_lift_only": LaunchConfiguration("soft_lift_only"),
            }
        ],
    )

    return LaunchDescription(args + [node])
