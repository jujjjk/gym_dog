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
        DeclareLaunchArgument("gait_hz", default_value="50.0"),
        DeclareLaunchArgument("step_hz", default_value="1.0"),
        DeclareLaunchArgument("motion_mode", default_value="trot"),
        DeclareLaunchArgument("stand_sec", default_value="3.0"),
        DeclareLaunchArgument("warmup_sec", default_value="2.0"),
        DeclareLaunchArgument("stand_kp", default_value="45.0"),
        DeclareLaunchArgument("stand_kd", default_value="5.0"),
        DeclareLaunchArgument("send_kp", default_value="40.0"),
        DeclareLaunchArgument("send_kd", default_value="5.0"),
        DeclareLaunchArgument("send_speed", default_value="0.0"),
        DeclareLaunchArgument("send_torque", default_value="0.0"),
        DeclareLaunchArgument("http_timeout", default_value="0.08"),
        DeclareLaunchArgument("hip_amp", default_value="0.0"),
        DeclareLaunchArgument("thigh_amp", default_value="0.18"),
        DeclareLaunchArgument("calf_lift_amp", default_value="0.60"),
        DeclareLaunchArgument("stance_calf_amp", default_value="0.08"),
        DeclareLaunchArgument("stride_sign", default_value="-1.0"),
        DeclareLaunchArgument("duty_factor", default_value="0.60"),
        DeclareLaunchArgument("sweep_min_hz", default_value="0.35"),
        DeclareLaunchArgument("sweep_max_hz", default_value="2.50"),
        DeclareLaunchArgument("sweep_period_sec", default_value="20.0"),
        DeclareLaunchArgument("excitation_hip_amp", default_value="0.08"),
        DeclareLaunchArgument("excitation_thigh_amp", default_value="0.18"),
        DeclareLaunchArgument("excitation_calf_amp", default_value="0.18"),
        DeclareLaunchArgument("squat_thigh_amp", default_value="0.08"),
        DeclareLaunchArgument("squat_calf_amp", default_value="0.18"),
        DeclareLaunchArgument("max_delta", default_value="1.50"),
    ]

    node = Node(
        package="mydog_policy",
        executable="mydog_openloop_gait_node",
        name="mydog_openloop_gait_node",
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
                "hip_amp": LaunchConfiguration("hip_amp"),
                "thigh_amp": LaunchConfiguration("thigh_amp"),
                "calf_lift_amp": LaunchConfiguration("calf_lift_amp"),
                "stance_calf_amp": LaunchConfiguration("stance_calf_amp"),
                "stride_sign": LaunchConfiguration("stride_sign"),
                "duty_factor": LaunchConfiguration("duty_factor"),
                "sweep_min_hz": LaunchConfiguration("sweep_min_hz"),
                "sweep_max_hz": LaunchConfiguration("sweep_max_hz"),
                "sweep_period_sec": LaunchConfiguration("sweep_period_sec"),
                "excitation_hip_amp": LaunchConfiguration("excitation_hip_amp"),
                "excitation_thigh_amp": LaunchConfiguration("excitation_thigh_amp"),
                "excitation_calf_amp": LaunchConfiguration("excitation_calf_amp"),
                "squat_thigh_amp": LaunchConfiguration("squat_thigh_amp"),
                "squat_calf_amp": LaunchConfiguration("squat_calf_amp"),
                "max_delta": LaunchConfiguration("max_delta"),
            }
        ],
    )

    return LaunchDescription(args + [node])
