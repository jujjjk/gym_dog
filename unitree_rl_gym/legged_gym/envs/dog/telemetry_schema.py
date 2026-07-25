"""CSV schema shared by the recorder and Stage-1 contract tests."""

JOINT_SIGNALS = (
    "policy_action",
    "dof_pos_rad",
    "dof_vel_rad_s",
    "target_pos_rad",
    "raw_pd_torque_nm",
    "motor_electromagnetic_torque_nm",
    "applied_joint_torque_nm",
    "peak_torque_limit_nm",
    "active_torque_limit_nm",
    "raw_over_17nm",
    "peak_saturation_flag",
    "active_saturation_flag",
    "motor_over_6nm",
    "motor_over_12nm",
    "motor_over_15nm",
    "motor_over_6_duration_s",
    "motor_over_12_duration_s",
    "motor_over_15_duration_s",
    "thermal_rms_ratio",
    "motor_temperature_c",
    "terminal_raw_pd_torque_nm",
    "terminal_motor_electromagnetic_torque_nm",
)


def mask_to_bits(mask):
    result = 0
    for index, value in enumerate(mask):
        result |= int(bool(value)) << index
    return result


def build_headers(
    joint_names,
    legs=("FL", "FR", "RL", "RR"),
    reward_names=(),
):
    headers = [
        "step", "time_s", "episode", "reset", "reset_reason", "reward",
        "cmd_vx_m_s", "cmd_vy_m_s", "cmd_yaw_rad_s",
        "base_x_m", "base_y_m", "base_z_m",
        "base_roll_rad", "base_pitch_rad", "base_yaw_rad",
        "body_vx_m_s", "body_vy_m_s", "body_vz_m_s",
        "body_roll_rate_rad_s", "body_pitch_rate_rad_s",
        "body_yaw_rate_rad_s", "gait_phase", "flight",
        "all_feet_contact", "foot_contact_mask", "desired_contact_mask",
        "illegal_contact_count", "terminal_contact_mask",
        "terminal_phase", "terminal_roll_rad", "terminal_pitch_rad",
        "terminal_yaw_rate_rad_s",
    ]
    headers += [f"foot_force_z_{leg}_n" for leg in legs]
    headers += [f"foot_contact_{leg}" for leg in legs]
    headers += [f"foot_z_{leg}_m" for leg in legs]
    headers += [f"foot_contact_duration_s_{leg}" for leg in legs]
    headers += [f"touchdown_event_{leg}" for leg in legs]
    headers += [f"takeoff_event_{leg}" for leg in legs]
    for signal in JOINT_SIGNALS:
        headers += [f"{signal}_{name}" for name in joint_names]
    headers += [f"reward_scaled_{name}_per_step" for name in reward_names]
    return headers
