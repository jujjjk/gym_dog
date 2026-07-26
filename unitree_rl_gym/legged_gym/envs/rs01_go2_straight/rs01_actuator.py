"""Small, testable RS01 actuator primitives used by the new Go2-style task."""

import torch


def limit_position_target(
    desired_target_rad,
    previous_target_rad,
    previous_target_rate_rad_s,
    rate_limit_rad_s,
    acceleration_limit_rad_s2,
    control_dt_s,
):
    """Apply the rate and acceleration contract of the real 50 Hz controller."""
    dt = float(control_dt_s)
    if dt <= 0.0:
        raise ValueError("control_dt_s must be positive")

    desired_rate = torch.clamp(
        (desired_target_rad - previous_target_rad) / dt,
        min=-rate_limit_rad_s,
        max=rate_limit_rad_s,
    )
    rate_step = torch.clamp(
        desired_rate - previous_target_rate_rad_s,
        min=-acceleration_limit_rad_s2 * dt,
        max=acceleration_limit_rad_s2 * dt,
    )
    limited_rate = torch.clamp(
        previous_target_rate_rad_s + rate_step,
        min=-rate_limit_rad_s,
        max=rate_limit_rad_s,
    )
    limited_target = previous_target_rad + limited_rate * dt

    # Do not cross the requested target because of the acceleration state.
    before = desired_target_rad - previous_target_rad
    after = desired_target_rad - limited_target
    crossed = before * after <= 0.0
    limited_target = torch.where(crossed, desired_target_rad, limited_target)
    limited_rate = torch.where(crossed, torch.zeros_like(limited_rate), limited_rate)
    return limited_target.clone(), limited_rate.clone()


def step_identified_position_response(
    previous_response_target_rad,
    delayed_target_rad,
    default_target_rad,
    response_gain,
    time_constant_s,
    physics_dt_s,
):
    """Advance the measured first-order position response by one physics step."""
    dt = float(physics_dt_s)
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive")
    if torch.any(time_constant_s <= 0.0):
        raise ValueError("time_constant_s must be positive")

    equilibrium = default_target_rad + response_gain * (
        delayed_target_rad - default_target_rad
    )
    alpha = 1.0 - torch.exp(-dt / time_constant_s)
    return (
        previous_response_target_rad
        + alpha * (equilibrium - previous_response_target_rad)
    ).clone()


def compute_rs01_joint_torques(
    response_target_rad,
    joint_position_rad,
    joint_velocity_rad_s,
    kp_nm_per_rad,
    kd_nm_per_rad_s,
    peak_torque_limit_nm,
    coulomb_friction_nm,
    friction_smoothing_rad_s,
):
    """Return raw PD, electromagnetic and applied joint torque separately."""
    raw_pd_torque_nm = (
        kp_nm_per_rad * (response_target_rad - joint_position_rad)
        - kd_nm_per_rad_s * joint_velocity_rad_s
    )
    motor_electromagnetic_torque_nm = torch.clamp(
        raw_pd_torque_nm,
        min=-peak_torque_limit_nm,
        max=peak_torque_limit_nm,
    ).clone()

    smoothing = max(float(friction_smoothing_rad_s), 1.0e-4)
    friction_nm = coulomb_friction_nm * torch.tanh(
        joint_velocity_rad_s / smoothing
    )
    applied_joint_torque_nm = (
        motor_electromagnetic_torque_nm - friction_nm
    ).clone()
    return (
        raw_pd_torque_nm.clone(),
        motor_electromagnetic_torque_nm,
        applied_joint_torque_nm,
    )
