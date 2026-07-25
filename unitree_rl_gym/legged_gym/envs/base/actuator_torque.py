"""Pure helpers defining the three RS01 torque domains."""

import torch


def limit_electromagnetic_torque(raw_pd_torques, active_torque_limits):
    """Apply the active electromagnetic motor limit without mutating inputs."""
    if torch.any(active_torque_limits < 0.0):
        raise ValueError("active torque limits must be non-negative")
    return torch.maximum(
        torch.minimum(raw_pd_torques, active_torque_limits),
        -active_torque_limits,
    ).clone()


def apply_coulomb_friction(
    motor_electromagnetic_torques,
    joint_velocity_rad_s,
    coulomb_friction_nm,
    velocity_smoothing_rad_s,
):
    """Return net joint torque; friction always opposes joint velocity."""
    smoothing = max(float(velocity_smoothing_rad_s), 1.0e-4)
    friction_torque = coulomb_friction_nm * torch.tanh(
        joint_velocity_rad_s / smoothing
    )
    return motor_electromagnetic_torques.clone() - friction_torque
