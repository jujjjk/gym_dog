"""Deterministic zero-to-stand trajectory; this is not learned by PPO.

The output is in URDF/policy coordinates.  A hardware node must apply the
verified per-motor sign/order mapping before transmitting RS01 commands.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
ZERO_POSE_RAD = (0.0,) * 12
STAND_POSE_RAD = (
    0.0, -0.32987297, 1.31853104,
    0.0, -0.32987297, 1.31853104,
    0.0, -0.32987297, 1.31853104,
    0.0, -0.32987297, 1.31853104,
)
KP = (60.0, 70.0, 70.0) * 4
KD = (1.2, 1.6, 1.6) * 4
SOFT_LOWER_RAD = (-0.60, -1.20, 0.0) * 4
SOFT_UPPER_RAD = (0.60, 0.45, 1.75) * 4
RATE_LIMIT_RAD_S = (0.35, 0.45, 0.65) * 4
ACCEL_LIMIT_RAD_S2 = (2.0, 2.5, 3.5) * 4
TORQUE_LIMIT_NM = (10.0, 10.0, 13.0) * 4


@dataclass(frozen=True)
class StandSetpoint:
    time_s: float
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    torque_limit_nm: tuple[float, ...]


def _quintic(u: float) -> tuple[float, float, float]:
    """Minimum-jerk position and derivatives with respect to normalized time."""
    u = min(max(u, 0.0), 1.0)
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    acceleration = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return position, velocity, acceleration


def _validate_pose(pose: Sequence[float]) -> tuple[float, ...]:
    if len(pose) != len(JOINT_ORDER):
        raise ValueError("pose must contain exactly 12 joint angles")
    values = tuple(float(value) for value in pose)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose contains a non-finite value")
    for index, value in enumerate(values):
        if not SOFT_LOWER_RAD[index] <= value <= SOFT_UPPER_RAD[index]:
            raise ValueError(
                f"{JOINT_ORDER[index]}={value:.6f} exceeds the stand-transition "
                f"envelope [{SOFT_LOWER_RAD[index]}, {SOFT_UPPER_RAD[index]}]"
            )
    return values


def generate_stand_transition(
    start_pose_rad: Sequence[float] = ZERO_POSE_RAD,
    stand_pose_rad: Sequence[float] = STAND_POSE_RAD,
    duration_s: float = 4.0,
    control_dt_s: float = 0.02,
) -> Iterable[StandSetpoint]:
    """Yield a 50 Hz minimum-jerk stand-up trajectory with gain ramping."""
    start = _validate_pose(start_pose_rad)
    target = _validate_pose(stand_pose_rad)
    if duration_s < 3.5:
        raise ValueError("duration_s must be at least 3.5 s for this robot")
    if abs(control_dt_s - 0.02) > 1.0e-9:
        raise ValueError("RS01 deployment contract requires control_dt_s=0.02")

    steps = int(math.ceil(duration_s / control_dt_s))
    actual_duration = steps * control_dt_s
    delta = tuple(end - begin for begin, end in zip(start, target))

    for step in range(steps + 1):
        time_s = step * control_dt_s
        blend, blend_d1, blend_d2 = _quintic(time_s / actual_duration)
        position = tuple(
            begin + blend * change for begin, change in zip(start, delta)
        )
        velocity = tuple(
            blend_d1 * change / actual_duration for change in delta
        )
        acceleration = tuple(
            blend_d2 * change / actual_duration**2 for change in delta
        )
        for index, value in enumerate(velocity):
            if abs(value) > RATE_LIMIT_RAD_S[index] + 1.0e-9:
                raise RuntimeError(f"rate limit exceeded by {JOINT_ORDER[index]}")
        for index, value in enumerate(acceleration):
            if abs(value) > ACCEL_LIMIT_RAD_S2[index] + 1.0e-9:
                raise RuntimeError(
                    f"acceleration limit exceeded by {JOINT_ORDER[index]}"
                )

        # Start with low position authority and smoothly reach the measured
        # real controller gains.  Damping is kept substantial from frame one.
        kp_scale = 0.15 + 0.85 * blend
        kd_scale = 0.60 + 0.40 * blend
        yield StandSetpoint(
            time_s=time_s,
            position_rad=position,
            velocity_rad_s=velocity,
            kp=tuple(value * kp_scale for value in KP),
            kd=tuple(value * kd_scale for value in KD),
            torque_limit_nm=TORQUE_LIMIT_NM,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the non-learned RS01 zero-to-stand trajectory"
    )
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["time_s"]
        + [f"{name}_position_rad" for name in JOINT_ORDER]
        + [f"{name}_velocity_rad_s" for name in JOINT_ORDER]
        + [f"{name}_kp" for name in JOINT_ORDER]
        + [f"{name}_kd" for name in JOINT_ORDER]
    )
    for point in generate_stand_transition(duration_s=args.duration):
        writer.writerow(
            [f"{point.time_s:.6f}"]
            + [f"{value:.9f}" for value in point.position_rad]
            + [f"{value:.9f}" for value in point.velocity_rad_s]
            + [f"{value:.6f}" for value in point.kp]
            + [f"{value:.6f}" for value in point.kd]
        )


if __name__ == "__main__":
    main()
