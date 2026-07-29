#!/usr/bin/env python3
"""Pure-Numpy deployment contract for the RS01 model_930 policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np


METADATA_KEY = "rs01_go2_deployment_config"
EXPECTED_TASK = "rs01_go2_sim2sim_heading52"
EXPECTED_ONNX_SHA256 = (
    "8496a7bf39dd6729b978171c1276ec475512138258f593dfb44568170bf04bd6"
)

REAL_MOTOR_IDS = (
    0x11, 0x12, 0x13,
    0x21, 0x22, 0x23,
    0x31, 0x32, 0x33,
    0x41, 0x42, 0x43,
)
REAL_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
POLICY_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

# User-verified on the new machine. q_policy = sign * q_real.
REVERSED_MOTOR_IDS = frozenset((0x11, 0x13, 0x21, 0x22, 0x32, 0x43))
SIGN_BY_MOTOR_ID = {
    motor_id: (-1.0 if motor_id in REVERSED_MOTOR_IDS else 1.0)
    for motor_id in REAL_MOTOR_IDS
}

URDF_LOWER_BY_TYPE = {
    "hip": -1.0472,
    "thigh": -2.76228694714,
    "calf": 0.0,
}
URDF_UPPER_BY_TYPE = {
    "hip": 1.0472,
    "thigh": 2.29921305286,
    "calf": 1.91986217719,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def estimate_stationary_gyro_bias(
    gyro_samples_rad_s,
    rpy_samples_rad,
    max_std_rad_s,
    max_rpy_span_rad,
    max_abs_bias_rad_s,
):
    """Validate a stationary sample window and return its gyro mean."""
    gyro = np.asarray(gyro_samples_rad_s, dtype=np.float64)
    rpy = np.asarray(rpy_samples_rad, dtype=np.float64)
    if (
        gyro.ndim != 2
        or rpy.ndim != 2
        or gyro.shape[1:] != (3,)
        or rpy.shape != gyro.shape
        or gyro.shape[0] < 2
    ):
        raise RuntimeError(
            "Gyro calibration samples must both have shape (N, 3)"
        )
    if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(rpy)):
        raise RuntimeError("Gyro calibration samples contain NaN/Inf")
    standard_deviation = np.std(gyro, axis=0)
    orientation_span = np.ptp(np.unwrap(rpy, axis=0), axis=0)
    if float(np.max(standard_deviation)) > float(max_std_rad_s):
        raise RuntimeError(
            "Robot moved during gyro calibration: std="
            f"{standard_deviation.tolist()} rad/s"
        )
    if float(np.max(orientation_span)) > float(max_rpy_span_rad):
        raise RuntimeError(
            "Robot orientation changed during gyro calibration: span="
            f"{orientation_span.tolist()} rad"
        )
    bias = np.mean(gyro, axis=0).astype(np.float32)
    if float(np.max(np.abs(bias))) > float(max_abs_bias_rad_s):
        raise RuntimeError(
            "Gyro bias exceeds configured safety bound: "
            f"{bias.tolist()} rad/s"
        )
    return bias


def _joint_type(name: str) -> str:
    matches = [
        joint_type
        for joint_type in ("hip", "thigh", "calf")
        if joint_type in name
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify joint type for {name!r}")
    return matches[0]


@dataclass(frozen=True)
class Model930Contract:
    expected_task: ClassVar[str] = EXPECTED_TASK
    expected_observations: ClassVar[int] = 52
    model_label: ClassVar[str] = "model_930"

    raw: dict
    joint_names: tuple[str, ...]
    default: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    rate_limit: np.ndarray
    accel_limit: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    command_scale: np.ndarray
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    obs_clip: float
    action_clip: float
    policy_dt: float
    gait_period: float
    continuous_torque: float
    peak_torque_limit: float

    @classmethod
    def from_onnx_session(
        cls,
        session,
        onnx_path: str | Path,
        expected_sha256: str = EXPECTED_ONNX_SHA256,
    ) -> "Model930Contract":
        actual_sha256 = sha256_file(onnx_path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{cls.model_label} ONNX SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        metadata = session.get_modelmeta().custom_metadata_map
        encoded = metadata.get(METADATA_KEY)
        if not encoded:
            raise RuntimeError(f"ONNX is missing {METADATA_KEY!r}")
        raw = json.loads(encoded)
        if raw.get("schema_version") != 2:
            raise RuntimeError(
                f"Unsupported RS01 contract schema {raw.get('schema_version')}"
            )
        if raw.get("task") != cls.expected_task:
            raise RuntimeError(
                f"Expected task {cls.expected_task!r}, "
                f"got {raw.get('task')!r}"
            )
        dimensions = raw.get("dimensions", {})
        if (
            int(dimensions.get("observations", -1))
            != cls.expected_observations
        ):
            raise RuntimeError(
                f"{cls.model_label} must have exactly "
                f"{cls.expected_observations} observations"
            )
        if int(dimensions.get("actions", -1)) != 12:
            raise RuntimeError(
                f"{cls.model_label} must have exactly 12 actions"
            )
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        if (
            input_shape[1] != cls.expected_observations
            or output_shape[1] != 12
        ):
            raise RuntimeError(
                f"ONNX graph dimensions disagree: {input_shape} -> {output_shape}"
            )

        joint_names = tuple(raw["joint_names"])
        if joint_names != POLICY_JOINT_NAMES:
            raise RuntimeError(
                "Unexpected policy joint order: "
                f"{joint_names}; expected {POLICY_JOINT_NAMES}"
            )
        expected_motor_ids = (
            "0x21", "0x22", "0x23",
            "0x11", "0x12", "0x13",
            "0x31", "0x32", "0x33",
            "0x41", "0x42", "0x43",
        )
        mapping = raw.get("motor_mapping", [])
        if tuple(item.get("motor_id") for item in mapping) != expected_motor_ids:
            raise RuntimeError("ONNX motor mapping does not match new-machine wiring")
        if tuple(item.get("action_index") for item in mapping) != tuple(range(12)):
            raise RuntimeError("ONNX motor mapping action indices are invalid")
        expected_signs = tuple(
            SIGN_BY_MOTOR_ID[int(motor_id, 16)]
            for motor_id in expected_motor_ids
        )
        actual_signs = tuple(
            float(item.get("real_to_policy_sign", 0.0))
            for item in mapping
        )
        if actual_signs != expected_signs:
            raise RuntimeError(
                "ONNX joint semantic signs do not match the verified "
                f"new-machine mapping: {actual_signs} != {expected_signs}"
            )

        control = raw["control"]
        observations = raw["observations"]
        if observations.get("heading_representation") != "sin_cos":
            raise RuntimeError(
                f"{cls.model_label} requires sin/cos heading observations"
            )
        policy_dt = float(control["policy_dt_s"])
        if abs(policy_dt - 0.02) > 1.0e-9:
            raise RuntimeError(f"Expected 50 Hz policy, got dt={policy_dt}")

        lower = np.asarray(
            [URDF_LOWER_BY_TYPE[_joint_type(name)] for name in joint_names],
            dtype=np.float32,
        )
        upper = np.asarray(
            [URDF_UPPER_BY_TYPE[_joint_type(name)] for name in joint_names],
            dtype=np.float32,
        )
        result = cls(
            raw=raw,
            joint_names=joint_names,
            default=np.asarray(
                raw["default_joint_angles_rad"], dtype=np.float32
            ).reshape(12),
            action_scale=np.asarray(
                control["action_scale_rad"], dtype=np.float32
            ).reshape(12),
            kp=np.asarray(
                control["kp_nm_per_rad"], dtype=np.float32
            ).reshape(12),
            kd=np.asarray(
                control["kd_nm_per_rad_s"], dtype=np.float32
            ).reshape(12),
            rate_limit=np.asarray(
                control["target_rate_limit_rad_s"], dtype=np.float32
            ).reshape(12),
            accel_limit=np.asarray(
                control["target_acceleration_limit_rad_s2"], dtype=np.float32
            ).reshape(12),
            lower=lower,
            upper=upper,
            command_scale=np.asarray(
                observations["command_scale"], dtype=np.float32
            ).reshape(3),
            lin_vel_scale=float(observations["lin_vel_scale"]),
            ang_vel_scale=float(observations["ang_vel_scale"]),
            dof_pos_scale=float(observations["dof_pos_scale"]),
            dof_vel_scale=float(observations["dof_vel_scale"]),
            obs_clip=abs(float(observations["clip"])),
            action_clip=abs(float(control["action_clip"])),
            policy_dt=policy_dt,
            gait_period=float(raw["gait"]["period_s"]),
            continuous_torque=float(control["continuous_torque_nm"]),
            peak_torque_limit=float(control["peak_torque_limit_nm"]),
        )
        arrays = (
            result.default,
            result.action_scale,
            result.kp,
            result.kd,
            result.rate_limit,
            result.accel_limit,
            result.lower,
            result.upper,
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise RuntimeError("ONNX contract contains NaN/Inf")
        if np.any(result.default < result.lower) or np.any(
            result.default > result.upper
        ):
            raise RuntimeError("Default pose lies outside the RS01 URDF limits")
        return result


class Rs01Model930Mapper:
    """Map physical FR,FL,RL,RR feedback to policy FL,FR,RL,RR semantics."""

    real_motor_ids = np.asarray(REAL_MOTOR_IDS, dtype=np.int32)
    real_joint_names = REAL_JOINT_NAMES

    def __init__(self, contract: Model930Contract):
        self.contract = contract
        self.policy_to_real = np.asarray(
            [REAL_JOINT_NAMES.index(name) for name in contract.joint_names],
            dtype=np.int64,
        )
        self.real_to_policy = np.empty(12, dtype=np.int64)
        self.real_to_policy[self.policy_to_real] = np.arange(12, dtype=np.int64)
        policy_motor_ids = [
            REAL_MOTOR_IDS[index] for index in self.policy_to_real
        ]
        self.sign_policy = np.asarray(
            [SIGN_BY_MOTOR_ID[motor_id] for motor_id in policy_motor_ids],
            dtype=np.float32,
        )

    def real_to_policy_abs(self, q_real, dq_real):
        q_real = np.asarray(q_real, dtype=np.float32).reshape(12)
        dq_real = np.asarray(dq_real, dtype=np.float32).reshape(12)
        q_policy = self.sign_policy * q_real[self.policy_to_real]
        dq_policy = self.sign_policy * dq_real[self.policy_to_real]
        return q_policy.astype(np.float32), dq_policy.astype(np.float32)

    def real_to_policy_observation(self, q_real, dq_real):
        q_policy, dq_policy = self.real_to_policy_abs(q_real, dq_real)
        return (
            (q_policy - self.contract.default).astype(np.float32),
            dq_policy,
        )

    def policy_target_to_real(self, q_policy):
        q_policy = np.asarray(q_policy, dtype=np.float32).reshape(12)
        q_policy = np.clip(
            q_policy, self.contract.lower, self.contract.upper
        )
        real = np.empty(12, dtype=np.float32)
        real[self.policy_to_real] = self.sign_policy * q_policy
        return real

    def policy_values_to_real(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(12)
        real = np.empty(12, dtype=np.float32)
        real[self.policy_to_real] = values
        return real


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


class Rs01NewMachineLegOdometry:
    """Body velocity estimator using the actual dog_rs01 URDF geometry."""

    LEG_ORDER = ("FL", "FR", "RL", "RR")
    AXES = (
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )
    ORIGINS = {
        "FL": (
            (0.216, 0.060, 0.0),
            (0.0, 0.08725, 0.0),
            (-0.155697793236241, 0.0, -0.0903227390049967),
            (0.201984878571, 0.0, 0.00837310150402),
        ),
        "FR": (
            (0.216, -0.060, 0.0),
            (0.0, -0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984878568, 0.0, 0.00837310156582),
        ),
        "RL": (
            (-0.216, 0.060, 0.0),
            (0.0, 0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984876549, 0.0, 0.00837315026116),
        ),
        "RR": (
            (-0.216, -0.060, 0.0),
            (0.0, -0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984876547, 0.0, 0.0083731503174),
        ),
    }

    def __init__(
        self,
        nominal_base_height=0.307,
        foot_radius=0.016,
        height_margin=0.030,
        vertical_speed_threshold=0.25,
        velocity_residual_threshold=0.35,
        filter_alpha=0.35,
        no_contact_decay=0.90,
        previous_stance_score_bonus=0.08,
    ):
        self.nominal_base_height = float(nominal_base_height)
        self.foot_radius = float(foot_radius)
        self.height_margin = float(height_margin)
        self.vertical_speed_threshold = float(vertical_speed_threshold)
        self.velocity_residual_threshold = float(
            velocity_residual_threshold
        )
        self.filter_alpha = float(filter_alpha)
        self.no_contact_decay = float(no_contact_decay)
        self.previous_stance_score_bonus = float(
            previous_stance_score_bonus
        )
        self.filtered = np.zeros(3, dtype=np.float32)
        self.last_stance = np.zeros(4, dtype=bool)

    def reset(self):
        """Clear velocity filtering and contact-selection hysteresis."""
        self.filtered.fill(0.0)
        self.last_stance.fill(False)

    @classmethod
    def foot_position_and_jacobian(cls, leg, q_leg):
        q_leg = np.asarray(q_leg, dtype=np.float64).reshape(3)
        origins = cls.ORIGINS[leg]
        rotation = np.eye(3)
        position = np.zeros(3)
        joint_positions = []
        joint_axes = []
        for index in range(3):
            position += rotation @ np.asarray(origins[index])
            joint_positions.append(position.copy())
            joint_axes.append(rotation @ cls.AXES[index])
            rotation = rotation @ _axis_angle(
                cls.AXES[index], float(q_leg[index])
            )
        foot = position + rotation @ np.asarray(origins[3])
        jacobian = np.zeros((3, 3))
        for index in range(3):
            jacobian[:, index] = np.cross(
                joint_axes[index], foot - joint_positions[index]
            )
        return foot.astype(np.float32), jacobian.astype(np.float32)

    def estimate(self, q_policy, dq_policy, omega_body):
        q_policy = np.asarray(q_policy, dtype=np.float32).reshape(12)
        dq_policy = np.asarray(dq_policy, dtype=np.float32).reshape(12)
        omega_body = np.asarray(omega_body, dtype=np.float32).reshape(3)
        foot_position = np.zeros((4, 3), dtype=np.float32)
        foot_velocity = np.zeros((4, 3), dtype=np.float32)
        velocity_by_foot = np.zeros((4, 3), dtype=np.float32)
        base_height_proxy = np.zeros(4, dtype=np.float32)
        for leg_index, leg in enumerate(self.LEG_ORDER):
            start = leg_index * 3
            position, jacobian = self.foot_position_and_jacobian(
                leg, q_policy[start:start + 3]
            )
            relative_velocity = jacobian @ dq_policy[start:start + 3]
            velocity_by_foot[leg_index] = -(
                relative_velocity + np.cross(omega_body, position)
            )
            foot_position[leg_index] = position
            foot_velocity[leg_index] = relative_velocity
            base_height_proxy[leg_index] = -position[2] + self.foot_radius

        lowest = float(np.max(base_height_proxy))
        candidates = []
        for leg_index in range(4):
            height_gap = lowest - float(base_height_proxy[leg_index])
            vertical_speed = abs(float(foot_velocity[leg_index, 2]))
            absolute_error = abs(
                float(base_height_proxy[leg_index])
                - self.nominal_base_height
            )
            if height_gap > self.height_margin:
                continue
            if vertical_speed > self.vertical_speed_threshold:
                continue
            if absolute_error > 0.10:
                continue
            score = (
                height_gap / max(self.height_margin, 1.0e-6)
                + vertical_speed
                / max(self.vertical_speed_threshold, 1.0e-6)
                - (
                    self.previous_stance_score_bonus
                    if self.last_stance[leg_index]
                    else 0.0
                )
            )
            candidates.append((score, leg_index))

        selected = []
        if candidates:
            candidates.sort()
            preselected = candidates[: min(3, len(candidates))]
            velocities = np.asarray(
                [velocity_by_foot[index] for _, index in preselected]
            )
            center = np.median(velocities, axis=0)
            accepted = []
            for score, index in preselected:
                residual = float(
                    np.linalg.norm(
                        (velocity_by_foot[index] - center)[:2]
                    )
                )
                if residual <= self.velocity_residual_threshold:
                    accepted.append((score, residual, index))
            if not accepted:
                accepted = [
                    min(
                        (
                            (
                                score,
                                float(
                                    np.linalg.norm(
                                        (
                                            velocity_by_foot[index] - center
                                        )[:2]
                                    )
                                ),
                                index,
                            )
                            for score, index in preselected
                        ),
                        key=lambda item: item[1],
                    )
                ]
            accepted.sort()
            selected = [item[2] for item in accepted[:2]]

        stance = np.zeros(4, dtype=bool)
        stance[selected] = True
        self.last_stance = stance
        if selected:
            raw = np.mean(velocity_by_foot[selected], axis=0)
            raw[:2] = np.clip(raw[:2], -1.0, 1.0)
            raw[2] = 0.0
            confidence = len(selected) / 2.0
            alpha = self.filter_alpha * confidence
            self.filtered = (
                (1.0 - alpha) * self.filtered + alpha * raw
            ).astype(np.float32)
        else:
            confidence = 0.0
            self.filtered *= self.no_contact_decay
        self.filtered[2] = 0.0
        return {
            "base_linear_velocity": self.filtered.copy(),
            "confidence": float(confidence),
            "stance_mask": stance,
            "foot_position": foot_position,
            "foot_velocity": foot_velocity,
            "velocity_by_foot": velocity_by_foot,
            "base_height_proxy": base_height_proxy,
        }


class Rs01Model930TargetLimiter:
    """Exact policy-rate target limits plus PD-equivalent peak protection."""

    def __init__(self, contract: Model930Contract):
        self.contract = contract
        self.target = contract.default.copy()
        self.target_velocity = np.zeros(12, dtype=np.float32)

    def reset(self, target):
        self.target = np.clip(
            np.asarray(target, dtype=np.float32).reshape(12),
            self.contract.lower,
            self.contract.upper,
        )
        self.target_velocity.fill(0.0)

    def step(self, desired):
        desired = np.clip(
            np.asarray(desired, dtype=np.float32).reshape(12),
            self.contract.lower,
            self.contract.upper,
        )
        dt = self.contract.policy_dt
        error = desired - self.target
        # Select a braking-aware speed.  The positive solution of
        # v^2 + 2*a*dt*v - 2*a*distance = 0 is the largest discrete-time
        # velocity that can still decelerate without snapping to the target.
        # This avoids the common "crossed target -> set exact target" shortcut,
        # whose last frame violates the configured acceleration limit.
        braking_speed = (
            -self.contract.accel_limit * dt
            + np.sqrt(
                (self.contract.accel_limit * dt) ** 2
                + 2.0 * self.contract.accel_limit * np.abs(error)
            )
        )
        desired_velocity = np.sign(error) * np.minimum(
            self.contract.rate_limit,
            braking_speed,
        )
        velocity_delta = np.clip(
            desired_velocity - self.target_velocity,
            -self.contract.accel_limit * dt,
            self.contract.accel_limit * dt,
        )
        next_velocity = self.target_velocity + velocity_delta
        next_target = self.target + next_velocity * dt
        self.target_velocity = ((next_target - self.target) / dt).astype(
            np.float32
        )
        self.target = next_target.astype(np.float32)
        return self.target.copy()

    def pd_equivalent_peak_limit(
        self,
        target,
        q_policy,
        dq_policy,
        active_limit_nm,
    ):
        target = np.asarray(target, dtype=np.float32).reshape(12)
        q_policy = np.asarray(q_policy, dtype=np.float32).reshape(12)
        dq_policy = np.asarray(dq_policy, dtype=np.float32).reshape(12)
        limit = np.asarray(active_limit_nm, dtype=np.float32)
        if limit.shape == ():
            limit = np.full(12, float(limit), dtype=np.float32)
        else:
            limit = limit.reshape(12)
        raw = (
            self.contract.kp * (target - q_policy)
            - self.contract.kd * dq_policy
        )
        safe = np.clip(raw, -limit, limit)
        safe_target = q_policy + (
            safe + self.contract.kd * dq_policy
        ) / self.contract.kp
        safe_target = np.clip(
            safe_target, self.contract.lower, self.contract.upper
        )
        return safe_target.astype(np.float32), {
            "raw_pd_torque_nm": raw.astype(np.float32),
            "safe_pd_torque_nm": safe.astype(np.float32),
            "limited_mask": np.abs(raw - safe) > 1.0e-6,
        }


class Rs01ContinuousTorqueGuard:
    """EWMA thermal proxy that derates peak authority toward 6 N.m."""

    def __init__(
        self,
        continuous_torque_nm,
        peak_torque_nm,
        derate_full_rms_nm=8.0,
        time_constant_s=2.0,
    ):
        self.continuous = float(continuous_torque_nm)
        self.peak = float(peak_torque_nm)
        self.derate_full = float(derate_full_rms_nm)
        self.time_constant = float(time_constant_s)
        if not (
            0.0 < self.continuous < self.derate_full <= self.peak
        ):
            raise ValueError("Invalid RS01 continuous-torque derating range")
        if self.time_constant <= 0.0:
            raise ValueError("Thermal RMS time constant must be positive")
        self.rms_sq = np.zeros(12, dtype=np.float32)

    def reset(self):
        self.rms_sq.fill(0.0)

    @property
    def rms(self):
        return np.sqrt(np.maximum(self.rms_sq, 0.0)).astype(np.float32)

    def active_limits(self):
        blend = np.clip(
            (self.rms - self.continuous)
            / (self.derate_full - self.continuous),
            0.0,
            1.0,
        )
        return (
            self.peak - blend * (self.peak - self.continuous)
        ).astype(np.float32)

    def update(self, torque_nm, dt):
        torque = np.asarray(torque_nm, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(torque)):
            raise RuntimeError("Thermal RMS input contains NaN/Inf")
        alpha = math.exp(-max(float(dt), 0.0) / self.time_constant)
        self.rms_sq = (
            alpha * self.rms_sq
            + (1.0 - alpha) * np.square(torque)
        ).astype(np.float32)
        return self.active_limits()


class Rs01Model930PolicyCore:
    """Observation, inference, and target path without ROS or hardware I/O."""

    def __init__(self, session, contract: Model930Contract):
        self.session = session
        self.contract = contract
        self.mapper = Rs01Model930Mapper(contract)
        self.limiter = Rs01Model930TargetLimiter(contract)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.phase_start = 0.0
        self.heading_target = 0.0

    def reset(self, now, yaw, q_policy=None):
        self.previous_action.fill(0.0)
        self.phase_start = float(now)
        self.heading_target = float(yaw)
        self.limiter.reset(
            self.contract.default if q_policy is None else q_policy
        )

    def build_observation(
        self,
        now,
        base_linear_velocity,
        base_angular_velocity,
        projected_gravity,
        command,
        q_policy,
        dq_policy,
        yaw,
    ):
        phase = (
            (float(now) - self.phase_start) / self.contract.gait_period
        ) % 1.0
        phase_angle = 2.0 * math.pi * phase
        heading_error = wrap_pi(self.heading_target - float(yaw))
        command = np.asarray(command, dtype=np.float32).reshape(3)
        observation = np.concatenate(
            [
                np.asarray(base_linear_velocity, dtype=np.float32).reshape(3)
                * self.contract.lin_vel_scale,
                np.asarray(base_angular_velocity, dtype=np.float32).reshape(3)
                * self.contract.ang_vel_scale,
                np.asarray(projected_gravity, dtype=np.float32).reshape(3),
                command * self.contract.command_scale,
                (np.asarray(q_policy, dtype=np.float32).reshape(12)
                 - self.contract.default)
                * self.contract.dof_pos_scale,
                np.asarray(dq_policy, dtype=np.float32).reshape(12)
                * self.contract.dof_vel_scale,
                self.previous_action,
                np.asarray(
                    [math.sin(phase_angle), math.cos(phase_angle)],
                    dtype=np.float32,
                ),
                np.asarray(
                    [math.sin(heading_error), math.cos(heading_error)],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if observation.shape != (52,):
            raise RuntimeError(
                f"model_930 observation shape is {observation.shape}, expected (52,)"
            )
        return np.clip(
            observation,
            -self.contract.obs_clip,
            self.contract.obs_clip,
        ).astype(np.float32)

    def step(self, observation, q_policy, dq_policy, active_limit_nm):
        output = self.session.run(
            ["actions"],
            {"observations": np.asarray(observation, dtype=np.float32)[None, :]},
        )[0][0]
        action = np.clip(
            np.asarray(output, dtype=np.float32).reshape(12),
            -self.contract.action_clip,
            self.contract.action_clip,
        )
        desired = self.contract.default + self.contract.action_scale * action
        limited = self.limiter.step(desired)
        safe, torque_info = self.limiter.pd_equivalent_peak_limit(
            limited,
            q_policy,
            dq_policy,
            active_limit_nm,
        )
        self.previous_action = action.copy()
        return {
            "action_raw": np.asarray(output, dtype=np.float32).reshape(12),
            "action": action,
            "desired_target_policy": desired.astype(np.float32),
            "limited_target_policy": limited,
            "safe_target_policy": safe,
            "torque_info": torque_info,
        }
