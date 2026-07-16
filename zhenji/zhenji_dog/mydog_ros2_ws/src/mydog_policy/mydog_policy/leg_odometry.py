#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np


POLICY_LEG_ORDER = ("FR", "FL", "RR", "RL")


@dataclass
class LegOdometryResult:
    base_lin_vel: np.ndarray
    raw_base_lin_vel: np.ndarray
    stance_mask: np.ndarray
    foot_pos: np.ndarray
    foot_vel_rel: np.ndarray
    foot_height: np.ndarray
    foot_speed: np.ndarray
    confidence: float


class FanfanLegKinematics:
    """
    Minimal FK/Jacobian model copied from fanfan.urdf.

    All vectors are expressed in the Trunk/body frame. Joint angle order is the
    policy order: FR, FL, RR, RL, each with hip/thigh/calf.
    """

    FOOT_RADIUS = 0.018

    LEG_DATA = {
        "FR": {
            "origins": (
                (0.19, -0.0451, 0.00075),
                (0.0, -0.076, 0.0),
                (0.0, 0.0005, -0.15606),
                (0.0, 0.0, -0.148941836101256),
            ),
        },
        "FL": {
            "origins": (
                (0.19, 0.0451, 0.00075),
                (0.0, 0.076, 0.0),
                (0.0, -0.0005, -0.15606),
                (0.0, 0.0, -0.148941836101256),
            ),
        },
        "RR": {
            "origins": (
                (-0.19, -0.06, 0.00075),
                (0.0, -0.076, 0.0),
                (0.0, 0.0005, -0.15606),
                (0.0, 0.0, -0.14894),
            ),
        },
        "RL": {
            "origins": (
                (-0.19, 0.06, 0.00075),
                (0.0, 0.076, 0.0),
                (0.0, -0.0005, -0.15606),
                (0.0, 0.0, -0.14894),
            ),
        },
    }

    JOINT_AXES = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )

    @staticmethod
    def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-8:
            return np.eye(3, dtype=np.float32)

        x, y, z = axis / norm
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        C = 1.0 - c

        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=np.float32,
        )

    def foot_position_and_jacobian(self, leg: str, q_leg: np.ndarray):
        q_leg = np.asarray(q_leg, dtype=np.float32).reshape(3)
        origins = self.LEG_DATA[leg]["origins"]

        R = np.eye(3, dtype=np.float32)
        p = np.zeros(3, dtype=np.float32)

        joint_pos = []
        joint_axis_world = []

        for i in range(3):
            p = p + R @ np.asarray(origins[i], dtype=np.float32)
            joint_pos.append(p.copy())
            joint_axis_world.append(R @ self.JOINT_AXES[i])
            R = R @ self._axis_angle(self.JOINT_AXES[i], float(q_leg[i]))

        foot_pos = p + R @ np.asarray(origins[3], dtype=np.float32)

        J = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            J[:, i] = np.cross(joint_axis_world[i], foot_pos - joint_pos[i])

        return foot_pos.astype(np.float32), J.astype(np.float32)


class LegOdometryEstimator:
    """
    Estimate body-frame base linear velocity from stance feet.

    Stance selection is intentionally simple for first real-machine debugging:
    a foot is treated as stance when its body-frame height is close to the
    nominal standing geometry and its relative foot speed is not too high.
    """

    def __init__(
        self,
        nominal_base_height: float = 0.293,
        foot_radius: float = FanfanLegKinematics.FOOT_RADIUS,
        stance_height_margin: float = 0.045,
        stance_speed_threshold: float = 0.65,
        filter_alpha: float = 0.25,
    ):
        self.kin = FanfanLegKinematics()
        self.nominal_base_height = float(nominal_base_height)
        self.foot_radius = float(foot_radius)
        self.stance_height_margin = float(stance_height_margin)
        self.stance_speed_threshold = float(stance_speed_threshold)
        self.filter_alpha = float(filter_alpha)

        self.filtered = np.zeros(3, dtype=np.float32)
        self.last_raw = np.zeros(3, dtype=np.float32)

    def estimate(
        self,
        q_abs_policy: np.ndarray,
        dq_policy: np.ndarray,
        omega_body: np.ndarray,
    ) -> LegOdometryResult:
        q_abs_policy = np.asarray(q_abs_policy, dtype=np.float32).reshape(12)
        dq_policy = np.asarray(dq_policy, dtype=np.float32).reshape(12)
        omega_body = np.asarray(omega_body, dtype=np.float32).reshape(3)

        foot_pos = np.zeros((4, 3), dtype=np.float32)
        foot_vel_rel = np.zeros((4, 3), dtype=np.float32)
        foot_height = np.zeros(4, dtype=np.float32)
        foot_speed = np.zeros(4, dtype=np.float32)
        candidates = []

        for leg_i, leg in enumerate(POLICY_LEG_ORDER):
            i0 = 3 * leg_i
            p, J = self.kin.foot_position_and_jacobian(
                leg,
                q_abs_policy[i0 : i0 + 3],
            )
            v_rel = J @ dq_policy[i0 : i0 + 3]

            foot_pos[leg_i] = p
            foot_vel_rel[leg_i] = v_rel
            foot_height[leg_i] = -float(p[2]) + self.foot_radius
            foot_speed[leg_i] = float(np.linalg.norm(v_rel))

            height_error = abs(float(foot_height[leg_i]) - self.nominal_base_height)
            speed_ok = foot_speed[leg_i] <= self.stance_speed_threshold
            height_ok = height_error <= self.stance_height_margin
            if height_ok and speed_ok:
                v_base_i = -(v_rel + np.cross(omega_body, p))
                weight = 1.0 / (1.0 + 30.0 * height_error + 2.0 * foot_speed[leg_i])
                candidates.append((v_base_i.astype(np.float32), weight))

        stance_mask = np.zeros(4, dtype=bool)
        for leg_i in range(4):
            height_error = abs(float(foot_height[leg_i]) - self.nominal_base_height)
            stance_mask[leg_i] = (
                height_error <= self.stance_height_margin
                and foot_speed[leg_i] <= self.stance_speed_threshold
            )

        if candidates:
            weights = np.asarray([w for _, w in candidates], dtype=np.float32)
            values = np.asarray([v for v, _ in candidates], dtype=np.float32)
            raw = np.sum(values * weights[:, None], axis=0) / max(float(np.sum(weights)), 1.0e-6)
            confidence = min(1.0, float(len(candidates)) / 2.0)
        else:
            raw = self.last_raw.copy()
            confidence = 0.0

        self.last_raw = raw.astype(np.float32)
        alpha = self.filter_alpha * confidence
        self.filtered = (1.0 - alpha) * self.filtered + alpha * raw

        # Flat walking policy normally expects little useful vertical velocity.
        self.filtered[2] = 0.0
        raw[2] = 0.0

        return LegOdometryResult(
            base_lin_vel=self.filtered.astype(np.float32).copy(),
            raw_base_lin_vel=raw.astype(np.float32).copy(),
            stance_mask=stance_mask.copy(),
            foot_pos=foot_pos.copy(),
            foot_vel_rel=foot_vel_rel.copy(),
            foot_height=foot_height.copy(),
            foot_speed=foot_speed.copy(),
            confidence=float(confidence),
        )
