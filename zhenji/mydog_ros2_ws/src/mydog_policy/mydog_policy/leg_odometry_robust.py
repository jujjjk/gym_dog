#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust leg-odometry estimator for real omnidirectional deployment.

Compared with the original estimator, stance selection:
1. uses relative foot height instead of only a fixed body-height window;
2. checks vertical foot speed rather than rejecting legitimate horizontal motion;
3. rejects inconsistent per-foot velocity estimates with a robust median;
4. limits the selected support set to the most plausible two feet.

The public result fields intentionally match leg_odometry.LegOdometryResult so
state_estimator_fixed_node.py can use this estimator without changing topics.
"""

from __future__ import annotations

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
    """Minimal FK/Jacobian model copied from the current fanfan URDF."""

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
        joint_axis_body = []

        for i in range(3):
            p = p + R @ np.asarray(origins[i], dtype=np.float32)
            joint_pos.append(p.copy())
            joint_axis_body.append(R @ self.JOINT_AXES[i])
            R = R @ self._axis_angle(self.JOINT_AXES[i], float(q_leg[i]))

        foot_pos = p + R @ np.asarray(origins[3], dtype=np.float32)
        J = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            J[:, i] = np.cross(joint_axis_body[i], foot_pos - joint_pos[i])

        return foot_pos.astype(np.float32), J.astype(np.float32)


class RobustLegOdometryEstimator:
    """Estimate body-frame planar velocity from robustly selected support feet."""

    def __init__(
        self,
        nominal_base_height: float = 0.293,
        foot_radius: float = FanfanLegKinematics.FOOT_RADIUS,
        stance_height_margin: float = 0.035,
        stance_vertical_speed_threshold: float = 0.22,
        stance_velocity_residual_threshold: float = 0.30,
        max_stance_feet: int = 2,
        filter_alpha: float = 0.35,
        absolute_height_guard: float = 0.10,
        planar_velocity_clip: float = 1.0,
    ):
        self.kin = FanfanLegKinematics()
        self.nominal_base_height = float(nominal_base_height)
        self.foot_radius = float(foot_radius)
        self.stance_height_margin = max(float(stance_height_margin), 1.0e-4)
        self.stance_vertical_speed_threshold = max(
            float(stance_vertical_speed_threshold), 1.0e-4
        )
        self.stance_velocity_residual_threshold = max(
            float(stance_velocity_residual_threshold), 1.0e-4
        )
        self.max_stance_feet = int(np.clip(int(max_stance_feet), 1, 4))
        self.filter_alpha = float(np.clip(float(filter_alpha), 0.0, 1.0))
        self.absolute_height_guard = max(float(absolute_height_guard), 0.02)
        self.planar_velocity_clip = max(float(planar_velocity_clip), 0.1)

        self.filtered = np.zeros(3, dtype=np.float32)
        self.last_raw = np.zeros(3, dtype=np.float32)
        self.last_stance_mask = np.zeros(4, dtype=bool)

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
        velocity_by_foot = np.zeros((4, 3), dtype=np.float32)

        for leg_i, leg in enumerate(POLICY_LEG_ORDER):
            i0 = 3 * leg_i
            p, J = self.kin.foot_position_and_jacobian(
                leg,
                q_abs_policy[i0 : i0 + 3],
            )
            v_rel = J @ dq_policy[i0 : i0 + 3]
            v_base_i = -(v_rel + np.cross(omega_body, p))

            foot_pos[leg_i] = p
            foot_vel_rel[leg_i] = v_rel
            foot_height[leg_i] = -float(p[2]) + self.foot_radius
            foot_speed[leg_i] = float(np.linalg.norm(v_rel))
            velocity_by_foot[leg_i] = v_base_i.astype(np.float32)

        # A support foot is normally among the physically lowest feet.  Using a
        # relative height gate tolerates small changes in body height and roll.
        lowest_height = float(np.max(foot_height))
        candidates = []
        for leg_i in range(4):
            height_gap = max(0.0, lowest_height - float(foot_height[leg_i]))
            absolute_height_error = abs(
                float(foot_height[leg_i]) - self.nominal_base_height
            )
            vertical_speed = abs(float(foot_vel_rel[leg_i, 2]))

            relative_height_ok = height_gap <= self.stance_height_margin
            absolute_height_ok = absolute_height_error <= self.absolute_height_guard
            vertical_ok = (
                vertical_speed <= self.stance_vertical_speed_threshold
            )
            if not (relative_height_ok and absolute_height_ok and vertical_ok):
                continue

            score = (
                height_gap / self.stance_height_margin
                + vertical_speed / self.stance_vertical_speed_threshold
            )
            # A small hysteresis preference prevents stance identity from
            # chattering when two feet have almost identical scores.
            if self.last_stance_mask[leg_i]:
                score -= 0.08
            candidates.append(
                {
                    "leg_i": leg_i,
                    "score": float(score),
                    "height_gap": float(height_gap),
                    "vertical_speed": float(vertical_speed),
                    "velocity": velocity_by_foot[leg_i].copy(),
                }
            )

        candidates.sort(key=lambda item: item["score"])

        # Keep one extra candidate for robust residual rejection, then retain
        # only the best two support feet for a diagonal trot.
        preselect_count = min(len(candidates), max(3, self.max_stance_feet + 1))
        candidates = candidates[:preselect_count]

        selected = []
        if candidates:
            values = np.asarray(
                [item["velocity"] for item in candidates],
                dtype=np.float32,
            )
            center = np.median(values, axis=0)
            for item in candidates:
                residual = float(
                    np.linalg.norm((item["velocity"] - center)[:2])
                )
                item["residual"] = residual
                if residual <= self.stance_velocity_residual_threshold:
                    selected.append(item)

            # Never discard every plausible foot solely because the residual
            # check became strict during a transient.
            if not selected:
                best = min(
                    candidates,
                    key=lambda item: float(
                        np.linalg.norm((item["velocity"] - center)[:2])
                    ),
                )
                best["residual"] = float(
                    np.linalg.norm((best["velocity"] - center)[:2])
                )
                selected = [best]

            selected.sort(key=lambda item: (item["score"], item["residual"]))
            selected = selected[: self.max_stance_feet]

        stance_mask = np.zeros(4, dtype=bool)
        for item in selected:
            stance_mask[item["leg_i"]] = True
        self.last_stance_mask = stance_mask.copy()

        if selected:
            values = np.asarray(
                [item["velocity"] for item in selected],
                dtype=np.float32,
            )
            weights = np.asarray(
                [
                    1.0
                    / (
                        1.0
                        + 8.0 * item["height_gap"]
                        + 2.0 * item["vertical_speed"]
                        + 2.0 * item["residual"]
                    )
                    for item in selected
                ],
                dtype=np.float32,
            )
            raw = np.sum(values * weights[:, None], axis=0) / max(
                float(np.sum(weights)),
                1.0e-6,
            )
            confidence = min(
                1.0,
                float(len(selected)) / float(max(self.max_stance_feet, 1)),
            )
        else:
            raw = self.last_raw.copy()
            confidence = 0.0

        raw = np.asarray(raw, dtype=np.float32)
        raw[:2] = np.clip(
            raw[:2],
            -self.planar_velocity_clip,
            self.planar_velocity_clip,
        )
        raw[2] = 0.0

        if confidence > 0.0:
            self.last_raw = raw.copy()

        alpha = self.filter_alpha * confidence
        self.filtered = (1.0 - alpha) * self.filtered + alpha * raw
        self.filtered[2] = 0.0

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
