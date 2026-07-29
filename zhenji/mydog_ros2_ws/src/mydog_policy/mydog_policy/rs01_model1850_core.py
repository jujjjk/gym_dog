#!/usr/bin/env python3
"""Pure-Numpy 54-D deployment contract for RS01 model_1850."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .rs01_model930_core import (
    Model930Contract,
    Rs01Model930PolicyCore,
)


EXPECTED_TASK = "rs01_go2_path54_sim2sim_transfer"
EXPECTED_ONNX_SHA256 = (
    "661415c63ebc75a5691a56782f281ea57b46d7dad13b2fd946a899dd42da1d06"
)


class Model1850Contract(Model930Contract):
    """Strictly validate the selected 54-observation deployment artifact."""

    expected_task = EXPECTED_TASK
    expected_observations = 54
    model_label = "model_1850"

    @classmethod
    def from_onnx_session(
        cls,
        session,
        onnx_path: str | Path,
        expected_sha256: str = EXPECTED_ONNX_SHA256,
    ) -> "Model1850Contract":
        result = super().from_onnx_session(
            session,
            onnx_path,
            expected_sha256=expected_sha256,
        )
        observations = result.raw["observations"]
        expected_tail = [
            ["straight_path_lateral_displacement_scaled", 1],
            ["straight_path_lateral_velocity_scaled", 1],
        ]
        if observations.get("layout", [])[-2:] != expected_tail:
            raise RuntimeError(
                "model_1850 observation tail must be lateral path "
                "displacement then path-frame lateral velocity"
            )
        if observations.get("straight_path_state_enabled") is not True:
            raise RuntimeError(
                "model_1850 requires the 54-D straight-path state"
            )
        for name in (
            "straight_path_lateral_position_scale",
            "straight_path_lateral_velocity_scale",
        ):
            value = float(observations.get(name, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"Invalid model_1850 observation scale {name}={value}"
                )
        return result


class Rs01StraightPathEstimator:
    """Integrate path-frame lateral displacement from body velocity and yaw."""

    def __init__(self, max_update_gap_s=0.10):
        self.max_update_gap_s = float(max_update_gap_s)
        if self.max_update_gap_s <= 0.0:
            raise ValueError("max_update_gap_s must be positive")
        self.heading_target = 0.0
        self.lateral_displacement_m = 0.0
        self.lateral_velocity_m_s = 0.0
        self.last_update_s = None

    def reset(self, now, yaw):
        values = np.asarray([now, yaw], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Path reset contains NaN/Inf")
        self.heading_target = float(yaw)
        self.lateral_displacement_m = 0.0
        self.lateral_velocity_m_s = 0.0
        self.last_update_s = float(now)

    def update(self, now, yaw, base_linear_velocity_body):
        values = np.asarray(
            [
                now,
                yaw,
                *np.asarray(
                    base_linear_velocity_body, dtype=np.float64
                ).reshape(3),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Path observation contains NaN/Inf")
        if self.last_update_s is None:
            self.reset(now, yaw)
        dt = float(now) - float(self.last_update_s)
        if dt < -1.0e-9:
            raise RuntimeError("Path estimator clock moved backwards")
        if dt > self.max_update_gap_s:
            raise RuntimeError(
                f"Path estimator update gap {dt:.3f}s exceeds "
                f"{self.max_update_gap_s:.3f}s"
            )
        velocity_body = values[2:5]
        relative_yaw = float(yaw) - self.heading_target
        lateral_velocity = (
            math.sin(relative_yaw) * float(velocity_body[0])
            + math.cos(relative_yaw) * float(velocity_body[1])
        )
        self.lateral_displacement_m += (
            0.5
            * (self.lateral_velocity_m_s + lateral_velocity)
            * max(dt, 0.0)
        )
        self.lateral_velocity_m_s = lateral_velocity
        self.last_update_s = float(now)
        return (
            float(self.lateral_displacement_m),
            float(self.lateral_velocity_m_s),
        )


class Rs01Model1850PolicyCore(Rs01Model930PolicyCore):
    """Exact 54-D observation, inference, and target path for model_1850."""

    def __init__(self, session, contract: Model1850Contract):
        super().__init__(session, contract)
        self.path_estimator = Rs01StraightPathEstimator()
        self.last_path_lateral_displacement_m = 0.0
        self.last_path_lateral_velocity_m_s = 0.0

    def reset(self, now, yaw, q_policy=None):
        super().reset(now, yaw, q_policy=q_policy)
        self.path_estimator.reset(now, yaw)
        self.last_path_lateral_displacement_m = 0.0
        self.last_path_lateral_velocity_m_s = 0.0

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
        proprioception = super().build_observation(
            now=now,
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=base_angular_velocity,
            projected_gravity=projected_gravity,
            command=command,
            q_policy=q_policy,
            dq_policy=dq_policy,
            yaw=yaw,
        )
        lateral_displacement, lateral_velocity = (
            self.path_estimator.update(
                now,
                yaw,
                base_linear_velocity,
            )
        )
        self.last_path_lateral_displacement_m = lateral_displacement
        self.last_path_lateral_velocity_m_s = lateral_velocity
        observations = self.contract.raw["observations"]
        path_state = np.asarray(
            [
                np.clip(
                    lateral_displacement
                    * float(
                        observations[
                            "straight_path_lateral_position_scale"
                        ]
                    ),
                    -1.0,
                    1.0,
                ),
                np.clip(
                    lateral_velocity
                    * float(
                        observations[
                            "straight_path_lateral_velocity_scale"
                        ]
                    ),
                    -1.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (proprioception, path_state)
        ).astype(np.float32)
        if observation.shape != (54,):
            raise RuntimeError(
                "model_1850 observation shape is "
                f"{observation.shape}, expected (54,)"
            )
        return np.clip(
            observation,
            -self.contract.obs_clip,
            self.contract.obs_clip,
        ).astype(np.float32)
