#!/usr/bin/env python3
"""Strict 54-D deployment contract for estimator-parity model_1950."""

from __future__ import annotations

from pathlib import Path

from .rs01_model1850_core import (
    Model1850Contract,
    Rs01Model1850PolicyCore,
)


EXPECTED_TASK = "rs01_go2_estimator_parity"
EXPECTED_ONNX_SHA256 = (
    "f78242f6ac60354421d7354b8a5f4b61284864c18be409aa0f854520f8d1202c"
)


class Model1950Contract(Model1850Contract):
    """Reject any artifact that is not the selected estimator-parity actor."""

    expected_task = EXPECTED_TASK
    expected_observations = 54
    model_label = "model_1950"

    @classmethod
    def from_onnx_session(
        cls,
        session,
        onnx_path: str | Path,
        expected_sha256: str = EXPECTED_ONNX_SHA256,
    ) -> "Model1950Contract":
        result = super().from_onnx_session(
            session,
            onnx_path,
            expected_sha256=expected_sha256,
        )
        observations = result.raw["observations"]
        if (
            observations.get("base_linear_velocity_source")
            != "rs01_leg_odometry"
        ):
            raise RuntimeError(
                "model_1950 requires RS01 leg-odometry base velocity"
            )
        if (
            observations.get("straight_path_state_source")
            != "rs01_leg_odometry_integral"
        ):
            raise RuntimeError(
                "model_1950 requires the RS01 leg-odometry path integral"
            )
        return result


class Rs01Model1950PolicyCore(Rs01Model1850PolicyCore):
    """Exact model_1950 inference path; no compensation or extra controller."""
