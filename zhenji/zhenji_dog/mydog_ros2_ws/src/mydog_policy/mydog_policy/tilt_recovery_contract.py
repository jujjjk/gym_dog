"""Fail-closed deployment contract for the selected tilt-recovery policy."""

from __future__ import annotations

from typing import Any

from .symmetric_transition_contract import (
    ACTIONS,
    ACTION_SCALE,
    DEFAULT_JOINT_ANGLES,
    DEPLOYMENT_LIMITS,
    DAMPING,
    JOINT_NAMES,
    OBSERVATIONS,
    POLICY_ACTION_ACCEL_LIMITS,
    POLICY_ACTION_RATE_LIMITS,
    STIFFNESS,
    TORQUE_LIMITS_POLICY,
    TRAINING_LIMITS,
    validate_metadata_against,
)

MODEL_TASK = "FanfanOmniTiltRecovery5530Cfg"
MODEL_FILENAME = "fanfan_tilt_recovery_5530_5650.onnx"
MODEL_SHA256 = (
    "8f370f9fd1165774426d80eef72391152137d28dc6b429af57af96284e68893f"
)

COMMAND_FEEDBACK = {
    "command_feedback_longitudinal_gain": 0.40,
    "command_feedback_lateral_gain": 0.55,
    "command_feedback_yaw_gain": 0.25,
    "command_feedback_heading_gain": 4.00,
    "command_feedback_heading_damping": 1.00,
    "command_feedback_diagonal_longitudinal_scale": 0.60,
}


def validate_metadata(contract: dict[str, Any]) -> bool:
    """Reject any artifact that differs from checkpoint 5650's contract."""
    return validate_metadata_against(
        contract,
        model_task=MODEL_TASK,
        command_feedback=COMMAND_FEEDBACK,
    )


__all__ = [
    "ACTIONS",
    "ACTION_SCALE",
    "COMMAND_FEEDBACK",
    "DEFAULT_JOINT_ANGLES",
    "DEPLOYMENT_LIMITS",
    "DAMPING",
    "JOINT_NAMES",
    "MODEL_FILENAME",
    "MODEL_SHA256",
    "MODEL_TASK",
    "OBSERVATIONS",
    "POLICY_ACTION_ACCEL_LIMITS",
    "POLICY_ACTION_RATE_LIMITS",
    "STIFFNESS",
    "TORQUE_LIMITS_POLICY",
    "TRAINING_LIMITS",
    "validate_metadata",
]
