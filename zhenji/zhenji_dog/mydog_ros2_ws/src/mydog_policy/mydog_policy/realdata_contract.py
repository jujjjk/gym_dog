"""Deployment contract for the real-CSV curriculum policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from .hardware_balance_contract import (
    ACTIONS,
    ACTION_SCALE,
    DEFAULT_JOINT_ANGLES,
    DAMPING,
    JOINT_NAMES,
    OBSERVATIONS,
    STIFFNESS,
    TORQUE_LIMITS_POLICY,
    _close,
    _require_list_close,
)


MODEL_TASK = "FanfanOmniRealDataPerformanceRecoveryCfg"
MODEL_FILENAME = "fanfan_realdata_best.onnx"
MODEL_SHA256 = (
    "c78fba8ffb24a1a598aa93c4aea94ca1b76ffe693eb4926d9df5e6b8048c05df"
)

FINAL_TARGET_RATE_LIMITS = [0.90, 1.30, 3.50] * 4
FINAL_TARGET_ACCEL_LIMITS = [28.0, 42.0, 80.0] * 4
REAR_CALF_TARGET_RATE_SCALE = 0.92
REALDATA_COMMAND_FEEDBACK = {
    "command_feedback_longitudinal_gain": 0.0,
    "command_feedback_lateral_gain": 0.0,
    "command_feedback_yaw_gain": 0.0,
    "command_feedback_heading_gain": 4.0,
    "command_feedback_heading_damping": 1.2,
    "command_feedback_diagonal_longitudinal_scale": 0.7,
}


def validate_metadata(contract: dict[str, Any]) -> bool:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported deployment schema")
    if contract.get("task") != MODEL_TASK:
        raise ValueError(f"realdata task mismatch: {contract.get('task')!r}")
    if contract.get("dimensions") != {
        "observations": OBSERVATIONS,
        "actions": ACTIONS,
    }:
        raise ValueError("policy dimension mismatch")
    if list(contract.get("joint_names", [])) != JOINT_NAMES:
        raise ValueError("policy joint order mismatch")
    _require_list_close(
        contract.get("default_joint_angles", []),
        DEFAULT_JOINT_ANGLES,
        "default_joint_angles",
    )

    control = contract.get("control", {})
    _require_list_close(control.get("stiffness", []), STIFFNESS, "stiffness")
    _require_list_close(control.get("damping", []), DAMPING, "damping")
    _require_list_close(
        control.get("action_scale", []), ACTION_SCALE, "action_scale"
    )
    _require_list_close(
        control.get("torque_limits", []),
        TORQUE_LIMITS_POLICY,
        "torque_limits",
    )
    _require_list_close(
        control.get("final_target_rate_limits", []),
        FINAL_TARGET_RATE_LIMITS,
        "final_target_rate_limits",
    )
    _require_list_close(
        control.get("final_target_accel_limits", []),
        FINAL_TARGET_ACCEL_LIMITS,
        "final_target_accel_limits",
    )
    if not _close(
        control.get("rear_calf_target_rate_scale", 0.0),
        REAR_CALF_TARGET_RATE_SCALE,
    ):
        raise ValueError("rear calf target limit scale mismatch")
    if control.get("filter_policy_actions") is not True:
        raise ValueError("policy action filter must be enabled")
    if not _close(control.get("policy_action_filter_alpha", 0.0), 0.26):
        raise ValueError("policy action filter alpha mismatch")
    if control.get("use_real_actuator_model") is not True:
        raise ValueError("training actuator model marker missing")
    if control.get("output_transform") != "tanh":
        raise ValueError("output transform must be tanh")
    if control.get("enforce_policy_symmetry") is not True:
        raise ValueError("strict policy symmetry must be enabled")
    if not _close(
        float(control.get("sim_dt", 0.0))
        * int(control.get("decimation", 0)),
        0.02,
    ):
        raise ValueError("control period must be exactly 20 ms")
    for name, expected in REALDATA_COMMAND_FEEDBACK.items():
        if not _close(control.get(name, np.nan), expected):
            raise ValueError(f"{name} mismatch")

    commands = contract.get("commands", {})
    if commands.get("ranges") != {
        "lin_vel_x": [-0.12, 0.45],
        "lin_vel_y": [-0.08, 0.08],
        "ang_vel_yaw": [-0.80, 0.80],
    }:
        raise ValueError("command range mismatch")
    if commands.get("heading_command") is not False:
        raise ValueError("heading_command must be false")
    if commands.get("observe_heading_error") is not True:
        raise ValueError("heading observation must be enabled")

    gait = contract.get("gait", {})
    if gait.get("continuous_scaling") is not True:
        raise ValueError("continuous gait scaling must be enabled")
    _require_list_close(
        gait.get("equivalent_speed_weights", []),
        [1.0, 1.5, 0.18],
        "equivalent_speed_weights",
    )
    _require_list_close(
        gait.get("speed_knots", []),
        [0.0, 0.01, 0.02, 0.05, 0.12, 0.20, 0.30],
        "speed_knots",
    )
    _require_list_close(
        gait.get("calf_amplitude_knots", []),
        [0.0, 0.0, 0.02, 0.042, 0.09, 0.15, 0.215],
        "calf_amplitude_knots",
    )
    if not _close(gait.get("stance_ratio", 0.0), 0.60):
        raise ValueError("gait stance ratio mismatch")
    for name, expected in {
        "thigh_amplitude": 0.15,
        "target_phase_lead": 0.12,
        "lateral_hip_amplitude": -0.15,
        "lateral_command_scale": 0.08,
        "lateral_diagonal_scale": 0.35,
        "thigh_lateral_scale": 2.0,
    }.items():
        if not _close(gait.get(name, np.nan), expected):
            raise ValueError(f"gait {name} mismatch")
    if control.get("enforce_swing_calf_reference") is not True:
        raise ValueError("swing calf reference guard must be enabled")
    if not _close(control.get("front_swing_calf_reference_scale", 0.0), 3.5):
        raise ValueError("front swing calf reference scale mismatch")
    if not _close(control.get("rear_swing_calf_reference_scale", 0.0), 4.0):
        raise ValueError("rear swing calf reference scale mismatch")
    if gait.get("phase_offsets") != {
        "FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0,
    }:
        raise ValueError("gait phase offsets mismatch")
    return True
