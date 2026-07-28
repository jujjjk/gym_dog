"""Immutable deployment contract for fanfan_force_coord_5280.onnx."""

from __future__ import annotations

import math
from typing import Any, Iterable

MODEL_TASK = "FanfanOmniForceCoordCfg"
OBSERVATIONS = 52
ACTIONS = 12

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT_JOINT_ANGLES = [0.0, 0.563, -0.95] * 4
STIFFNESS = [60.0, 70.0, 70.0] * 4
DAMPING = [1.2, 1.6, 1.6] * 4
ACTION_SCALE = [
    0.092, 0.215, 0.215,
    0.092, 0.215, 0.215,
    0.092, 0.235, 0.235,
    0.092, 0.235, 0.235,
]
TORQUE_LIMITS_POLICY = [
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
]
POLICY_ACTION_RATE_LIMITS = [
    8.333333333333334, 7.073170731707317, 11.463414634146343,
] * 4
POLICY_ACTION_ACCEL_LIMITS = [
    288.8888888888889, 234.14634146341464, 400.0,
] * 4

TRAINING_LIMITS = {
    "vx": (-0.12, 0.46),
    "vy": (-0.12, 0.12),
    "yaw": (-0.85, 0.85),
}

# Conservative first-ground-test envelope. These are launch defaults, not
# changes to the ONNX observation contract.
DEPLOYMENT_LIMITS = {
    "vx": (-0.08, 0.12),
    "vy": (-0.025, 0.025),
    "yaw": (-0.20, 0.20),
}

PRESETS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_006": (0.06, 0.0, 0.0),
    "forward_008": (0.08, 0.0, 0.0),
    "forward_010": (0.10, 0.0, 0.0),
    "forward_012": (0.12, 0.0, 0.0),
    "backward_004": (-0.04, 0.0, 0.0),
    "left_0015": (0.0, 0.015, 0.0),
    "right_0015": (0.0, -0.015, 0.0),
    "turn_left_012": (0.0, 0.0, 0.12),
    "turn_right_012": (0.0, 0.0, -0.12),
}


def _close(a: float, b: float, *, atol: float = 1.0e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def _require_list_close(
    actual: Iterable[Any],
    expected: Iterable[Any],
    name: str,
    *,
    atol: float = 1.0e-7,
) -> None:
    actual_list = list(actual)
    expected_list = list(expected)
    if len(actual_list) != len(expected_list):
        raise ValueError(
            f"{name} length mismatch: expected {len(expected_list)}, "
            f"got {len(actual_list)}"
        )
    bad = [
        (i, float(a), float(e))
        for i, (a, e) in enumerate(zip(actual_list, expected_list))
        if not _close(a, e, atol=atol)
    ]
    if bad:
        raise ValueError(f"{name} mismatch at {bad[:4]}")


def validate_metadata(contract: dict[str, Any]) -> bool:
    """Validate the exact runtime-relevant contract exported with model 5280."""
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported deployment schema")
    if contract.get("task") != MODEL_TASK:
        raise ValueError(
            f"task mismatch: expected {MODEL_TASK!r}, got {contract.get('task')!r}"
        )
    if contract.get("dimensions") != {
        "observations": OBSERVATIONS,
        "actions": ACTIONS,
    }:
        raise ValueError(f"dimension mismatch: {contract.get('dimensions')!r}")

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
    _require_list_close(control.get("action_scale", []), ACTION_SCALE, "action_scale")
    _require_list_close(
        control.get("torque_limits", []),
        TORQUE_LIMITS_POLICY,
        "torque_limits",
    )
    _require_list_close(
        control.get("policy_action_rate_limits", []),
        POLICY_ACTION_RATE_LIMITS,
        "policy_action_rate_limits",
        atol=1.0e-6,
    )
    _require_list_close(
        control.get("policy_action_accel_limits", []),
        POLICY_ACTION_ACCEL_LIMITS,
        "policy_action_accel_limits",
        atol=1.0e-5,
    )
    if control.get("filter_policy_actions") is not True:
        raise ValueError("policy action filter must be enabled")
    if not _close(control.get("policy_action_filter_alpha", 0.0), 0.26):
        raise ValueError("policy_action_filter_alpha mismatch")
    if control.get("output_transform") != "tanh":
        raise ValueError("output transform must be tanh")
    if not _close(
        float(control.get("sim_dt", 0.0))
        * int(control.get("decimation", 0)),
        0.02,
    ):
        raise ValueError("control period must be 0.02 s")

    ranges = contract.get("commands", {}).get("ranges", {})
    expected_ranges = {
        "lin_vel_x": [-0.12, 0.46],
        "lin_vel_y": [-0.12, 0.12],
        "ang_vel_yaw": [-0.85, 0.85],
    }
    if ranges != expected_ranges:
        raise ValueError(f"command range mismatch: {ranges!r}")

    gait = contract.get("gait", {})
    expected_phase_offsets = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
    if not _close(gait.get("period", 0.0), 0.54):
        raise ValueError("gait period mismatch")
    if not _close(gait.get("stance_ratio", 0.0), 0.62):
        raise ValueError("gait stance ratio mismatch")
    if not _close(gait.get("thigh_amplitude", 0.0), 0.0):
        raise ValueError("gait thigh amplitude mismatch")
    if not _close(gait.get("calf_amplitude", 0.0), -0.3):
        raise ValueError("gait calf amplitude mismatch")
    if gait.get("gate_with_command") is not True:
        raise ValueError("gait command gate must be enabled")
    if not _close(gait.get("command_gate_sigma", 0.0), 0.0004):
        raise ValueError("gait command gate sigma mismatch")
    if gait.get("phase_offsets") != expected_phase_offsets:
        raise ValueError("gait phase offsets mismatch")
    return True
