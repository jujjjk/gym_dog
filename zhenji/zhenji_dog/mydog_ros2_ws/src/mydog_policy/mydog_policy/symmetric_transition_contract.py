"""Immutable deployment contract for fanfan_symmetric_transition_5530.onnx."""

from __future__ import annotations

import math
from typing import Any, Iterable

MODEL_TASK = "FanfanOmniSymmetricTransitionCfg"
MODEL_FILENAME = "fanfan_symmetric_transition_5530.onnx"
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
    0.115, 0.215, 0.215,
    0.115, 0.215, 0.215,
    0.115, 0.235, 0.235,
    0.115, 0.235, 0.235,
]
TORQUE_LIMITS_POLICY = [
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
    10.0, 10.0, 13.0,
]
POLICY_ACTION_RATE_LIMITS = [
    9.130434782608695, 7.906976744186046, 12.325581395348838,
] * 4
POLICY_ACTION_ACCEL_LIMITS = [
    347.82608695652175, 269.7674418604651, 437.2093023255814,
] * 4

TRAINING_LIMITS = {
    "vx": (-0.25, 0.60),
    "vy": (-0.26, 0.26),
    "yaw": (-1.30, 1.30),
}

# The deployment node accepts the same command range as the exported model.
# A low-rack test changes the published command values, not this contract.
DEPLOYMENT_LIMITS = {
    "vx": (-0.25, 0.60),
    "vy": (-0.26, 0.26),
    "yaw": (-1.30, 1.30),
}

COMMAND_FEEDBACK = {
    "command_feedback_longitudinal_gain": 0.40,
    "command_feedback_lateral_gain": 0.80,
    "command_feedback_yaw_gain": 0.25,
    "command_feedback_heading_gain": 4.00,
    "command_feedback_heading_damping": 1.00,
    "command_feedback_diagonal_longitudinal_scale": 0.60,
}

PRESETS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.45, 0.0, 0.0),
    "backward": (-0.18, 0.0, 0.0),
    "left_lateral": (0.0, 0.15, 0.0),
    "right_lateral": (0.0, -0.15, 0.0),
    "turn_left": (0.0, 0.0, 0.90),
    "turn_right": (0.0, 0.0, -0.90),
    "diagonal_left": (0.35, 0.15, 0.0),
    "diagonal_right": (0.35, -0.15, 0.0),
    "arc_left": (0.35, 0.0, 0.90),
    "arc_right": (0.35, 0.0, -0.90),
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
    """Validate every runtime-relevant field exported with checkpoint 5530."""
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
    if not _close(control.get("policy_action_filter_alpha", 0.0), 0.30):
        raise ValueError("policy_action_filter_alpha mismatch")
    if control.get("output_transform") != "tanh":
        raise ValueError("output transform must be tanh")
    if control.get("enforce_policy_symmetry") is not True:
        raise ValueError("strict policy symmetry is not enabled in ONNX metadata")
    for name, expected in COMMAND_FEEDBACK.items():
        if not _close(control.get(name, float("nan")), expected, atol=1.0e-7):
            raise ValueError(
                f"{name} mismatch: expected {expected}, got {control.get(name)!r}"
            )
    if not _close(
        float(control.get("sim_dt", 0.0))
        * int(control.get("decimation", 0)),
        0.02,
    ):
        raise ValueError("control period must be 0.02 s")

    observations = contract.get("observations", {})
    if not _close(observations.get("lin_vel_scale", 0.0), 2.0):
        raise ValueError("lin_vel_scale mismatch")
    if not _close(observations.get("ang_vel_scale", 0.0), 0.25):
        raise ValueError("ang_vel_scale mismatch")
    if not _close(observations.get("dof_pos_scale", 0.0), 1.0):
        raise ValueError("dof_pos_scale mismatch")
    if not _close(observations.get("dof_vel_scale", 0.0), 0.05):
        raise ValueError("dof_vel_scale mismatch")
    _require_list_close(
        observations.get("command_scale", []),
        [2.0, 2.0, 0.25],
        "command_scale",
    )

    commands = contract.get("commands", {})
    if commands.get("heading_command") is not False:
        raise ValueError("heading_command must be false")
    if commands.get("observe_heading_error") is not True:
        raise ValueError("heading error observation must be enabled")
    expected_ranges = {
        "lin_vel_x": [-0.25, 0.60],
        "lin_vel_y": [-0.26, 0.26],
        "ang_vel_yaw": [-1.30, 1.30],
    }
    if commands.get("ranges", {}) != expected_ranges:
        raise ValueError(f"command range mismatch: {commands.get('ranges')!r}")

    gait = contract.get("gait", {})
    if not _close(gait.get("period", 0.0), 0.45):
        raise ValueError("gait period mismatch")
    if not _close(gait.get("stance_ratio", 0.0), 0.62):
        raise ValueError("gait stance ratio mismatch")
    if not _close(gait.get("thigh_amplitude", 0.0), 0.0):
        raise ValueError("gait thigh amplitude mismatch")
    if not _close(gait.get("calf_amplitude", 0.0), -0.30):
        raise ValueError("gait calf amplitude mismatch")
    if gait.get("gate_with_command") is not True:
        raise ValueError("gait command gate must be enabled")
    if not _close(gait.get("command_gate_sigma", 0.0), 0.0004):
        raise ValueError("gait command gate sigma mismatch")
    if gait.get("phase_offsets") != {
        "FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0,
    }:
        raise ValueError("gait phase offsets mismatch")
    return True
