"""Exact deployment contract for fanfan_clearance_robust_5730.onnx."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

MODEL_TASK = "FanfanOmniClearanceRobustCfg"
MODEL_FILENAME = "fanfan_clearance_robust_5730.onnx"
MODEL_SHA256 = "a7dd106fb5df1385cbc3c4a0be38916f8e75506a9b2026d87e7671704a9a9b39"
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
DEPLOYMENT_LIMITS = dict(TRAINING_LIMITS)

COMMAND_FEEDBACK = {
    "command_feedback_longitudinal_gain": 0.40,
    "command_feedback_lateral_gain": 0.80,
    "command_feedback_yaw_gain": 0.25,
    "command_feedback_heading_gain": 4.00,
    "command_feedback_heading_damping": 1.00,
    "command_feedback_diagonal_longitudinal_scale": 0.60,
}

GAIT_CONTRACT = {
    "period": 0.45,
    "stance_ratio": 0.62,
    "thigh_amplitude": 0.0,
    "calf_amplitude": -0.30,
    "motion_calf_amplitude": -0.42,
    "yaw_calf_amplitude": -0.35,
    "motion_lateral_threshold": 0.03,
    "motion_yaw_threshold": 0.08,
    "clearance_transition_duration": 0.90,
    "clearance_transition_boost": 1.10,
    "clearance_tilt_boost_gain": 1.40,
    "clearance_tilt_boost_max": 1.28,
    "gate_with_command": True,
    "command_gate_sigma": 0.0004,
    "phase_offsets": {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0},
}

# Commands used by gym_dog/mujoko/sim2sim.py --demo-matrix for the fast model.
PARITY_PRESETS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.35, 0.0, 0.0),
    "backward": (-0.10, 0.0, 0.0),
    "left_lateral": (0.0, 0.07, 0.0),
    "right_lateral": (0.0, -0.07, 0.0),
    "turn_left": (0.0, 0.0, 0.70),
    "turn_right": (0.0, 0.0, -0.70),
    "diagonal_left": (0.20, 0.07, 0.0),
    "diagonal_right": (0.20, -0.07, 0.0),
    "arc_left": (0.25, 0.0, 0.50),
    "arc_right": (0.25, 0.0, -0.50),
    "backward_arc_left": (-0.10, 0.0, 0.35),
    "backward_arc_right": (-0.10, 0.0, -0.35),
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
        (index, float(value), float(reference))
        for index, (value, reference) in enumerate(zip(actual_list, expected_list))
        if not _close(value, reference, atol=atol)
    ]
    if bad:
        raise ValueError(f"{name} mismatch at {bad[:4]}")


def clearance_calf_amplitude(
    command: Iterable[float],
    transition_age_s: float,
    projected_gravity: Iterable[float],
    gait: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the exact calf reference amplitude used by Gym and MuJoCo.

    Lateral motion overrides the yaw-only amplitude, matching fanfan_env.py and
    sim2sim.py. Tilt is computed from projected_gravity[:2], not Euler angles.
    """
    cfg = GAIT_CONTRACT if gait is None else gait
    cmd = np.asarray(command, dtype=np.float32).reshape(3)
    gravity = np.asarray(projected_gravity, dtype=np.float32).reshape(3)

    lateral_motion = abs(float(cmd[1])) > float(
        cfg.get("motion_lateral_threshold", 0.03)
    )
    yaw_motion = abs(float(cmd[2])) > float(
        cfg.get("motion_yaw_threshold", 0.08)
    )
    motion = lateral_motion or yaw_motion

    amplitude = float(cfg["calf_amplitude"])
    if yaw_motion:
        amplitude = float(
            cfg.get(
                "yaw_calf_amplitude",
                cfg.get("motion_calf_amplitude", amplitude),
            )
        )
    if lateral_motion:
        amplitude = float(cfg.get("motion_calf_amplitude", amplitude))

    transition = (
        float(transition_age_s)
        < float(cfg.get("clearance_transition_duration", 0.0))
        and float(np.linalg.norm(cmd)) > 0.05
    )
    if transition:
        amplitude *= float(cfg.get("clearance_transition_boost", 1.0))

    tilt_norm = float(np.linalg.norm(gravity[:2]))
    tilt_boost = 1.0
    if motion or transition:
        tilt_boost = 1.0 + float(
            cfg.get("clearance_tilt_boost_gain", 0.0)
        ) * tilt_norm
        tilt_boost = min(
            tilt_boost,
            float(cfg.get("clearance_tilt_boost_max", 1.0)),
        )
        amplitude *= tilt_boost

    return float(amplitude), {
        "lateral_motion": lateral_motion,
        "yaw_motion": yaw_motion,
        "motion": motion,
        "transition": transition,
        "transition_age_s": float(transition_age_s),
        "tilt_norm": tilt_norm,
        "tilt_boost": float(tilt_boost),
    }


def gait_command_gate(
    command: Iterable[float],
    gait: Mapping[str, Any] | None = None,
) -> float:
    """Return the command gate from Gym/MuJoCo for the reference gait."""
    cfg = GAIT_CONTRACT if gait is None else gait
    if not bool(cfg.get("gate_with_command", False)):
        return 1.0

    cmd = np.asarray(command, dtype=np.float32).reshape(3)
    energy = float(np.sum(np.square(cmd[:2])) + 0.04 * cmd[2] ** 2)
    sigma = float(cfg.get("command_gate_sigma", 0.0))
    if sigma <= 0.0:
        return 1.0 if energy > 0.0 else 0.0
    return float(1.0 - np.exp(-energy / sigma))


def clearance_gait_offset(
    phase: float,
    command: Iterable[float],
    transition_age_s: float,
    projected_gravity: Iterable[float],
    gait: Mapping[str, Any] | None = None,
    joint_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the exact 12-joint reference offset used by Gym and MuJoCo."""
    cfg = GAIT_CONTRACT if gait is None else gait
    names = JOINT_NAMES if joint_names is None else list(joint_names)
    if len(names) != ACTIONS:
        raise ValueError(f"expected {ACTIONS} joint names, got {len(names)}")

    result = np.zeros(ACTIONS, dtype=np.float32)
    stance_ratio = float(cfg["stance_ratio"])
    offsets = cfg["phase_offsets"]
    amplitude, diagnostics = clearance_calf_amplitude(
        command,
        transition_age_s,
        projected_gravity,
        cfg,
    )

    for index, name in enumerate(names):
        leg = str(name)[:2]
        leg_phase = (float(phase) + float(offsets[leg])) % 1.0
        swing = np.clip(
            (leg_phase - stance_ratio) / (1.0 - stance_ratio),
            0.0,
            1.0,
        )
        smooth = swing * swing * (3.0 - 2.0 * swing)
        if "thigh" in name:
            if leg_phase < stance_ratio:
                profile = -1.0 + 2.0 * np.clip(
                    leg_phase / stance_ratio,
                    0.0,
                    1.0,
                )
            else:
                profile = 1.0 - 2.0 * smooth
            result[index] = float(cfg["thigh_amplitude"]) * profile
        elif "calf" in name:
            result[index] = (
                amplitude
                * np.sin(np.pi * smooth)
                * float(leg_phase >= stance_ratio)
            )

    gate = gait_command_gate(command, cfg)
    diagnostics = dict(diagnostics)
    diagnostics["gait_gate"] = gate
    diagnostics["calf_amplitude"] = amplitude
    return (result * gate).astype(np.float32), diagnostics


def validate_metadata(contract: dict[str, Any]) -> bool:
    """Validate every runtime-relevant field exported with checkpoint 5730."""
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
        if not _close(control.get(name, float("nan")), expected):
            raise ValueError(
                f"{name} mismatch: expected {expected}, got {control.get(name)!r}"
            )
    if not _close(
        float(control.get("sim_dt", 0.0)) * int(control.get("decimation", 0)),
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
    scalar_fields = (
        "period",
        "stance_ratio",
        "thigh_amplitude",
        "calf_amplitude",
        "motion_calf_amplitude",
        "yaw_calf_amplitude",
        "motion_lateral_threshold",
        "motion_yaw_threshold",
        "clearance_transition_duration",
        "clearance_transition_boost",
        "clearance_tilt_boost_gain",
        "clearance_tilt_boost_max",
        "command_gate_sigma",
    )
    for name in scalar_fields:
        if not _close(gait.get(name, float("nan")), GAIT_CONTRACT[name]):
            raise ValueError(
                f"gait {name} mismatch: expected {GAIT_CONTRACT[name]}, "
                f"got {gait.get(name)!r}"
            )
    if gait.get("gate_with_command") is not True:
        raise ValueError("gait command gate must be enabled")
    if gait.get("phase_offsets") != GAIT_CONTRACT["phase_offsets"]:
        raise ValueError("gait phase offsets mismatch")
    return True
