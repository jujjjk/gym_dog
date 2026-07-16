"""Immutable deployment contract for the Fanfan Omni yaw-clean model."""

MODEL_TASK = "FanfanOmniYawDriftCleanCfg"
MODEL_SHA256 = "c9c9621c97620100b9f61bb9c508bedad80d7edde3db03d8279218a1e6946cc8"
OBSERVATIONS = 52
ACTIONS = 12

TRAINING_LIMITS = {
    "vx": (-0.12, 0.46),
    "vy": (-0.12, 0.12),
    "yaw": (-0.85, 0.85),
}

# Initial real-robot limits requested for model_5100.
DEPLOYMENT_LIMITS = {
    "vx": (-0.12, 0.12),
    "vy": (-0.025, 0.025),
    "yaw": (-0.25, 0.25),
}

PRESETS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_slow": (0.06, 0.0, 0.0),
    "forward": (0.10, 0.0, 0.0),
    "forward_fast": (0.12, 0.0, 0.0),
    "backward_slow": (-0.04, 0.0, 0.0),
    "backward": (-0.08, 0.0, 0.0),
    "backward_fast": (-0.12, 0.0, 0.0),
    "left_slow": (0.0, 0.015, 0.0),
    "left": (0.0, 0.025, 0.0),
    "right_slow": (0.0, -0.015, 0.0),
    "right": (0.0, -0.025, 0.0),
    "turn_left_slow": (0.0, 0.0, 0.12),
    "turn_left": (0.0, 0.0, 0.25),
    "turn_right_slow": (0.0, 0.0, -0.12),
    "turn_right": (0.0, 0.0, -0.25),
    "forward_turn_left_slow": (0.06, 0.0, 0.12),
    "forward_turn_left": (0.10, 0.0, 0.20),
    "forward_turn_right_slow": (0.06, 0.0, -0.12),
    "forward_turn_right": (0.10, 0.0, -0.20),
    "backward_turn_left_slow": (-0.04, 0.0, 0.10),
    "backward_turn_left": (-0.08, 0.0, 0.18),
    "backward_turn_right_slow": (-0.04, 0.0, -0.10),
    "backward_turn_right": (-0.08, 0.0, -0.18),
    "diagonal_left_slow": (0.06, 0.015, 0.0),
    "diagonal_left": (0.10, 0.025, 0.0),
    "diagonal_right_slow": (0.06, -0.015, 0.0),
    "diagonal_right": (0.10, -0.025, 0.0),
}


def validate_metadata(contract):
    """Raise ValueError when ONNX metadata differs from the deployed policy."""
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported deployment schema")
    if contract.get("task") != MODEL_TASK:
        raise ValueError(f"task mismatch: {contract.get('task')!r}")
    if contract.get("dimensions") != {"observations": OBSERVATIONS, "actions": ACTIONS}:
        raise ValueError(f"dimension mismatch: {contract.get('dimensions')!r}")

    control = contract.get("control", {})
    expected_kp = [60.0, 70.0, 70.0] * 4
    expected_kd = [1.2, 1.6, 1.6] * 4
    expected_scale = [0.092, 0.215, 0.215, 0.092, 0.215, 0.215,
                      0.092, 0.235, 0.235, 0.092, 0.235, 0.235]
    if control.get("stiffness") != expected_kp:
        raise ValueError("Kp contract mismatch")
    if control.get("damping") != expected_kd:
        raise ValueError("Kd contract mismatch")
    if control.get("action_scale") != expected_scale:
        raise ValueError("action scale contract mismatch")
    if control.get("output_transform") != "tanh":
        raise ValueError("output transform must be tanh")
    if float(control.get("sim_dt", 0.0)) * int(control.get("decimation", 0)) != 0.02:
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
    if gait.get("period") != 0.54 or gait.get("calf_amplitude") != -0.3:
        raise ValueError("gait contract mismatch")
    return True
