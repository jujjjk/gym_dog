import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

from mydog_policy.hardware_balance_contract import (
    BACKWARD_REAR_CALF_TARGET_MIN,
    MODEL_SHA256,
    guard_backward_rear_calf_target,
    validate_metadata,
)


RESOURCE = Path(__file__).resolve().parents[1] / "resource"


def test_manifest_and_embedded_onnx_contract_match():
    sidecar = json.loads(
        (RESOURCE / "fanfan_hardware_balance_5530_best.json").read_text(
            encoding="utf-8"
        )
    )
    validate_metadata(sidecar)

    model = onnx.load(
        str(RESOURCE / "fanfan_hardware_balance_5530_best.onnx"),
        load_external_data=False,
    )
    metadata = {item.key: item.value for item in model.metadata_props}
    embedded = json.loads(metadata["fanfan_deployment_config"])
    assert embedded == sidecar
    validate_metadata(embedded)
    onnx.checker.check_model(model)


def test_exact_selected_model_sha256():
    path = RESOURCE / "fanfan_hardware_balance_5530_best.onnx"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256


def test_backward_rear_calf_guard_matches_gym_and_mujoco():
    target = np.array([
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.70,
        0.0, 0.5, -1.55,
    ], dtype=np.float32)

    guarded = guard_backward_rear_calf_target(target, [-0.10, 0.0, 0.0])
    np.testing.assert_allclose(
        guarded[[8, 11]],
        [BACKWARD_REAR_CALF_TARGET_MIN] * 2,
    )
    np.testing.assert_array_equal(guarded[:8], target[:8])

    # Gym uses a strict vx < -0.03 condition.
    unchanged = guard_backward_rear_calf_target(target, [-0.03, 0.0, 0.0])
    np.testing.assert_array_equal(unchanged, target)
    unchanged = guard_backward_rear_calf_target(target, [0.0, 0.08, 0.0])
    np.testing.assert_array_equal(unchanged, target)


def test_launch_enforces_real_machine_contract_and_fail_safes():
    launch = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "sim2real_hardware_balance.launch.py"
    ).read_text(encoding="utf-8")
    assert '"policy_hz": 50.0' in launch
    assert '"cmd_min_x": -0.12' in launch
    assert '"cmd_max_x": 0.45' in launch
    assert '"cmd_min_y": -0.08' in launch
    assert '"cmd_max_y": 0.08' in launch
    assert '"cmd_min_yaw": -0.80' in launch
    assert '"cmd_max_yaw": 0.80' in launch
    assert '"enable_tilt_protection": True' in launch
    assert 'default_value="0.45"' in launch
    assert '"enable_command_timeout_stand_hold": True' in launch
    assert '"use_model_pd_gains": True' in launch
    assert '"motor_torque_limit_nm": 13.0' in launch
    assert '"motion_torque_ff_limit_nm": 13.0' in launch
    assert '"use_hardware_torque_limits": True' in launch
    assert '"require_verified_hardware_limits": True' in launch
    assert '"enable_rear_torque_boost": False' in launch
    assert '"enable_target_smoothing": False' in launch
    assert '"enable_velocity_ff": False' in launch
    assert 'default_value=MODEL_SHA256' in launch
    assert (
        'DeclareLaunchArgument("enable_send", default_value="false")'
        in launch
    )
