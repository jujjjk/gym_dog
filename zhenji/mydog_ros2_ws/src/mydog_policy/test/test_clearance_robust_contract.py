import json
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import struct
import threading

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from mydog_policy.clearance_robust_contract import (
    MODEL_SHA256,
    clearance_calf_amplitude,
    clearance_gait_offset,
    validate_metadata,
)
from mydog_policy.state_estimator_contract import (
    SHARED_FRAME_SIZE,
    append_shared_motor_snapshot,
    unpack_shared_motor_snapshot,
)


def test_manifest_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "resource"
        / "fanfan_clearance_robust_5730.json"
    )
    validate_metadata(json.loads(path.read_text(encoding="utf-8")))


def test_onnx_metadata_matches_sidecar_manifest():
    resource = Path(__file__).resolve().parents[1] / "resource"
    model = onnx.load(
        str(resource / "fanfan_clearance_robust_5730.onnx"),
        load_external_data=False,
    )
    metadata = {item.key: item.value for item in model.metadata_props}
    embedded = json.loads(metadata["fanfan_deployment_config"])
    sidecar = json.loads(
        (resource / "fanfan_clearance_robust_5730.json").read_text(
            encoding="utf-8"
        )
    )
    assert embedded == sidecar
    validate_metadata(embedded)
    onnx.checker.check_model(model)


def test_dynamic_clearance_amplitudes():
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    amp, info = clearance_calf_amplitude([0.35, 0.0, 0.0], 10.0, gravity)
    assert np.isclose(amp, -0.30)
    assert not info["motion"]

    amp, info = clearance_calf_amplitude([0.0, 0.07, 0.0], 10.0, gravity)
    assert np.isclose(amp, -0.42)
    assert info["lateral_motion"]

    amp, info = clearance_calf_amplitude([0.0, 0.0, 0.70], 10.0, gravity)
    assert np.isclose(amp, -0.35)
    assert info["yaw_motion"]

    amp, info = clearance_calf_amplitude([0.35, 0.0, 0.0], 0.0, gravity)
    assert np.isclose(amp, -0.33)
    assert info["transition"]

    tilted = np.array([0.10, 0.0, -0.995], dtype=np.float32)
    amp, info = clearance_calf_amplitude([0.0, 0.07, 0.0], 10.0, tilted)
    assert np.isclose(amp, -0.42 * 1.14, atol=1.0e-6)
    assert np.isclose(info["tilt_boost"], 1.14, atol=1.0e-6)


def test_exact_5730_model_sha256():
    path = (
        Path(__file__).resolve().parents[1]
        / "resource"
        / "fanfan_clearance_robust_5730.onnx"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256


def test_reference_gait_matches_mujoco_mid_swing():
    stance_ratio = 0.62
    phase = stance_ratio + 0.5 * (1.0 - stance_ratio)
    command = np.array([0.0, 0.07, 0.0], dtype=np.float32)
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    offset, info = clearance_gait_offset(
        phase,
        command,
        transition_age_s=10.0,
        projected_gravity=gravity,
    )
    gate = 1.0 - np.exp(-(0.07 ** 2) / 0.0004)
    expected = -0.42 * gate

    # At this base phase FL/RR are at swing apex; FR/RL are in stance.
    np.testing.assert_allclose(offset[[2, 11]], expected, atol=1.0e-7)
    np.testing.assert_allclose(offset[[5, 8]], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(offset[[0, 1, 3, 4, 6, 7, 9, 10]], 0.0)
    assert np.isclose(info["gait_gate"], gate)
    assert np.isclose(info["calf_amplitude"], -0.42)


def test_zero_command_gates_reference_gait_only():
    offset, info = clearance_gait_offset(
        0.81,
        [0.0, 0.0, 0.0],
        transition_age_s=10.0,
        projected_gravity=[0.0, 0.0, -1.0],
    )
    np.testing.assert_array_equal(offset, np.zeros(12, dtype=np.float32))
    assert info["gait_gate"] == 0.0


def test_onnx_left_right_mirror_error_is_zero():
    resource = Path(__file__).resolve().parents[1] / "resource"
    model = onnx.load(
        str(resource / "fanfan_clearance_robust_5730.onnx"),
        load_external_data=False,
    )
    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(5730)
    observations = rng.normal(0.0, 0.2, (4, 52)).astype(np.float32)
    observations[:, 8] = -1.0
    observations[:, 49] = 1.0

    mirrored = observations.copy()
    mirrored[:, [1, 7, 10, 3, 5, 11]] *= -1.0
    leg_mirror = np.array([1, 0, 3, 2])
    joint_sign = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    for start in (12, 24, 36):
        block = observations[:, start:start + 12].reshape(-1, 4, 3)
        mirrored[:, start:start + 12] = (
            block[:, leg_mirror, :] * joint_sign
        ).reshape(-1, 12)
    mirrored[:, 48:50] *= -1.0
    mirrored[:, 50] *= -1.0

    actions = evaluator.run(None, {"observations": observations})[0]
    mirrored_actions = evaluator.run(None, {"observations": mirrored})[0]
    mirrored_back = (
        mirrored_actions.reshape(-1, 4, 3)[:, leg_mirror, :] * joint_sign
    ).reshape(-1, 12)
    np.testing.assert_array_equal(actions, mirrored_back)


def test_launch_is_exact_13_nm_simulation_profile():
    launch = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "sim2real_clearance_robust_5730.launch.py"
    ).read_text(encoding="utf-8")
    assert '"policy_hz": 50.0' in launch
    assert '"zero_cmd_inhibits_policy": False' in launch
    assert '"enable_zero_cmd_stand_protection": False' in launch
    assert '"motor_torque_limit_nm": 13.0' in launch
    assert '"torque_safety_budget_nm": 13.0' in launch
    assert '"motion_torque_ff_limit_nm": 13.0' in launch
    assert '"use_hardware_torque_limits": True' in launch
    assert '"require_verified_hardware_limits": True' in launch
    assert '"hip_current_limit_amp": 12.0' in launch
    assert '"thigh_current_limit_amp": 12.0' in launch
    assert '"calf_current_limit_amp": 16.0' in launch
    assert '"max_imu_sample_age_sec": 0.10' in launch
    assert '"critical_state_failure_stop_cycles": 5' in launch
    assert '"max_estimator_tick_lag_ms": 35.0' in launch
    assert '"enable_rear_torque_boost": False' in launch
    assert '"deployment_gait_phase_period_scale": 1.0' in launch
    assert '"gait_phase_lead_sec": 0.0' in launch
    assert '"enable_target_smoothing": False' in launch
    assert '"enable_velocity_ff": False' in launch
    assert 'default_value=MODEL_SHA256' in launch
    assert "FindPackageShare(\"mydog_policy\")" in launch


def test_rs01_server_exposes_internal_torque_limit_handshake():
    workspace = Path(__file__).resolve().parents[4]
    motor_source = (workspace / "text" / "lingzu_motor.py").read_text(
        encoding="utf-8"
    )
    server_source = (workspace / "text" / "app.py").read_text(
        encoding="utf-8"
    )
    assert "IDX_TORQUE_LIMIT  = 0x700B" in motor_source
    assert "def set_torque_limit" in motor_source
    assert "/api/rs04/configure_motion_torque_limits" in server_source
    assert "all 12 motors exactly once" in server_source
    assert "require_hardware_torque_limits" in server_source


def test_verified_rs01_current_and_torque_readback_contract():
    workspace = Path(__file__).resolve().parents[4]
    motor_source = (workspace / "text" / "lingzu_motor.py").read_text(
        encoding="utf-8"
    )
    server_source = (workspace / "text" / "app.py").read_text(
        encoding="utf-8"
    )
    firmware = (workspace / "DOG_0.1A" / "Core" / "Src" / "main.c").read_text(
        encoding="utf-8"
    )
    assert "SPI_OP_READ_MOTOR_PARAM = 0x13" in motor_source
    assert "PARAM_RESPONSE_MAGIC = 0x5B" in motor_source
    assert "def read_param_f32" in motor_source
    assert "def write_and_verify_param_f32" in motor_source
    assert "/api/rs04/configure_verified_motion_safety_limits" in server_source
    assert '0x7018' in server_source
    assert "verified RS01 safety configuration failed" in server_source
    assert "SPI_OP_READ_MOTOR_PARAM 0x13" in firmware
    assert "PARAM_RESPONSE_MAGIC    0x5B" in firmware
    assert "cmd_type == 0x11" in firmware


def test_startup_reenables_after_verified_limit_handshake():
    workspace = Path(__file__).resolve().parents[4]
    server_source = (workspace / "text" / "app.py").read_text(
        encoding="utf-8"
    )
    policy_source = (
        workspace
        / "mydog_ros2_ws"
        / "src"
        / "mydog_policy"
        / "mydog_policy"
        / "mydog_policy_node.py"
    ).read_text(encoding="utf-8")
    assert "enable_first requires all 12 motors exactly once" in server_source
    assert "target_primed = True" in server_source
    assert "enable_applied = True" in server_source
    assert "motion startup handshake failed; all motors stopped" in server_source
    assert 'body.get("enable_applied", False)' in policy_source
    assert 'body.get("target_primed", False)' in policy_source
    assert "one-shot live-target prime and all-motor" in policy_source
    assert "max(float(self.http_timeout), 0.35)" in policy_source


def test_stm32_parameter_readback_frame_layout():
    workspace = Path(__file__).resolve().parents[4]
    path = workspace / "text" / "lingzu_motor.py"
    old_spidev = sys.modules.get("spidev")
    sys.modules["spidev"] = types.SimpleNamespace(SpiDev=object)
    try:
        spec = importlib.util.spec_from_file_location(
            "lingzu_motor_protocol_test",
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old_spidev is None:
            sys.modules.pop("spidev", None)
        else:
            sys.modules["spidev"] = old_spidev

    request = module.build_spi_param_read_frame(0x23, 0x7018)
    assert len(request) == 96
    assert request[:6] == bytes([0xFA, 0x13, 0x23, 0x00, 0x18, 0x70])

    response = bytearray(96)
    response[0:6] = bytes([0x5B, 0xA1, 0x23, 0x00, 0x18, 0x70])
    struct.pack_into("<f", response, 8, 16.0)
    struct.pack_into("<I", response, 12, 123456)
    parsed = module.LingZuMotorController._parse_param_response_frame(response)
    assert parsed["motor_id"] == 0x23
    assert parsed["index"] == 0x7018
    assert parsed["value"] == 16.0
    assert parsed["response_tick_ms"] == 123456

    class FakeTransport:
        def __init__(self):
            self.lock = threading.Lock()
            self.calls = 0

        def _xfer_to_one(self, bus, device, payload):
            self.calls += 1
            if self.calls < 3:
                return bytes(96)
            return bytes(response)

    controller = module.LingZuMotorController(
        motor_id=0x23,
        transport=FakeTransport(),
    )
    controller.state.board_capabilities = (
        module.SNAPSHOT_CAP_PARAM_READBACK
    )
    assert controller.read_param_f32(0x7018, timeout=0.1) == 16.0


def test_estimator_shared_motor_snapshot_roundtrip():
    base = np.arange(17, dtype=np.float32)
    motor = SimpleNamespace(
        q_real=np.linspace(-1.0, 1.0, 12, dtype=np.float32),
        dq_real=np.linspace(-2.0, 2.0, 12, dtype=np.float32),
        torque=np.linspace(-3.0, 3.0, 12, dtype=np.float32),
        temp=np.linspace(30.0, 41.0, 12, dtype=np.float32),
        online=np.ones(12, dtype=bool),
        error_code=np.arange(12, dtype=np.int32),
        age_ms=np.linspace(1.0, 12.0, 12, dtype=np.float32),
    )
    frame = append_shared_motor_snapshot(base, motor)
    assert frame.shape == (SHARED_FRAME_SIZE,)
    unpacked = unpack_shared_motor_snapshot(frame)
    np.testing.assert_array_equal(unpacked["q_real"], motor.q_real)
    np.testing.assert_array_equal(unpacked["dq_real"], motor.dq_real)
    np.testing.assert_array_equal(unpacked["online"], motor.online)
    np.testing.assert_array_equal(unpacked["error_code"], motor.error_code)


def test_imu_and_policy_fail_safe_source_contracts():
    package = Path(__file__).resolve().parents[1] / "mydog_policy"
    imu_source = (package / "imu_serial_interface.py").read_text(
        encoding="utf-8"
    )
    fixed_source = (package / "sim2real_parity_fixed_node.py").read_text(
        encoding="utf-8"
    )
    obs_source = (package / "obs_builder.py").read_text(encoding="utf-8")
    assert "threading.excepthook" in imu_source
    assert "_backend_is_alive" in imu_source
    assert "LOCK_EX | fcntl.LOCK_NB" in imu_source
    assert "[PARITY_FIXED][FAIL_SAFE_STOP]" in fixed_source
    assert 'f"{self.motor_base_url}/api/stop"' in fixed_source
    assert "unpack_shared_motor_snapshot" in obs_source
    assert "and not self.use_external_motor_snapshot" in obs_source
