from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from mydog_policy.rs01_model1950_core import (
    EXPECTED_ONNX_SHA256,
    Model1950Contract,
    Rs01Model1950PolicyCore,
)
from mydog_policy.rs01_model930_core import (
    POLICY_JOINT_NAMES,
    REAL_MOTOR_IDS,
    REVERSED_MOTOR_IDS,
    Rs01Model930Mapper,
    sha256_file,
)


PACKAGE = Path(__file__).resolve().parents[1]
ONNX = PACKAGE / "resource" / "model_1950_rs01_estimator_parity.onnx"
NODE = PACKAGE / "mydog_policy" / "rs01_model1950_node.py"
LAUNCH = PACKAGE / "launch" / "rs01_model1950.launch.py"


def load_contract():
    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )
    return session, Model1950Contract.from_onnx_session(
        session, ONNX, EXPECTED_ONNX_SHA256
    )


def test_model1950_hash_dimensions_estimator_sources_and_mapping():
    session, contract = load_contract()
    assert sha256_file(ONNX) == EXPECTED_ONNX_SHA256
    assert contract.raw["task"] == "rs01_go2_estimator_parity"
    assert session.get_inputs()[0].shape == ["batch", 54]
    assert session.get_outputs()[0].shape == ["batch", 12]
    assert tuple(contract.joint_names) == POLICY_JOINT_NAMES
    observations = contract.raw["observations"]
    assert (
        observations["base_linear_velocity_source"]
        == "rs01_leg_odometry"
    )
    assert (
        observations["straight_path_state_source"]
        == "rs01_leg_odometry_integral"
    )
    mapping = contract.raw["motor_mapping"]
    assert [item["real_feedback_index"] for item in mapping] == [
        3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11
    ]
    reversed_ids = {
        int(item["motor_id"], 16)
        for item in mapping
        if float(item["real_to_policy_sign"]) == -1.0
    }
    assert reversed_ids == set(REVERSED_MOTOR_IDS)


def test_model1950_rejects_wrong_hash():
    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        Model1950Contract.from_onnx_session(
            session, ONNX, expected_sha256="0" * 64
        )


def test_model1950_dry_inference_is_finite_and_limited_to_14_nm():
    session, contract = load_contract()
    core = Rs01Model1950PolicyCore(session, contract)
    core.reset(now=0.0, yaw=0.0, q_policy=contract.default)
    observation = core.build_observation(
        now=contract.policy_dt,
        base_linear_velocity=[0.23, 0.0, 0.0],
        base_angular_velocity=[0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.23, 0.0, 0.0],
        q_policy=contract.default,
        dq_policy=np.zeros(12, dtype=np.float32),
        yaw=0.0,
    )
    result = core.step(
        observation,
        contract.default,
        np.zeros(12, dtype=np.float32),
        active_limit_nm=14.0,
    )
    assert observation.shape == (54,)
    assert result["action"].shape == (12,)
    assert np.all(np.isfinite(observation))
    assert np.all(np.isfinite(result["action"]))
    assert np.max(
        np.abs(result["torque_info"]["safe_pd_torque_nm"])
    ) <= 14.0 + 1.0e-6


def test_model1950_mapping_round_trip_preserves_motor_semantics():
    _, contract = load_contract()
    mapper = Rs01Model930Mapper(contract)
    rng = np.random.default_rng(1950)
    q_policy = rng.uniform(
        contract.lower, contract.upper
    ).astype(np.float32)
    q_real = mapper.policy_target_to_real(q_policy)
    recovered, _ = mapper.real_to_policy_abs(
        q_real, np.zeros(12, dtype=np.float32)
    )
    np.testing.assert_allclose(recovered, q_policy, atol=1.0e-6)
    assert tuple(mapper.real_motor_ids.tolist()) == REAL_MOTOR_IDS


def test_model1950_launch_defaults_to_dry_stand_only():
    node = NODE.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "class Rs01Model1950Node(Rs01Model930Node)" in node
    assert "observation_count = 54" in node
    assert "include_path_state = True" in node
    assert "calibrate_gyro_bias = True" in node
    assert "strict_diagonal_odometry = True" in node
    assert "heading_consistency_enabled = True" in node
    assert "soft_inhibit_enabled = True" in node
    assert (
        'DeclareLaunchArgument("enable_send", default_value="false")'
        in launch
    )
    assert (
        'DeclareLaunchArgument("stand_only", default_value="true")'
        in launch
    )
    assert '"hardware_torque_limit_nm": 14.0' in launch
    assert '"startup_ready_error_rad": 0.12' in launch
    assert '"startup_ready_hold_sec": 2.0' in launch
    assert '"walk_start_stable_sec": 1.0' in launch
    assert (
        '"heading_consistency_max_mean_error_rad_s": 0.06'
        in launch
    )
    assert EXPECTED_ONNX_SHA256 in launch
