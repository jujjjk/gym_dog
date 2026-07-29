from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from mydog_policy.rs01_model1850_core import (
    EXPECTED_ONNX_SHA256,
    Model1850Contract,
    Rs01Model1850PolicyCore,
    Rs01StraightPathEstimator,
)
from mydog_policy.rs01_model930_core import (
    POLICY_JOINT_NAMES,
    REAL_MOTOR_IDS,
    REVERSED_MOTOR_IDS,
    Rs01Model930Mapper,
    Rs01Model930PolicyCore,
    estimate_stationary_gyro_bias,
    sha256_file,
)


PACKAGE = Path(__file__).resolve().parents[1]
ONNX = PACKAGE / "resource" / "model_1850_rs01_path54.onnx"
NODE = PACKAGE / "mydog_policy" / "rs01_model1850_node.py"
SHARED_NODE = PACKAGE / "mydog_policy" / "rs01_model930_node.py"
LAUNCH = PACKAGE / "launch" / "rs01_model1850.launch.py"


def load_contract():
    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )
    return session, Model1850Contract.from_onnx_session(
        session, ONNX, EXPECTED_ONNX_SHA256
    )


def test_model1850_hash_dimensions_task_and_mapping():
    session, contract = load_contract()
    assert sha256_file(ONNX) == EXPECTED_ONNX_SHA256
    assert contract.raw["task"] == "rs01_go2_path54_sim2sim_transfer"
    assert session.get_inputs()[0].shape == ["batch", 54]
    assert session.get_outputs()[0].shape == ["batch", 12]
    assert tuple(contract.joint_names) == POLICY_JOINT_NAMES
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


def test_model1850_rejects_wrong_hash():
    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        Model1850Contract.from_onnx_session(
            session, ONNX, expected_sha256="0" * 64
        )


def test_model1850_mapping_round_trip_preserves_verified_motor_semantics():
    _, contract = load_contract()
    mapper = Rs01Model930Mapper(contract)
    rng = np.random.default_rng(1850)
    q_policy = rng.uniform(
        contract.lower, contract.upper
    ).astype(np.float32)
    q_real = mapper.policy_target_to_real(q_policy)
    recovered, _ = mapper.real_to_policy_abs(
        q_real, np.zeros(12, dtype=np.float32)
    )
    np.testing.assert_allclose(recovered, q_policy, atol=1.0e-6)
    assert tuple(mapper.real_motor_ids.tolist()) == REAL_MOTOR_IDS


def test_path_estimator_uses_initial_heading_frame_and_resets():
    estimator = Rs01StraightPathEstimator()
    estimator.reset(now=1.0, yaw=0.0)
    displacement, velocity = estimator.update(
        now=1.02,
        yaw=0.0,
        base_linear_velocity_body=[0.23, 0.04, 0.0],
    )
    assert velocity == pytest.approx(0.04)
    assert displacement == pytest.approx(0.0004)
    displacement, velocity = estimator.update(
        now=1.04,
        yaw=np.pi / 2.0,
        base_linear_velocity_body=[0.20, 0.0, 0.0],
    )
    assert velocity == pytest.approx(0.20)
    assert displacement == pytest.approx(0.0028)
    estimator.reset(now=2.0, yaw=1.2)
    assert estimator.lateral_displacement_m == 0.0
    assert estimator.heading_target == pytest.approx(1.2)


def test_path_estimator_rejects_stale_or_nonfinite_updates():
    estimator = Rs01StraightPathEstimator(max_update_gap_s=0.10)
    estimator.reset(now=0.0, yaw=0.0)
    with pytest.raises(RuntimeError, match="update gap"):
        estimator.update(0.11, 0.0, [0.0, 0.0, 0.0])
    estimator.reset(now=1.0, yaw=0.0)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        estimator.update(1.02, 0.0, [np.nan, 0.0, 0.0])


def test_stationary_gyro_bias_is_recovered_and_motion_is_rejected():
    rng = np.random.default_rng(1850)
    expected = np.asarray([0.012, -0.018, -0.133])
    gyro = expected + rng.normal(0.0, 0.002, size=(500, 3))
    rpy = rng.normal(0.0, 0.001, size=(500, 3))
    actual = estimate_stationary_gyro_bias(
        gyro,
        rpy,
        max_std_rad_s=0.05,
        max_rpy_span_rad=0.08,
        max_abs_bias_rad_s=0.35,
    )
    np.testing.assert_allclose(actual, expected, atol=5.0e-4)
    moving_rpy = rpy.copy()
    moving_rpy[:, 2] += np.linspace(0.0, 0.2, len(moving_rpy))
    with pytest.raises(RuntimeError, match="orientation changed"):
        estimate_stationary_gyro_bias(
            gyro,
            moving_rpy,
            max_std_rad_s=0.05,
            max_rpy_span_rad=0.08,
            max_abs_bias_rad_s=0.35,
        )


def test_54d_observation_inference_and_path_tail_are_exact():
    session, contract = load_contract()
    core = Rs01Model1850PolicyCore(session, contract)
    legacy_prefix = Rs01Model930PolicyCore(session, contract)
    core.reset(now=10.0, yaw=0.3, q_policy=contract.default)
    legacy_prefix.reset(
        now=10.0, yaw=0.3, q_policy=contract.default
    )
    observation = core.build_observation(
        now=10.02,
        base_linear_velocity=[0.23, 0.05, 0.0],
        base_angular_velocity=[0.0, 0.0, 0.1],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.23, 0.0, 0.0],
        q_policy=contract.default,
        dq_policy=np.zeros(12, dtype=np.float32),
        yaw=0.3,
    )
    assert observation.shape == (54,)
    assert np.all(np.isfinite(observation))
    expected_prefix = legacy_prefix.build_observation(
        now=10.02,
        base_linear_velocity=[0.23, 0.05, 0.0],
        base_angular_velocity=[0.0, 0.0, 0.1],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.23, 0.0, 0.0],
        q_policy=contract.default,
        dq_policy=np.zeros(12, dtype=np.float32),
        yaw=0.3,
    )
    np.testing.assert_array_equal(observation[:52], expected_prefix)
    assert observation[-2] == pytest.approx(0.001)
    assert observation[-1] == pytest.approx(0.10)
    result = core.step(
        observation,
        contract.default,
        np.zeros(12, dtype=np.float32),
        active_limit_nm=14.0,
    )
    assert result["action"].shape == (12,)
    assert np.all(np.isfinite(result["action"]))
    assert np.max(
        np.abs(result["torque_info"]["safe_pd_torque_nm"])
    ) <= 14.0 + 1.0e-6


def test_model1850_node_and_launch_default_to_no_send_and_stand_only():
    node = NODE.read_text(encoding="utf-8")
    shared_node = SHARED_NODE.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "class Rs01Model1850Node(Rs01Model930Node)" in node
    assert "observation_count = 54" in node
    assert "include_path_state = True" in node
    assert "calibrate_gyro_bias = True" in node
    assert (
        'DeclareLaunchArgument("enable_send", default_value="false")'
        in launch
    )
    assert (
        'DeclareLaunchArgument("stand_only", default_value="true")'
        in launch
    )
    assert '"hardware_torque_limit_nm": 14.0' in launch
    assert '"gyro_bias_calibration_sec": 5.0' in launch
    assert 'self.declare_parameter("enable_send", False)' in shared_node
    assert 'self.declare_parameter("stand_only", True)' in shared_node
    assert "walk_authorized" in shared_node
    assert "self.leg_odometry.reset()" in shared_node
    assert "walk_start_stable_sec" in shared_node
    assert '"walk_start_stable_sec": 1.0' in launch
    assert '"walk_start_max_odom_speed_mps": 0.05' in launch
    assert '"require_hardware_torque_limits": True' in shared_node
    assert '"require_verified_hardware_safety_limits": True' in shared_node
