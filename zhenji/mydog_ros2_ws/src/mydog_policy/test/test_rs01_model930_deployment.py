from pathlib import Path

import numpy as np
import onnxruntime as ort

from mydog_policy.rs01_model930_core import (
    EXPECTED_ONNX_SHA256,
    Model930Contract,
    POLICY_JOINT_NAMES,
    REAL_MOTOR_IDS,
    REVERSED_MOTOR_IDS,
    Rs01ContinuousTorqueGuard,
    Rs01Model930Mapper,
    Rs01Model930PolicyCore,
    Rs01Model930TargetLimiter,
    Rs01NewMachineLegOdometry,
    sha256_file,
)


PACKAGE = Path(__file__).resolve().parents[1]
ONNX = PACKAGE / "resource" / "model_930_rs01_heading52.onnx"
NODE = PACKAGE / "mydog_policy" / "rs01_model930_node.py"
LAUNCH = PACKAGE / "launch" / "rs01_model930.launch.py"


def load_contract():
    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )
    contract = Model930Contract.from_onnx_session(
        session, ONNX, EXPECTED_ONNX_SHA256
    )
    return session, contract


def test_model_hash_dimensions_and_embedded_mapping():
    session, contract = load_contract()
    assert sha256_file(ONNX) == EXPECTED_ONNX_SHA256
    assert tuple(contract.joint_names) == POLICY_JOINT_NAMES
    assert session.get_inputs()[0].shape == ["batch", 52]
    assert session.get_outputs()[0].shape == ["batch", 12]
    mapping = contract.raw["motor_mapping"]
    assert [item["real_feedback_index"] for item in mapping] == [
        3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11
    ]
    signs = {
        int(item["motor_id"], 16): item["real_to_policy_sign"]
        for item in mapping
    }
    assert {
        motor_id for motor_id, sign in signs.items() if sign == -1.0
    } == set(REVERSED_MOTOR_IDS)


def test_new_machine_mapping_round_trip_and_stand_target():
    _, contract = load_contract()
    mapper = Rs01Model930Mapper(contract)
    rng = np.random.default_rng(930)
    q_policy = rng.uniform(contract.lower, contract.upper).astype(
        np.float32
    )
    q_real = mapper.policy_target_to_real(q_policy)
    recovered, recovered_dq = mapper.real_to_policy_abs(
        q_real, np.zeros(12, dtype=np.float32)
    )
    np.testing.assert_allclose(recovered, q_policy, atol=1.0e-6)
    np.testing.assert_array_equal(recovered_dq, 0.0)
    np.testing.assert_allclose(
        mapper.policy_target_to_real(contract.default),
        [
            0.0, -0.32987297, -1.31853104,
            0.0, 0.32987297, 1.31853104,
            0.0, 0.32987297, 1.31853104,
            0.0, -0.32987297, -1.31853104,
        ],
        atol=1.0e-6,
    )
    assert tuple(mapper.real_motor_ids.tolist()) == REAL_MOTOR_IDS


def test_actual_urdf_fk_reproduces_supplied_default_foot_centers():
    _, contract = load_contract()
    expected = {
        "FL": [0.216, 0.14725, -0.300],
        "FR": [0.216, -0.14725, -0.300],
        "RL": [-0.216, 0.14725, -0.300],
        "RR": [-0.216, -0.14725, -0.300],
    }
    for index, leg in enumerate(("FL", "FR", "RL", "RR")):
        foot, jacobian = (
            Rs01NewMachineLegOdometry.foot_position_and_jacobian(
                leg, contract.default[index * 3:index * 3 + 3]
            )
        )
        np.testing.assert_allclose(foot, expected[leg], atol=3.0e-6)
        assert jacobian.shape == (3, 3)
        assert np.all(np.isfinite(jacobian))


def test_observation_inference_and_target_limits_are_finite():
    session, contract = load_contract()
    core = Rs01Model930PolicyCore(session, contract)
    core.reset(now=10.0, yaw=0.25, q_policy=contract.default)
    observation = core.build_observation(
        now=10.15,
        base_linear_velocity=[0.1, 0.0, 0.0],
        base_angular_velocity=[0.0, 0.0, 0.1],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.23, 0.0, 0.0],
        q_policy=contract.default,
        dq_policy=np.zeros(12),
        yaw=0.20,
    )
    assert observation.shape == (52,)
    assert np.all(np.isfinite(observation))
    result = core.step(
        observation,
        contract.default,
        np.zeros(12, dtype=np.float32),
        active_limit_nm=14.0,
    )
    for key in (
        "action",
        "desired_target_policy",
        "limited_target_policy",
        "safe_target_policy",
    ):
        assert result[key].shape == (12,)
        assert np.all(np.isfinite(result[key]))
    assert np.max(np.abs(
        result["torque_info"]["safe_pd_torque_nm"]
    )) <= 14.0 + 1.0e-6


def test_limiter_respects_rate_acceleration_and_pd_budget():
    _, contract = load_contract()
    limiter = Rs01Model930TargetLimiter(contract)
    limiter.reset(contract.default)
    previous_target = limiter.target.copy()
    previous_velocity = limiter.target_velocity.copy()
    for _ in range(20):
        target = limiter.step(contract.upper)
        velocity = (
            target - previous_target
        ) / contract.policy_dt
        acceleration = (
            velocity - previous_velocity
        ) / contract.policy_dt
        assert np.all(
            np.abs(velocity) <= contract.rate_limit + 2.0e-5
        )
        assert np.all(
            np.abs(acceleration) <= contract.accel_limit + 2.0e-4
        )
        previous_target = target
        previous_velocity = velocity
    _, info = limiter.pd_equivalent_peak_limit(
        contract.upper,
        contract.default,
        np.full(12, 8.0, dtype=np.float32),
        14.0,
    )
    assert np.all(np.abs(info["safe_pd_torque_nm"]) <= 14.0)
    assert np.count_nonzero(info["limited_mask"]) > 0


def test_continuous_torque_guard_preserves_bursts_and_derates_sustained_load():
    guard = Rs01ContinuousTorqueGuard(
        continuous_torque_nm=6.0,
        peak_torque_nm=14.0,
        derate_full_rms_nm=8.0,
        time_constant_s=2.0,
    )
    np.testing.assert_allclose(guard.active_limits(), 14.0)
    for _ in range(25):
        limit = guard.update(np.full(12, 14.0), 0.02)
    assert np.all(guard.rms > 6.0)
    assert np.all(limit < 14.0)
    for _ in range(200):
        limit = guard.update(np.full(12, 14.0), 0.02)
    np.testing.assert_allclose(limit, 6.0, atol=1.0e-5)
    for _ in range(500):
        limit = guard.update(np.zeros(12), 0.02)
    assert np.all(limit > 13.0)


def test_hardware_node_defaults_are_double_interlocked():
    node = NODE.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'self.declare_parameter("enable_send", False)' in node
    assert 'self.declare_parameter("stand_only", True)' in node
    assert '"require_hardware_torque_limits": True' in node
    assert '"require_verified_hardware_safety_limits": True' in node
    assert "/api/rs04/configure_verified_motion_safety_limits" in node
    assert 'DeclareLaunchArgument("enable_send", default_value="false")' in launch
    assert 'DeclareLaunchArgument("stand_only", default_value="true")' in launch
