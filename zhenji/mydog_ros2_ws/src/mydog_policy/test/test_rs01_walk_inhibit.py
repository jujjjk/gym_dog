from pathlib import Path

import numpy as np

from mydog_policy.rs01_model1850_core import Rs01StraightPathEstimator
from mydog_policy.rs01_model930_core import (
    Rs01NewMachineLegOdometry,
    Rs01YawGyroConsistencyMonitor,
)


PACKAGE = Path(__file__).resolve().parents[1]
NODE = PACKAGE / "mydog_policy" / "rs01_model930_node.py"


def _synthetic_odometry(velocity_by_foot):
    estimator = Rs01NewMachineLegOdometry(
        strict_diagonal_pairs=True
    )
    positions = {
        leg: np.asarray([0.0, 0.0, -0.291], dtype=np.float32)
        for leg in estimator.LEG_ORDER
    }
    estimator.foot_position_and_jacobian = (
        lambda leg, _q: (
            positions[leg],
            np.eye(3, dtype=np.float32),
        )
    )
    dq = -np.asarray(velocity_by_foot, dtype=np.float32).reshape(12)
    return estimator.estimate(
        np.zeros(12, dtype=np.float32),
        dq,
        np.zeros(3, dtype=np.float32),
    )


def test_only_agreeing_diagonal_pair_is_accepted():
    result = _synthetic_odometry(
        [
            [0.20, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [-0.50, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        result["stance_mask"], [True, False, False, True]
    )
    assert result["selected_pair_index"] == 0
    assert result["legal_diagonal_support"]
    assert result["confidence"] == 1.0


def test_same_side_agreement_cannot_authorize_odometry():
    result = _synthetic_odometry(
        [
            [0.20, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
        ]
    )
    assert not result["legal_diagonal_support"]
    assert result["selected_pair_index"] == -1
    assert result["confidence"] == 0.0
    np.testing.assert_array_equal(result["stance_mask"], False)


def test_yaw_gyro_monitor_accepts_consistency_and_rejects_dc_mismatch():
    monitor = Rs01YawGyroConsistencyMonitor(warmup_sec=0.5)
    yaw = 0.0
    monitor.reset(0.0, yaw)
    for step in range(1, 101):
        state = monitor.update(step * 0.02, yaw, 0.0)
    assert state["ready"]
    assert state["healthy"]

    monitor.reset(0.0, 0.0)
    yaw_rate = 0.15
    for step in range(1, 151):
        yaw = yaw_rate * step * 0.02
        state = monitor.update(step * 0.02, yaw, 0.0)
    assert state["ready"]
    assert not state["healthy"]
    assert state["mean_error_rad_s"] > 0.12


def test_path_freezes_without_legal_support():
    estimator = Rs01StraightPathEstimator()
    estimator.reset(now=0.0, yaw=0.0)
    displacement, velocity = estimator.update(
        now=0.02,
        yaw=0.0,
        base_linear_velocity_body=[0.23, 0.10, 0.0],
        update_enabled=False,
    )
    assert displacement == 0.0
    assert velocity == 0.0
    displacement, velocity = estimator.update(
        now=0.04,
        yaw=0.0,
        base_linear_velocity_body=[0.23, 0.10, 0.0],
        update_enabled=True,
    )
    assert displacement > 0.0
    assert velocity == 0.10


def test_estimator_rejection_uses_soft_hold_not_direct_stop():
    source = NODE.read_text(encoding="utf-8")
    start = source.index("    def _enter_soft_hold")
    end = source.index("    def _update_walk_inhibitors", start)
    soft_hold_source = source[start:end]
    assert 'self.mode = "soft_hold"' in soft_hold_source
    assert 'f"{self.motor_base_url}/api/stop' not in soft_hold_source
    assert "self._emergency_stop" not in soft_hold_source
