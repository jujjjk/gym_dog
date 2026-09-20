from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_rs01_model6850_node_offline import make_node, advance


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_stand_then_explicit_calibration_and_motion_reset(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 200)
    assert n.mode == 'ready'
    assert not n.imu_calibrated
    assert not n.arm_callback(NS(data=True), NS()).success
    assert n.calibrate_callback(NS(data=True), NS()).success
    advance(n, clock, 100)
    assert not n.imu_calibrated
    n.motor.offset[1] = .2
    advance(n, clock, 1)
    assert not n._calibration_samples
    n.motor.offset[:] = 0
    advance(n, clock, 270)
    assert n.imu_calibrated
    assert n.calibration_state == 'complete'
    assert not calls


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_timing_loss_between_arm_and_walk_never_enters_policy(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 200)
    n.imu_calibrated = True
    assert n.arm_callback(NS(data=True), NS()).success
    for _ in range(100):
        n._b_loop_intervals.clear()
        n._b_loop_intervals.extend([.04]*50)
        n.trial.receive([0,0,0], clock.t)
        advance(n, clock, 1)
    assert n.mode == 'ready'
    assert n.core.actor.steps == 0
    assert not calls


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_missing_rt_endpoint_prevents_enable_handshake(make_node, monkeypatch):
    n, clock, calls = make_node(send=False, stand=False)
    def unavailable():
        raise ConnectionError('missing local socket')
    monkeypatch.setattr(n._rt, 'exchange', unavailable)
    with pytest.raises(ConnectionError):
        n._prime_live_enable()
    assert not calls


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_noisy_velocity_with_stable_positions_can_calibrate(make_node, monkeypatch):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 200)
    original = n.motor.get_latest
    def noisy_feedback():
        snapshot = original()
        snapshot.dq_real[:] = .23 * np.sin(clock.t * 80)
        return snapshot
    monkeypatch.setattr(n.motor, 'get_latest', noisy_feedback)
    assert n.calibrate_callback(NS(data=True), NS()).success
    for i in range(280):
        n.motor.offset[:] = .0005 * np.sin(i)
        advance(n, clock, 1)
    assert n.imu_calibrated
    assert n.calibration_reason == ''
    assert n.calibration_joint_span_rad < .002
    assert not calls


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_real_joint_motion_resets_collection_even_with_zero_velocity(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 200)
    assert n.calibrate_callback(NS(data=True), NS()).success
    advance(n, clock, 100)
    # Still inside the stand position tolerance, but actual displacement is real.
    n.motor.offset[1] = .03
    advance(n, clock, 1)
    assert not n.imu_calibrated and not n._calibration_samples
    assert n.calibration_reason == 'joint_position_span_exceeds_0.02rad'
    n.motor.offset[:] = 0
    advance(n, clock, 260)
    assert n.imu_calibrated
    assert not calls


@pytest.mark.parametrize('make_node', ['B18000'], indirect=True)
def test_slow_joint_drift_cannot_accumulate_five_second_window(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 200)
    assert n.calibrate_callback(NS(data=True), NS()).success
    for i in range(500):
        n.motor.offset[1] = i * .00012  # .006 rad/s: below former velocity gate.
        advance(n, clock, 1)
    assert not n.imu_calibrated
    assert not calls
