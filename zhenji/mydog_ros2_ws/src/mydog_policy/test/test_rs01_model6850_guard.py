"""Offline protection tests: no ROS node, serial port or HTTP connection."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from mydog_policy.rs01_model6850_core import Model6850Contract, Rs01Model6850Core
from mydog_policy.rs01_model6850_guard import (
    Guarded6850PolicyCore, ReceptionGuard, TrialCommand,
)


@pytest.fixture(scope='module')
def actor():
    path = Path(__file__).resolve().parents[1] / 'resource/stand_only_6850.onnx'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    return session, Model6850Contract.from_onnx_session(session, path)


@pytest.mark.parametrize('command', ([.1, 0, 0], [-.1, 0, 0], [0, .1, 0],
                                     [0, -.1, 0], [0, 0, .2], [0, 0, -.2], [.1, .1, .2]))
def test_guarded_actor_preserves_61d_inference_and_limits_pd(actor, command):
    session, contract = actor
    original = Rs01Model6850Core(session, contract)
    guarded = Guarded6850PolicyCore(session, contract)
    q, dq = contract.default.copy(), np.zeros(12)
    guarded.reset(0, 0, q)
    for index in range(25):
        expected = original.tick(q, dq, [0, 0, 0], [0, 0, -1], 0, command)
        obs = guarded.build_observation(index * .02, [99, 99, 99], [0, 0, 0],
                                        [0, 0, -1], command, q, dq, 0)
        result = guarded.step(obs, q, dq, np.full(12, 6.))
        np.testing.assert_array_equal(obs, expected['observation'])
        np.testing.assert_array_equal(result['action'], expected['action'])
        np.testing.assert_array_equal(guarded.actor.target, expected['target_policy'])
        torque = contract.kp * (result['safe_target_policy'] - q) - contract.kd * dq
        assert np.max(np.abs(torque)) <= 6.00001
        assert np.isfinite(result['safe_target_policy']).all()


def test_trial_requires_explicit_arm_and_expires_without_rearm():
    trial = TrialCommand()
    assert not trial.receive([.1, 0, 0], 1.)
    trial.arm(2.)
    assert not trial.active(2.)  # old command never carried into a new arm
    assert trial.receive([0, -.1, .1], 2.1)
    assert trial.active(2.2)
    assert not trial.active(2.46)
    assert not trial.receive([.1, 0, 0], 2.5)
    assert trial.reason == 'command timeout'


def test_command_newer_than_control_tick_now_is_not_timeout():
    trial = TrialCommand()
    trial.arm(1.0)
    assert trial.receive([0.1, 0.0, 0.0], 1.20)
    # Control tick captured `now` before the concurrent callback stamped.
    assert trial.active(1.19)
    assert trial.armed
    assert trial.active(1.20)
    assert not trial.active(1.20 + trial.timeout_sec + 0.01)
    assert trial.reason == 'command timeout'


def test_arm_newer_than_control_tick_now_is_not_lease_expiry():
    trial = TrialCommand()
    trial.arm(2.0)
    assert trial.receive([0.1, 0.0, 0.0], 2.01)
    assert trial.active(1.99)
    assert trial.armed

def test_zero_command_is_step_in_place_while_armed():
    trial = TrialCommand()
    trial.arm(1.)
    assert trial.receive([.2, 0, 0], 1.1)
    assert trial.receive([0, 0, 0], 1.2)
    assert trial.active(1.25)
    np.testing.assert_array_equal(trial.vector, np.zeros(3))
    assert trial.receive([.3, .1, .25], 1.3)
    assert trial.active(1.35)


def test_isaac_sequence_stays_armed_across_step_and_combo():
    trial = TrialCommand()
    trial.arm(1.)
    trial.receive([0, 0, 0], 1.01)
    trial.start(1.02)
    for now in np.arange(1.04, 86., 1.):
        command = [0, 0, 0] if int(now) % 2 == 0 else [.3, .2, .3]
        assert trial.receive(command, now)
        assert trial.active(now)
    trial.receive([.2, 0, 0], 1. + trial.lease_sec + .01)
    assert not trial.active(1. + trial.lease_sec + .01)
    assert not trial.armed


@pytest.mark.parametrize('command', ([float('nan'), 0, 0],
                                     [.31, 0, 0], [0, -.21, 0], [0, 0, .31]))
def test_invalid_commands_disarm(command):
    trial = TrialCommand()
    trial.arm(1.)
    trial.receive([.1, 0, 0], 1.1)
    trial.receive(command, 1.2)
    assert not trial.active(1.2)
    np.testing.assert_array_equal(trial.vector, np.zeros(3))


def snapshots():
    motor = SimpleNamespace(stamp=9.99, cache_age_ms=10., age_ms=np.full(12, 10.),
                            snapshot_seq=np.full(12, 5), board_tick_ms=np.full(12, 100),
                            q_real=np.zeros(12), dq_real=np.zeros(12), torque=np.zeros(12))
    imu = SimpleNamespace(stamp=9.99, gyro_rad_s=np.zeros(3), rpy_deg=np.zeros(3),
                          projected_gravity=np.array([0, 0, -1]), quat_wxyz=np.array([1, 0, 0, 0]))
    return motor, imu


def test_transport_age_includes_both_caches_and_does_not_claim_acquisition_sync():
    motor, imu = snapshots()
    guard = ReceptionGuard()
    guard.check(motor, imu, 10., 1.)
    assert guard.metrics['effective_motor_age_ms'] == pytest.approx(30.)
    assert guard.metrics['acquisition_sync_verified'] is False
    motor.cache_age_ms = 75.
    guard.check(motor, imu, 10., 1.01)
    motor.cache_age_ms = 240.
    with pytest.raises(RuntimeError, match='stale'):
        guard.check(motor, imu, 10., 1.02)


def test_stalled_and_reset_boards_rejected_despite_fresh_http_timestamps():
    motor, imu = snapshots()
    guard = ReceptionGuard()
    guard.check(motor, imu, 10., 1.)
    with pytest.raises(RuntimeError, match='progressing'):
        guard.check(motor, imu, 10., 1.26)
    guard = ReceptionGuard()
    guard.check(motor, imu, 10., 1.)
    motor.snapshot_seq[0] = 0
    with pytest.raises(RuntimeError, match='regressed'):
        guard.check(motor, imu, 10., 1.02)


@pytest.mark.parametrize('change', ('future', 'old_imu', 'nan_torque', 'bad_quat'))
def test_invalid_feedback_is_rejected(change):
    motor, imu = snapshots()
    if change == 'future':
        motor.stamp = 10.01
    elif change == 'old_imu':
        imu.stamp = 9.7
    elif change == 'nan_torque':
        motor.torque[0] = np.nan
    else:
        imu.quat_wxyz = np.zeros(4)
    with pytest.raises(RuntimeError):
        ReceptionGuard().check(motor, imu, 10., 1.)


@pytest.mark.parametrize('new_seq', [0, 3])
def test_uint16_wrap_on_one_board_refreshes_progress(new_seq):
    motor, imu = snapshots()
    motor.snapshot_seq[:6] = 65514
    motor.snapshot_seq[6:] = 65535
    guard = ReceptionGuard()
    guard.check(motor, imu, 10., 1.)
    motor.snapshot_seq[:6] += 1
    motor.snapshot_seq[6:] = new_seq
    motor.board_tick_ms += 20
    guard.check(motor, imu, 10., 1.20)
    # More than 250ms since initialization, but wrap was valid progress.
    guard.check(motor, imu, 10., 1.30)
    with pytest.raises(RuntimeError, match='progressing'):
        guard.check(motor, imu, 10., 1.46)


@pytest.mark.parametrize('old_seq,new_seq,old_tick,new_tick', [
    (100, 99, 1000, 1020),       # stale/out-of-order frame
    (65535, 0, 1000, 0),        # reboot at sequence boundary
    (65535, 0, 1000, 1000),     # wrapping sequence without clock progress
    (100, 101, 1000, 0),        # clock reset despite forward sequence
    (0, 32768, 1000, 1020),     # ambiguous half-range jump
])
def test_counter_reset_or_regression_still_rejected(old_seq, new_seq, old_tick, new_tick):
    motor, imu = snapshots()
    motor.snapshot_seq[:] = old_seq
    motor.board_tick_ms[:] = old_tick
    guard = ReceptionGuard()
    guard.check(motor, imu, 10., 1.)
    motor.snapshot_seq[0] = new_seq
    motor.board_tick_ms[0] = new_tick
    with pytest.raises(RuntimeError, match='regressed/reset: motor_indices='):
        guard.check(motor, imu, 10., 1.02)
    assert guard.seq[0] == old_seq
    assert guard.tick[0] == old_tick


def test_out_of_range_board_sequence_rejected():
    motor, imu = snapshots()
    motor.snapshot_seq[0] = 65536
    with pytest.raises(RuntimeError, match='16-bit'):
        ReceptionGuard().check(motor, imu, 10., 1.)


def test_shared_kinematics_match_internal_foot_fk(actor):
    from mydog_policy.rs01_model930_core import Rs01NewMachineLegOdometry
    session, contract = actor
    q = contract.default.copy()
    dq = np.zeros(12)
    command = np.array([0.1, 0.0, 0.0])
    gyro = np.zeros(3)
    gravity = np.array([0.0, 0.0, -1.0])
    with_shared = Rs01Model6850Core(session, contract)
    without = Rs01Model6850Core(session, contract)
    with_shared.tick(q, dq, gyro, gravity, 0.0, command)
    without.tick(q, dq, gyro, gravity, 0.0, command)
    kinematics = with_shared.odometry.compute_kinematics(q, dq, gyro)
    shared = with_shared.tick(
        q, dq, gyro, gravity, 0.0, command, kinematics=kinematics)
    internal = without.tick(q, dq, gyro, gravity, 0.0, command)
    np.testing.assert_array_equal(shared['observation'], internal['observation'])
    first = Rs01NewMachineLegOdometry(strict_diagonal_pairs=True)
    reused = first.estimate(q, dq, gyro, kinematics=kinematics)
    fresh = Rs01NewMachineLegOdometry(strict_diagonal_pairs=True).estimate(
        q, dq, gyro)
    np.testing.assert_allclose(
        reused['base_linear_velocity'], fresh['base_linear_velocity'])
    assert reused['confidence'] == pytest.approx(fresh['confidence'])
