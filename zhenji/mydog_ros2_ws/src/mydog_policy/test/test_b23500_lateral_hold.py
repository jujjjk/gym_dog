"""Lateral position hold: bounded march drift correction, no real device access."""
import math
from pathlib import Path
import numpy as np
import pytest
import onnxruntime as ort
from mydog_policy.lateral_position_hold import LateralPositionHold
from mydog_policy.rs01_model23500_core import Model23500Contract, Rs01Model23500Core
from mydog_policy.rs01_model6850_core import Rs01Model6850Core

DT = .02
LEFT_DRIFT = [0., .03, 0.]
MARCH = [0., 0., 0.]


def advance(hold, steps, velocity=LEFT_DRIFT, confidence=.8, command=MARCH,
            turning=False, gait=1., heading_error=0., dt=DT):
    for _ in range(steps):
        hold.update(dt, velocity, confidence, command, turning, gait, heading_error)


def test_left_drift_builds_offset_and_a_bounded_right_correction():
    hold = LateralPositionHold()
    advance(hold, 50)
    assert hold.reason == 'holding' and hold.active
    assert hold.offset == pytest.approx(.03, abs=1e-9)
    assert hold.correction == pytest.approx(-LateralPositionHold.gain_per_s*.01, abs=1e-9)
    assert -LateralPositionHold.correction_limit_mps <= hold.correction < 0.


def test_deadband_leaves_small_offsets_uncorrected():
    hold = LateralPositionHold()
    advance(hold, 30)
    assert 0. < hold.offset < LateralPositionHold.deadband_m
    assert hold.correction == 0. and hold.active
    advance(hold, 10)
    assert hold.correction < 0.
    assert hold.correction == pytest.approx(
        -LateralPositionHold.gain_per_s*(hold.offset-LateralPositionHold.deadband_m))


def test_offset_and_correction_stay_clamped_under_a_persistent_drift():
    hold = LateralPositionHold()
    advance(hold, 5000)
    assert hold.offset == LateralPositionHold.offset_limit_m
    assert hold.correction == -LateralPositionHold.correction_limit_mps


def test_right_drift_is_corrected_to_the_left():
    hold = LateralPositionHold()
    advance(hold, 50, velocity=[0., -.03, 0.])
    assert hold.offset < 0. and hold.correction > 0.


def test_velocity_is_projected_onto_the_heading_axis():
    hold = LateralPositionHold()
    # A body-forward velocity seen under a 90 deg heading error is lateral.
    advance(hold, 50, velocity=[.03, 0., 0.], heading_error=math.pi/2)
    assert hold.offset == pytest.approx(-.03, abs=1e-9)
    assert hold.correction > 0.


def test_turns_stand_and_operator_lateral_commands_re_reference_the_offset():
    for kwargs in [dict(turning=True), dict(gait=0.), dict(command=[0., .3, 0.])]:
        hold = LateralPositionHold()
        advance(hold, 50)
        advance(hold, 1, **kwargs)
        assert hold.offset == 0. and hold.correction == 0. and not hold.active
        expected = ('operator_lateral_command' if 'command' in kwargs
                    else 'not_pure_march')
        assert hold.reason == expected


def test_small_operator_lateral_command_below_the_gate_keeps_holding():
    hold = LateralPositionHold()
    advance(hold, 50, command=[0., LateralPositionHold.operator_vy_gate_mps/2, 0.])
    assert hold.active and hold.offset > 0.


def test_unconfident_odometry_freezes_the_offset_instead_of_resetting_it():
    hold = LateralPositionHold()
    advance(hold, 50)
    offset, correction = hold.offset, hold.correction
    for velocity, confidence in [(LEFT_DRIFT, 0.), (None, .8)]:
        advance(hold, 5, velocity=velocity, confidence=confidence)
        assert hold.offset == offset and hold.correction == correction
        assert not hold.active and hold.reason == 'odometry_unconfident'
    advance(hold, 5)
    assert hold.offset > offset and hold.active


def test_nonfinite_input_resets_and_out_of_range_dt_is_rejected():
    hold = LateralPositionHold()
    advance(hold, 50)
    hold.update(DT, [0., float('nan'), 0.], .8, MARCH, False, 1., 0.)
    assert hold.offset == 0. and hold.correction == 0. and hold.reason == 'nonfinite_input'
    advance(hold, 50)
    offset = hold.offset
    hold.update(LateralPositionHold.max_dt_sec*2, LEFT_DRIFT, .8, MARCH, False, 1., 0.)
    assert hold.offset == offset and hold.reason == 'dt_out_of_range' and not hold.active


def test_state_is_json_scalar_only():
    state = LateralPositionHold().state()
    assert sorted(state) == ['lateral_hold_active', 'lateral_hold_correction_mps',
                             'lateral_hold_offset_m', 'lateral_hold_reason']
    assert all(isinstance(v, (bool, float, str)) for v in state.values())


@pytest.fixture(scope='module')
def session_and_contract():
    path = Path(__file__).resolve().parents[1]/'resource/B23500.onnx'
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opt,
                                   providers=['CPUExecutionProvider'])
    return session, Model23500Contract.from_onnx_session(session, path)


@pytest.fixture
def actor(session_and_contract):
    session, contract = session_and_contract
    return Rs01Model23500Core(session, contract)


def stub_odometry(actor, velocity, confidence=.8):
    vel = np.asarray(velocity, dtype=float).reshape(3)

    def estimate(q_policy, dq_policy, omega_body, kinematics=None):
        return dict(raw_base_velocity=vel.copy(), base_linear_velocity=vel.copy(),
                    confidence=confidence, stance_mask=np.array([True, False, False, True]),
                    foot_position=np.zeros((4, 3)), foot_velocity=np.zeros((4, 3)),
                    velocity_by_foot=np.tile(vel, (4, 1)), base_height_proxy=np.full(4, .30),
                    selected_pair_index=0, pair_residual_m_s=0., legal_diagonal_support=True)

    actor.odometry.estimate = estimate


def march(actor, contract, steps, command=MARCH, gait=1.):
    rows = []
    for _ in range(steps):
        result = actor.tick(contract.default, np.zeros(12), np.zeros(3), [0., 0., -1.],
                            0., command, gait=gait)
        rows.append(result['observation'])
    return np.asarray(rows)


def test_fresh_actor_reproduces_the_trained_direction_mixer(actor):
    cfg = actor.contract.raw['v13']['commands']
    assert actor.lateral_hold.correction == 0.
    for cmd in ([.2, 0., 0.], [0., 0., 0.], [0., .1, 0.]):
        cmd = np.asarray(cmd)
        assert np.array_equal(actor._mix_direction_command(cmd, .1, False, cfg),
                              Rs01Model6850Core._mix_direction_command(actor, cmd, .1, False, cfg))


def test_correction_enters_the_mixed_target_only(actor):
    cfg = actor.contract.raw['v13']['commands']
    actor.lateral_hold.correction = -.04
    target = actor._mix_direction_command(np.asarray([.2, 0., 0.]), 0., False, cfg)
    baseline = Rs01Model6850Core._mix_direction_command(actor, np.asarray([.2, 0., 0.]),
                                                        0., False, cfg)
    assert target[1] == pytest.approx(baseline[1]-.04)
    assert target[0] == baseline[0] and target[2] == baseline[2]
    actor.lateral_hold.correction = -10.
    lo, hi = actor.contract.raw['commands']['ranges']['lin_vel_y']
    assert actor._mix_direction_command(np.zeros(3), 0., False, cfg)[1] == pytest.approx(lo)
    actor.lateral_hold.correction = 10.
    assert actor._mix_direction_command(np.zeros(3), 0., False, cfg)[1] == pytest.approx(hi)


def test_march_with_left_drift_pushes_right_without_touching_raw_commands(actor):
    contract = actor.contract
    stub_odometry(actor, LEFT_DRIFT)
    obs = march(actor, contract, 150)
    scale = float(contract.command_scale[1])
    hold = actor.lateral_hold
    limit = LateralPositionHold.correction_limit_mps
    assert hold.active and hold.offset > 0.
    assert hold.correction == pytest.approx(-limit)
    assert np.allclose(obs[:, 57:60], 0.), 'raw command channels must stay untouched'
    # obs 10 is the mixed lateral target, obs 53 the lateral velocity error.
    # Both saturate with the hold, so the one-tick injection lag is invisible.
    assert np.all(np.abs(obs[:, 10]/scale) <= limit+1e-6)
    assert obs[-1, 10]/scale == pytest.approx(-limit, abs=1e-6)
    assert obs[-1, 53] == pytest.approx(2*(LEFT_DRIFT[1]+limit), abs=1e-3)
    assert np.isfinite(obs).all()


def test_correction_ramps_monotonically_and_stays_bounded(actor):
    contract = actor.contract
    stub_odometry(actor, [.0, .006, 0.])
    corrections = []
    for _ in range(200):
        actor.tick(contract.default, np.zeros(12), np.zeros(3), [0., 0., -1.],
                   0., MARCH, gait=1.)
        corrections.append(actor.lateral_hold.correction)
    limit = LateralPositionHold.correction_limit_mps
    assert all(-limit <= c <= 0. for c in corrections)
    assert corrections[0] == 0.
    assert all(b <= a+1e-12 for a, b in zip(corrections, corrections[1:]))
    # 0.006 m/s needs ~167 ticks to leave the 2 cm deadband, then follows the
    # proportional law exactly; saturation is covered by the unit tests above.
    assert corrections[100] == 0.
    assert corrections[179] == pytest.approx(
        -LateralPositionHold.gain_per_s*(.006*DT*180-LateralPositionHold.deadband_m),
        abs=1e-6)
    assert corrections[-1] == pytest.approx(
        -LateralPositionHold.gain_per_s*(.006*DT*200-LateralPositionHold.deadband_m),
        abs=1e-6)
    assert abs(corrections[-1]) < limit


def test_a_turn_after_drift_re_references_the_offset(actor):
    contract = actor.contract
    stub_odometry(actor, LEFT_DRIFT)
    march(actor, contract, 60)
    assert actor.lateral_hold.offset > 0.
    yaw_limit = contract.raw['commands']['ranges']['ang_vel_yaw'][1]
    march(actor, contract, 5, command=[0., 0., min(.6, yaw_limit)])
    hold = actor.lateral_hold
    assert hold.offset == 0. and hold.correction == 0. and hold.reason == 'not_pure_march'
    obs = march(actor, contract, 5)
    assert np.allclose(obs[:, 9:12], 0., atol=1e-6)


def test_stand_and_reset_clear_the_hold(actor):
    contract = actor.contract
    stub_odometry(actor, LEFT_DRIFT)
    march(actor, contract, 60)
    assert actor.lateral_hold.offset > 0.
    march(actor, contract, 2, gait=0.)
    assert actor.lateral_hold.offset == 0. and not actor.lateral_hold.active
    march(actor, contract, 60)
    actor.reset(0.)
    assert actor.lateral_hold.offset == 0. and actor.lateral_hold.correction == 0.


def test_unconfident_odometry_keeps_the_last_correction_bounded(actor):
    contract = actor.contract
    stub_odometry(actor, LEFT_DRIFT)
    march(actor, contract, 60)
    correction = actor.lateral_hold.correction
    assert correction < 0.
    stub_odometry(actor, LEFT_DRIFT, confidence=0.)
    obs = march(actor, contract, 10)
    scale = float(contract.command_scale[1])
    assert actor.lateral_hold.correction == correction
    assert not actor.lateral_hold.active
    assert obs[-1, 10]/scale == pytest.approx(correction, abs=1e-4)
