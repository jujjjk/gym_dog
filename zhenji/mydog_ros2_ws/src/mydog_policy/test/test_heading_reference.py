"""Direction-controller heading follows the gyro while the motor field disturbs the fused yaw."""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from mydog_policy.heading_reference import GyroHeadingReference, wrap_pi


def run(reference, seconds, fused, gyro, mag, walking, start=0., dt=.02):
    t = start
    heading = None
    for _ in range(int(round(seconds/dt))):
        t += dt
        heading = reference.update(t, fused(t), gyro(t), mag(t), walking)
    return heading, t


def test_standing_quiet_field_tracks_fused_yaw():
    r = GyroHeadingReference()
    heading, _ = run(r, 3., fused=lambda t: .3, gyro=lambda t: 0., mag=lambda t: 80., walking=False)
    assert heading == pytest.approx(.3, abs=1e-3)
    assert r.baseline_ut == pytest.approx(80.)
    assert r.diagnostics['heading_source'] == 'fused_tracking'


def test_walking_with_motor_field_integrates_gyro_and_ignores_fused_yaw():
    r = GyroHeadingReference()
    _, t = run(r, 2., fused=lambda t: 0., gyro=lambda t: 0., mag=lambda t: 80., walking=False)
    # Capture 20260926_181039: gyro -0.12 rad/s for 4.4 s (-30 deg), fused yaw +7 deg,
    # magnetometer norm 76-172 uT against an 80 uT standing baseline.
    heading, _ = run(r, 4.4, fused=lambda t: math.radians(7.)*(t-2.)/4.4, gyro=lambda t: -.12,
                     mag=lambda t: 80.+90.*abs(math.sin(6.*t)), walking=True, start=t)
    assert heading == pytest.approx(-.12*4.4, abs=.02)
    assert r.diagnostics['heading_source'] == 'gyro_integrated'
    assert not r.diagnostics['mag_field_quiet']
    assert r.baseline_ut == pytest.approx(80.)


def test_walking_quiet_field_pulls_slowly_toward_fused_yaw():
    r = GyroHeadingReference(walking_pull_tau_sec=5.)
    _, t = run(r, 2., fused=lambda t: 0., gyro=lambda t: 0., mag=lambda t: 80., walking=False)
    heading, _ = run(r, 5., fused=lambda t: .5, gyro=lambda t: 0., mag=lambda t: 82., walking=True, start=t)
    assert 0.25 < heading < 0.4
    assert r.diagnostics['heading_source'] == 'gyro_with_slow_fused_pull'


def test_wrap_clock_jump_and_nonfinite_inputs():
    r = GyroHeadingReference()
    r.update(1., 3.1, 0., 80., False)
    assert r.update(1.02, 3.1, 10., 150., True) == pytest.approx(wrap_pi(3.1+.2))
    # A 4 s clock jump integrates nothing instead of inventing 40 rad.
    assert r.update(5., 3.1, 10., 150., True) == pytest.approx(wrap_pi(3.1+.2))
    with pytest.raises(ValueError):
        r.update(5.02, float('nan'), 0., 80., True)
    r.update(5.04, 0., 0., float('nan'), True)
    assert not r.diagnostics['mag_field_quiet']


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance


@pytest.mark.parametrize('make_node', ['B23500'], indirect=True)
def test_node_steers_on_gyro_heading_while_monitor_watches_fused_yaw(make_node, monkeypatch):
    from mydog_policy import rs01_model23500_node as module
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 120)
    state = dict(gyro=0., yaw_deg=0., mag=np.array([80., 0., 0.]))
    imu_latest = n.imu.get_latest
    def imu():
        s = imu_latest()
        s.gyro_rad_s = np.array([0., 0., state['gyro']])
        s.rpy_deg = np.array([0., 0., state['yaw_deg']])
        s.mag_uT = state['mag'].copy()
        return s
    n.imu.get_latest = imu
    advance(n, clock, 500)
    assert n.heading_reference.baseline_ut == pytest.approx(80., abs=1.)
    n.mode = 'walk'
    n.trial.arm(clock.t); n.trial.receive([.1, 0, 0], clock.t)
    n._reset_walk_session(clock.t, n.heading_reference.heading, n.contract.default)
    target = float(n.core.heading_target)
    state.update(gyro=-.12, mag=np.array([170., 0., 0.]))
    for _ in range(100):
        clock.t += .02
        state['yaw_deg'] += .03
        n.trial.receive([.1, 0, 0], clock.t)
        n.control_loop()
        assert not n.faulted
    heading = n.heading_reference.heading
    assert wrap_pi(heading-target) == pytest.approx(-.12*2., abs=.03)
    # The monitor compares the fused yaw (+1.5 deg/s) with the gyro (-6.9 deg/s).
    assert abs(n.heading_consistency_state['mean_error_rad_s']) > .08
    assert not calls
