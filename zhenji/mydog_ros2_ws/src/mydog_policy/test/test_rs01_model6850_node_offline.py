"""Exercise the real constructor/control loop with all ROS/device I/O mocked.

No ROS graph, serial open, HTTP request, hardware configure or enable occurs.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from mydog_policy import rs01_model930_node as base
from mydog_policy import rs01_model6850_node as mod


@pytest.fixture
def make_node(monkeypatch, tmp_path):
    clock = NS(t=100.)
    fake_time = NS(time=lambda: clock.t, monotonic=lambda: clock.t,
                   sleep=lambda seconds: None)
    monkeypatch.setattr(base, 'time', fake_time)
    monkeypatch.setattr(mod, 'time', fake_time)
    calls, nodes = [], []
    resource = Path(__file__).resolve().parents[1] / 'resource/stand_only_6850.onnx'
    logger = NS(info=lambda *a: None, warn=lambda *a: None,
                warning=lambda *a: None, error=lambda *a: None)
    overrides = dict(onnx_path=str(resource), max_motor_age_ms=80., max_imu_age_sec=.06,
                     http_timeout_sec=.04, max_abs_roll_rad=.45, max_abs_pitch_rad=.45,
                     startup_ready_error_rad=.12, startup_ready_hold_sec=2.)

    def init_ros(self, name):
        self.fake_parameters = {}
        self.messages = []

    def declare(self, name, default):
        self.fake_parameters[name] = overrides.get(name, default)

    def publish(self, topic, msg):
        self.messages.append((topic, msg))

    monkeypatch.setattr(base.Node, '__init__', init_ros)
    cls = mod.Rs01Model6850Node
    monkeypatch.setattr(cls, 'declare_parameter', declare)
    monkeypatch.setattr(cls, 'get_parameter', lambda s, n: NS(value=s.fake_parameters[n]))
    monkeypatch.setattr(cls, 'get_logger', lambda s: logger)
    monkeypatch.setattr(cls, 'create_publisher', lambda s, ty, topic, depth:
                        NS(publish=lambda msg: publish(s, topic, msg)))
    monkeypatch.setattr(cls, 'create_subscription', lambda *a: None)
    monkeypatch.setattr(cls, 'create_service', lambda *a: None)
    monkeypatch.setattr(cls, 'create_timer', lambda *a: None)
    monkeypatch.setattr(cls, '_calibrate_gyro_bias', lambda s: np.zeros(3))
    monkeypatch.setattr(cls, '_start_watchdog', lambda s: None)
    monkeypatch.setattr(base, 'get_package_share_directory', lambda _: str(tmp_path))

    class Http:
        def post(self, url, **kw):
            calls.append((url, kw))
            return NS(status_code=200, json=lambda: dict(verified=True, count=12))

        def close(self):
            pass

    monkeypatch.setattr(base.requests, 'Session', Http)

    class Motor:
        def __init__(self, **kw):
            self.owner = None
            self.stale = self.offline = False

        def get_latest(self):
            n = self.owner
            q = n.mapper.policy_target_to_real(n.contract.default)
            return NS(valid=True, stamp=clock.t - (.1 if self.stale else .001),
                      age_ms=np.ones(12), online=np.full(12, not self.offline),
                      error_code=np.zeros(12), temp=np.full(12, 29.),
                      q_real=q, dq_real=np.zeros(12), torque=np.zeros(12),
                      snapshot_seq=np.full(12, int(clock.t * 50)),
                      board_tick_ms=np.full(12, int(clock.t * 1000)),
                      cache_age_ms=1., poll_dt_ms=20.)

        def close(self):
            pass

    class Imu:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def wait_until_ready(self, **kw):
            return True

        def get_latest(self):
            return NS(valid=True, backend_alive=True, backend_error='', stamp=clock.t - .001,
                      rpy_deg=np.zeros(3), gyro_rad_s=np.zeros(3),
                      projected_gravity=np.array([0., 0., -1.]), quat_wxyz=np.array([1., 0., 0., 0.]))

    monkeypatch.setattr(base, 'MotorStateHttpInterface', Motor)
    monkeypatch.setattr(cls, 'imu_interface_type', Imu)

    def make(send=False, stand=True):
        overrides.update(enable_send=send, stand_only=stand)
        node = cls()
        node.motor.owner = node
        nodes.append(node)
        return node, clock, calls

    yield make
    for node in nodes:
        # Constructor lock is real but only an advisory file, not hardware.
        if node._owner:
            node._owner.close()


def advance(node, clock, count):
    for _ in range(count):
        clock.t += .02
        node.control_loop()
        assert not node.faulted


def test_dry_run_constructor_and_loop_never_post_http(make_node):
    node, clock, calls = make_node()
    advance(node, clock, 180)
    assert node.mode == 'ready'
    assert not calls
    assert node.core.actor.steps == 0


def arm(node):
    response = node.arm_callback(NS(data=True), SetBool.Response())
    assert response.success, response.message


@pytest.mark.parametrize('axis,value', [('x', .1), ('x', -.1), ('y', .1),
                                       ('y', -.1), ('yaw', .2), ('yaw', -.2)])
def test_omni_start_and_zero_return_use_live_stand_ramp(make_node, axis, value):
    node, clock, calls = make_node(send=True, stand=False)
    assert 'configure_verified_motion_safety_limits' in calls[0][0]
    advance(node, clock, 180)
    assert node.mode == 'ready' and node.walk_start_stable
    arm(node)
    msg = Twist()
    if axis == 'yaw':
        msg.angular.z = value
    else:
        setattr(msg.linear, axis, value)
    for _ in range(65):
        node.command_callback(msg)
        advance(node, clock, 1)
    assert node.mode == 'walk'
    assert node.core.actor.steps > 0
    idx = {'x': 0, 'y': 1, 'yaw': 2}[axis]
    assert node.core.actor.command[idx] == pytest.approx(value)
    node.command_callback(Twist())
    advance(node, clock, 1)
    assert node.mode == 'soft_hold'
    assert not node.trial.armed
    assert not any('/api/stop' in url for url, _ in calls)
    np.testing.assert_allclose(node.stand_target_policy, node.contract.default)
    advance(node, clock, 110)
    assert node.mode == 'ready'
    node.command_callback(msg)
    advance(node, clock, 65)
    assert node.mode == 'ready'  # requires a NEW explicit arm


@pytest.mark.parametrize('problem', ['stale', 'offline', 'loop_delay'])
def test_hard_fault_sends_stop_and_never_resumes(make_node, problem):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 4)
    if problem == 'loop_delay':
        clock.t += .2
    else:
        setattr(node.motor, problem, True)
    clock.t += .02
    node.control_loop()
    assert node.faulted
    assert '/api/stop' in calls[-1][0]
    count = len(calls)
    clock.t += .02
    node.control_loop()
    assert len(calls) == count


def test_stand_only_refuses_arm(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 180)
    response = node.arm_callback(NS(data=True), SetBool.Response())
    assert not response.success
    msg = Twist()
    msg.linear.y = .1
    node.command_callback(msg)
    advance(node, clock, 100)
    assert node.mode == 'ready'
    assert node.core.actor.steps == 0


def test_watchdog_latches_stop_without_next_control_tick(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 3)
    clock.t += .151
    node._watchdog_check()
    assert node.faulted
    assert '/api/stop' in calls[-1][0]


@pytest.mark.parametrize('stop', ['timeout', 'budget', 'disarm'])
def test_trial_exit_paths_return_softly_and_require_new_arm(make_node, stop):
    node, clock, calls = make_node(send=True, stand=False)
    advance(node, clock, 180)
    arm(node)
    msg = Twist()
    msg.linear.y = .1
    for _ in range(65):
        node.command_callback(msg)
        advance(node, clock, 1)
    assert node.mode == 'walk'
    if stop == 'timeout':
        advance(node, clock, 19)
    elif stop == 'budget':
        for _ in range(195):
            node.command_callback(msg)
            advance(node, clock, 1)
    else:
        node.arm_callback(NS(data=False), SetBool.Response())
        advance(node, clock, 1)
    assert node.mode == 'soft_hold'
    assert not node.trial.armed
    assert not any('/api/stop' in url for url, _ in calls)


def test_limits_cannot_be_raised_through_parameters(make_node):
    node, _, calls = make_node()
    node.fake_parameters['hardware_torque_limit_nm'] = 17.
    with pytest.raises(RuntimeError, match='hardware_torque_limit_nm'):
        node._validate_deployment_parameters()
    assert not calls


def test_failed_stop_is_not_reported_as_accepted_and_is_retried(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 3)
    original = node.http.post
    node.http.post = lambda *a, **kw: NS(status_code=503)
    node._emergency_stop('test fault')
    assert node.faulted and not node.stop_sent
    node.http.post = original
    clock.t += .51
    node._watchdog_check()
    assert node.stop_sent
    assert '/api/stop' in calls[-1][0]
