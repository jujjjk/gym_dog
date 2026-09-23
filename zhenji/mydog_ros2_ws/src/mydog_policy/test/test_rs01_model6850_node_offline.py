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
def make_node(monkeypatch, tmp_path, request):
    clock = NS(t=100.)
    fake_time = NS(time=lambda: clock.t, monotonic=lambda: clock.t,
                   sleep=lambda seconds: None, perf_counter=lambda: clock.t)
    monkeypatch.setattr(base, 'time', fake_time)
    monkeypatch.setattr(mod, 'time', fake_time)
    calls, nodes = [], []
    use_capture = getattr(request, 'param', None) in ('capture61', 'capture23500')
    use_v22 = getattr(request, 'param', None) in ('B23500', 'capture23500')
    use_b = use_capture or use_v22 or getattr(request, 'param', None) == 'B18000'
    if use_b:
        from mydog_policy import rs01_model18000_node as bmod
        monkeypatch.setattr(bmod, 'time', fake_time)
    resource = Path(__file__).resolve().parents[1] / ('resource/B18000.onnx' if use_b else 'resource/stand_only_6850.onnx')
    if use_v22:
        resource = resource.with_name('B23500.onnx')
    logger = NS(info=lambda *a: None, warn=lambda *a: None,
                warning=lambda *a: None, error=lambda *a: None)
    overrides = dict(onnx_path=str(resource), max_motor_age_ms=80., max_imu_age_sec=.06,
                     http_timeout_sec=.04, max_abs_roll_rad=.45, max_abs_pitch_rad=.45,
                     startup_ready_error_rad=.12, startup_ready_hold_sec=2.)

    if use_capture:
        overrides['capture_dir']=str(tmp_path/'capture')

    def init_ros(self, name):
        self.fake_parameters = {}
        self.messages = []

    def declare(self, name, default):
        self.fake_parameters[name] = overrides.get(name, default)

    def publish(self, topic, msg):
        self.messages.append((topic, msg))

    monkeypatch.setattr(base.Node, '__init__', init_ros)
    cls = bmod.Rs01Model18000Node if use_b else mod.Rs01Model6850Node
    if use_v22:
        from mydog_policy.rs01_model23500_node import Rs01Model23500Node
        cls = Rs01Model23500Node
    if use_capture:
        from mydog_policy import capture61_node as capmod
        monkeypatch.setattr(capmod, 'time', fake_time)
        cls=capmod.Capture61Node
        if use_v22:
            from mydog_policy.capture23500_node import Capture23500Node
            cls = Capture23500Node
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
            self.offset = np.zeros(12)

        def get_latest(self):
            n = self.owner
            q = n.mapper.policy_target_to_real(n.contract.default + self.offset)
            return NS(valid=True, stamp=clock.t - (.1 if self.stale else .001),
                      age_ms=np.ones(12), online=np.full(12, not self.offline),
                      error_code=np.zeros(12), temp=np.full(12, 29.),
                      q_real=q, dq_real=np.zeros(12), torque=np.zeros(12),
                      snapshot_seq=np.full(12, int(clock.t * 50)),
                      board_tick_ms=np.full(12, int(clock.t * 1000)),
                      cache_age_ms=1., poll_dt_ms=20., last_update_ts=np.full(12,clock.t-.001))

        def close(self):
            pass

        def pause_async_poll(self):
            pass

        def resume_async_poll(self):
            pass

        def refresh_latest(self):
            return self.get_latest()

        def snapshot_from_payload(self, data):
            return None

    class Imu:
        R_BASE_IMU=np.eye(3)
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
                      rpy_deg=np.zeros(3), gyro_rad_s=np.zeros(3), acc_g=np.array([0.,0.,1.]), mag_uT=np.zeros(3),
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
        if getattr(node,"capture",None) is not None:node.capture.close()
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
    # Zero velocity is step-in-place, matching Isaac; stay in walk.
    node.command_callback(Twist())
    advance(node, clock, 1)
    assert node.mode == 'walk'
    assert node.trial.armed
    np.testing.assert_allclose(node.trial.vector, 0)
    # Stop from a pose different from both default and the old stand target.
    node.motor.offset.fill(.05)
    node.arm_callback(NS(data=False), SetBool.Response())
    advance(node, clock, 1)
    assert node.mode == 'soft_hold'
    assert not node.trial.armed
    assert not any('/api/stop' in url for url, _ in calls)
    expected = node.contract.default + .05 - node.startup_rate * node.contract.policy_dt
    np.testing.assert_allclose(node.stand_target_policy, expected, atol=1e-7)
    node.motor.offset.fill(0.)
    advance(node, clock, 110)
    assert node.mode == 'ready'
    node.command_callback(msg)
    advance(node, clock, 65)
    assert node.mode == 'ready'  # requires a NEW explicit arm


@pytest.mark.parametrize('problem', ['stale', 'offline', 'loop_delay'])
def test_hard_fault_keeps_enable_and_does_not_stop(make_node, problem):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 4)
    count = len(calls)
    if problem == 'loop_delay':
        clock.t += .2
    else:
        setattr(node.motor, problem, True)
    clock.t += .02
    node.control_loop()
    assert not node.faulted
    assert not any('/api/stop' in url for url, _ in calls)
    clock.t += .02
    node.control_loop()
    assert not node.faulted
    assert len(calls) >= count
    assert not any('/api/stop' in url for url, _ in calls)


def test_watchdog_resends_hold_without_disabling(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 3)
    count = len(calls)
    clock.t += .151
    node._watchdog_check()
    assert not node.faulted
    assert not any('/api/stop' in url for url, _ in calls)
    assert len(calls) >= count


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


def test_emergency_path_never_posts_stop(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 3)
    node._emergency_stop('test fault')
    assert not node.faulted
    assert not node.stop_sent
    assert not any('/api/stop' in url for url, _ in calls)


@pytest.mark.parametrize('stop', ['timeout', 'disarm'])
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
    else:
        node.arm_callback(NS(data=False), SetBool.Response())
        advance(node, clock, 1)
    assert node.mode == 'soft_hold'
    assert not node.trial.armed
    assert not any('/api/stop' in url for url, _ in calls)


def test_sim_combo_and_step_stay_in_walk(make_node):
    node, clock, calls = make_node(send=True, stand=False)
    advance(node, clock, 180)
    arm(node)
    combo = Twist()
    combo.linear.x, combo.linear.y, combo.angular.z = .3, .1, .25
    for _ in range(65):
        node.command_callback(combo)
        advance(node, clock, 1)
    assert node.mode == 'walk'
    for _ in range(50):
        node.command_callback(Twist())
        advance(node, clock, 1)
    assert node.mode == 'walk'
    assert node.trial.armed
    np.testing.assert_allclose(node.core.actor.command, 0)
    assert not any('/api/stop' in url for url, _ in calls)


def test_limits_cannot_be_raised_through_parameters(make_node):
    node, _, calls = make_node()
    node.fake_parameters['hardware_torque_limit_nm'] = 17.
    with pytest.raises(RuntimeError, match='hardware_torque_limit_nm'):
        node._validate_deployment_parameters()
    assert not calls


def test_failed_stop_is_not_used_because_enable_is_kept(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 3)
    node._emergency_stop('test fault')
    clock.t += .51
    node._watchdog_check()
    assert not node.faulted
    assert not any('/api/stop' in url for url, _ in calls)


def test_send_is_queued_when_pump_is_running(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 2)
    count = len(calls)
    node._send_pump = NS(is_alive=lambda: True)
    clock.t += .02
    node.control_loop()
    assert not node.faulted
    assert len(calls) == count
    assert node._pending_target is not None
    node._dispatch_pending_send()
    assert len(calls) == count + 1
    assert node._pending_target is None
    assert 'motion_batch_fast' in calls[-1][0]


def test_guard_uses_one_node_estimator(make_node):
    node, _, _ = make_node()
    assert node.leg_odometry is node.walk_guard_odometry


def test_odometry_kinematics_can_be_reused():
    from mydog_policy.rs01_model930_core import Rs01NewMachineLegOdometry
    q = np.array([0.0, -0.33, 1.32] * 4, dtype=np.float32)
    dq = np.zeros(12, dtype=np.float32)
    omega = np.zeros(3, dtype=np.float32)
    first = Rs01NewMachineLegOdometry(strict_diagonal_pairs=True)
    kinematics = first.compute_kinematics(q, dq, omega)
    reused = first.estimate(q, dq, omega, kinematics=kinematics)
    second = Rs01NewMachineLegOdometry(strict_diagonal_pairs=True)
    fresh = second.estimate(q, dq, omega)
    np.testing.assert_allclose(
        reused['base_linear_velocity'], fresh['base_linear_velocity'])
    assert reused['confidence'] == pytest.approx(fresh['confidence'])
    np.testing.assert_array_equal(reused['stance_mask'], fresh['stance_mask'])


def test_telemetry_is_queued_when_pump_is_running(make_node):
    node, clock, calls = make_node(send=True)
    advance(node, clock, 2)
    before = len(node.messages)
    node._telemetry_pump = NS(is_alive=lambda: True)
    clock.t += .02
    node.control_loop()
    assert not node.faulted
    assert node._pending_telemetry is not None
    assert len(node.messages) == before
    node._dispatch_pending_telemetry()
    assert node._pending_telemetry is None
    assert len(node.messages) > before


def test_heading_mismatch_does_not_soft_hold_while_commanded(make_node):
    node, clock, calls = make_node(send=True, stand=False)
    advance(node, clock, 180)
    arm(node)
    msg = Twist()
    msg.linear.x = .1
    for _ in range(65):
        node.command_callback(msg)
        advance(node, clock, 1)
    assert node.mode == 'walk'
    node.heading_consistency_state = dict(
        ready=True, healthy=False, mean_error_rad_s=-0.139,
        abs_error_rad_s=0.139, filtered_yaw_rate_rad_s=0.)
    for _ in range(40):
        node.command_callback(msg)
        advance(node, clock, 1)
    assert node.mode == 'walk'
    assert not node.walk_inhibit_latched
    assert not any('/api/stop' in url for url, _ in calls)
