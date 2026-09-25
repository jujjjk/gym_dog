import json
from types import SimpleNamespace as NS
import pytest
from test_rs01_model6850_node_offline import make_node
from mydog_policy.rs01_model23500_node import Rs01Model23500Node
from mydog_policy.rs01_model6850_guard import TrialCommand
from mydog_policy import rs01_model6850_command as command


@pytest.mark.parametrize('make_node', ['B23500', 'capture23500'], indirect=True)
@pytest.mark.parametrize('continuous', [False, True])
def test_lease_and_deadman(make_node, monkeypatch, continuous):
    original_init = Rs01Model23500Node.__init__
    def init(self):
        declare = self.declare_parameter
        self.declare_parameter = lambda name, default: declare(
            name, continuous if name == 'continuous_commands' else default)
        original_init(self)
    monkeypatch.setattr(Rs01Model23500Node, '__init__', init)
    n, clock, calls = make_node(send=False, stand=False)
    assert n._extra_status()['continuous_commands'] == continuous
    assert TrialCommand().lease_sec == 120.
    assert not n.trial.armed and not calls
    n.trial.arm(clock.t)
    for dt in [0., 5., 180., 181., 3600., 86400.]:
        n.trial.receive([.2, 0, 0], clock.t+dt)
        if not continuous and dt > 180.:
            assert not n.trial.active(clock.t+dt)
            break
        assert n.trial.active(clock.t+dt)
    if continuous:
        assert not n.trial.active(clock.t+86400.+.36)
        assert not n.trial.armed
        n.trial.arm(clock.t+86401.)
        n.trial.receive([.2, 0, 0], clock.t+86401.)
        n.trial.disarm('operator')
        assert not n.trial.active(clock.t+86401.)


@pytest.mark.parametrize('case', ['ctrl_c', 'stale', 'fault', 'old_controller', 'timing_not_ready'])
def test_continuous_command_cleanup(monkeypatch, case):
    clock = NS(t=0.)
    state = dict(armed=False, callback=None, sent=[], arm_calls=[], destroyed=False)
    def publish(msg):
        state['sent'].append((clock.t, msg.linear.x))
    def call_async(req):
        state['arm_calls'].append(req.data)
        state['armed'] = req.data
        return NS(done=lambda: True, result=lambda: NS(success=True, message='ok'))
    client = NS(wait_for_service=lambda **kw: True, call_async=call_async,
                service_is_ready=lambda: True)
    def subscribe(ty, topic, cb, depth):
        state['callback'] = cb
    node = NS(create_publisher=lambda *a: NS(publish=publish, get_subscription_count=lambda: 1),
              create_client=lambda *a: client, create_subscription=subscribe,
              destroy_node=lambda: state.update(destroyed=True))
    def spin(node, timeout_sec):
        clock.t += .05
        if state['armed'] and case == 'ctrl_c' and clock.t > 181.:
            raise KeyboardInterrupt
        if state['armed'] and case == 'stale':
            return
        state['callback'](NS(data=json.dumps(dict(
            mode='fault' if state['armed'] and case == 'fault' else 'ready',
            walk_start_stable=True, send=True, stand_only=False,
            continuous_commands=case != 'old_controller',
            observation_temporal_ok=case != 'timing_not_ready'))))
    monkeypatch.setattr(command.time, 'monotonic', lambda: clock.t)
    monkeypatch.setattr(command.rclpy, 'init', lambda **kw: state.update(init=kw))
    monkeypatch.setattr(command.rclpy, 'create_node', lambda *a: node)
    monkeypatch.setattr(command.rclpy, 'spin_once', spin)
    monkeypatch.setattr(command.rclpy, 'spin_until_future_complete', lambda *a, **kw: None)
    monkeypatch.setattr(command.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(command.rclpy, 'shutdown', lambda: None)
    args = ['--continuous', '--vx', '.4', '--namespace', '/mydog/model23500']
    if case == 'ctrl_c':
        command.main(args, speed_caps=(.4,.3,.6), allow_continuous=True)
        assert max(t for t, vx in state['sent'] if vx == .4) > 180.
    else:
        with pytest.raises(RuntimeError):
            command.main(args, speed_caps=(.4,.3,.6), allow_continuous=True)
    assert state['arm_calls'][-1] is False
    assert state['destroyed']
    assert all(vx == 0 for _, vx in state['sent'][-3:])
    assert state['init']['signal_handler_options'] == command.SignalHandlerOptions.NO
    if case in ('old_controller', 'timing_not_ready'):
        assert True not in state['arm_calls']


def test_duration_flags_are_exclusive():
    with pytest.raises(SystemExit):
        command.main(['--continuous', '--seconds', '5'], allow_continuous=True)
