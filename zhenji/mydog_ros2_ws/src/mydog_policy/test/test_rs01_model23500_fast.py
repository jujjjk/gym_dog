import numpy as np
import pytest
from mydog_policy.rs01_model6850_guard import TrialCommand
from test_rs01_model6850_node_offline import make_node
from mydog_policy.rs01_model23500_node import Rs01Model23500Node
from mydog_policy import rs01_model23500_command as command

@pytest.mark.parametrize('make_node', ['B23500', 'capture23500'], indirect=True)
@pytest.mark.parametrize('fast', [False, True])
def test_fast_opt_in_retains_deadman_and_default(make_node, monkeypatch, fast):
    original_init = Rs01Model23500Node.__init__
    def init(self):
        original_declare = self.declare_parameter
        self.declare_parameter = lambda name, default: original_declare(name, fast if name == 'fast_commands' else default)
        original_init(self)
    monkeypatch.setattr(Rs01Model23500Node, '__init__', init)
    n, clock, calls = make_node(send=False, stand=False)
    assert n.fast_commands == fast
    assert np.array_equal(n.trial.caps, [.4,.3,.6] if fast else [.3,.2,.3])
    assert np.array_equal(TrialCommand().caps, [.3,.2,.3])
    assert not n.trial.armed and not calls
    n.trial.arm(clock.t)
    assert n.trial.receive([.4,.3,.6], clock.t) == fast
    if fast:
        assert not n.trial.active(clock.t+.36)
        n.trial.arm(clock.t)
        assert not n.trial.receive([.401,0,0], clock.t)
        assert not n.trial.armed
    assert n._extra_status()['fast_commands'] == fast

@pytest.mark.parametrize('fast', [False, True])
def test_command_opt_in(monkeypatch, fast):
    got={}
    def capture(args, **kwargs):
        got.update(args=args, **kwargs)
    monkeypatch.setattr(command, 'command_main', capture)
    command.main((['--fast'] if fast else [])+['--vx','.4','--seconds','5'])
    assert '--fast' not in got['args']
    assert got['args'][-1] == '/mydog/model23500'
    assert got['speed_caps'] == ((.4,.3,.6) if fast else (.3,.2,.3))
