from types import SimpleNamespace as NS
import numpy as np
import pytest
from mydog_policy.heading_recovery import HeadingRecovery
from mydog_policy.rs01_model23500_core import Rs01Model23500Core
from mydog_policy.observation_pipeline import ObservationPipeline


def monitor(mean=.08345513, absolute=.148998235, healthy=False):
    return dict(ready=True, healthy=healthy, mean_error_rad_s=mean, abs_error_rad_s=absolute)


def test_disagreement_keeps_full_heading_correction():
    recovery = HeadingRecovery()
    pipeline = ObservationPipeline()
    pipeline.diagnostics['observation_temporal_ok'] = True
    for tick in range(501):
        status = recovery.update(tick*.02, monitor(), True)
        assert not pipeline.quality(tick*.02, 1., True, True)
    assert status['heading_recovery_state'] == 'degraded'
    assert status['heading_severe'] is False
    assert status['heading_correction_weight'] == pytest.approx(1.)
    status = recovery.update(10.02, monitor(0., .01, True), True)
    assert status['heading_recovery_state'] == 'normal'
    assert status['heading_correction_weight'] == pytest.approx(1.)


@pytest.mark.parametrize('bad', [monitor(.13), monitor(.01, .31), monitor(float('nan'))])
def test_severe_heading_does_not_stop_or_drop_correction(bad):
    r = HeadingRecovery(); p = ObservationPipeline()
    p.diagnostics['observation_temporal_ok'] = True
    for tick in range(12):
        s = r.update(tick*.02, bad, True)
        assert not p.quality(tick*.02, 1., True, True)
        assert s['heading_correction_weight'] == pytest.approx(1.)
        assert s['heading_severe'] is True
    assert p.diagnostics['obs_quality_stop_reason'] == ''
    assert r.update(.3, bad, False)['heading_recovery_state'] == 'idle'
    assert r.update(.3, bad, False)['heading_correction_weight'] == 1.


def test_heading_channels_keep_full_error_without_resetting_target():
    actor = object.__new__(Rs01Model23500Core)
    actor.heading = .24
    actor.command = np.array([.3, 0., 0.])  # straight walk: correction active
    assert actor._direction_heading_error(.083) == .083
    actor.heading_correction_weight = 1.
    assert actor._direction_heading_error(.083) == .083
    assert actor.heading == .24


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance
from mydog_policy.observation_pipeline import ObservationPipeline


@pytest.mark.parametrize('make_node', ['B23500', 'capture23500'], indirect=True)
def test_node_heading_disagreement_corrects_and_keeps_walking(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    n.observation_pipeline = ObservationPipeline()
    n.observation_pipeline.diagnostics['observation_temporal_ok'] = True
    n.mode = 'walk'; n.trial.arm(clock.t)
    n.heading_consistency_state = monitor()
    for tick in range(51):
        assert not n._update_walk_inhibitors(clock.t+tick*.02, {'confidence':1.}, n.contract.default)
    assert n.trial.armed and n.mode == 'walk'
    assert n.core.actor.heading_correction_weight == pytest.approx(1.)
    actor = n.core.actor
    actor.reset(.083)
    result = actor.tick(n.contract.default, np.zeros(12), np.zeros(3), [0,0,-1], 0., [.4,0,0])
    assert result['observation'][50] == pytest.approx(np.sin(.083))
    assert result['observation'][57] == pytest.approx(.4*n.contract.command_scale[0])
    n.heading_consistency_state = monitor(.13)
    assert not n._update_walk_inhibitors(clock.t+1.1, {'confidence':1.}, n.contract.default)
    assert not n._update_walk_inhibitors(clock.t+1.32, {'confidence':1.}, n.contract.default)
    assert n.mode == 'walk' and n.trial.armed
    assert n.core.actor.heading_correction_weight == pytest.approx(1.)
    assert not calls
