"""No real device access. Contract and real-node tests use mocked inputs."""
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import onnxruntime as ort
from mydog_policy.rs01_model23500_core import Model23500Contract, Rs01Model23500Core
from mydog_policy.rs01_model6850_core import Rs01Model6850Core
from mydog_policy.rs01_model18000_core import Rs01Model18000Core


def test_locked_artifact_and_original_direction_controller():
    path = Path(__file__).resolve().parents[1]/'resource/B23500.onnx'
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
    c = Model23500Contract.from_onnx_session(s, path)
    with pytest.raises(RuntimeError, match='override'):
        Model23500Contract.from_onnx_session(s, path, '0'*64)
    actor = Rs01Model23500Core(s, c)
    cfg = c.raw['v13']['commands']
    cmd = np.array([.2, 0., 0.])
    actual = actor._mix_direction_command(cmd, .1, False, cfg)
    assert np.array_equal(actual, Rs01Model6850Core._mix_direction_command(actor, cmd, .1, False, cfg))
    assert not np.allclose(actual, Rs01Model18000Core._mix_direction_command(actor, cmd, .1, False, cfg))
    r = actor.tick(c.default, np.zeros(12), np.zeros(3), [0, 0, -1], 0., cmd)
    assert r['observation'].shape == (61,)
    assert r['confidence'] > 0
    assert r['phase'] == 0
    assert np.isfinite(r['target_policy']).all()
    assert not Rs01Model18000Core.initialize_odometry_on_first_tick
    assert Rs01Model18000Core.target_reached_atol == 0


def test_roundoff_arrival_resets_rate():
    # The tolerance is numerical, not a change to physical limits.
    assert Rs01Model23500Core.target_reached_atol == 1e-12


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance
from mydog_policy.rs01_model23500_node import Rs01Model23500Node
from mydog_policy.rs01_model18000_node import Rs01Model18000Node


@pytest.mark.parametrize('make_node', ['B23500', 'capture23500'], indirect=True)
def test_v22_dry_run_calibration_timing_and_namespace(make_node):
    n, clock, calls = make_node(send=False, stand=False)
    advance(n, clock, 50)
    assert n.model_filename == 'B23500.onnx'
    assert n.topic_namespace == '/mydog/model23500'
    assert n.core.actor.initialize_odometry_on_first_tick
    assert not n.trial.armed
    assert not calls
    r = n.arm_callback(NS(data=True), NS(success=True, message=''))
    assert not r.success and 'calibration' in r.message
    n.imu_calibrated = True
    n._b_loop_intervals.clear()
    n._b_loop_intervals.extend([.04]*50)
    r = n.arm_callback(NS(data=True), NS(success=True, message=''))
    assert not r.success and '50Hz' in r.message
    n.trial.arm(clock.t)
    n.trial.receive([.1, 0., 0.], clock.t)
    n.mode = 'walk'
    n._previous_control = clock.t-.08
    n.control_loop()
    assert not n.trial.armed and not calls
    status = n._extra_status()
    assert status['model'] == 'B23500'
    assert 'lateral_correction_mps' not in status
    assert not status['hardware_motion_validated']
    assert Rs01Model23500Node._dispatch_pending_send is Rs01Model18000Node._dispatch_pending_send
