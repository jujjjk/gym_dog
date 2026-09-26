"""ROS constructor/control tests with mocked HTTP, IMU and motor feedback."""
import numpy as np
import pytest
from types import SimpleNamespace as NS
pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node,advance
from mydog_policy.rs01_model18000_node import timing_window_ready,Rs01Model18000Node
from mydog_policy.rs01_model6850_node import Rs01Model6850Node

def test_reuses_upstream_fixed_scheduler_and_send_pump():
    assert Rs01Model18000Node._control_thread_loop is Rs01Model6850Node._control_thread_loop
    assert Rs01Model18000Node._send_pump_loop is Rs01Model6850Node._send_pump_loop

@pytest.mark.parametrize('make_node',['B18000'],indirect=True)
def test_b_send_cadence_and_freshness(make_node):
    n,clock,calls=make_node(send=False,stand=True)
    n._b_loop_intervals.extend([.02]*50);n._b_send_intervals.extend([.02]*50)
    n.enable_send=True;n._b_last_success=clock.t
    try:
        assert n._b_timing_ready()
        n._b_last_success=clock.t-.1
        assert not n._b_timing_ready()
        n._b_last_success=clock.t
        n._b_send_intervals.clear();n._b_send_intervals.extend([.04]*50)
        assert not n._b_timing_ready()
    finally:n.enable_send=False
    assert not calls

def test_timing_window_rejects_25hz_and_stalls():
    assert timing_window_ready([.02]*50)
    assert not timing_window_ready([.04]*50)
    assert not timing_window_ready([.02]*39)
    assert not timing_window_ready([.02]*49+[.081])
    assert not timing_window_ready([.02]*49+[float('nan')])
    # Measured Jetson send windows: median 19.6 ms with p95 30.17 ms, and
    # median 20.44 ms with one 40.17 ms first-walk-tick sample.
    assert timing_window_ready([.0196]*46+[.03017]*4)
    assert timing_window_ready([.02044]*49+[.04017])
    assert not timing_window_ready([.0196]*46+[.036]*4)
    assert not timing_window_ready([.02]*49+[.051])
    assert not timing_window_ready([.02]*49+[.060])

@pytest.mark.parametrize('make_node',['B18000'],indirect=True)
def test_b_constructor_dry_run_and_timing_gate(make_node):
    n,clock,calls=make_node(send=False,stand=True)
    advance(n,clock,50)
    assert n.model_filename=='B18000.onnx'
    assert n.contract.policy_dt==.02
    assert not calls
    assert timing_window_ready(n._b_loop_intervals)
    n._b_loop_intervals.clear();n._b_loop_intervals.extend([.04]*50)
    response=n.arm_callback(NS(data=True),NS(success=True,message=''))
    assert not response.success
    assert 'calibration' in response.message
    n.imu_calibrated = True
    response = n.arm_callback(NS(data=True), NS(success=True, message=''))
    assert not response.success
    assert '50Hz' in response.message
    assert n.arm_callback(NS(data=False),NS(success=False,message='')).success

@pytest.mark.parametrize('make_node',['B18000'],indirect=True)
def test_b_slow_loop_disarms_before_inference(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,50)
    n.trial.arm(clock.t);n.trial.receive([.1,0,0],clock.t)
    n.mode='walk';n._previous_control=clock.t-.08
    n.control_loop()
    assert not n.trial.armed
    assert n.core.actor.steps==0
    assert not calls
