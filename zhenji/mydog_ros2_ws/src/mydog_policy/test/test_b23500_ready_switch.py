from types import SimpleNamespace as NS
import numpy as np
import pytest
from mydog_policy.observation_pipeline import ObservationPipeline
from mydog_policy.heading_reference import GyroHeadingReference


def test_support_recovery_is_bounded_and_does_not_hide_other_faults():
    p=ObservationPipeline();p.diagnostics['observation_temporal_ok']=True
    for t in (0.,.2,.4,.58):
        assert not p.quality(t,0.,True,True,odometry_support_usable=False,odometry_timeout_sec=.6)
        assert not p.diagnostics['obs_quality_ok']
    assert p.quality(.61,0.,True,True,odometry_support_usable=False,odometry_timeout_sec=.6)
    assert p.diagnostics['obs_quality_stop_reason']=='odometry'
    assert not p.quality(.62,.2,True,True,odometry_support_usable=True,odometry_timeout_sec=.6)
    assert p.diagnostics['obs_quality_state']=='degraded'
    p.diagnostics['observation_temporal_ok']=False
    assert not p.quality(1.,.8,True,True,odometry_support_usable=True,odometry_timeout_sec=.6)
    assert p.quality(1.21,.8,True,True,odometry_support_usable=True,odometry_timeout_sec=.6)
    assert p.diagnostics['obs_quality_stop_reason']=='timing'


def test_soft_hold_does_not_learn_a_disturbed_magnetic_baseline():
    r=GyroHeadingReference()
    r.update(0.,0.,0.,80.,False,stationary=True)
    for i in range(1,201):
        r.update(i*.02,1.,0.,180.,False,stationary=False)
    assert r.baseline_ut==80.
    assert r.heading==pytest.approx(0.)
    assert r.diagnostics['heading_source']=='gyro_integrated'


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance
from std_srvs.srv import SetBool


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_ready_recovers_using_active_heading_without_erasing_bad_monitor(make_node,monkeypatch):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,180)
    bad=n.heading_consistency.snapshot().copy();bad.update(ready=True,healthy=False)
    monkeypatch.setattr(n.heading_consistency,'update',lambda *a:bad.copy())
    n._enter_soft_hold('operator disarm',clock.t,n.contract.default)
    advance(n,clock,150)
    assert n.mode=='ready' and n.walk_start_stable
    assert not n.heading_consistency_state['healthy']
    assert n._heading_ready_for_stand()
    assert not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_recalibration_is_transactional_and_does_not_arm_or_rebase_to_magnetic_yaw(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,180)
    n.imu_calibrated=True;n.gyro_bias_rad_s=np.array([.001,.002,.003])
    before=n.gyro_bias_rad_s.copy()
    assert n.calibrate_callback(NS(data=True),SetBool.Response()).success
    assert n.imu_calibrated and n.calibration_requested
    assert not n.arm_callback(NS(data=True),SetBool.Response()).success
    assert not n._walk_entry_allowed()
    assert n.calibrate_callback(NS(data=False),SetBool.Response()).success
    assert n.imu_calibrated and not n.calibration_requested
    np.testing.assert_array_equal(n.gyro_bias_rad_s,before)
    n.heading_reference.heading=.8
    n._on_gyro_calibration_complete(clock.t,-.9,n.contract.default)
    assert n.core.actor.heading==pytest.approx(.8)
    assert n.heading_reference.heading==pytest.approx(.8)
    assert n.calibration_generation==1
    assert not n.trial.armed and not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_march_preserves_actual_heading_error_and_actor_velocity_input(make_node):
    n,clock,calls=make_node(send=False,stand=False);a=n.core.actor;c=n.contract
    a.reset(0.,q_policy=c.default)
    r=a.tick(c.default,np.zeros(12),np.zeros(3),[0,0,-1],.25,[0,0,0])
    assert r['observation'][50]==pytest.approx(np.sin(-.25))
    assert a.heading_correction_active()
    assert not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_joint_position_and_timing_faults_still_prevent_ready(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,180)
    n.motor.offset[2]=-.21
    n._enter_soft_hold('operator disarm',clock.t,n.contract.default)
    advance(n,clock,180)
    assert n.mode=='soft_hold'
    d=n._extra_status()
    assert d['stand_max_error_rad']==pytest.approx(.21)
    assert d['stand_worst_joint']==n.contract.joint_names[2]
    assert d['stand_not_at_target_joints']==[n.contract.joint_names[2]]
    assert not calls
