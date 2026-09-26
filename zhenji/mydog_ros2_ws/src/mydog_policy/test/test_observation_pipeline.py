from copy import deepcopy
from types import SimpleNamespace as NS
import csv
import json
import numpy as np
import pytest
from mydog_policy.observation_pipeline import ObservationPipeline, mount_rotation
from mydog_policy.observation_audit import propose_mount


def sample(t=10.):
    m=NS(stamp=t,cache_age_ms=2.,age_ms=np.full(12,3.),dq_real=np.zeros(12))
    i=NS(stamp=t-.005,valid=True,gyro_rad_s=np.array([1.,0.,0.]),
         projected_gravity=np.array([0.,0.,-1.]),rpy_deg=np.zeros(3),
         quat_wxyz=np.array([1.,0.,0.,0.]))
    return m,i


def test_nearest_imu_all_motor_ages_and_no_acquisition_claim():
    m,i=sample(); latest=deepcopy(i); latest.stamp=9.999
    p=ObservationPipeline(timing_mode='strict_host_alignment'); unchanged,result=p.process(m,latest,[i],10.,1.)
    assert unchanged is m and result.stamp == i.stamp
    d=p.diagnostics
    assert d['observation_temporal_ok'] and d['imu_history_selected']
    assert d['aligned_sensor_skew_ms'] == pytest.approx(0.)
    assert d['motor_acquisition_timestamp'] is None
    assert d['imu_acquisition_timestamp'] is None
    assert not d['acquisition_sync_verified']
    m.age_ms[0]=30.
    p.process(m,latest,[i],10.,1.02)
    assert not p.diagnostics['observation_temporal_ok']


@pytest.mark.parametrize('bad', ['old','future','internal_skew','regression'])
def test_timestamp_failure_is_not_hidden_by_matching(bad):
    m,i=sample(); p=ObservationPipeline(timing_mode='strict_host_alignment');p.process(m,i,[],10.,1.)
    if bad == 'old': i.stamp=9.90
    if bad == 'future': i.stamp=10.01
    if bad == 'internal_skew': i.frame_receive_wall=[9.995,9.970,9.994]
    if bad == 'regression': i.stamp-=.001
    p.process(m,i,[],10.,1.02)
    assert not p.diagnostics['observation_temporal_ok']


def test_mount_rotation_is_consistent_and_inputs_remain_raw():
    m,i=sample();p=ObservationPipeline(mount=(10.,0.,90.))
    _,result=p.process(m,i,[],10.,1.)
    np.testing.assert_allclose(result.gyro_rad_s,p.rotation@i.gyro_rad_s)
    np.testing.assert_allclose(result.projected_gravity,p.rotation@i.projected_gravity)
    np.testing.assert_allclose(mount_rotation(*result.rpy_deg),p.rotation.T,atol=1e-7)
    np.testing.assert_array_equal(i.gyro_rad_s,[1,0,0])
    assert not p.diagnostics['filter_applied_to_policy']
    m.dq_real[:]=1.;i.gyro_rad_s[:]=2.;m.stamp+=.02;i.stamp+=.02
    _,result=p.process(m,i,[],10.02,1.02)
    assert np.all(p.diagnostics['preview_filtered_dq'] < 1.)
    np.testing.assert_array_equal(m.dq_real,np.ones(12))
    np.testing.assert_allclose(result.gyro_rad_s,p.rotation@i.gyro_rad_s)


def test_quality_transient_recovery_and_persistent_stop():
    m,i=sample();p=ObservationPipeline();p.process(m,i,[],10.,1.)
    assert not p.quality(1.,.1,True,True)
    assert not p.quality(1.08,.1,True,True)
    assert not p.quality(1.10,1.,True,True)
    assert not p.quality(1.2,.1,True,True)
    assert p.quality(1.41,.1,True,True)
    assert not p.quality(2.,.1,True,False)


def test_alternating_faults_cannot_extend_each_others_timer():
    m,i=sample();p=ObservationPipeline();p.process(m,i,[],10.,1.)
    # One-second union of bad states, but no single cause lasts 200ms.
    for step in range(60):
        cause=(step//5)%3
        p.diagnostics['observation_temporal_ok']=cause != 0
        assert not p.quality(1.+step*.02,.1 if cause==1 else 1.,cause!=2,True)
    assert p.diagnostics['obs_quality_stop_reason']==''


@pytest.mark.parametrize('cause',['timing','odometry','heading'])
def test_each_persistent_fault_still_stops_independently(cause):
    m,i=sample();p=ObservationPipeline();p.process(m,i,[],10.,1.)
    p.diagnostics['observation_temporal_ok']=cause!='timing'
    conf=.1 if cause=='odometry' else 1.
    heading=cause!='heading'
    assert not p.quality(1.,conf,heading,True)
    assert p.quality(1.21,conf,heading,True)
    assert p.diagnostics['obs_quality_stop_reason']==cause
    assert p.diagnostics['obs_quality_'+cause+'_bad_sec']==pytest.approx(.21)
    p.quality(1.22,conf,heading,False)
    assert all(since is None for since in p.quality_since.values())


def test_level_mount_proposal_and_motion_rejection():
    rotation=mount_rotation(2.,-1.,0.)
    gravity=rotation.T@np.array([0.,0.,-1.])
    rows=[]
    for index in range(501):
        row=dict(timestamp_policy=str(index*.02),mode='ready',observation_temporal_ok='True')
        for axis,value in zip('xyz',gravity): row['raw_gravity_'+axis]=str(value)
        for axis in 'xyz': row['raw_gyro_'+axis]='0'
        for leg in ['FR','FL','RL','RR']:
            for joint in ['hip','thigh','calf']: row['q_'+leg+'_'+joint]='0'
        rows.append(row)
    angles,proposal=propose_mount(rows)
    np.testing.assert_allclose(proposal@gravity,[0,0,-1],atol=1e-10)
    rows[-1]['q_FR_hip']='.1'
    with pytest.raises(ValueError,match='Motion'):
        propose_mount(rows)


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance
from mydog_policy import rs01_model23500_node as node_module


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_pipeline_node_and_capture(make_node,monkeypatch):
    cls=node_module.Rs01Model23500Node; original=cls.__init__
    def init(self):
        declare=self.declare_parameter
        self.declare_parameter=lambda name,default: declare(name,True if name=='observation_pipeline_enabled' else ('reception' if name=='observation_timing_mode' else default))
        original(self)
    monkeypatch.setattr(cls,'__init__',init)
    n,clock,calls=make_node(send=False,stand=False)
    monkeypatch.setattr(node_module,'time',NS(time=lambda:clock.t,monotonic=lambda:clock.t))
    advance(n,clock,10)
    assert not n.faulted
    assert n.observation_pipeline.diagnostics['observation_temporal_ok']
    n.observation_pipeline.diagnostics['observation_temporal_ok']=False
    response=n.arm_callback(NS(data=True),NS(success=True,message=''))
    assert not response.success and 'Observation timing' in response.message
    n.mode='walk';n.trial.arm(clock.t)
    n.trial.receive([.1,0,0],clock.t)
    q=n.contract.default.copy()
    assert not n._update_walk_inhibitors(clock.t,{'confidence':0.},q)
    assert n._update_walk_inhibitors(clock.t+.21,{'confidence':0.},q)
    assert n.mode=='soft_hold' and not n.trial.armed
    if getattr(n,'capture',None) is not None:
        n.capture.close()
        with (n.capture.path/'cycles.csv').open() as f: rows=list(csv.DictReader(f))
        assert rows and 'raw_gyro_x' in rows[0] and 'guard_raw_velocity_x' in rows[0]
        assert rows[0]['motor_acquisition_timestamp']==''
        assert n.capture.error == ''
        meta=json.loads((n.capture.path/'metadata.json').read_text())
        assert meta['observation_pipeline']['acquisition_sync_verified'] is False
    assert not calls


def test_reception_mode_uses_latest_and_keeps_skew_diagnostic():
    m,i=sample()
    i.stamp=9.970
    i.frame_receive_wall=(9.995,9.990,9.970)
    old=deepcopy(i);old.stamp=9.960
    p=ObservationPipeline()
    _, selected=p.process(m,i,[old],10.,1.)
    assert selected.stamp==i.stamp
    assert p.diagnostics['observation_reception_ok']
    assert p.diagnostics['observation_temporal_ok']
    assert not p.diagnostics['observation_alignment_target_ok']
    assert not p.diagnostics['acquisition_sync_verified']
    strict=ObservationPipeline(timing_mode='strict_host_alignment')
    strict.process(m,i,[old],10.,1.)
    assert not strict.diagnostics['observation_temporal_ok']
    # The deployed reception mode limit is 60 ms.
    p.process(m,i,[],10.031,1.031)
    assert not p.diagnostics['observation_reception_ok']
