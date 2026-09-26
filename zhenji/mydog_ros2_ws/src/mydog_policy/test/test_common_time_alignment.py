from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
from mydog_policy.observation_time_alignment import BoardClock, CommonTimeAlignment, interpolate


def test_clock_wrap_reset_drift_and_repeated_snapshot():
    c = BoardClock()
    c.update(2**32-10, 100.)
    assert c.update(10, 100.02) == pytest.approx((2**32+10)/1000.)
    count = len(c.points)
    c.update(10, 100.03)
    assert len(c.points) == count
    with pytest.raises(ValueError, match='reset'):
        c.update(0, 100.04)
    for i in range(251): c.update(i*20, 200.+i*.02*1.0002)
    assert c.ready and c.slope == pytest.approx(1.0002, abs=1e-8)
    assert math.isfinite(c.residual_ms) and c.residual_ms < 1.


def test_outlier_and_command_jitter_do_not_drop_a_ready_clock():
    c = BoardClock()
    for i in range(40):
        c.update(i*20, 50.+i*.02)
    assert c.ready
    # 8 ms early, still strictly increasing and inside the 100 ms reset.
    c.update(40*20, 50.+40*.02-.008)
    for i in range(41, 80):
        c.update(i*20, 50.+i*.02+(.008 if i % 5 == 0 else 0.))
    assert c.ready
    assert math.isfinite(c.residual_ms) and c.residual_ms <= BoardClock.hold_residual_ms


def test_diverging_host_and_tick_is_not_ready():
    c = BoardClock()
    for i in range(40):
        # Tick advances 20 ms while the host stamp advances 50 ms.
        c.update(i*20, 80.+i*.05)
    assert not c.ready
    assert c.residual_ms > BoardClock.hold_residual_ms


def test_interpolation_no_extrapolation_gap_and_quaternion_sign():
    v, gap = interpolate([(1., [1., 2.]), (1.02, [3., 4.])], 1.01)
    np.testing.assert_allclose(v, [2, 3]); assert gap == pytest.approx(20.)
    with pytest.raises(ValueError): interpolate([(1., [1.]), (1.02, [2.])], 1.03)
    with pytest.raises(ValueError): interpolate([(1., [1.]), (1.2, [2.])], 1.1)
    q, _ = interpolate([(1., [1,0,0,0]), (1.02, [-1,0,0,0])], 1.01, quaternion=True)
    np.testing.assert_allclose(q, [1,0,0,0])
    q, _ = interpolate([(1., [1,0,0,0]), (1.02, [0,0,0,1])], 1.01, quaternion=True)
    np.testing.assert_allclose(q, [2**-.5,0,0,2**-.5])


def feed(a, i, offline=False):
    now=10.+i*.02
    ages=np.arange(12)%6
    stamps=now-ages/1000.
    m=NS(state_epoch_monotonic=now, board_tick_ms=np.r_[np.full(6,i*20),np.full(6,1000+i*20)],
        age_ms=ages, online=np.ones(12,dtype=bool), q_real=stamps*2.,dq_real=np.full(12,2.))
    if offline: m.online[0]=False
    frames=dict(gyro=[(now-.06,[0,0,now-.06]),(now-.03,[0,0,now-.03]),(now,[0,0,now])],
        quat=[(now-.06,[1,0,0,0]),(now-.03,[1,0,0,0]),(now,[1,0,0,0])])
    return a.update(m,frames,now),m,frames,now


def test_common_time_values_not_just_changed_labels():
    a=CommonTimeAlignment(np.eye(3))
    for i in range(20): sample,m,f,now=feed(a,i)
    assert sample is not None
    np.testing.assert_allclose(sample['q_real'],(now-.05)*2.,atol=1e-8)
    assert not np.array_equal(sample['q_real'],m.q_real)
    assert sample['gyro'][2] == pytest.approx(now-.05)
    assert a.diagnostics['observation_alignment_verified']
    assert not a.diagnostics['acquisition_sync_verified']
    assert a.diagnostics['observation_bracket_max_ms'] > 0
    sample,_,_,_=feed(a,20,offline=True)
    assert sample is None and 0 in a.diagnostics['unaligned_motor_indices']


def test_imu_gap_and_missing_monotonic_fail_closed():
    a=CommonTimeAlignment(np.eye(3))
    for i in range(20): _,m,f,now=feed(a,i)
    m.state_epoch_monotonic=None
    assert a.update(m,f,now+.02) is None
    assert 'epoch' in a.diagnostics['observation_alignment_reason']


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_actor_delayed_inputs_keep_live_pd_inputs(make_node,monkeypatch):
    n,clock,calls=make_node(send=False,stand=False)
    c=n.core
    q=n.contract.default.copy()
    q_real=c.mapper.policy_target_to_real(q)
    c.common_time_required=True
    c.aligned_sample=dict(timestamp=clock.t-.04,q_real=q_real,dq_real=np.zeros(12),
        gyro=np.array([.1,.2,.3]),gravity=np.array([0.,0.,-1.]),yaw=.2)
    c.aligned_gyro_bias=np.array([.01,.02,.03])
    seen={}
    tick=c.actor.tick
    def spy(qp,dqp,gyro,gravity,yaw,*args,**kwargs):
        seen.update(q=qp.copy(),gyro=gyro.copy(),yaw=yaw,kinematics=kwargs.get('kinematics'))
        return tick(qp,dqp,gyro,gravity,yaw,*args,**kwargs)
    monkeypatch.setattr(c.actor,'tick',spy)
    live=q+.001
    obs=c.build_observation(clock.t,np.zeros(3),np.zeros(3),[0,0,-1],[.1,0,0],live,np.zeros(12),.5)
    np.testing.assert_allclose(seen['q'],q,atol=1e-7)
    np.testing.assert_allclose(seen['gyro'],[.09,.18,.27])
    assert seen['yaw']==.2 and seen['kinematics'] is None
    original=c.limiter.pd_equivalent_peak_limit
    def pd(target,qp,dqp,limits):
        np.testing.assert_array_equal(qp,live)
        return original(target,qp,dqp,limits)
    monkeypatch.setattr(c.limiter,'pd_equivalent_peak_limit',pd)
    c.step(obs,live,np.zeros(12),np.full(12,14.))
    c.aligned_sample=None
    with pytest.raises(RuntimeError,match='common-time'):
        c.build_observation(clock.t,np.zeros(3),np.zeros(3),[0,0,-1],[.1,0,0],live,np.zeros(12),.5)
    assert not calls


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_common_time_node_warmup_capture_and_missing_bracket_hold(make_node,monkeypatch):
    import csv
    from mydog_policy import rs01_model23500_node as module
    from test_rs01_model6850_node_offline import advance
    cls=module.Rs01Model23500Node; original=cls.__init__
    def init(self):
        declare=self.declare_parameter
        self.declare_parameter=lambda name,default:declare(name,True if name=='observation_pipeline_enabled' else default)
        original(self)
    monkeypatch.setattr(cls,'__init__',init)
    n,clock,calls=make_node(send=False,stand=False)
    monkeypatch.setattr(module,'time',NS(time=lambda:clock.t,monotonic=lambda:clock.t))
    latest=n.motor.get_latest
    def motor():
        m=latest();m.state_epoch_monotonic=clock.t;return m
    n.motor.get_latest=motor
    n.imu.get_frame_history=lambda:dict(
        gyro=[(clock.t-d,[0,0,0]) for d in [.06,.03,0]],
        quat=[(clock.t-d,[1,0,0,0]) for d in [.06,.03,0]])
    advance(n,clock,60)
    assert not n.faulted
    assert n.observation_pipeline.diagnostics['observation_alignment_target_ok']
    assert n.observation_pipeline.diagnostics['aligned_sensor_skew_ms'] == 0.
    assert 'raw_reception_skew_ms' in n.observation_pipeline.diagnostics
    assert n.core.common_time_required
    n.mode='walk';n.trial.arm(clock.t);n.trial.receive([.1,0,0],clock.t)
    n.imu.get_frame_history=lambda:{}
    # 50 ms lookback + 140 ms hold: ages 70..130 ms reuse the last sample.
    advance(n,clock,4)
    assert n.mode=='walk' and n.trial.armed and not n.faulted
    advance(n,clock,1)
    assert n.mode=='soft_hold' and not n.trial.armed and not n.faulted
    if getattr(n,'capture',None) is not None:
        n.capture.close()
        assert n.capture.error==''
        with (n.capture.path/'cycles.csv').open() as f: rows=list(csv.DictReader(f))
        assert rows and 'aligned_q_FR_hip' in rows[-1]
        assert any(r['observation_alignment_verified']=='True' for r in rows)
    assert not calls
