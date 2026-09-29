from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from mydog_policy.motion_session import CommandRamp,CommandOwner,actions,ready
from mydog_policy.support_plane_odometry import SupportPlaneOdometry
from mydog_policy.rs01_model930_core import Rs01NewMachineLegOdometry
from mydog_policy.odometry_quality import support_quality


def test_tilted_level_support_is_not_rejected_by_body_z(monkeypatch):
    theta=.15;c=np.cos(theta);s=np.sin(theta)
    rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    world=np.array([[.216,.15,-.291],[.216,-.15,-.291],[-.216,.15,-.291],[-.216,-.15,-.291]])
    body=world@rotation;gravity=rotation.T@np.array([0,0,-1.])
    omega=np.array([0,1.5,0])
    kin=dict(foot_position=body,foot_velocity=-np.cross(omega,body),velocity_by_foot=np.zeros((4,3)))
    old=Rs01NewMachineLegOdometry(strict_diagonal_pairs=True)
    assert not old.estimate(None,None,None,kinematics=kin)['legal_diagonal_support']
    new=SupportPlaneOdometry.from_estimator(old)
    monkeypatch.setattr(new,'compute_kinematics',lambda *a:kin)
    od=new.estimate_aligned(None,None,omega,gravity)
    assert support_quality(od,new)['obs_odometry_support_usable']
    np.testing.assert_allclose(od['base_height_proxy'],.307)
    np.testing.assert_array_equal(od['foot_position'],body)
    np.testing.assert_allclose(od['support_vertical_speed'],0.)
    assert new.height_margin==old.height_margin==.03
    with pytest.raises(ValueError):new.estimate_aligned(None,None,omega,[0,0,0])


def test_command_ramp_reverses_without_step_and_has_no_duration_limit():
    r=CommandRamp([.4,.3,.6]);sequence=[]
    for i in range(101):sequence.append(r.update([.4,0,0],i*.02))
    for i in range(101,202):sequence.append(r.update([-.3,.3,.6],i*.02))
    sequence=np.array(sequence)
    assert np.max(np.abs(np.diff(sequence,axis=0))/np.array([.8,.6,1.2])) <= .02000001
    np.testing.assert_allclose(sequence[-1],[-.3,.3,.6])
    np.testing.assert_allclose(r.update([-.3,.3,.6],86400.),[-.3,.3,.6])
    assert actions([.4,.3,.6])['s']==(-.3,0,0)


def test_exclusive_command_session_releases_lock():
    import uuid
    namespace='/test_'+uuid.uuid4().hex
    owner=CommandOwner(namespace)
    try:
        with pytest.raises(RuntimeError,match='Another'):CommandOwner(namespace)
    finally:owner.close()
    second=CommandOwner(namespace);second.close()


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_guard_uses_aligned_support_and_resets_between_actions(make_node):
    from mydog_policy.observation_pipeline import ObservationPipeline
    n,clock,calls=make_node(send=False,stand=False)
    n.observation_pipeline=ObservationPipeline();n.common_time_alignment=object()
    n.observation_pipeline.diagnostics['observation_temporal_ok']=True
    n._latest_base_gyro=np.zeros(3);n.corrected_gyro_rad_s=np.zeros(3)
    q=n.mapper.policy_target_to_real(n.contract.default)
    n.core.aligned_sample=dict(timestamp=clock.t-.05,q_real=q,dq_real=np.zeros(12),
                               gyro=np.zeros(3),gravity=np.array([0,0,-1.]))
    n.mode='walk';n.trial.arm(clock.t);n.trial.receive([0,0,0],clock.t)
    assert not n._update_walk_inhibitors(clock.t,{'confidence':0.},n.contract.default)
    d=n.observation_pipeline.diagnostics
    assert d['obs_support_source']=='common_time_gravity'
    assert d['obs_odometry_support_usable'] and d['obs_support_sample_age_ms']==pytest.approx(50.)
    n._ground_support.filtered[:]=1.
    n.observation_pipeline.quality_since['odometry']=clock.t-.1
    n._reset_walk_session(clock.t,0.,n.contract.default)
    assert n.observation_pipeline.quality_since['odometry'] is None
    np.testing.assert_array_equal(n._ground_support.filtered,np.zeros(3))
    assert not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_translation_switch_preserves_heading_after_march_and_rebases_after_turn(make_node):
    n,clock,calls=make_node(send=False,stand=False);a=n.core.actor;c=n.contract
    def tick(command,yaw=0.):return a.tick(c.default,np.zeros(12),np.zeros(3),[0,0,-1],yaw,command)
    tick([.4,0,0]);old_phase=a.phase;steps=a.steps
    tick([0,.3,0]);assert a.steps==steps+1 and a.phase!=old_phase
    tick([0,0,0],.4);old_phase=a.phase
    r=tick([-.3,0,0],.4)
    assert a.phase!=old_phase and a.heading==pytest.approx(0.)
    assert r['observation'][50]==pytest.approx(np.sin(-.4),abs=1e-7)
    tick([0,0,.6],.4)
    r=tick([0,0,0],.5)
    assert r['observation'][50]==pytest.approx(0.,abs=1e-7)
    assert not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_forward_inference_is_identical_to_previous_tick_path(make_node):
    from mydog_policy.rs01_model6850_core import Rs01Model6850Core
    n,clock,calls=make_node(send=False,stand=False);a=n.core.actor
    b=type(a)(a.session,a.contract);c=n.contract
    for i in range(40):
        args=(c.default,np.zeros(12),np.zeros(3),[0,0,-1],i*.001,[.4,0,0])
        actual=a.tick(*args);expected=Rs01Model6850Core.tick(b,*args)
        for k in ['observation','action','target_policy']:np.testing.assert_array_equal(actual[k],expected[k])


@pytest.mark.parametrize('fault',[False,True])
def test_interactive_waits_for_ready_between_actions_and_requires_manual_recovery(monkeypatch,fault):
    import rclpy
    from mydog_policy import rs01_interactive_command as m
    state=dict(t=0.,step=0,armed=False,arm=[],sent=[],callback=None,disarm_at=-100)
    schedule={1:'m',40:'w',80:'a',120:'s',180:'z',210:'w',260:'x'}
    if fault:schedule={1:'m',90:'x'}
    def call(req):
        if req.data: assert state['step'] > state['disarm_at']+5
        else: state['disarm_at']=state['step']
        state['armed']=req.data;state['arm'].append(req.data)
        return NS(done=lambda:True,result=lambda:NS(success=True,message='ok'))
    client=NS(wait_for_service=lambda **kw:True,call_async=call)
    def subscribe(ty,topic,cb,depth):state['callback']=cb
    node=NS(create_client=lambda *a:client,create_subscription=subscribe,
            create_publisher=lambda *a:NS(get_subscription_count=lambda:1,publish=lambda msg:state['sent'].append([msg.linear.x,msg.linear.y,msg.angular.z])),destroy_node=lambda:None)
    def spin(*a,**kw):
        state['step']+=1;state['t']+=.02
        status=dict(mode='walk' if state['armed'] else 'ready',walk_start_stable=True,
                    send=True,stand_only=False,timing_ready=True,observation_temporal_ok=True,
                    imu_calibrated=True,continuous_commands=True,walk_inhibit_latched=False,
                    trial_armed=state['armed'],time_monotonic_s=state['t'])
        if not state['armed'] and state['step']<=state['disarm_at']+5:
            # Queue one old ready sample, then ramp home, then fresh ready.
            if state['step']==state['disarm_at']+1:status['time_monotonic_s']=state['t']-.1
            else:status['mode']='soft_hold'
        if fault and state['step']>=30:status['mode']='soft_hold';status['walk_inhibit_reason']='test fault'
        state['callback'](NS(data=json.dumps(status)))
    monkeypatch.setattr(m.time,'monotonic',lambda:state['t'])
    monkeypatch.setattr(m.sys,'stdin',NS(isatty=lambda:True,readline=lambda:schedule.pop(state['step'])+'\n'))
    monkeypatch.setattr(m.select,'select',lambda *a:([m.sys.stdin],[],[]) if state['step'] in schedule else ([],[],[]))
    monkeypatch.setattr(m,'CommandOwner',lambda *a:NS(close=lambda:None))
    for name,func in dict(init=lambda **kw:None,create_node=lambda *a:node,spin_once=spin,
                          spin_until_future_complete=lambda *a,**kw:None,ok=lambda:True,shutdown=lambda:None).items():
        monkeypatch.setattr(rclpy,name,func)
    m.main((.4,.3,.6))
    assert state['arm']==([True,False] if fault else [True,False]*5)
    if not fault:
        v=np.array(state['sent']);assert v[:,0].max()>.3 and v[:,0].min()<-.2 and v[:,1].max()>.25
