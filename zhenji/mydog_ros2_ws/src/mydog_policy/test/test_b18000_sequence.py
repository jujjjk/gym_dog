import pytest
from mydog_policy.rs01_model18000_sequence import build_plan, check_active, play_plan, main, ACTION_KEYS, wait_for_stand


def healthy(mode='walk'):
    return dict(model='B18000', mode=mode, send=True, stand_only=False,
                imu_calibrated=True, trial_armed=True, timing_ready=True,
                walk_inhibit_latched=False)


def test_plan_durations_speeds_and_transitions():
    plan = build_plan()
    assert len(plan) == 17
    assert sum(s[1] for s in plan) == 165
    for i, (_, seconds, vec) in enumerate(plan):
        assert seconds == (5 if i % 2 == 0 else 15)
        if i % 2 == 0: assert vec == (0., 0., 0.)
        assert all(abs(x)<=limit for x,limit in zip(vec,(.30,.20,.30)))


@pytest.mark.parametrize('change', [dict(timing_ready=False), dict(mode='soft_hold'),
    dict(trial_armed=False), dict(imu_calibrated=False), dict(walk_inhibit_latched=True),
    dict(send=False), dict(stand_only=True), dict(model='A6850')])
def test_abort_on_protection(change):
    status=healthy(); status.update(change)
    with pytest.raises(RuntimeError): check_active(status, .01)


def test_stale_status_and_reentry_rejected():
    with pytest.raises(RuntimeError): check_active(healthy(), .251)
    with pytest.raises(RuntimeError): check_active(healthy('ready'), .01)


def test_initial_gate_not_charged_to_segments_and_abort_stops_publishing():
    now=[0.]; sent=[]
    def snapshot(): return healthy('ready' if now[0]<1 else 'walk'),now[0]
    def pump(dt): now[0]+=dt
    play_plan([('move',15.,(.10,0.,0.)),('march',5.,(0.,0.,0.))],
              snapshot, lambda v:sent.append((now[0],v)), pump, lambda:now[0])
    moves=[t for t,v in sent if v[0]>.0]
    assert min(moves)>=1. and max(moves)-min(moves)>=14.97
    assert now[0]>=21.
    now[0]=0.;sent.clear()
    def fault():
        status=healthy()
        if now[0]>=.10: status['timing_ready']=False
        return status,now[0]
    with pytest.raises(RuntimeError):
        play_plan(build_plan(),fault,lambda v:sent.append(v),pump,lambda:now[0])
    assert all(v==(0.,0.,0.) for v in sent)
    assert now[0]<.15


def test_preview_no_ros(capsys):
    main(['--print-plan'])
    assert '165' in capsys.readouterr().out


@pytest.mark.parametrize('action', ('march',)+ACTION_KEYS)
def test_single_action_has_no_next_move(action):
    plan=build_plan(action)
    assert len(plan)==1 and plan[0][1]==15.


def test_wait_stand_requires_disarm_and_continuous_stability():
    now=[0.]
    def pump(dt): now[0]+=dt
    def snapshot():
        s=healthy('ready');s['trial_armed']=now[0]<1.
        s['walk_start_stable']=now[0]>=1.5
        return s,now[0]
    wait_for_stand(snapshot,pump,lambda:now[0])
    assert now[0]>=3.5
    with pytest.raises(RuntimeError):
        wait_for_stand(lambda:(healthy(),0.),pump,lambda:now[0])
