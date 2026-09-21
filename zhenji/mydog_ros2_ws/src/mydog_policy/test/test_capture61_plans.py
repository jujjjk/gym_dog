import json
import pytest
from mydog_policy.rs01_model18000_sequence import main

@pytest.mark.parametrize('action,speed,axis,sign', [('forward',.15,0,1),('backward',.10,0,-1),('left',.10,1,1),('right',.10,1,-1),('turn-left',.20,2,1),('turn-right',.20,2,-1)])
def test_collection_plan_keeps_caps_and_requested_duration(action,speed,axis,sign,capsys):
    main(['--action',action,'--speed',str(speed),'--seconds','30','--print-plan'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['total_seconds']==30
    assert plan['segments'][0][2][axis]==pytest.approx(sign*speed)

@pytest.mark.parametrize('args',[
 ['--action','forward','--speed','.31'],['--action','left','--speed','.21'],
 ['--action','forward','--seconds','61'],['--action','forward','--speed','nan'],
 ['--action','forward','--seconds','nan'],['--seconds','30'],
 ['--action','combo','--speed','.1']])
def test_invalid_trial_override_exits_before_ros_import(args):
    with pytest.raises(SystemExit):main(args+['--print-plan'])
