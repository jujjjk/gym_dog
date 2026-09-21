import csv
import numpy as np
import pytest
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from test_rs01_model6850_node_offline import make_node,advance

@pytest.mark.parametrize('make_node',['capture61'],indirect=True)
def test_capture_stand_and_policy_are_distinct_and_coherent(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,180)
    assert n.mode=='ready' and not calls
    n.imu_calibrated=True
    n._b_timing_ready=lambda:True
    response=n.arm_callback(SetBool.Request(data=True),SetBool.Response())
    assert response.success
    cmd=Twist();cmd.linear.x=.15
    for _ in range(90):
        n.command_callback(cmd);advance(n,clock,1)
    assert n.mode=='walk'
    expected_obs=n.core.pending['observation'] if n.core.pending else None
    n.capture.close()
    rows=list(csv.DictReader((n.capture.path/'cycles.csv').open()))
    assert not calls
    assert not n.capture.error and not n.capture.dropped
    ready=[r for r in rows if r['mode']=='ready'];walk=[r for r in rows if r['mode']=='walk']
    assert ready and walk
    assert all(r['policy_evaluated']=='False' for r in ready)
    assert all(r['policy_evaluated']=='True' for r in walk)
    assert all(float(r['cmd_vx'])==pytest.approx(.15) for r in walk)
    assert all(float(r['projected_gravity_z'])==-1 for r in rows)
    assert np.isnan(float(ready[-1]['raw_action_01']))
    assert np.isfinite(float(walk[-1]['raw_action_01']))
    obs=[m.data for topic,m in n.messages if topic.endswith('/observation')][-1]
    np.testing.assert_array_equal([float(walk[-1][f'obs_{i:02d}']) for i in range(61)],obs)
    target=[m.data for topic,m in n.messages if topic.endswith('/target_real')][-1]
    np.testing.assert_allclose(float(walk[-1]['target_q_FR_hip']),target[0])
