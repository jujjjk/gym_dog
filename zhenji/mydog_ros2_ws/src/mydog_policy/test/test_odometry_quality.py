from copy import deepcopy
import numpy as np
import pytest
from mydog_policy.odometry_quality import support_quality
from mydog_policy.observation_pipeline import ObservationPipeline
from mydog_policy.rs01_model930_core import Rs01NewMachineLegOdometry


def valid_pair(confidence=.4):
    residual=(1-confidence)*.35
    return dict(confidence=confidence, base_height_proxy=np.full(4,.307),
                foot_velocity=np.zeros((4,3)),
                velocity_by_foot=np.array([[residual/2,0,0],[0,0,0],[0,0,0],[-residual/2,0,0]]),
                selected_pair_index=0, legal_diagonal_support=True,
                stance_mask=np.array([True,False,False,True]),
                pair_residual_m_s=residual, raw_base_velocity=np.zeros(3))


def test_low_confidence_valid_pair_stays_degraded_and_is_not_relabelled_healthy():
    estimator=Rs01NewMachineLegOdometry(strict_diagonal_pairs=True)
    odometry=valid_pair();before=deepcopy(odometry)
    p=ObservationPipeline();p.diagnostics['observation_temporal_ok']=True
    for i in range(501):
        support=support_quality(odometry,estimator)
        assert support['obs_odometry_support_usable']
        assert not p.quality(i*.02,odometry['confidence'],True,True,
                             odometry_support_usable=support['obs_odometry_support_usable'])
    assert p.diagnostics['obs_quality_state']=='degraded'
    assert not p.diagnostics['obs_quality_ok']
    assert p.diagnostics['obs_quality_odometry_low_conf_sec']==10.
    assert p.diagnostics['obs_quality_odometry_bad_sec']==0.
    assert odometry['confidence']==before['confidence']==.4
    np.testing.assert_array_equal(odometry['raw_base_velocity'],before['raw_base_velocity'])


@pytest.mark.parametrize('bad', ['missing','pair','height','vertical','residual','nan','zero_confidence','mask'])
def test_invalid_support_is_not_rescued_by_a_high_confidence_number(bad):
    e=Rs01NewMachineLegOdometry(strict_diagonal_pairs=True);o=valid_pair(.9)
    if bad=='missing': o={}
    elif bad=='pair': o['selected_pair_index']=-1
    elif bad=='height': o['base_height_proxy'][0]-=.05
    elif bad=='vertical': o['foot_velocity'][0,2]=.26
    elif bad=='residual': o['velocity_by_foot'][0,0]=1.
    elif bad=='nan': o['raw_base_velocity'][0]=np.nan
    elif bad=='zero_confidence': o['confidence']=0.
    elif bad=='mask': o['stance_mask']=np.array([True,True,False,False])
    support=support_quality(o,e)
    assert not support['obs_odometry_support_usable']
    p=ObservationPipeline();p.diagnostics['observation_temporal_ok']=True
    assert not p.quality(1.,o.get('confidence',0.),True,True,odometry_support_usable=False)
    assert p.quality(1.21,o.get('confidence',0.),True,True,odometry_support_usable=False)
    assert p.diagnostics['obs_quality_stop_reason']=='odometry'


def test_short_support_transition_recovers_and_timing_guard_remains_independent():
    p=ObservationPipeline();p.diagnostics['observation_temporal_ok']=True
    assert not p.quality(1.,0.,True,True,odometry_support_usable=False)
    assert not p.quality(1.18,.4,True,True,odometry_support_usable=True)
    assert p.diagnostics['obs_quality_odometry_bad_sec']==0.
    p.diagnostics['observation_temporal_ok']=False
    assert not p.quality(2.,.4,True,True,odometry_support_usable=True)
    assert p.quality(2.21,.4,True,True,odometry_support_usable=True)
    assert p.diagnostics['obs_quality_stop_reason']=='timing'


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_node_preserves_march_for_usable_pair_and_disarms_on_support_loss(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    n.observation_pipeline=ObservationPipeline()
    n.observation_pipeline.diagnostics['observation_temporal_ok']=True
    n.mode='walk';n.trial.arm(clock.t);n.trial.receive([0,0,0],clock.t)
    o=valid_pair()
    for i in range(51):
        assert not n._update_walk_inhibitors(clock.t+i*.02,o,n.contract.default)
    assert n.mode=='walk' and n.trial.armed
    assert n.observation_pipeline.diagnostics['obs_quality_state']=='degraded'
    o['legal_diagonal_support']=False
    assert not n._update_walk_inhibitors(clock.t+1.1,o,n.contract.default)
    assert not n._update_walk_inhibitors(clock.t+1.31,o,n.contract.default)
    assert n._update_walk_inhibitors(clock.t+1.71,o,n.contract.default)
    assert n.mode=='soft_hold' and not n.trial.armed and not calls
