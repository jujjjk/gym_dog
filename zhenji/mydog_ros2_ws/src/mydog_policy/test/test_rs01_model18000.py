from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
from mydog_policy.rs01_model18000_core import Model18000Contract,Rs01Model18000Core,Guarded18000PolicyCore
from mydog_policy.rs01_model6850_core import Model6850Contract,Rs01Model6850Core

@pytest.fixture
def contract_session():
    path=Path(__file__).resolve().parents[1]/'resource/B18000.onnx'
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    return Model18000Contract.from_onnx_session(s,path),s,path

def _mix(core, command, error_deg, turning=False):
    conf=core.contract.raw['v13']['commands']
    return core._mix_direction_command(np.asarray(command,dtype=float), np.deg2rad(error_deg), turning, conf)

def _legacy_mix(core, command, error_deg, turning=False):
    conf=core.contract.raw['v13']['commands']
    return Rs01Model6850Core._mix_direction_command(
        core, np.asarray(command,dtype=float), np.deg2rad(error_deg), turning, conf)

def test_contract_and_wrong_model(contract_session):
    c,s,p=contract_session
    assert c.policy_dt==.02 and c.peak_torque_limit==14.
    with pytest.raises(RuntimeError):Model6850Contract.from_onnx_session(s,p)
    with pytest.raises(RuntimeError):Model18000Contract.from_onnx_session(s,p,'')

def test_mapping_and_unchanged_outward(contract_session):
    c,s,_=contract_session;core=Rs01Model18000Core(s,c)
    ids=[0,3,6,9];sides=np.array([1.,-1.,1.,-1.]);target=c.default.copy();target[ids]=-.22*sides
    out=core.project_policy_target(target,np.zeros(3))
    np.testing.assert_allclose(out[ids],-np.deg2rad(8)*sides)
    np.testing.assert_equal(core.project_policy_target(target,[0,.2,0]),target)
    target[ids]=.22*sides
    np.testing.assert_equal(core.project_policy_target(target,[0,0,0]),target)

def test_guard_preserves_actor_history(contract_session):
    c,s,_=contract_session;core=Guarded18000PolicyCore(s,c)
    q=c.default.copy();q[1]+=.1
    obs=core.build_observation(0,np.zeros(3),np.zeros(3),[0,0,-1],[.2,0,0],q,np.zeros(12),0)
    target=core.actor.target.copy();r=core.step(obs,q,np.zeros(12),np.full(12,6.))
    np.testing.assert_equal(core.actor.target,target)
    assert np.max(abs(r['torque_info']['safe_pd_torque_nm']))<=6.00001
    assert np.isfinite(r['safe_target_policy']).all()

def test_straight_forward_does_not_inject_lateral(contract_session):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    b=_mix(core, [.2,0,0], -4.3)
    legacy=_legacy_mix(core, [.2,0,0], -4.3)
    np.testing.assert_allclose(b[:2], [.2,0.], atol=1e-12)
    assert b[2] < 0
    assert legacy[1] < -0.01
    assert abs(b[2]) < abs(legacy[2])
    assert abs(b[2]) < 0.05

def test_straight_heading_deadband_ignores_gait_wobble(contract_session):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    np.testing.assert_allclose(_mix(core, [.2,0,0], 1.5), [.2,0,0], atol=1e-12)
    np.testing.assert_allclose(_mix(core, [0,0,0], -1.5), [0,0,0], atol=1e-12)

def test_straight_heading_feedback_is_symmetric_without_bias(contract_session):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    for vx in [-.2,0.,.2]:
        np.testing.assert_allclose(_mix(core,[vx,0,0],0),[vx,0,0])
        for angle in [1.5,3.,10.]:
            left=_mix(core,[vx,0,0],angle)
            right=_mix(core,[vx,0,0],-angle)
            assert left[1]==right[1]==0.
            assert left[2]==pytest.approx(-right[2])


def test_omni_and_turn_keep_legacy_planar_mix(contract_session):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    for cmd in ([.2,.12,0],[0,.12,0],[.2,0,.25]):
        np.testing.assert_allclose(_mix(core, cmd, -4.3), _legacy_mix(core, cmd, -4.3))

def test_tick_straight_observation_keeps_zero_vy(contract_session):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    core.reset(0.)
    result=core.tick(c.default, np.zeros(12), np.zeros(3), [0,0,-1], np.deg2rad(4.3), [.2,0,0], gait=1)
    assert result['observation'][10] == pytest.approx(0., abs=1e-6)
    assert result['observation'][58] == pytest.approx(0., abs=1e-6)
    assert result['observation'][11] / 0.25 < 0
