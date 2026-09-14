from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
from mydog_policy.rs01_model18000_core import Model18000Contract,Rs01Model18000Core,Guarded18000PolicyCore
from mydog_policy.rs01_model6850_core import Model6850Contract

@pytest.fixture
def contract_session():
    path=Path(__file__).resolve().parents[1]/'resource/B18000.onnx'
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    return Model18000Contract.from_onnx_session(s,path),s,path

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
