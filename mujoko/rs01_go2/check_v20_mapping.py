"""Deterministic NumPy bridge versus training torch algebra parity."""
import isaacgym  # must precede torch
import numpy as np
import torch
from sim2sim_v20 import map_targets, project_guard, update_guard
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import map_hip_targets
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import guarded_target, update_guard as torch_update


def main():
    rng=np.random.default_rng(42); n=1000
    q=rng.normal(0,.2,(n,12)); dq=rng.normal(0,3,(n,12))
    target=rng.normal(0,.2,(n,12)); command=rng.normal(0,.1,(n,3))
    ids=[0,3,6,9]; sides=np.array([1.,-1.,1.,-1.])
    cfg=dict(vy_gate=.08,wz_gate=.20,inward_target_rad=np.deg2rad(8))
    a=np.array([map_targets(t,c,ids,sides,np.full(4,.22),cfg) for t,c in zip(target,command)])
    b=map_hip_targets(torch.tensor(target),torch.tensor(command),ids,torch.tensor(sides),.22,cfg['inward_target_rad'])
    np.testing.assert_allclose(a,b.numpy(),atol=1e-12,rtol=0)
    limit=rng.uniform(6,14,(n,12))
    a=project_guard(target,q,dq,40.,1.,limit,-1.,1.)
    b=guarded_target(*[torch.tensor(x) for x in (target,q,dq,40.,1.,limit,-1.,1.)])
    for x,y in zip(a,b):np.testing.assert_allclose(x,y.numpy(),atol=1e-12,rtol=0)
    rms=rng.uniform(0,64,(n,12));torque=rng.normal(0,8,(n,12))
    a=update_guard(rms,torque,.02,dict(continuous=6.,peak=14.,full=8.,tau=2.))
    b=torch_update(torch.tensor(rms),torch.tensor(torque),.02)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y.numpy(),atol=1e-12,rtol=0)
    print('PASS: 1000 mapping/guard/thermal cases, absolute tolerance1e-12')


if __name__=='__main__':main()
