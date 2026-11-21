"""One conditional AMP discriminator update using V3 steady and transition data.

No policy, PPO optimizer, environment reward, or long training is started.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from reference_v3 import V3Reference

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261116_v3';data=BASE/'amp_reference_v3'
urdf=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf';rng=np.random.default_rng(76);torch.manual_seed(76);torch.set_num_threads(1)
reference=V3Reference(data,urdf,'mujoco');gym_reference=V3Reference(data/'gym_verified',urdf,'isaac_gym')
batch=reference.sample(4096,rng);real=torch.tensor(np.c_[batch['previous'],batch['next']],dtype=torch.float32);command=torch.tensor(batch['command'],dtype=torch.float32)
live=np.load(BASE/'gym_transitions/live_rollout.npz');valid=live['valid'][:-1]&live['valid'][1:]&(live['time_s'][:-1,None]>=0)
fakepairs=np.concatenate([live['amp_features'][:-1][valid],live['amp_features'][1:][valid]],axis=1);fc=(live['frame_commands'][:-1][valid]+live['frame_commands'][1:][valid])*.5
ids=rng.choice(len(fakepairs),4096,replace=False);fake=torch.tensor(fakepairs[ids],dtype=torch.float32);fakecommand=torch.tensor(fc[ids],dtype=torch.float32)
states=torch.tensor(reference.previous,dtype=torch.float32);floor=states.new_tensor([.1]*12+[1.]*12+[.03]*12+[.1]*3+[.2]*3+[.1]*3+[.03]);mean=states.mean(0);std=torch.maximum(states.std(0),floor)
def inputs(pair,c):return torch.cat([((pair-mean.repeat(2))/std.repeat(2)).clamp(-10,10),c/c.new_tensor([.5,.2,.8])],1)
net=nn.Sequential(nn.Linear(95,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1));optimizer=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
ri=inputs(real,command).detach().requires_grad_(True);fi=inputs(fake,fakecommand);dr=net(ri);df=net(fi);grad=torch.autograd.grad(dr.sum(),ri,create_graph=True)[0][:,:92];gp=grad.square().sum(1).mean();loss=.5*((dr-1).square().mean()+(df+1).square().mean())+10*gp
optimizer.zero_grad();loss.backward();finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());assert finite and torch.isfinite(loss);nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
with torch.no_grad():bonus=(1-.25*(net(fi).flatten()-1).square()).clamp(0,1)
zero=reference.sample(128,rng,command_query=np.zeros((128,3)));assert not zero['phase_valid'].any() and reference.contacts[zero['pair_index']].all()
old_guard=False
try:V3Reference(data,ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01_no_battery_legacy.urdf',audit_only=True)
except ValueError:old_guard=True
coverage_guard=False
try:reference.sample(1,rng,command_query=np.array([[.6,0,0]]))
except ValueError:coverage_guard=True
assert old_guard and coverage_guard
report=dict(passed=True,full_dataset_pairs=len(reference.previous),gym_qualified_pairs=len(gym_reference.previous),
    clips=len(reference.clips),gym_clips=len(gym_reference.clips),state_features=46,pair_features=92,conditional_input_features=95,
    sampled_real_pairs=4096,real_transition_pairs=int(sum(reference.clips[i].get('variable_command',False) for i in batch['clip_index'])),
    sampled_live_Gym_pairs=4096,discriminator_updates=1,loss=float(loss.detach()),gradient_penalty=float(gp.detach()),gradients_finite=bool(finite),reward_finite=bool(torch.isfinite(bonus).all()),
    old_mass_URDF_guard_preserved=old_guard,unsupported_point6_command_rejected=coverage_guard,zero_command_samples_four_foot_stand=True,
    variable_cadence_and_commands_loaded=True,ppo_training_performed=False,policy_weights_changed=False,note='Numerical smoke only; this does not show a newly trained gait.')
(data/'amp_adapter_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
