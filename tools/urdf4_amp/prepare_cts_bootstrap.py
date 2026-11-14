"""Collect actual RS01 teacher rollouts, reset snapshots and a finite BC seed.

No reference controller is embedded in the exported student policy.
"""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--commands-dir',type=Path,default=ROOT/'artifacts/urdf4_amp_isaac_validation_20261113/commands60')
p.add_argument('--output',type=Path,default=ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/bootstrap')
p.add_argument('--bc_steps',type=int,default=1500)
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
if not 1<=own.bc_steps<=3000:raise ValueError('BC preparation capped at 3000 optimizer steps')
args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True;args.num_envs=48
cfg,train=task_registry.get_cfgs(args.task);cfg.amp_cts.reference_reset_probability=0.;cfg.env.episode_length_s=30.
manifest_path=Path(cfg.amp_cts.reference)/'manifest.json';manifest=json.loads(manifest_path.read_text())
contract=json.loads((own.commands_dir/'contract.json').read_text())
if contract['dataset_manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():raise ValueError('Teacher commands do not match V2')
data=[dict(np.load(own.commands_dir/c['clip']/'physical.npz',allow_pickle=False)) for c in manifest['clips']]
if any(abs(float(z['command_dt_s'])-.02)>1e-8 for z in data):raise ValueError('Teacher period mismatch')
env,cfg=task_registry.make_env(args.task,args=args,env_cfg=cfg);device=env.device
ids=torch.arange(48,device=device)%12
commands=env.reference_commands[ids]
q0=torch.tensor(np.stack([z['actuator_default'] for z in data]),device=device,dtype=torch.float32)[ids]
correction=(q0-env.default_dof_pos)*(1/cfg.rs01_actuator.response_gain-1)
targets=torch.tensor(np.stack([z['command_targets'][:500] for z in data]),device=device,dtype=torch.float32)[ids]
env.set_command_override(torch.zeros_like(commands),gait_enabled=False);obs,_=env.reset();env.clock_offset[:]=0
dataset_obs=[];dataset_actions=[];snapshots={};snapshot_clips=[];teacher_resets=0;peak=0.;illegal=0.;samples=0

def cycle_values(mean):
 ix=(mean.head-torch.arange(mean.window.shape[1],device=device))%mean.window.shape[1]
 return mean.window[:12,ix].clone()

def snapshot():
 values={name:getattr(env,name)[:12].clone() for name in ['dof_pos','dof_vel','position_target','target_rate','response_target','actions','last_actions','torques','raw_pd_torques','motor_torques','feet_vel','obs_history','velocity_drift','heading_error']}
 root=env.root_states[:12].clone();shift=root[:,:3].clone();shift[:,2]=env.env_origins[:12,2]
 root[:,:3]-=shift
 values.update(root=root,feet_pos=env.feet_pos[:12]-shift[:,None,:],contact=env.get_foot_contact_mask()[:12].clone(),
               delay_buffer=env.delay_buffer[:,:12].transpose(0,1).clone(),clock=env._clock_phase()[:12].clone(),
               velocity_window=cycle_values(env.velocity_error),yaw_window=cycle_values(env.measured_yaw_error))
 for name,value in values.items():snapshots.setdefault(name,[]).append(value.cpu())
 snapshot_clips.append(torch.arange(12))

with torch.no_grad():
 for k in range(500):
  if k==100:
   env.set_command_override(commands,gait_enabled=True);env.heading_target[:]=yaw_from_quat(env.root_states[:,3:7]);env.heading_error[:]=0
   env.compute_observations();obs=env.get_observations()
  desired=targets[:,k]+correction
  action=((desired-env.default_dof_pos)/env.action_scale).clamp(-cfg.normalization.clip_actions,cfg.normalization.clip_actions)
  if k>=100 or k%5==0:
   dataset_obs.append(obs[:24].cpu().clone());dataset_actions.append(action[:24].cpu().clone())
  obs,_,_,done,_=env.step(action);teacher_resets+=int(done.sum())
  if k>=150:
   peak=max(peak,float(env.raw_pd_torques.abs().max()));illegal+=float(env._reward_illegal_support().sum());samples+=48
  if 450<=k<500:snapshot()
  if (k+1)%100==0:print('teacher_steps',k+1,'resets',teacher_resets,flush=True)
if teacher_resets:raise RuntimeError('Reference replay failed; refusing to build a bootstrap seed')
own.output.mkdir(parents=True,exist_ok=True)
obs=torch.cat(dataset_obs);actions=torch.cat(dataset_actions)
snap={name:torch.cat(values) for name,values in snapshots.items()};clips=torch.cat(snapshot_clips)
indices=torch.stack([(clips==i).nonzero().flatten() for i in range(12)])
bank=dict(contract=env.reference_state_contract(),snapshots=snap,clip_indices=indices,
          source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          teacher_commands_contract=contract,teacher_resets=teacher_resets)
bank_path=Path(cfg.amp_cts.reference_bank);bank_path.parent.mkdir(parents=True,exist_ok=True)
torch.save(bank,str(bank_path));torch.save(dict(observations=obs,actions=actions),str(own.output/'samples.pt'))
env.reference_bank={k:({n:v.to(device) for n,v in value.items()} if k=='snapshots' else value.to(device) if torch.is_tensor(value) else value) for k,value in bank.items()}
env.reference_bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest();env.reference_reset_probability=.8;cfg.amp_cts.reference_reset_probability=.8
runner=Urdf4AMPCTSRunner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
model=runner.alg.actor_critic;obs=obs.to(device);actions=actions.to(device)
# Only the initialization uses direct action supervision. PPO/AMP remains on-policy.
params=list(model.actor.parameters())+list(model.teacher_encoder.parameters())+list(model.student_encoder.parameters())
optimizer=torch.optim.Adam(params,lr=5e-4)
valid=torch.arange(len(obs),device=device)%5==0;train_ids=(~valid).nonzero().flatten();valid_ids=valid.nonzero().flatten()

def loss_on(batch,augment=False):
 public,state,_=model.split(obs[batch]);public=public.clone()
 if augment:
  # Match the public reset contract: some examples have a repeated first frame.
  cold=torch.rand(len(batch),device=device)<.15
  public[cold]=public[cold,:51].repeat(1,20)
 teacher=model.teacher_context(state);student=model.student_encoder(public)
 target=actions[batch]
 t=model.actor(torch.cat((model.control_frame(public),teacher),1));s=model.actor(torch.cat((model.control_frame(public),student),1))
 bc=.5*((t-target).square().mean()+(s-target).square().mean())
 return bc+.1*(student-teacher.detach()).square().mean(),(s-target).square().mean()

with torch.no_grad():before=float(loss_on(valid_ids)[1])
for k in range(own.bc_steps):
 batch=train_ids[torch.randint(len(train_ids),(512,),device=device)]
 loss,mse=loss_on(batch,True);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
 if (k+1)%250==0:
  with torch.no_grad():print('BC',k+1,'student_holdout_mse',float(loss_on(valid_ids)[1]),flush=True)
with torch.no_grad():after=float(loss_on(valid_ids)[1])
runner.save(str(own.output/'run/model_0.pt'))
result=dict(seed=train.seed,teacher_steps=500,teacher_seconds=10.,teacher_resets=teacher_resets,
            teacher_raw_peak_Nm=peak,teacher_illegal_support_fraction=illegal/max(samples,1),
            samples=len(obs),snapshots=len(clips),bc_steps=own.bc_steps,
            student_holdout_mse_before=before,student_holdout_mse_after=after,
            reference_bank=str(bank_path),reference_bank_sha256=env.reference_bank_sha256,
            checkpoint=str(own.output/'run/model_0.pt'),behavior_accepted=False)
(own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
