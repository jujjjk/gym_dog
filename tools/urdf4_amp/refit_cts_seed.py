"""Fit a network seed to validated control waveforms, with no runtime lookup.

The initial actor depends on command/clock/gait. PPO can subsequently learn
state feedback using the unchanged full observation and CTS contexts.
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
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner
p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,default=ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/phase_seed');p.add_argument('--steps',type=int,default=2500)
p.add_argument('--load_seed',type=Path)
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
if not 1<=own.steps<=4000:raise ValueError('Finite initialization budget exceeded')
args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True;args.num_envs=48
cfg,train=task_registry.get_cfgs(args.task);env,cfg=task_registry.make_env(args.task,args=args)
own.output.mkdir(parents=True,exist_ok=True);runner=Urdf4AMPCTSRunner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
model=runner.alg.actor_critic;device=env.device;frames=[];labels=[]
if own.load_seed:model.load_state_dict(torch.load(str(own.load_seed),map_location=device)["model_state_dict"])
with torch.no_grad():model.std.fill_(train.policy.init_noise_std)
manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text())
for c in manifest['clips']:
 with np.load(ROOT/'artifacts/urdf4_amp_isaac_validation_20261113/commands60'/c['clip']/'physical.npz') as data:
  q0=data['actuator_default'];correction=(q0-env.default_dof_pos[0].cpu().numpy())*(1/cfg.rs01_actuator.response_gain-1)
  actions=(data['command_targets']+correction-env.default_dof_pos[0].cpu().numpy())/env.action_scale[0].cpu().numpy() if env.action_scale.ndim>1 else (data['command_targets']+correction-env.default_dof_pos[0].cpu().numpy())/env.action_scale.cpu().numpy()
  # Use steady cycles only, avoiding the reference controller's startup transient.
  count=(len(actions)-1100)//25;wave=actions[1100:1100+count*25].reshape(count,25,12).mean(0)
  for phase in range(25):
   frame=np.zeros(51,np.float32);frame[6:9]=np.array(c['command'])*[2,2,.25];frame[45:47]=[np.sin(2*np.pi*phase/25),np.cos(2*np.pi*phase/25)];frame[50]=1
   frames.append(frame);labels.append(wave[phase])
for phase in range(25):
 frame=np.zeros(51,np.float32);frame[45:47]=[np.sin(2*np.pi*phase/25),np.cos(2*np.pi*phase/25)]
 frames.append(frame);labels.append(np.zeros(12,np.float32))
x=torch.tensor(np.array(frames),device=device);y=torch.tensor(np.array(labels),device=device).clamp(-2,2)
context=torch.zeros(len(x),19,device=device);inputs=torch.cat((model.control_frame(x),context),1)
keep=torch.zeros_like(model.actor[0].weight);keep[:,[6,7,8,45,46,50]]=1
with torch.no_grad():model.actor[0].weight.mul_(keep)
hook=model.actor[0].weight.register_hook(lambda g:g*keep)
opt=torch.optim.Adam(model.actor.parameters(),lr=1e-3)
for k in range(own.steps):
 loss=(model.actor(inputs)-y).square().mean();opt.zero_grad();loss.backward();opt.step()
 if (k+1)%500==0:print('phase_BC',k+1,'MSE',float(loss),flush=True)
# Deterministic full-batch polish, still using only actual reference actions.
opt=torch.optim.LBFGS(model.actor.parameters(),lr=.5,max_iter=300,history_size=15,line_search_fn='strong_wolfe')
def closure():
 opt.zero_grad();loss=(model.actor(inputs)-y).square().mean();loss.backward();return loss
opt.step(closure);hook.remove()
with torch.no_grad():mse=float((model.actor(inputs)-y).square().mean())
# Initialize the student context on physically collected histories as well.
samples=torch.load(ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/bootstrap/samples.pt',map_location=device)['observations']
opt=torch.optim.Adam(model.student_encoder.parameters(),lr=5e-4)
for _ in range(500):
 batch=samples[torch.randint(len(samples),(512,),device=device)].clone();batch[:,-1]=0
 loss,_,_=model.student_loss(batch);opt.zero_grad();loss.backward();opt.step()
runner.save(str(own.output/'run/model_0.pt'))
(own.output/'result.json').write_text(json.dumps(dict(reference_action_mse=mse,steps=own.steps,LBFGS_max_steps=300,
 checkpoint=str(own.output/'run/model_0.pt'),actor_inactive_inputs_initially_zero=True,behavior_accepted=False),indent=2)+'\n')
print('PHASE SEED MSE',mse,flush=True)
