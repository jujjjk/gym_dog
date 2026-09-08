"""Nominal PhysX control: exact default joints, zero velocity, phase zero."""
import json
from pathlib import Path
import isaacgym
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'unitree_rl_gym/legged_gym/scripts'))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args,task_registry

args=get_args()
cfg,train=task_registry.get_cfgs(args.task)
_set_nominal_eval_cfg(cfg,30.,4)
cfg.init_state.reset_dof_position_noise_rad=0.
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
train.runner.resume=True
runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
policy=runner.get_inference_policy(device=env.device)
ids=torch.arange(4,device=env.device)
env.reset_idx(ids)
env.base_lin_vel.zero_();env.base_ang_vel.zero_();env.rpy.zero_();env.actions.zero_()
env.projected_gravity[:]=torch.tensor([0.,0.,-1.],device=env.device)
env.wide_gait_phase.zero_()
commands=torch.tensor([[0.,0.,0.],[.2,0.,0.],[.4,0.,0.],[0.,0.,0.]],device=env.device)
gaits=torch.tensor([0.,1.,1.,1.],device=env.device)
names=['zero_action_stand','forward02','forward04','march']
records=[[] for _ in names]; first=[None]*4
with torch.no_grad():
 for step in range(round(30/env.dt)):
  env.set_evaluation_command(commands,gaits)
  env.compute_observations()
  action=policy(env.get_observations());action[0]=0.
  _,_,_,done,_=env.step(action)
  for i in range(4):
   if first[i] is not None:continue
   if done[i]:first[i]=(step+1)*env.dt;continue
   records[i].append([*env.base_lin_vel[i,:2].cpu().tolist(),env.base_ang_vel[i,2].item(),
                      env.rpy[i,0].item(),env.rpy[i,1].item(),env.root_states[i,2].item()])
result=[]
for name,rows,stop in zip(names,records,first):
 x=torch.tensor(rows);steady=x[100:]
 result.append(dict(case=name,first_environment_reset_s=stop,recorded_steps=len(rows),
  mean_vx_vy_wz=steady[:,:3].mean(0).tolist() if len(steady) else None,
  max_abs_roll_pitch=x[:,3:5].abs().max(0).values.tolist(),min_height=x[:,5].min().item(),
  final_height=x[-1,5].item()))
print(json.dumps(result,indent=2))
out=Path(__file__).resolve().parents[2]/'artifacts/rs01_v13_a6850_sim2sim/physx_control.json'
out.write_text(json.dumps(result,indent=2)+'\n')
