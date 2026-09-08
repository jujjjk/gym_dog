"""Read-only experiment: same actor actions in both engines, ground and air."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import isaacgym
from isaacgym import gymtorch
import torch
import numpy as np
import mujoco
from sim2sim_v13 import V13Sim
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'unitree_rl_gym/legged_gym/scripts'))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args,task_registry

root=Path(__file__).resolve().parents[2];out=root/'artifacts/rs01_v13_a6850_sim2sim'
args=get_args();cfg,train=task_registry.get_cfgs(args.task)
_set_nominal_eval_cfg(cfg,10.,2);cfg.init_state.reset_dof_position_noise_rad=0.
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
train.runner.resume=True
runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
policy=runner.get_inference_policy(device=env.device)
ids=torch.arange(2,device=env.device);env.reset_idx(ids)
env.root_states[1,2]+=2.
env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states))
env.base_lin_vel.zero_();env.base_ang_vel.zero_();env.rpy.zero_();env.actions.zero_()
env.projected_gravity[:]=torch.tensor([0.,0.,-1.],device=env.device);env.wide_gait_phase.zero_()
sims=[V13Sim(out/'scene.xml',out/'model_6850.onnx',[.2,0,0]) for _ in range(2)]
sims[1].data.qpos[2]+=2.;mujoco.mj_forward(sims[1].model,sims[1].data)
rows=[]
with torch.no_grad():
 for step in range(25):
  env.set_evaluation_command(torch.tensor([.2,0,0],device=env.device),1.)
  env.compute_observations();actions=policy(env.get_observations());a=actions.cpu().numpy()
  _,_,_,dones,_=env.step(actions)
  for i,sim in enumerate(sims):
   sim.session=SimpleNamespace(run=lambda *unused,a=a[i].copy():[a[None]])
   sim.control_step()
   q=env.dof_pos[i].cpu().numpy();dq=env.dof_vel[i].cpu().numpy()
   target=env.rs01_response_target_rad[i].cpu().numpy()
   row=dict(t=(step+1)*env.dt,case=['ground','air'][i],reset=bool(dones[i]),
    target_max_error=float(np.max(abs(target-sim.response_target))),
    q_max_error=float(np.max(abs(q-sim.data.qpos[sim.qpos_indices]))),
    dq_max_error=float(np.max(abs(dq-sim.data.qvel[sim.qvel_indices]))),
    physx_q=q.tolist(),mj_q=sim.data.qpos[sim.qpos_indices].tolist(),
    physx_dq=dq.tolist(),mj_dq=sim.data.qvel[sim.qvel_indices].tolist(),
    physx_height=env.root_states[i,2].item(),mj_height=float(sim.data.qpos[2]),
    physx_force=env.contact_forces[i,env.feet_indices,2].cpu().tolist(),
    mj_force=sim.contact_forces().tolist())
   rows.append(row)
   if step in (0,1,2,4,9,14,24):print({k:v for k,v in row.items() if k not in ('physx_q','mj_q','physx_dq','mj_dq')},flush=True)
(out/'same_action_replay.json').write_text(json.dumps(rows,indent=2)+'\n')
