"""Bounded corrective-label initialization. Assistance is NEVER evaluation.

Teacher joint commands label actual student/mixture-visited states. All physical
steps use the unchanged RS01 actuator; only ordinary episode resets write poses.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner,stand_blend
from teacher_v3 import Teacher,rot


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-fraction',type=float,default=.75);p.add_argument('--seconds',type=float,default=16.);p.add_argument('--bc-steps',type=int,default=1000)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    if not 0<=own.teacher_fraction<=1 or not 1<=own.bc_steps<=2000 or own.seconds>30:raise ValueError('Corrective initialization must remain finite')
    own.output.mkdir(parents=True,exist_ok=False);args=get_args();args.task='urdf4_amp_cts_v3';args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10
    manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text());specs=copy.deepcopy(manifest['clips'])
    slow=copy.deepcopy(min(specs,key=lambda c:np.linalg.norm(np.array(c['command'])-[.18,0,0])));slow['command']=[.18,0,0];specs.append(slow)
    args.num_envs=len(specs);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device);runner.load(own.model,load_optimizer=False)
    model=runner.alg.actor_critic;command=torch.tensor([c['command'] for c in specs],device=env.device,dtype=torch.float);env.set_command_override(command);env.reset()
    plant=ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant';teachers=[];corrections=[]
    for c in specs:
        c['design'].update(ramp_s=2.,stop_anchor=True,route_progress_projection=True)
        teacher=Teacher(plant,c);teachers.append(teacher)
        corrections.append((teacher.q0-env.default_dof_pos[0].cpu().numpy())*(1/.928-1))
    corrections=np.asarray(corrections);default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy()
    samples=[];targets=[];resets=0;teacher_raw_max=0.;executed_peak=0.;clip_excess=0
    for k in range(round((own.seconds+6)/env.dt)):
        if k==round(own.seconds/env.dt):env.set_command_override(torch.zeros_like(command))
        obs=env.obs_buf.clone();roots=env.root_states.cpu().numpy().copy();roots[:,:3]-=env.env_origins.cpu().numpy();positions=env.dof_pos.cpu().numpy();velocities=env.dof_vel.cpu().numpy();external=env.external_commands.cpu().numpy();phase=env.motion_phase.cpu().numpy();episode=env.episode_steps.cpu().numpy()
        qcommands=[]
        for i,teacher in enumerate(teachers):
            qpos=np.zeros(teacher.m.nq);qvel=np.zeros(teacher.m.nv);qpos[:3]=roots[i,:3];qpos[3:7]=roots[i,[6,3,4,5]];qpos[teacher.qa]=positions[i];qvel[:3]=roots[i,7:10];qvel[3:6]=rot(qpos[3:7]).T@roots[i,10:13];qvel[teacher.va]=velocities[i]
            if episode[i]<2:
                teacher.ff[:]=0.;teacher.yaw_mean=0.;teacher.yaw_integral=0.;teacher.heading_integral=0.;teacher.velocity_mean[:]=0.;teacher.velocity_integral[:]=0.;teacher.reference_xy=roots[i,:2].copy();teacher.reference_yaw=float(np.arctan2(2*(qpos[3]*qpos[6]+qpos[4]*qpos[5]),1-2*(qpos[5]**2+qpos[6]**2)))
            amplitude=np.linalg.norm(external[i])/max(np.linalg.norm(teacher.cmd),1e-9)
            clock=(phase[i]-float(env.reference_phase_offsets[env.reference_id[i]]))%1
            qcommands.append(teacher.command(episode[i]*env.dt-cfg.expand.cold_settle_s,qpos,qvel,dt=env.dt,phase_override=clock,planned_override=external[i],amplitude_override=amplitude))
        label=torch.tensor((np.array(qcommands)+corrections-default)/scale,device=env.device,dtype=torch.float);teacher_raw_max=max(teacher_raw_max,float(abs(label).max()));clip_excess+=int((label.abs()>cfg.normalization.clip_actions).sum());label.clamp_(-cfg.normalization.clip_actions,cfg.normalization.clip_actions)
        blend=stand_blend(obs)[:,None];label=blend*label+(1-blend)*model.stand_action
        with torch.no_grad():actor=model.act_inference(obs);action=own.teacher_fraction*label+(1-own.teacher_fraction)*actor
        keep=~env.reset_buf.bool()
        if k%2==0:samples.append(obs[keep].cpu());targets.append(label[keep].cpu())
        with torch.no_grad():_,_,_,done,_=env.step(action)
        resets+=int(done.sum());executed_peak=max(executed_peak,float(env.raw_pd_torques.abs().max()))
        if (k+1)%250==0:print('CORRECTIVE',k+1,'resets',resets,'peak',executed_peak,flush=True)
    online_obs=torch.cat(samples).to(args.rl_device);online_target=torch.cat(targets).to(args.rl_device)
    torch.save(dict(observations=online_obs.cpu(),actions=online_target.cpu()),own.output/'corrective_samples.pt')
    old=torch.load(ROOT/'artifacts/urdf4_amp_v3_training/bootstrap/samples.pt',map_location='cpu');offline_obs=old['observations'].to(args.rl_device);offline_target=old['actions'].to(args.rl_device)
    model.teacher_encoder.requires_grad_(True);params=list(model.actor.parameters())+list(model.teacher_encoder.parameters())+list(model.student_encoder.parameters());optimizer=torch.optim.Adam(params,lr=2e-4)
    def loss_on(obs,target):
        public,state,_=model.split(obs);tc=model.teacher_context(state);sc=model.student_encoder(public);frame=model.control_frame(public)
        t=model.executed_mean(public,model.actor(torch.cat((frame,tc),1)));s=model.executed_mean(public,model.actor(torch.cat((frame,sc),1)))
        blend=stand_blend(public)[:,None];desired=blend*target+(1-blend)*model.stand_action
        return .5*((t-desired).square().mean()+(s-desired).square().mean())+.1*(sc-tc.detach()).square().mean()
    for k in range(own.bc_steps):
        a=torch.randint(len(online_obs),(768,),device=args.rl_device);b=torch.randint(len(offline_obs),(256,),device=args.rl_device)
        loss=loss_on(torch.cat((online_obs[a],offline_obs[b])),torch.cat((online_target[a],offline_target[b])));optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (k+1)%200==0:print('CORRECTIVE BC',k+1,float(loss),flush=True)
    runner.seed_migration=dict(source=str(own.model),kind='finite corrective BC',teacher_fraction=own.teacher_fraction,teacher_in_deployed_student=False)
    runner.save(str(own.output/'model_0.pt'))
    (own.output/'result.json').write_text(json.dumps(dict(samples=len(online_obs),assisted_rollout_seconds=own.seconds+6,teacher_fraction=own.teacher_fraction,
        resets=resets,raw_sample_peak_Nm=executed_peak,uncapped_label_action_peak=teacher_raw_max,label_clip_count=clip_excess,
        BC_updates=own.bc_steps,PPO_updates=0,not_a_behavior_acceptance=True,teacher_in_exported_student=False),indent=2)+'\n')


if __name__=='__main__':main()
