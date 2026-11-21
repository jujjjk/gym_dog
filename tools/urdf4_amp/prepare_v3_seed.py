"""Finite BC initialization from already measured new-mass Gym trajectories.

History at control tick k uses only measured state k-1 and earlier. The label is
the joint command issued at k. No reference controller enters the exported actor.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner
from legged_gym.envs.urdf4.amp_v3_env import V3PathController
from legged_gym.envs.urdf4.path_tracking import PathGains,wrap

BASE=ROOT/'artifacts/urdf4_amp_v3_training';DATA=ROOT/'artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3'


def collect_samples(env,output,manifest_clock=False,include_transitions=True):
    manifests=[(DATA/'gym_verified',json.loads((DATA/'gym_verified/manifest.json').read_text())),
        (DATA/'gym_verified/transitions',json.loads((DATA/'gym_verified/transitions/manifest.json').read_text()))]
    if not include_transitions:manifests=manifests[:1]
    specs=[c for _,m in manifests for c in m['clips']];cache={};states=[];labels=[];caseids=[];validation=[];provenance=[]
    default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy();clip=float(env.cfg.normalization.clip_actions)
    for ci,c in enumerate(specs):
        receipt_path=Path(c['isaac_gym_validation']['receipt']);receipt=json.loads(receipt_path.read_text());case_name=c['isaac_gym_validation']['case']['clip']
        actor=next(i for i,x in enumerate(receipt['cases']) if x['clip']==case_name)
        source=receipt_path.parent/'live_rollout.npz'
        if source not in cache:
            if hashlib.sha256(source.read_bytes()).hexdigest()!=receipt['live_rollout_sha256']:raise ValueError('Live Gym record changed')
            z=np.load(source);cache[source]={key:z[key] for key in ['amp_features','contact','root_state_xyzw_world','feet_pos_world_m','issued_joint_targets','applied_torque_Nm','valid','time_s']}
        z=cache[source];F=z['amp_features'][:,actor];root=z['root_state_xyzw_world'][:,actor];ct=z['contact'][:,actor];issued=z['issued_joint_targets'][:,actor];T=c['design']['period_s']
        physical=DATA/c['source_folder'].removeprefix('../')/'physical.npz' if sys.version_info>=(3,9) else DATA/c['source_folder'].replace('../','',1)/'physical.npz'
        q0=np.load(physical)['actuator_default'];correction=(q0-default)*(1/.928-1)
        actions=(issued+correction-default)/scale
        if abs(actions).max()>clip+.001:raise ValueError('Teacher command exceeds new action space: '+c['clip']+' '+str(abs(actions).max()))
        path=V3PathController(1,.02,'cpu',gains=PathGains(**env.cfg.direction.gains));history=torch.zeros(20,51);yaw_window=[];vel_window=[]
        times=np.arange(len(F))*.02-2.;mask=(times>=5)&(times<min(c['seconds'],55.));duty=float(ct[mask].mean()) if mask.any() else .68
        clock=(times[:,None]/T+np.array([0,.5,.5,0]))%1
        _,offset=max((float((ct[mask]==((clock[mask]+v)%1<duty)).mean()),float(v)) for v in np.linspace(-.35,.35,141)) if mask.any() and c['family']!='stand' else (1.,0.)
        if manifest_clock:offset=c.get('phase_alignment_offset_cycles',offset)
        for k in range(1,len(F),1):
            t=k*.02-2.;state=torch.from_numpy(F[k-1]);r=torch.from_numpy(root[k-1]);q=r[3:7];yaw=torch.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
            if c.get('variable_command'):
                schedule=c['design']['command_schedule'];tt=np.array([v['time_s'] for v in schedule]);ia=int(np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-1));ib=min(ia+1,len(tt)-1)
                u=np.clip((t-tt[ia])/max(tt[ib]-tt[ia],1e-9),0,1) if ia!=ib else 0.;u=u*u*(3-2*u)
                user=(1-u)*np.array(schedule[ia]['command'])+u*np.array(schedule[ib]['command'])
            else:
                u=np.clip(t/max(c['design'].get('ramp_s',2.),1e-9),0,1);user=np.array(c['command'])*(u*u*(3-2*u))
            user=torch.tensor(user,dtype=torch.float32)[None];corrected=path.update(user,r[:2][None],yaw[None],r[7:9][None],state[41:42])[0]
            moving=bool(user.abs().sum()>.001);phase=(t/T+offset)%1 if moving else 0.
            heading=wrap(path.reference_yaw[0]-yaw);yaw_window.append(corrected[2]-state[41]);vel_window.append(corrected-state[[36,37,41]])
            ym=torch.stack(yaw_window[-25:]).mean();vm=torch.stack(vel_window[-25:]).mean(0)
            frame=torch.cat((state[39:42]*.25,state[42:45],corrected*corrected.new_tensor([2.,2.,.25]),
                (state[:12]-torch.tensor(default)),state[12:24]*.05,torch.tensor(actions[k-1],dtype=torch.float),
                torch.tensor([np.sin(phase*2*np.pi),np.cos(phase*2*np.pi)],dtype=torch.float),ym.reshape(1)*4,
                heading.sin().reshape(1),(heading.cos()-1).reshape(1),torch.tensor([float(moving)])))
            history=torch.roll(history,1,0);history[0]=frame
            if k==1:history[:]=frame
            scheduled=torch.tensor(((phase+np.array([0,.5,.5,0]))%1<duty)|~np.full(4,moving),dtype=torch.float)
            private=frame.clone();private[47]=vm[2]*4
            private=torch.cat((private,state[36:39]*2,(state[45:46]-env.cfg.rewards.base_height_target)*5,
                torch.tensor(ct[k-1],dtype=torch.float),scheduled,torch.tensor(z['feet_pos_world_m'][k-1,actor,:,2])*10,
                torch.zeros(3),torch.tensor([.8,0.]),torch.zeros(3),torch.tensor(z['applied_torque_Nm'][k-1,actor])*.05,torch.zeros(12)))
            assert frame.shape==(51,) and private.shape==(99,)
            if not z['valid'][k-1,actor] or not z['valid'][k,actor]:continue
            if k%2!=(k//50)%2:continue
            chunk=int((t+2)//4);hold=chunk%5==4
            # Remove training samples close to held-out history boundaries.
            if not hold and (t+2)%4<.4 and (chunk-1)%5==4:continue
            extra=torch.empty(0)
            if getattr(env.cfg.amp_cts,'feedback_v4',False):
                goal=torch.tensor(c['command'],dtype=torch.float)
                amplitude=(user.norm()/goal.norm().clamp(min=1e-6)).clamp(0,1).reshape(1)
                extra=torch.cat((goal,user.flatten(),state[36:38],amplitude))
            packed=torch.cat((history.flatten(),extra,private,torch.zeros(2)))
            states.append(packed);labels.append(torch.tensor(actions[k],dtype=torch.float32));caseids.append(ci);validation.append(hold)
        provenance.append(dict(clip=c['clip'],receipt=str(receipt_path),live_sha256=receipt['live_rollout_sha256'],actor=actor,phase_offset=offset))
        print('BC DATA',c['clip'],len(states),flush=True)
    samples=dict(observations=torch.stack(states),actions=torch.stack(labels),clip=torch.tensor(caseids),holdout=torch.tensor(validation),provenance=provenance)
    torch.save(samples,output/'samples.pt');return samples


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,default=BASE/'bootstrap');p.add_argument('--bc_steps',type=int,default=2000);p.add_argument('--samples',type=Path);p.add_argument('--warm-start',type=Path,default=ROOT/'artifacts/urdf4_direction_selection_20261116/best/model_100.pt')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    if not 1<=own.bc_steps<=3000:raise ValueError('Finite BC initialization is capped at 3000 steps')
    own.output.mkdir(parents=True,exist_ok=False);args=get_args();args.task='urdf4_amp_cts_v3';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV3Runner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
    runner.warm_start(own.warm_start)
    data=torch.load(own.samples,map_location='cpu') if own.samples else collect_samples(env,own.output)
    obs=data['observations'].to(args.rl_device);target=data['actions'].to(args.rl_device);hold=data['holdout'].to(args.rl_device);clips=data['clip'].to(args.rl_device)
    groups=[((clips==i)&~hold).nonzero().flatten() for i in clips.unique()];valid=hold.nonzero().flatten();model=runner.alg.actor_critic
    model.teacher_encoder.requires_grad_(True)
    params=list(model.actor.parameters())+list(model.teacher_encoder.parameters())+list(model.student_encoder.parameters());optimizer=torch.optim.Adam(params,lr=3e-4)
    def loss(ids):
        public,state,_=model.split(obs[ids]);teacher=model.teacher_context(state);student=model.student_encoder(public)
        t=model.executed_mean(public,model.actor(torch.cat((model.control_frame(public),teacher),1)));s=model.executed_mean(public,model.actor(torch.cat((model.control_frame(public),student),1)))
        from legged_gym.algorithms.urdf4_amp_v3 import stand_blend
        blend=stand_blend(public)[:,None];desired=blend*target[ids]+(1-blend)*model.stand_action
        bc=.5*((t-desired).square().mean()+(s-desired).square().mean())
        return bc+.1*(student-teacher.detach()).square().mean(),(s-desired).square().mean()
    with torch.no_grad():before=float(loss(valid)[1])
    for k in range(own.bc_steps):
        chosen=torch.randint(len(groups),(16,),device=args.rl_device);ids=torch.cat([g[torch.randint(len(g),(64,),device=args.rl_device)] for g in [groups[int(i)] for i in chosen]])
        total,mse=loss(ids);optimizer.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (k+1)%200==0:
            with torch.no_grad():print('BC',k+1,'holdout',float(loss(valid)[1]),flush=True)
    with torch.no_grad():after=float(loss(valid)[1])
    runner.save(str(own.output/'run/model_0.pt'))
    result=dict(samples=len(obs),bc_updates=own.bc_steps,before_mse=before,holdout_mse=after,checkpoint=str(own.output/'run/model_0.pt'),
        causal_history=True,source='actual qualified new-mass Gym trajectories',PPO_updates=0,behavior_accepted=False,contract=runner.contract)
    (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('BC FINISHED',after,flush=True)


if __name__=='__main__':main()
