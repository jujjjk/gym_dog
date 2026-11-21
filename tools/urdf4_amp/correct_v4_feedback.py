"""Finite corrective distillation on visited states. Assistance is not acceptance."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner
from teacher_v3 import Teacher,rot,rpy


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--seconds',type=float,default=18.)
    p.add_argument('--assistance',default='.95,.9,.8,.65');p.add_argument('--migrate-experimental',action='store_true')
    p.add_argument('--v5',action='store_true');p.add_argument('--vx',type=float,default=.5);p.add_argument('--data-only',action='store_true')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;assert 1<=own.steps<=2000 and own.seconds<=30
    own.output.mkdir(parents=True,exist_ok=False);args=get_args();args.task='urdf4_amp_cts_v5' if own.v5 else 'urdf4_amp_cts_v4';args.num_envs=8;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10
    saved=torch.load(own.model,map_location='cpu')
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
    runner_class=Urdf4AMPV4Runner
    if own.v5:
        from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner
        runner_class=Urdf4AMPV5Runner
        cfg.amp_cts.extra_reference=saved['experiment']['env']['amp_cts']['extra_reference']
        cfg.amp_cts.adaptive_feedback_phase=saved['experiment']['env']['amp_cts']['adaptive_feedback_phase']
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=runner_class(env,class_to_dict(train),None,device=args.rl_device);model=runner.alg.actor_critic
    if own.migrate_experimental:
        saved=torch.load(own.model,map_location=env.device)
        assert saved['contract']['native_prior_sha256']==runner.contract['native_prior_sha256'] and saved['contract']['public_size']==1029
        missing,unexpected=model.load_state_dict(saved['model_state_dict'],strict=False)
        assert not unexpected and all(k.startswith('legacy_policy.') for k in missing)
    else:runner.load(own.model,False)
    model.teacher_encoder.requires_grad_(False)
    command=torch.tensor([own.vx,0.,0.],device=env.device).repeat(env.num_envs,1);env.set_command_override(command);env.reset()
    if own.v5 and abs(own.vx-.4)<1e-6:spec=copy.deepcopy(torch.load(cfg.amp_cts.extra_reference,map_location='cpu')['manifest']['case']['spec'])
    else:spec=copy.deepcopy(next(c for c in json.load(open(Path(cfg.amp_cts.reference)/'manifest.json'))['clips'] if c['command']==[own.vx,0,0]))
    teachers=[Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec) for _ in range(env.num_envs)]
    default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy();fix=(teachers[0].q0-default)*(1/.928-1)
    levels=list(map(float,own.assistance.split(',')));mix=torch.tensor([levels[i%len(levels)] for i in range(env.num_envs)],device=env.device)[:,None]
    assert bool(((mix>=0)&(mix<=1)).all());observations=[];labels=[];peak=0.;resets=0
    for k in range(round(own.seconds/env.dt)):
        obs=env.obs_buf.clone();root=env.root_states.cpu().numpy().copy();root[:,:3]-=env.env_origins.cpu().numpy();q=env.dof_pos.cpu().numpy();dq=env.dof_vel.cpu().numpy()
        public=obs[:,:1029].cpu().numpy();phase=env.motion_phase.cpu().numpy();qcommands=[]
        for i,teacher in enumerate(teachers):
            qp=np.zeros(teacher.m.nq);qv=np.zeros(teacher.m.nv);qp[:3]=root[i,:3];qp[3:7]=root[i,[6,3,4,5]];qp[teacher.qa]=q[i]
            qv[:3]=root[i,7:10];qv[3:6]=rot(qp[3:7]).T@root[i,10:13];qv[teacher.va]=dq[i]
            # Direction comes exclusively from the same corrected public twist.
            # No hidden global route error enters the teacher labels.
            teacher.reference_xy=qp[:2].copy();teacher.reference_yaw=rpy(qp[3:7])[2];teacher.heading_integral=0.
            theta=(phase[i]-float(env.reference_phase_offsets[env.reference_id[i]]))%1
            qcommands.append(teacher.command(float(env.episode_steps[i])*env.dt-2.,qp,qv,phase_override=theta,
                planned_override=public[i,6:9]/np.array([2.,2.,.25]),amplitude_override=public[i,1028]))
        label=torch.tensor((np.stack(qcommands)+fix-default)/scale,device=env.device,dtype=torch.float)
        assert torch.isfinite(label).all();label=label.clamp(-cfg.normalization.clip_actions,cfg.normalization.clip_actions)
        with torch.no_grad():pred=model.act_inference(obs);action=mix*label+(1-mix)*pred
        if k%2==0:observations.append(obs.cpu());labels.append(label.cpu())
        with torch.no_grad():_,_,_,done,_=env.step(action)
        resets+=int(done.sum());peak=max(peak,float(env.raw_pd_torques.abs().max()))
        if (k+1)%250==0:print('CORRECTIVE',k+1,'assisted_peak',peak,'resets',resets,flush=True)
    online=torch.cat(observations).to(env.device);targets=torch.cat(labels).to(env.device)
    torch.save(dict(observations=online.cpu(),actions=targets.cpu(),assistance=levels,training_labels=True),own.output/'corrective_samples.pt')
    if own.data_only:
        (own.output/'result.json').write_text(json.dumps(dict(samples=len(online),PPO_updates=0,BC_updates=0,
            assisted_end_step_raw_peak_Nm=peak,assisted_resets=resets,assistance=levels,teacher_spec=spec,
            training_labels_only=True,not_AMP_expert_data=True,policy_accepted=False),indent=2)+'\n');return
    params=list(model.actor.parameters())+list(model.student_encoder.parameters());optimizer=torch.optim.Adam(params,lr=1e-4)
    def loss(obs,target):
        public,state,_=model.split(obs);student=model.public_mean(public)
        tc=model.teacher_context(state);teacher=model.executed_mean(public,model.actor(torch.cat((model.control_frame(public),tc),1)))
        context=model.student_encoder(public)
        return .5*((student-target).square().mean()+(teacher-target).square().mean())+.03*(context-tc.detach()).square().mean()
    for k in range(own.steps):
        a=torch.randint(len(online),(1536,),device=env.device);b=torch.multinomial(runner.anchor_weights,512,replacement=True)
        total=loss(torch.cat((online[a],runner.anchor_observations[b])),torch.cat((targets[a],runner.anchor_actions[b])))
        optimizer.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (k+1)%200==0:print('CORRECTIVE FIT',k+1,float(total),flush=True)
    runner.seed_migration=dict(source=str(own.model),source_sha256=hashlib.sha256(own.model.read_bytes()).hexdigest(),corrective_data_sha256=hashlib.sha256((own.output/'corrective_samples.pt').read_bytes()).hexdigest(),assistance=levels)
    runner._freeze_seed_policy();runner.save(str(own.output/'model_0.pt'))
    (own.output/'result.json').write_text(json.dumps(dict(samples=len(online),BC_updates=own.steps,PPO_updates=0,assisted_raw_peak_Nm=peak,
        assisted_resets=resets,behavior_accepted=False,teacher_used_only_for_training_labels=True,assistance=levels),indent=2)+'\n')


if __name__=='__main__':main()
