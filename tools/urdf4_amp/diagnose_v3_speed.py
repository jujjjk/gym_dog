"""Causal high-speed ablations and AMP simulator-domain audit. No training."""
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
from legged_gym.algorithms.urdf4_v3_prior import reference_actions
from legged_gym.envs.urdf4.amp_reference import amp_features


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--focused',action='store_true')
    p.add_argument('--native',action='store_true')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    own.output.mkdir(parents=True,exist_ok=False);args=get_args();args.task='urdf4_amp_cts_v3';args.headless=True
    commands=[(.25,0,0),(.35,0,0),(.5,0,0),(.6,0,0),(0,0,-.8)]
    modes=['student','prior_only','target_ramp','target_ramp_student']
    if own.focused:commands=[(.35,0,0),(.5,0,0),(.6,0,0)];modes=['student','prior_only','fixed_query','canonical_cycle','recorded','fixed_query_student']
    if own.native:commands=[(.35,0,0),(.5,0,0),(.6,0,0)];modes=['prior_only','native','native_fixed','native_canonical']
    args.num_envs=len(commands)*len(modes)
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device)
    model_path=ROOT/'artifacts/urdf4_v3_selection_20261117/best/model_700.pt';runner.load(model_path,False);model=runner.alg.actor_critic
    if own.native:
        from build_native_prior import build
        from legged_gym.algorithms.urdf4_native_prior import native_actions
        native={key:value.to(env.device) if isinstance(value,torch.Tensor) else value for key,value in build(env,own.output/'native_prior.pt').items()}
    manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text());cache={};stand=[];audit=[];issued_bank=[]
    for c in manifest['clips']:
        rp=Path(c['isaac_gym_validation']['receipt']);r=json.loads(rp.read_text());file=rp.parent/'live_rollout.npz'
        if file not in cache:cache[file]=np.load(file)
        live=cache[file];index=next(i for i,v in enumerate(r['cases']) if v['clip']==c['isaac_gym_validation']['case']['clip'])
        raw=np.load(Path(cfg.amp_cts.reference)/c['source_folder']/'physical.npz')
        correction=(raw['actuator_default']-env.default_dof_pos[0].cpu().numpy())*(1/.928-1)
        t=live['time_s']-.02;mask=(t>-.5)&(t<-.05)
        stand.append((live['issued_joint_targets'][mask,index].mean(0)+correction-env.default_dof_pos[0].cpu().numpy())/env.action_scale.cpu().numpy())
        issued_bank.append((live['issued_joint_targets'][:,index]+correction-env.default_dof_pos[0].cpu().numpy())/env.action_scale.cpu().numpy())
        if c['family'] not in ['stand','forward','turn_right']:continue
        real=np.load(Path(cfg.amp_cts.reference)/(c['clip']+'.npz'))['amp_features'];mask=(live['time_s']>=5)&(live['time_s']<55)
        actual=live['amp_features'][mask,index];count=min(len(real),len(actual))-1
        q=torch.tensor(c['command'],device=env.device).repeat(count,1)
        er=torch.tensor(np.c_[real[:count],real[1:count+1]],device=env.device,dtype=torch.float)
        eg=torch.tensor(np.c_[actual[:count],actual[1:count+1]],device=env.device,dtype=torch.float)
        with torch.no_grad():
            real_input=runner.style.inputs(er,q);gym_input=runner.style.inputs(eg,q)
            before=runner.style.net(gym_input).flatten();expert=runner.style.net(real_input).flatten()
        audit.append(dict(clip=c['clip'],command=c['command'],Mj_mean=real[:count].mean(0).tolist(),Gym_mean=actual[:count].mean(0).tolist(),
            Gym_std=actual[:count].std(0).tolist(),Mj_std=real[:count].std(0).tolist(),D700_real=float(expert.mean()),D700_live_teacher=float(before.mean()),real_input=real_input.cpu(),gym_input=gym_input.cpu()))
    latest=torch.load(ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_v3/Nov16_19-46-19_newmass_omni_v3/model_30000.pt',map_location=env.device)
    runner.style.net.load_state_dict(latest['amp_state']['net'])
    for row in audit:
        with torch.no_grad():
            er=row.pop('real_input').to(env.device);eg=row.pop('gym_input').to(env.device)
            row.update(D30000_real=float(runner.style.net(er).mean()),D30000_live_teacher=float(runner.style.net(eg).mean()))
            fixes={}
            for name,lo,hi in [('q',0,12),('dq',12,24),('feet',24,36),('v',36,39),('omega',39,42),('gravity',42,45),('height',45,46)]:
                swapped=eg.clone();swapped[:,lo:hi]=er[:,lo:hi];swapped[:,46+lo:46+hi]=er[:,46+lo:46+hi]
                fixes[name]=float(runner.style.net(swapped).mean())
            row['D30000_live_teacher_with_source_feature']=fixes
    (own.output/'domain_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    target=torch.tensor(commands,device=env.device,dtype=torch.float).repeat(len(modes),1)
    stand=torch.tensor(np.stack(stand),device=env.device,dtype=torch.float)
    ids=env._select_reference(target);initial=stand[ids];env.set_command_override(target);env.reset()
    issued_bank=torch.tensor(np.stack(issued_bank),device=env.device,dtype=torch.float)
    fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool);first=torch.full((env.num_envs,),float('nan'),device=env.device);clock=[0.]
    def no_reset(which):
        fresh=which[~fallen[which]];first[fresh]=clock[0];fallen[which]=True
    env.reset_idx=no_reset;original=env._compute_torques;peak=torch.zeros(env.num_envs,device=env.device);counts=torch.zeros(env.num_envs,12,device=env.device);ticks=peak.clone()
    raw17=counts.clone();limited=counts.clone();position_error=counts.clone()
    def torques(actions):
        value=original(actions);alive=(~fallen).float();peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*alive));raw17.add_((env.raw_pd_torques.abs()>17)*alive[:,None]);counts.add_(env.motor_torques.square()*alive[:,None]);position_error.add_((env.response_target-env.dof_pos).abs()*alive[:,None]);ticks.add_(alive);return value
    env._compute_torques=torques;records=[]
    with torch.no_grad():
        for k in range(round(own.seconds/env.dt)):
            clock[0]=k*env.dt;obs=env.obs_buf.clone();obs[fallen]=0.
            prior=reference_actions(obs[:,:51],model.prior_commands,model.prior_table,model.stand_action)
            student=model.act_inference(obs);residual=student-prior
            hint=target+env.commands[:,:3]-env.external_commands
            frame=obs[:,:51].clone();frame[:,6:9]=hint*hint.new_tensor([2.,2.,.25]);frame[:,50]=1.
            planned=reference_actions(frame,model.prior_commands,model.prior_table,model.stand_action)
            amplitude=(env.external_commands.norm(dim=1)/target.norm(dim=1).clamp(min=1e-6)).clamp(0,1)[:,None]
            ramped=initial+amplitude*(planned-initial)
            fixed=frame.clone();fixed[:,6:9]=target*target.new_tensor([2.,2.,.25])
            fixed_walk=reference_actions(fixed,model.prior_commands,model.prior_table,model.stand_action)
            fixed_walk=initial+amplitude*(fixed_walk-initial)
            canonical=fixed.clone();periods=1/env.reference_frequencies[ids].clamp(min=1e-6)
            phase=torch.remainder(max(k*env.dt-2.,0.)/periods+env.reference_phase_offsets[ids],1.)
            canonical[:,45]=torch.sin(phase*2*np.pi);canonical[:,46]=torch.cos(phase*2*np.pi)
            u=min(max((k*env.dt-2.)/3.,0.),1.);blend=u*u*(3-2*u)
            canonical_walk=initial+blend*(reference_actions(canonical,model.prior_commands,model.prior_table,model.stand_action)-initial)
            recorded=issued_bank[ids,min(k,issued_bank.shape[1]-1)]
            candidates=dict(student=student,prior_only=prior,target_ramp=ramped,target_ramp_student=ramped+residual,
                fixed_query=fixed_walk,fixed_query_student=fixed_walk+residual,canonical_cycle=canonical_walk,recorded=recorded)
            if own.native:
                def evaluate_native(f):return native_actions(f,model.prior_commands,native['actions'],native['counts'],native['offsets'],model.stand_action)
                candidates.update(native=evaluate_native(obs[:,:51]),native_fixed=initial+amplitude*(evaluate_native(fixed)-initial),
                    native_canonical=initial+blend*(evaluate_native(canonical)-initial))
            action=student.clone();n=len(commands)
            for mi,mode in enumerate(modes):action[mi*n:(mi+1)*n]=candidates[mode][mi*n:(mi+1)*n]
            action[fallen]=0.
            desired=env.default_dof_pos+env.action_scale*action
            _,_,_,_,_=env.step(action)
            limited.add_((desired-env.position_target).abs()>.005)
            records.append(dict(features=amp_features(env).cpu().numpy(),roots=env.root_states.cpu().numpy().copy(),contact=env.get_foot_contact_mask().cpu().numpy(),
                height=(env.feet_pos[:,:,2]-env.env_origins[:,None,2]-.02).cpu().numpy(),phase=env.motion_phase.cpu().numpy().copy(),
                action=action.cpu().numpy(),prior=prior.cpu().numpy(),desired=desired.cpu().numpy(),limited=env.position_target.cpu().numpy().copy(),
                response=env.response_target.cpu().numpy().copy(),q=env.dof_pos.cpu().numpy().copy(),raw=env.raw_pd_torques.cpu().numpy().copy(),
                external=env.external_commands.cpu().numpy().copy(),servo=env.commands[:,:3].cpu().numpy().copy(),cross=env.path.cross_error.cpu().numpy().copy(),heading=env.path.heading_error.cpu().numpy().copy(),valid=(~fallen).cpu().numpy()))
    a={key:np.stack([r[key] for r in records]) for key in records[0]};t=np.arange(1,len(records)+1)*env.dt;rows=[]
    for i in range(env.num_envs):
        mask=(t>=6)&a['valid'][:,i];f=a['features'][mask,i];ct=a['contact'][mask,i];nr=ct.sum(1);dg=(ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2])
        row=dict(mode=modes[i//len(commands)],command=target[i].tolist(),fallen=bool(fallen[i]),first_fall_s=float(first[i]),raw_peak_Nm=float(peak[i]),
            raw_over17_fraction=float((raw17[i]/ticks[i].clamp(min=1)).mean()),motor_joint_rms_Nm=torch.sqrt(counts[i]/ticks[i].clamp(min=1)).tolist(),
            target_rate_limited_fraction=(limited[i]/len(t)).tolist(),mean_abs_PD_error_rad=(position_error[i]/ticks[i].clamp(min=1)).tolist(),
            mean_velocity=f[:,[36,37,41]].mean(0).tolist() if len(f) else None,mean_height_m=float(f[:,45].mean()) if len(f) else None,
            illegal_support_fraction=float(((nr==0)|(nr==1)|((nr==2)&~dg)).mean()) if len(nr) else 1.,
            cross_max_m=float(abs(a['cross'][mask,i]).max()) if mask.any() else None,heading_mean_deg=float(abs(a['heading'][mask,i]).mean()*180/np.pi) if mask.any() else None)
        rows.append(row);print('DIAG',json.dumps(row),flush=True)
    np.savez_compressed(own.output/'rollout.npz',**a,time_s=t)
    (own.output/'result.json').write_text(json.dumps(dict(cases=rows,checkpoint=str(model_path),seconds=own.seconds,pure_public_state_ablations=True,training=False),indent=2)+'\n')


if __name__=='__main__':main()
