"""Bounded reference search on the actual Gym plant; never policy acceptance.

Only Cartesian gait geometry/timing and teacher balance parameters vary.
Mass, PD, actuator limits, free root physics and contact thresholds are fixed.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.urdf4.amp_reference import amp_features
from teacher_v3 import Teacher,rot
from evaluate_v3_policy import starts
from probe_public_balance import public_state


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--speeds',default='.4,.45,.6')
    p.add_argument('--population',type=int,default=16);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--ramp-s',type=float,default=3.,help='Reference startup ramp; must be recorded separately from student acceptance')
    p.add_argument('--specs',type=Path,help='Replay a previously screened list of candidate specifications')
    p.add_argument('--public-held',action='store_true',help='Use public FK height/vz, corrected horizontal velocity, and held Cartesian geometry')
    p.add_argument('--mid-cross',action='store_true',help='Use V5 extra cross-track recovery only between .35 and .50')
    p.add_argument('--mid-integral',type=float,default=0.,help='Use the same desired heading correction in the public path integral')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    assert 1<=own.population<=32 and 8<=own.seconds<=60
    own.output.mkdir(parents=True,exist_ok=False)
    manifest=json.load(open(ROOT/'artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/gym_verified/manifest.json'))
    parent=next(c for c in manifest['clips'] if c['command']==[.5,0,0])
    source={key:copy.deepcopy(parent[key]) for key in ('clip','family','command','design','lead','feed','phase_alignment_offset_cycles')}
    rng=np.random.default_rng(74117);specs=[]
    if own.specs:
        specs=[{key:copy.deepcopy(c[key]) for key in [*source,*[k for k in ('geometry_command','geometry_period_s','height_estimate_bias_m') if k in c]]} for c in json.loads(own.specs.read_text())]
    else:
        specs=[copy.deepcopy(source)]
        for speed in map(float,own.speeds.split(',')):
            for j in range(own.population):
                c=copy.deepcopy(source);c['clip']='forward_%03d_trial%02d'%(round(speed*1000),j);c['command']=[speed,0.,0.];d=c['design']
                # Same useful stride, varied period and calibrated fixed-time
                # lead. Joint-angle amplitude scaling would also shrink lift.
                if j<4:d['period_s']=round((.48*.5/speed+[-.04,-.02,0.,.02][j])/.02)*.02
                else:d['period_s']=float(rng.choice([.44,.48,.52,.56,.60])) if speed<.5 else float(rng.choice([.40,.42,.44,.46,.48]))
                d['body_height_m']+=float(rng.choice([-.002,0.,.002,.004]))
                c['lead']+=float(rng.choice([-.012,0.,.012,.024]));c['feed']+=float(rng.choice([-.025,0.,.025]))
                if j>=4:
                    duty=float(rng.choice([.52,.55,.58]));old=d['stance_ratio'];d['stance_ratio']=duty
                    d['leg_stance_ratio']=(np.asarray(d['leg_stance_ratio'])+duty-old).tolist()
                    clear=np.asarray(d['leg_clearance_m']);clear[2:]=rng.choice([.028,.032,.036,.040],2)
                    d['leg_clearance_m']=clear.tolist();d['velocity_gain']=float(rng.choice([2.,3.4,5.]))
                specs.append(c)
    assert len(specs)<=97
    args=get_args();args.task='urdf4_amp_cts_v4';args.num_envs=len(specs);args.headless=True
    if own.mid_cross:args.task='urdf4_amp_cts_v5'
    cfg,_=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10.;cfg.commands.fast_ramp_s=own.ramp_s
    if own.mid_integral:
        assert own.mid_cross and 0<own.mid_integral<=.5
        cfg.direction.mid_cross_integral_gain=own.mid_integral
    if own.public_held:cfg.direction.speed_feedback_enabled=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    teacher_specs=copy.deepcopy(specs)
    for c in teacher_specs:
        if own.public_held or 'geometry_command' in c:c['command']=c.get('geometry_command',[.5,0.,0.])
        if 'geometry_period_s' in c:c['design']['period_s']=c['geometry_period_s']
    teachers=[Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',c) for c in teacher_specs]
    for teacher,spec in zip(teachers,specs):teacher.T=spec['design']['period_s']
    target=torch.tensor([c['command'] for c in specs],device=env.device);env.set_command_override(target);env.reset()
    default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy()
    correction=np.stack([(t.q0-default)*(1/.928-1) for t in teachers])
    fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool);peak=torch.zeros(env.num_envs,device=env.device)
    env.reset_idx=lambda ids:fallen.__setitem__(ids,True)
    original=env._compute_torques
    def torques(action):
        value=original(action);peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*(~fallen)));return value
    env._compute_torques=torques;records=[];slots=[env.foot_legs.index(x) for x in ('FL','FR','RL','RR')]
    canonical=torch.zeros(env.num_envs,device=env.device);was_active=torch.zeros_like(fallen)
    frequencies=torch.tensor([1./c['design']['period_s'] for c in specs],device=env.device)
    for k in range(round(own.seconds/env.dt)):
        root=env.root_states.cpu().numpy().copy();root[:,:3]-=env.env_origins.cpu().numpy()
        q=env.dof_pos.cpu().numpy();dq=env.dof_vel.cpu().numpy();external=env.external_commands.cpu().numpy();commands=[]
        t=k*env.dt-2.;phase=(canonical.cpu().numpy().astype(np.float64)-source['phase_alignment_offset_cycles'])%1
        sine=torch.sin(2*torch.pi*canonical);cosine=torch.cos(2*torch.pi*canonical)
        env.obs_history[:,0,45]=sine;env.obs_history[:,0,46]=cosine;env.obs_buf[:,45]=sine;env.obs_buf[:,46]=cosine
        observations=env.obs_buf.cpu().numpy().copy()
        for i,teacher in enumerate(teachers):
            if own.public_held:
                qp,qv,_,_=public_state(teacher,observations[i,:51].copy(),default)
                qp[2]-=specs[i].get('height_estimate_bias_m',0.)
                qv[:2]=observations[i,1026:1028]
                teacher.reference_xy[:]=0.;teacher.reference_yaw=0.;teacher.heading_integral=0.
                amplitude=float(observations[i,1028]);planned=observations[i,6:9]/np.array([2.,2.,.25])
            else:
                qp=np.zeros(teacher.m.nq);qv=np.zeros(teacher.m.nv);qp[:3]=root[i,:3];qp[3:7]=root[i,[6,3,4,5]];qp[teacher.qa]=q[i]
                qv[:3]=root[i,7:10];qv[3:6]=rot(qp[3:7]).T@root[i,10:13];qv[teacher.va]=dq[i]
                if k<2:teacher.reference_xy=qp[:2].copy();teacher.reference_yaw=0.
                amplitude=float(np.linalg.norm(external[i])/max(np.linalg.norm(teacher.cmd),1e-9));planned=external[i]
            commands.append(teacher.command(t,qp,qv,phase_override=phase[i],planned_override=planned,amplitude_override=amplitude))
        actions=torch.tensor((np.stack(commands)+correction-default)/scale,device=env.device,dtype=torch.float)
        actions[fallen]=0.
        with torch.no_grad():env.step(actions)
        active=env.phase_was_active;canonical[active&~was_active]=source['phase_alignment_offset_cycles']%1
        canonical=torch.remainder(canonical+frequencies*env.dt*active.float(),1.)
        canonical[~active]=0.;was_active=active.clone()
        if specs[0]['command']==[.5,0.,0.] and specs[0]['design']['period_s']==.48:
            assert torch.equal(canonical[0],env.motion_phase[0]),'Positive-control clock must exactly match validated playback'
        records.append(dict(observations=observations,actions=actions.cpu().numpy(),features=amp_features(env).cpu().numpy(),
            contact=env.get_foot_contact_mask()[:,slots].cpu().numpy(),height=(env.feet_pos[:,slots,2]-env.env_origins[:,None,2]-.02).cpu().numpy(),
            root=env.root_states.cpu().numpy().copy(),valid=(~fallen).cpu().numpy(),phase=phase))
        if (k+1)%250==0:print('GYM SEARCH',round(k*env.dt,2),'falls',int(fallen.sum()),flush=True)
    a={key:np.stack([r[key] for r in records]) for key in records[0]};time=np.arange(1,len(records)+1)*env.dt;moving=time>=6;rows=[]
    for i,c in enumerate(specs):
        mask=moving&a['valid'][:,i];features=a['features'][mask,i];ct=a['contact'][mask,i];height=a['height'][mask,i]
        if not len(ct):rows.append(dict(spec=c,passed=False,rejection_reasons=['fall'],score=1e6));continue
        n=ct.sum(1);di=(ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2]);gaps=[];lifts=[]
        for active in [ct,~ct]:
            values=[]
            for aa,bb in [(0,3),(1,2)]:
                ea=starts(active[:,aa]);eb=starts(active[:,bb])
                if len(ea) and len(eb):values.extend(np.min(abs(eb-e))*.02 for e in ea)
            gaps.append(float(np.quantile(values,.95)*1000) if values else 1e4)
        for leg in range(4):lifts.append(len(starts((height[:,leg]>.012)&~ct[:,leg])))
        velocity=features[:,[36,37,41]].mean(0);quats=a['root'][mask,i,3:7];x,y,z,w=quats.T
        heading=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));origin_y=float(env.env_origins[i,1]);cross=float(abs(a['root'][mask,i,1]-origin_y).max())
        clearance=np.quantile(height,.95,axis=0)*1000;illegal=float(((n==1)|((n==2)&~di)).mean());flight=float((n==0).mean())
        checks=dict(fall=bool(fallen[i]),raw_torque=float(peak[i])>17,flight=flight>0,support=illegal>.01,
            tracking=np.linalg.norm(velocity[:2]-c['command'][:2])>.05 or abs(velocity[2])>.12,
            direction=cross>.05 or np.mean(abs(heading))*180/np.pi>2.,clearance=min(clearance)<12,
            diagonal_timing=max(gaps)>60.001,alternation=min(lifts)<(own.seconds-6)/c['design']['period_s']*.65)
        reasons=[name for name,bad in checks.items() if bad]
        score=100*len(reasons)+max(0,float(peak[i])-17)+100*np.linalg.norm(velocity[:2]-c['command'][:2])+cross
        rows.append(dict(spec=c,passed=not reasons,rejection_reasons=reasons,score=score,raw_peak_Nm=float(peak[i]),
            mean_velocity=velocity.tolist(),clearance_p95_mm=clearance.tolist(),flight_fraction=flight,illegal_support_fraction=illegal,
            cross_max_m=cross,heading_mean_abs_deg=float(np.mean(abs(heading))*180/np.pi),diagonal_event_p95_ms=gaps,lift_events_per_foot=lifts))
    np.savez_compressed(own.output/'rollout.npz',**a,time_s=time)
    result=dict(seconds=own.seconds,ramp_s=own.ramp_s,urdf_sha256=env.urdf_sha256,teacher_controller_used=not own.public_held,
        mid_cross_integral_gain=own.mid_integral,
        public_model_based_controller=own.public_held,policy_accepted=False,
        plant_settings=dict(control=class_to_dict(cfg.control),actuator=class_to_dict(cfg.rs01_actuator)),
        root_or_joint_overwrite_during_rollout=False,automatic_resets=False,cases=rows)
    (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for speed in sorted(set(c['command'][0] for c in specs)):
        best=min((r for r in rows if r['spec']['command'][0]==speed),key=lambda r:r['score'])
        print('BEST',speed,best['spec']['clip'],best['passed'],best['rejection_reasons'],best.get('mean_velocity'),best.get('raw_peak_Nm'),flush=True)


if __name__=='__main__':main()
