"""Read-only, frozen-policy path tests with explicit pose-source labeling."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.envs.urdf4.path_runtime import PathTrackingSession
from legged_gym.envs.urdf4.path_tracking import PathGains,rotate_xy,wrap
from evaluate_cts_behavior import events
from direction_rollout import DirectionRolloutSession


def run(env,policy,cases,seconds,seed,pose_source,gains,stop_s=0.,scope='experimental_all'):
    torch.manual_seed(seed);np.random.seed(seed)
    labels=torch.arange(env.num_envs,device=env.device)//4
    modes=torch.tensor([c['mode'] for c in cases],device=env.device)[labels]
    commands=torch.tensor([c['command'] for c in cases],device=env.device)[labels]
    moving=torch.tensor([c.get('moving',True) for c in cases],device=env.device)[labels]
    env.reference_reset_probability=0.;env.command_resample_steps=10**9
    env.max_episode_length=round((seconds+stop_s+10)/env.dt)
    env.set_command_override(torch.zeros_like(commands),gait_enabled=False);obs,_=env.reset()
    with torch.no_grad():
        for _ in range(100):obs,_,_,_,_=env.step(torch.zeros(env.num_envs,12,device=env.device))
    holder={};queue=[]
    def simulated_localization(public,dt):
        xy,yaw,velocity=holder['session'].truth_pose()
        sample=(xy+torch.randn_like(xy)*.003,wrap(yaw+torch.randn_like(yaw)*np.deg2rad(.2)),velocity+torch.randn_like(velocity)*.005)
        queue.append(tuple(x.clone() for x in sample))
        if len(queue)>3:queue.pop(0)
        return queue[0]  # causal 40ms latency after the queue fills
    session_class=DirectionRolloutSession if hasattr(env,'path') else PathTrackingSession
    session=session_class(env,policy,mode=modes,
        pose_source='external' if pose_source=='localization_sim' else pose_source,
        pose_provider=simulated_localization if pose_source=='localization_sim' else None,gains=gains,control_scope=scope)
    holder['session']=session
    env.episode_length_buf.zero_();env.clock_offset.zero_();env.velocity_error.reset(torch.arange(env.num_envs,device=env.device))
    env.velocity_drift.zero_();env.push_wait[~env.nominal_mask]=5.;env.push_wait[env.nominal_mask]=1.e9
    env.compute_observations();obs=env.obs_buf
    data={k:[] for k in ['time','xy','yaw','command','estimated_xy','measurement_truth_xy','contact','feet','done','state','reference_xy','reference_yaw','push_count','push_force']}
    raw_peak=torch.zeros(env.num_envs,device=env.device);motor_sum=torch.zeros(env.num_envs,12,device=env.device);motor_sq=motor_sum.clone();ticks=0
    original=env._compute_torques
    def torque(action):
        nonlocal ticks
        result=original(action);raw_peak.copy_(torch.maximum(raw_peak,env.raw_pd_torques.abs().amax(1)))
        motor_sum.add_(env.motor_torques.abs());motor_sq.add_(env.motor_torques.square());ticks+=1
        return result
    env._compute_torques=torque;start=time.time()
    try:
        with torch.no_grad():
            for step in range(round((seconds+stop_s)/env.dt)):
                external=commands if step*env.dt<seconds else torch.zeros_like(commands)
                measurement_xy=session.truth_pose()[0].clone()
                old_head=env.measured_yaw_error.head
                enabled=moving if step*env.dt<seconds else torch.zeros_like(moving)
                (obs,_,_,done,info),telemetry=session.step(external,obs,moving=enabled)
                if step==0:
                    assert not bool(done.any())
                    assert env.measured_yaw_error.head==(old_head+1)%env.measured_yaw_error.window.shape[1], 'Sensor window was advanced more than once'
                xy,yaw,_=session.truth_pose();slots=[env.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
                values=[step*env.dt,xy,yaw,telemetry['command'],telemetry['estimated_xy'],measurement_xy,info['amp']['contact'],
                    env.feet_pos[:,slots,2]-.02,done,info['amp']['next'],session.controller.reference_xy,session.controller.reference_yaw,env.push_count,env.push_force]
                for k,v in zip(data,values):data[k].append(v if isinstance(v,float) else v.cpu().numpy().copy())
    finally:env._compute_torques=original;session.close()
    a={k:np.asarray(v) for k,v in data.items()};cut=round(3/env.dt);end=round(seconds/env.dt);rows=[]
    for i,case in enumerate(cases):
        external=np.array(case['command']);norm=np.linalg.norm(external[:2]);tangent=external[:2]/max(norm,1.e-6)
        # Tests initially cover fixed-heading straight segments; reference yaw
        # is kept explicitly so arc/turn variants remain auditable.
        normal=np.array([-tangent[1],tangent[0]])
        for kind,ids in [('nominal',[4*i]),('robust',[4*i+1,4*i+2,4*i+3])]:
            if env.cfg.env.test and kind=='robust':kind='nominal_replicas'
            xy=a['xy'][cut:end,ids];reference=a['reference_xy'][cut:end,ids]
            refyaw=a['reference_yaw'][cut:end,ids]
            normals=np.stack((-np.sin(refyaw)*tangent[0]-np.cos(refyaw)*tangent[1],np.cos(refyaw)*tangent[0]-np.sin(refyaw)*tangent[1]),-1)
            cross=((xy-reference)*normals).sum(-1)
            position_error=np.linalg.norm(xy-reference,axis=-1)
            if norm<1.e-6:
                # A turn in place still has a position target. A zero tangent
                # must not make every drifting rotation pass with zero error.
                cross=position_error
            heading=np.arctan2(np.sin(a['yaw'][cut:end,ids]-refyaw),np.cos(a['yaw'][cut:end,ids]-refyaw))
            cycle=25;headmean=np.stack([np.convolve(heading[:,j],np.ones(cycle)/cycle,'valid') for j in range(len(ids))],1)
            contact=a['contact'][cut:end,ids];n=contact.sum(2);diagonal=(contact[:,:,0]&contact[:,:,3])|(contact[:,:,1]&contact[:,:,2])
            bad=(n==0)|(n==1)|((n==2)&~diagonal);gaps=[];lifts=[];alternations=[]
            for j in range(len(ids)):
                c=contact[:,j];lifts.append([len(events(~c[:,leg])) for leg in range(4)])
                group=np.full(len(c),-1);group[c[:,0]&c[:,3]&~c[:,1]&~c[:,2]]=0;group[c[:,1]&c[:,2]&~c[:,0]&~c[:,3]]=1
                seq=group[group>=0];alternations.append(int(np.sum(seq[1:]!=seq[:-1])))
                for aa,bb in [(0,3),(1,2)]:
                    ea,eb=events(c[:,aa]),events(c[:,bb])
                    if len(ea) and len(eb):gaps.extend(float(np.min(abs(eb-k)))*env.dt for k in ea)
            clear=np.percentile(a['feet'][cut:end,ids].reshape(-1,4),95,axis=0)*1000
            reset=int(a['done'][:,ids].sum());flight=float((n==0).mean());illegal=float(bad.mean())
            gap=float(np.percentile(gaps,95)) if gaps else 1.
            peak=float(raw_peak[ids].max());reasons=[]
            if reset:reasons.append('reset')
            if flight:reasons.append('flight')
            if illegal>(.01 if kind=='robust' else .001):reasons.append('support')
            if kind!='robust' and gap>.040001:reasons.append('diagonal_timing')
            elapsed=(end-cut)*env.dt
            if min(min(x) for x in lifts)<.7*elapsed*2 or min(alternations)<.7*elapsed*4:reasons.append('gait_not_alternating')
            if min(clear)<10 or max(clear)>55:reasons.append('clearance')
            if peak>17:reasons.append('raw_torque')
            stop={}
            if stop_s:
                tail=a['state'][end:,ids];stop_v=tail[:,:,[36,37,41]]
                stop=dict(resets=int(a['done'][end:,ids].sum()),
                    mean_linear_m_s=float(np.linalg.norm(stop_v[:,:,:2],axis=-1).mean()),
                    mean_abs_yaw_rad_s=float(abs(stop_v[:,:,2]).mean()),
                    displacement_m=float(np.linalg.norm(a['xy'][-1,ids]-a['xy'][end-1,ids],axis=-1).max()))
                averaged=np.stack([np.stack([np.convolve(stop_v[:,j,k],np.ones(25)/25,'valid') for k in range(3)],1) for j in range(len(ids))],1)
                stop['cycle_linear_error_m_s']=float(np.linalg.norm(averaged[:,:,:2],axis=-1).mean())
                stop['cycle_yaw_error_rad_s']=float(abs(averaged[:,:,2]).mean())
            recoveries=[];recovery_failed=0
            for j,slot in enumerate(ids):
                active=np.linalg.norm(a['push_force'][:end,slot],axis=-1)>.01
                endings=np.where(active[:-1]&~active[1:])[0]+1
                for ending in endings:
                    if ending<cut or ending+round(5/env.dt)>=end:continue
                    start_index=ending-cut;success=None
                    for offset in range(round(5/env.dt)-25):
                        k=start_index+offset
                        if np.max(abs(cross[k:k+25,j]))<=.05 and k<len(headmean) and abs(headmean[k,j])<=np.deg2rad(3):
                            success=(offset+25)*env.dt;break
                    if success is None:recovery_failed+=1
                    else:recoveries.append(success)
            crossmax=float(abs(cross).max());crossterm=float(abs(cross[-1]).max())
            headingmean=float(abs(headmean).mean()*180/np.pi)
            path_ok=crossmax<=(.10 if kind=='robust' else .05) and crossterm<=(.05 if kind=='robust' else .03) and headingmean<=2.
            rows.append(dict(name=case['name'],command=case['command'],mode=case['mode'],kind=kind,
                cross_max_m=crossmax,cross_terminal_m=crossterm,cross_rms_m=float(np.sqrt((cross**2).mean())),
                position_max_m=float(position_error.max()),position_terminal_m=float(position_error[-1].max()),
                heading_cycle_mean_abs_deg=headingmean,heading_cycle_p95_abs_deg=float(np.percentile(abs(headmean),95)*180/np.pi),
                odometry_terminal_error_m=float(np.linalg.norm(a['estimated_xy'][end-1,ids]-a['measurement_truth_xy'][end-1,ids],axis=-1).max()),
                actual_body_velocity=a['state'][cut:end,ids][:,:,[36,37,41]].mean((0,1)).tolist(),
                raw_peak_Nm=peak,motor_mean_abs_Nm=float((motor_sum[ids]/ticks).mean()),motor_rms_Nm=float((motor_sq[ids]/ticks).mean().sqrt()),
                per_joint_motor_rms_Nm=(motor_sq[ids].mean(0)/ticks).sqrt().tolist(),resets=reset,flight_fraction=flight,
                illegal_support_fraction=illegal,diagonal_touchdown_p95_s=gap,clearance_p95_mm=clear.tolist(),
                push_events=int(a['push_count'][-1,ids].sum()),push_recovery_max_s=max(recoveries) if recoveries else None,
                failed_push_recoveries=recovery_failed,stop=stop,gait_passed=not reasons,rejection_reasons=reasons,path_passed=path_ok and recovery_failed==0))
    return dict(seconds=seconds,stop_s=stop_s,seed=seed,pose_source=pose_source,
        privileged_pose=pose_source=='oracle',frozen_lower_policy=True,wall_seconds=time.time()-start,
        simulated_localization_assumption=dict(position_std_m=.003,yaw_std_deg=.2,velocity_std_m_s=.005,latency_s=.04) if pose_source=='localization_sim' else None,
        controller=dict(session.controller.metadata(),control_scope=scope),results=rows),a


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15.);p.add_argument('--seeds',default='1')
    p.add_argument('--pose_sources',default='oracle,student');p.add_argument('--robust',action='store_true')
    p.add_argument('--speeds',default='.18,.28');p.add_argument('--gains',type=Path)
    p.add_argument('--stop_s',type=float,default=0.)
    p.add_argument('--suite',choices=['straight','omni'],default='straight')
    p.add_argument('--scope',choices=['forward','experimental_all'],default='experimental_all')
    p.add_argument('--omni_forward_speed',type=float,default=.28)
    p.add_argument('--skip_march',action='store_true')
    p.add_argument('--direction_task',action='store_true')
    p.add_argument('--policy',type=Path)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    if own.direction_task:
        expected='localization_sim' if own.robust else 'oracle'
        if own.pose_sources!=expected:
            raise ValueError('The native direction environment uses %s localization; select --pose_sources %s'%(expected,expected))
    args=get_args();args.task='urdf4_amp_cts_direction_v1' if own.direction_task else 'urdf4_amp_cts_expand_x_v1';args.headless=True
    cases=[dict(name='%s_%03d'%(label,round(100*v)),mode=mode,command=[v,0.,0.]) for v in map(float,own.speeds.split(',')) for mode,label in enumerate(['off','heading','path'])]
    cfg,_=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg);args.num_envs=len(cases)*4
    if own.direction_task and own.gains:cfg.direction.gains.update(json.loads(own.gains.read_text()))
    if own.suite=='omni':
        manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text())
        cases=[dict(name=label+'_'+c['clip'],mode=mode,command=c['command'],moving=True) for c in manifest['clips'] for mode,label in [(0,'off'),(2,'path')]]
        for case in cases:
            if case['name'].endswith('_forward'):case['command']=[own.omni_forward_speed,0.,0.]
        if own.skip_march:cases=[c for c in cases if not c['name'].endswith('_march')]
        args.num_envs=len(cases)*4
    cfg.env.test=not own.robust;cfg.noise.add_noise=own.robust;cfg.push.enabled=own.robust
    cfg.amp_cts.reference_reset_probability=0.;cfg.rewards.heading_gain=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    env._push_strength=lambda:1.
    checkpoint=own.policy or ROOT/'artifacts/urdf4_selection_x_20261115/best/student_0.pt'
    policy=torch.jit.load(str(checkpoint),map_location=env.device).eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    weights=lambda:hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in policy.state_dict().values())).hexdigest()
    weight_hash=weights()
    own.output.mkdir(parents=True,exist_ok=True)
    gains=PathGains(**json.loads(own.gains.read_text())) if own.gains else PathGains()
    results=[]
    for source in own.pose_sources.split(','):
        for seed in map(int,own.seeds.split(',')):
            report,arrays=run(env,policy,cases,own.seconds,seed,source,gains,own.stop_s,own.scope)
            results.append(report);np.savez_compressed(str(own.output/('%s_seed%d.npz'%(source,seed))),**arrays)
            (own.output/'result.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),policy_sha256=before,
                reports=results,robust=own.robust,frozen_weights_unchanged=weights()==weight_hash,long_training_started=False),indent=2)+'\n')
            print('PATH',source,seed,[(r['name'],r['kind'],round(r['cross_max_m'],3),round(r['heading_cycle_mean_abs_deg'],2),r['rejection_reasons']) for r in report['results']],flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    assert weights()==weight_hash
    print('FROZEN PATH EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
