"""Pure deployable-student cold starts, world-direction tracking and actual stop."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner
from legged_gym.envs.urdf4.amp_reference import amp_features

CASES={'stand':(0,0,0),'forward_slow':(.18,0,0),'forward':(.25,0,0),'backward':(-.15,0,0),
    'left':(0,.1,0),'right':(0,-.1,0),'turn_left':(0,0,.4),'turn_right':(0,0,-.4),
    'forward_left':(.15,.08,0),'forward_right':(.15,-.08,0),'backward_left':(-.15,.08,0),'backward_right':(-.15,-.08,0),
    'arc_left':(.2,0,.3),'arc_right':(.2,0,-.3),'back_arc_left':(-.15,0,.3),'back_arc_right':(-.15,0,-.3)}
FRONTIER={'fast_forward':(.5,0,0),'fast_backward':(-.3,0,0),'fast_left':(0,.2,0),'fast_right':(0,-.2,0),'fast_turn_left':(0,0,.6),'fast_turn_right':(0,0,-.6)}


def starts(mask):
    pad=np.r_[False,mask,False];s=np.flatnonzero(~pad[:-1]&pad[1:]);e=np.flatnonzero(pad[:-1]&~pad[1:]);return s[(e-s>=2)&(s>0)]


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--copies',type=int,default=2);p.add_argument('--robust',action='store_true');p.add_argument('--old-seed',action='store_true');p.add_argument('--frontier',action='store_true')
    p.add_argument('--full-push',action='store_true');p.add_argument('--extended-frontier',action='store_true')
    p.add_argument('--v4',action='store_true')
    p.add_argument('--v5',action='store_true')
    p.add_argument('--rough',action='store_true', help='Evaluate the rough-ground V4 task')
    p.add_argument('--transition',action='store_true', help='Evaluate the synchronized rough-ground controller')
    p.add_argument('--transition-flat',action='store_true', help='Evaluate the synchronized controller on a PhysX plane')
    p.add_argument('--motions',help='Comma-separated diagnostic subset; does not count as full core acceptance')
    p.add_argument('--public-policy',type=Path,help='Evaluate an exported 1029D public-only V4 student, with its explicit phase metadata')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task='urdf4_amp_cts_v5' if own.v5 else ('urdf4_amp_cts_v4' if own.v4 else 'urdf4_amp_cts_v3');args.headless=True
    if own.transition_flat:own.transition=True
    if own.transition:own.rough=True
    if own.rough:args.task='urdf4_amp_cts_transition' if own.transition else 'urdf4_amp_cts_rough'
    if own.extended_frontier:
        FRONTIER.update(forward_035=(.35,0,0),forward_0375=(.375,0,0),forward_040=(.4,0,0),forward_0425=(.425,0,0),forward_045=(.45,0,0),forward_0475=(.475,0,0),
            forward_060=(.6,0,0),turn_left_080=(0,0,.8),turn_right_080=(0,0,-.8))
        own.frontier=True
    cases=dict(CASES,**(FRONTIER if own.frontier else {}))
    if own.motions:cases={name:cases[name] for name in own.motions.split(',')}
    names=list(cases);per=own.copies;args.num_envs=len(names)*per
    own.output.mkdir(parents=True,exist_ok=False);cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=not own.robust;cfg.env.episode_length_s=own.seconds+10.
    if (own.v4 or own.v5 or own.rough) and not own.old_seed:
        saved=torch.load(own.model,map_location='cpu');cfg.amp_cts.balance_variant=saved['contract']['balance_variant']
        if own.transition and 'transition_training' in saved['contract']:
            from legged_gym.envs.urdf4.amp_transition_training import restore_transition_contract
            args.task='urdf4_amp_cts_transition_training';cfg,train=task_registry.get_cfgs(args.task)
            cfg.env.test=not own.robust;cfg.env.episode_length_s=own.seconds+10.
            restore_transition_contract(cfg,train,saved)
            cfg.amp_cts.balance_variant=saved['contract']['balance_variant']
        cfg.amp_cts.balance_archive=saved['experiment']['env']['amp_cts']['balance_archive']
        cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
        if own.rough:
            for key,value in saved['contract']['rough_ground'].items():setattr(cfg.rough,key,value)
        if saved['experiment']['env']['expand']['max_stage']==1 and saved['experiment']['env']['expand']['stages'][-1]==[-.30,.50,.20,.60]:
            from check_robust050_launch import restore
            for section in ('commands','expand'):restore(getattr(cfg,section),saved['experiment']['env'][section])
        if own.v5:
            cfg.amp_cts.extra_reference=saved['experiment']['env']['amp_cts']['extra_reference']
            cfg.amp_cts.adaptive_feedback_phase=saved['experiment']['env']['amp_cts']['adaptive_feedback_phase']
            cfg.amp_cts.use_public_prior=saved['experiment']['env']['amp_cts'].get('use_public_prior',False)
            cfg.amp_cts.reward_revision=saved['experiment']['env']['amp_cts']['reward_revision']
            cfg.amp_cts.public_prior_archive=saved['experiment']['env']['amp_cts'].get('public_prior_archive',cfg.amp_cts.public_prior_archive)
            cfg.direction.mid_cross_integral_gain=saved['experiment']['env']['direction'].get('mid_cross_integral_gain',0.)
            cfg.amp_cts.public_feedback_history=saved['experiment']['env']['amp_cts'].get('public_feedback_history',False)
            for key in ('public_obs_size','num_observations','num_privileged_obs'):
                setattr(cfg.env,key,saved['experiment']['env']['env'][key])
    if own.transition_flat:
        from legged_gym.envs.urdf4.urdf4_omni_env import Urdf4OmniRobot
        robot=task_registry.get_task_class(args.task)
        robot._create_ground_plane=Urdf4OmniRobot._create_ground_plane
        robot.ground_height_at=lambda self,xy:xy[...,0].new_zeros(xy.shape[:-1])
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    public_policy=None;public_metadata=None
    if own.public_policy:
        assert own.v4 or own.v5
        public_metadata=json.loads(Path(str(own.public_policy)+'.json').read_text())
        assert public_metadata['public_size']==1029 and public_metadata['urdf_sha256']==env.urdf_sha256
        public_policy=torch.jit.load(str(own.public_policy),map_location=env.device).eval()
        if public_metadata.get('fast_straight_reference'):
            original_select=env._select_reference
            reference=env.reference_commands.new_tensor(public_metadata['fast_straight_reference'])
            fast_id=(env.reference_commands-reference).square().sum(1).argmin()
            def select(command):
                ids=original_select(command)
                fast=(command[:,0]>public_metadata['fast_straight_threshold_m_s'])&(command[:,1:].abs().sum(1)<.001)
                ids[fast]=fast_id
                return ids
            env._select_reference=select
    runner_class=Urdf4AMPV3Runner
    if own.v4:
        from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner
        runner_class=Urdf4AMPV4Runner
    if own.v5:
        from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner
        runner_class=Urdf4AMPV5Runner
    if own.rough:
        from legged_gym.algorithms.urdf4_amp_rough import Urdf4AMPRoughRunner
        runner_class=Urdf4AMPRoughRunner
    if args.task=='urdf4_amp_cts_transition_training':
        from legged_gym.algorithms.urdf4_amp_transition_training import Urdf4AMPTransitionTrainingRunner
        runner_class=Urdf4AMPTransitionTrainingRunner
    runner=runner_class(env,class_to_dict(train),None,device=args.rl_device)
    if own.old_seed:runner.warm_start(own.model)
    else:runner.load(own.model,load_optimizer=False)
    command=torch.tensor([cases[k] for k in names for copy in range(per)],device=env.device,dtype=torch.float);env.set_command_override(command);env.reset()
    if own.full_push:env.common_step_counter=max(env.common_step_counter,round(cfg.push.ramp_s/env.dt))
    fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool);firstfall=torch.full((env.num_envs,),float('nan'),device=env.device);clock=[0.]
    def no_reset(ids):
        fresh=ids[~fallen[ids]];firstfall[fresh]=clock[0];fallen[ids]=True
    env.reset_idx=no_reset
    peak=torch.zeros(env.num_envs,device=env.device);mpeak=peak.clone();apeak=peak.clone();sum_square=torch.zeros(env.num_envs,12,device=env.device);absolute=sum_square.clone();applied_square=sum_square.clone();raw17=sum_square.clone();motor6=sum_square.clone();ticks=peak.clone();original=env._compute_torques
    def torques(actions):
        value=original(actions);alive=(~fallen).float();peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*alive));mpeak.copy_(torch.maximum(mpeak,env.motor_torques.abs().amax(1)*alive));sum_square.add_(env.motor_torques.square()*alive[:,None]);ticks.add_(alive)
        apeak.copy_(torch.maximum(apeak,value.abs().amax(1)*alive));absolute.add_(env.motor_torques.abs()*alive[:,None]);applied_square.add_(value.square()*alive[:,None]);raw17.add_((env.raw_pd_torques.abs()>17).float()*alive[:,None]);motor6.add_((env.motor_torques.abs()>6).float()*alive[:,None]);return value
    env._compute_torques=torques;records=[];slots=[env.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
    with torch.no_grad():
        for k in range(round((own.seconds+6)/env.dt)):
            clock[0]=k*env.dt
            if k==round(own.seconds/env.dt):env.set_command_override(torch.zeros_like(command))
            actions=public_policy(env.obs_buf[:,:1029]) if public_policy is not None else runner.alg.actor_critic.act_inference(env.obs_buf)
            actions[fallen]=0.;env.step(actions)
            ground=env.ground_height_at(env.feet_pos[:,slots,:2]) if own.rough else env.env_origins[:,None,2]
            records.append(dict(features=amp_features(env).cpu().numpy(),contact=env.get_foot_contact_mask()[:,slots].cpu().numpy(),
                height=(env.feet_pos[:,slots,2]-ground-.02).cpu().numpy(),cross=env.path.cross_error.cpu().numpy().copy(),
                heading=env.path.heading_error.cpu().numpy().copy(),valid=(~fallen).cpu().numpy(),phase=env.motion_phase.cpu().numpy().copy()))
    a={key:np.stack([r[key] for r in records]) for key in records[0]};t=np.arange(1,len(records)+1)*env.dt;moving=(t>=6)&(t<own.seconds);stopping=t>=own.seconds+4;rows=[]
    for i,name in enumerate(names):
        sl=slice(i*per,(i+1)*per);mask=moving[:,None]&a['valid'][:,sl];f=a['features'][:,sl];ct=a['contact'][:,sl];cc=ct[mask];n=cc.sum(1);diag=(cc[:,0]&cc[:,3])|(cc[:,1]&cc[:,2]);height=a['height'][:,sl]
        row=dict(motion=name,command=list(cases[name]),frontier=name in FRONTIER,fallen_count=int(fallen[sl].sum()),
            raw_peak_Nm=float(peak[sl].max()),motor_peak_Nm=float(mpeak[sl].max()),
            applied_peak_Nm=float(apeak[sl].max()),motor_mean_absolute_Nm=float((absolute[sl]/ticks[sl,None].clamp(min=1)).mean()),
            applied_joint_rms_Nm=torch.sqrt(applied_square[sl]/ticks[sl,None].clamp(min=1)).mean(0).tolist(),
            raw_over_17_fraction=float((raw17[sl]/ticks[sl,None].clamp(min=1)).mean()),motor_over_6_fraction=float((motor6[sl]/ticks[sl,None].clamp(min=1)).mean()),
            motor_joint_rms_Nm=torch.sqrt(sum_square[sl]/ticks[sl,None].clamp(min=1)).mean(0).tolist(),
            mean_velocity=f[mask][:,[36,37,41]].mean(0).tolist() if mask.any() else None,
            clearance_p95_mm=(np.quantile(height[mask],.95,axis=0)*1000).tolist() if mask.any() else None,
            flight_fraction=float((n==0).mean()) if len(n) else 1.,illegal_support_fraction=float(((n==1)|((n==2)&~diag)).mean()) if len(n) else 1.,
            cross_max_m=float(abs(a['cross'][:,sl][mask]).max()) if mask.any() else 1e3,
            heading_mean_abs_deg=float(abs(a['heading'][:,sl][mask]).mean()*180/np.pi) if mask.any() else 1e3)
        gaps=[];lifts=[]
        for active in [ct,~ct]:
            values=[]
            for copy in range(per):
                c=active[moving,copy];times=t[moving]
                for aa,bb in [(0,3),(1,2)]:
                    ea=starts(c[:,aa]);eb=starts(c[:,bb])
                    if len(ea) and len(eb):values.extend(np.min(abs(times[eb]-times[e])) for e in ea)
            gaps.append(float(np.quantile(values,.95)*1000) if values else None)
        for j in range(4):lifts.append(min(len(starts((height[moving,copy,j]>.012)&~ct[moving,copy,j])) for copy in range(per)))
        sf=f[stopping];stopvelocity=sf[:,:,[36,37,41]].mean((0,1));stopcontact=float(ct[stopping].all(2).mean())
        row.update(diagonal_event_p95_ms=gaps,lift_events_per_foot=lifts,stop_mean_velocity=stopvelocity.tolist(),stop_four_foot_fraction=stopcontact)
        reasons=[]
        if row['fallen_count']:reasons.append('fall')
        if row['raw_peak_Nm']>17:reasons.append('raw_torque')
        if row['flight_fraction']>0:reasons.append('flight')
        if row['illegal_support_fraction']>.01:reasons.append('support')
        if name!='stand':
            if row['clearance_p95_mm'] is None or min(row['clearance_p95_mm'])<12:reasons.append('clearance')
            rid=env._select_reference(command)[i*per];rate=float(env.reference_frequencies[rid])
            if min(lifts)<(own.seconds-6)*rate*.65:reasons.append('alternation')
            if any(g is None or g>60.001 for g in gaps):reasons.append('diagonal_timing')
            if row['cross_max_m']>.05 or row['heading_mean_abs_deg']>2:reasons.append('direction')
            if row['mean_velocity'] is None or np.linalg.norm(np.array(row['mean_velocity'])[:2]-cases[name][:2])>.05 or abs(row['mean_velocity'][2]-cases[name][2])>.12:reasons.append('tracking')
        if np.linalg.norm(stopvelocity[:2])>.02 or abs(stopvelocity[2])>.04 or stopcontact<.95:reasons.append('stop')
        row.update(passed=not reasons,rejection_reasons=reasons);rows.append(row);print('POLICY',name,row['passed'],reasons,row['mean_velocity'],row['clearance_p95_mm'],flush=True)
    np.savez_compressed(own.output/'rollout.npz',**a,time_s=t)
    result=dict(checkpoint=str(own.model),checkpoint_sha256=hashlib.sha256(own.model.read_bytes()).hexdigest(),seed=args.seed,seconds=own.seconds,
        copies=per,randomized=own.robust,pure_student=True,teacher_controller_used=False,automatic_resets=False,root_or_joint_overwrite_during_rollout=False,
        core_passed=set(CASES).issubset(cases) and all(r['passed'] for r in rows if not r['frontier']),
        tested_subset_passed=all(r['passed'] for r in rows),cases=rows,contract=runner.contract,source_sha256=runner.experiment['source_sha256'],
        push_count=float(env.push_count.sum()),nominal_fraction=float(env.nominal_mask.float().mean()),full_strength_push=own.full_push,
        push_strength_at_end=float(env._push_strength()),evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if own.public_policy:
        result.update(public_policy=str(own.public_policy),public_policy_sha256=hashlib.sha256(own.public_policy.read_bytes()).hexdigest(),
            public_policy_metadata=public_metadata,public_policy_metadata_sha256=hashlib.sha256(Path(str(own.public_policy)+'.json').read_bytes()).hexdigest())
    if own.rough:
        result.update(rough_ground=not own.transition_flat,synchronized_transitions=own.transition,
                      rough_fraction=0. if own.transition_flat else float(env.rough_env.float().mean()),
                      terrain_height_bounds_m=[0.,0.] if own.transition_flat else [float(env.terrain_heights.min()),float(env.terrain_heights.max())])
    (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
