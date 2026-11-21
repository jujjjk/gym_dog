"""Finite student-only nominal/robust checks, optionally after <=500 updates."""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_expand import Urdf4AMPExpandRunner
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat
from evaluate_cts_behavior import events

def evaluate(env,runner,seconds=10.,seed=1,replicas=1):
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();numpy_rng=np.random.get_state()
    torch.manual_seed(seed);np.random.seed(seed)
    manifest=json.loads((Path(env.cfg.amp_cts.reference)/'manifest.json').read_text())
    cases=[dict(name=x['clip'],command=x['command'],moving=True) for x in manifest['clips']]
    if getattr(env.cfg.expand,'x_only',False):
        cases += [dict(name='stand',command=[0,0,0],moving=False)]
        cases += [dict(name='forward_%03d'%round(100*v),command=[v,0,0],moving=True) for v in [.35,.4,.5,.6]]
    else:
        cases += [dict(name='stand',command=[0,0,0],moving=False),dict(name='frontier_forward',command=[.4,0,0],moving=True),
            dict(name='frontier_left',command=[0,.15,0],moving=True),dict(name='frontier_turn',command=[0,0,.6],moving=True),
            dict(name='frontier_mixed',command=[.3,.08,.25],moving=True)]
    count=len(cases)*4*replicas;assert env.num_envs>=count
    labels=(torch.arange(env.num_envs,device=env.device)//4)%len(cases)
    commands=torch.tensor([x['command'] for x in cases],device=env.device)[labels]
    enabled=torch.tensor([x['moving'] for x in cases],device=env.device)[labels]
    old=(env.reference_reset_probability,env.command_resample_steps,env.max_episode_length,env.common_step_counter,env._push_strength,env.curriculum_meter.clone())
    env.reference_reset_probability=0.;env.command_resample_steps=10**9;env.max_episode_length=round((seconds+5)/env.dt)
    env._push_strength=lambda:1.
    env.set_command_override(torch.zeros_like(commands),False);obs,_=env.reset()
    with torch.no_grad():
        for _ in range(100):obs,_,_,_,_=env.step(torch.zeros(env.num_envs,12,device=env.device))
    env.set_command_override(commands,True);env.target_gait[:]=enabled;env.gait_enabled[:]=enabled
    env.episode_length_buf[:]=0;env.clock_offset[:]=0
    env.heading_target[:]=yaw_from_quat(env.root_states[:,3:7]);env.heading_error[:]=0
    env.public_heading_target[:]=env._public_yaw();env.public_heading_error[:]=0
    env.velocity_error.reset(torch.arange(env.num_envs,device=env.device));env.velocity_drift[:]=0
    env.push_wait[~env.nominal_mask]=3.5;env.push_wait[env.nominal_mask]=1.e9
    env.compute_observations();obs=env.obs_buf
    data={k:[] for k in ['state','contact','clearance','done','recovering','push_count','limiter']}
    raw_peak=torch.zeros(env.num_envs,device=env.device);motor_peak=raw_peak.clone()
    sums=torch.zeros(env.num_envs,12,device=env.device);squares=sums.clone();samples=0;record=False
    original=env._compute_torques
    def torque(action):
        nonlocal samples
        applied=original(action);raw_peak.copy_(torch.maximum(raw_peak,env.raw_pd_torques.abs().amax(1)))
        motor_peak.copy_(torch.maximum(motor_peak,env.motor_torques.abs().amax(1)))
        if record:sums.add_(env.motor_torques.abs());squares.add_(env.motor_torques.square());samples+=1
        return applied
    env._compute_torques=torque
    try:
        with torch.no_grad():
            for step in range(round(seconds/env.dt)):
                record=step>=150
                obs,_,_,done,info=env.step(runner.alg.actor_critic.act_inference(obs));a=info['amp']
                slots=[env.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
                values=[a['next'],a['contact'],env.feet_pos[:,slots,2]-.02,done,env.recovery_left>0,env.push_count,
                    (env.target_rate.abs()>=env.target_rate_limit-.001).float().mean(1)]
                for k,v in zip(data,values):data[k].append(v[:count].cpu().numpy().copy())
    finally:
        env._compute_torques=original
        env.reference_reset_probability,env.command_resample_steps,env.max_episode_length,env.common_step_counter,env._push_strength,meter=old
        env.curriculum_meter.copy_(meter);env.command_override=None
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);np.random.set_state(numpy_rng)
    arrays={k:np.stack(v) for k,v in data.items()};cut=min(150,len(arrays['state'])//3);rows=[]
    for i,case in enumerate(cases*replicas):
        for kind,slots in [('nominal',[4*i]),('robust',[4*i+1,4*i+2,4*i+3])]:
            state=arrays['state'][cut:,slots];contact=arrays['contact'][cut:,slots];n=contact.sum(2)
            diagonal=(contact[:,:,0]&contact[:,:,3])|(contact[:,:,1]&contact[:,:,2]);bad=(n==0)|(n==1)|((n==2)&~diagonal)
            v=state[:,:,[36,37,41]];command=np.array(case['command']);cycle=25
            mean=np.stack([np.stack([np.convolve(v[:,j,k],np.ones(cycle)/cycle,'valid') for k in range(3)],1) for j in range(len(slots))],1)
            foot=np.percentile(arrays['clearance'][cut:,slots].reshape(-1,4),95,axis=0)*1000
            lifts=[];alternations=[];gaps=[]
            for copy in range(len(slots)):
                c=contact[:,copy];lifts.append([len(events(~c[:,leg])) for leg in range(4)])
                group=np.full(len(c),-1);group[c[:,0]&c[:,3]&~c[:,1]&~c[:,2]]=0;group[c[:,1]&c[:,2]&~c[:,0]&~c[:,3]]=1
                seq=group[group>=0];alternations.append(int(np.sum(seq[1:]!=seq[:-1])))
                for aa,bb in [(0,3),(1,2)]:
                    ea,eb=events(c[:,aa]),events(c[:,bb])
                    if len(ea) and len(eb):gaps.extend(float(np.min(abs(eb-k)))*env.dt for k in ea)
            reasons=[];reset=int(arrays['done'][:,slots].sum());flight=float((n==0).mean());illegal=float(bad.mean())
            if reset:reasons.append('reset')
            if flight:reasons.append('flight')
            if illegal>(.001 if kind=='nominal' else .01):reasons.append('illegal_support')
            touchdown=float(np.percentile(gaps,95)) if gaps else None
            if case['moving']:
                elapsed=len(state)*env.dt
                if min(min(x) for x in lifts)<.7*elapsed*2:reasons.append('not_stepping')
                if min(alternations)<.7*elapsed*4:reasons.append('not_alternating_diagonals')
                if kind=='nominal' and (touchdown is None or touchdown>.040001):reasons.append('diagonal_timing')
                if min(foot)<10 or max(foot)>55:reasons.append('clearance')
            else:
                if float((n==4).mean())<.95 or max(foot)>5:reasons.append('standing_not_static')
            indices=torch.tensor(slots,device=env.device);peak=float(raw_peak[indices].max())
            if peak>17:reasons.append('raw_torque')
            rows.append(dict(name=case['name'],kind=kind,command=case['command'],mean_velocity=v.mean((0,1)).tolist(),
                replica=i//len(cases),
                linear_error_m_s=float(np.linalg.norm(mean[:,:,:2]-command[:2],axis=2).mean()),
                yaw_error_rad_s=float(abs(mean[:,:,2]-command[2]).mean()),foot_clearance_p95_mm=foot.tolist(),
                diagonal_touchdown_p95_s=touchdown,resets=reset,flight_fraction=flight,illegal_support_fraction=illegal,
                push_events=int(arrays['push_count'][-1,slots].sum()),raw_peak_Nm=peak,motor_peak_Nm=float(motor_peak[indices].max()),
                motor_mean_abs_Nm=float((sums[indices]/max(1,samples)).mean()),motor_rms_Nm=float((squares[indices]/max(1,samples)).mean().sqrt()),
                per_joint_rms_Nm=(squares[indices].mean(0)/max(1,samples)).sqrt().tolist(),
                limiter_fraction=float(arrays['limiter'][cut:,slots].mean()),rejection_reasons=reasons,gait_passed=not reasons))
    return dict(seed=seed,seconds=seconds,cold_settle_s=2.,push_peak_range_N=[5.,10.],push_full_strength=True,
        results=rows,tracking_used_as_gait_gate=False),arrays

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000/model_17800.pt')
    p.add_argument('--seconds',type=float,default=10.);p.add_argument('--updates',type=int,default=0);p.add_argument('--seeds',default='1')
    p.add_argument('--x_only',action='store_true')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    if not 0<=own.updates<=500:raise ValueError('Finite preflight capped at 500 updates')
    args=get_args();args.task='urdf4_amp_cts_expand_x_v1' if own.x_only else 'urdf4_amp_cts_expand_v1';args.headless=True
    if args.num_envs is None:args.num_envs=1024 if own.updates else 128
    if args.num_envs<68:raise ValueError('Need >=68 environments')
    cfg,train=task_registry.get_cfgs(args.task);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    own.output.mkdir(parents=True,exist_ok=True)
    runner=Urdf4AMPExpandRunner(env,class_to_dict(train),str(own.output/'run') if own.updates else None,device=args.rl_device)
    runner.load(str(own.checkpoint),False);checkpoint=own.checkpoint
    if own.updates:
        before,arrays=evaluate(env,runner,own.seconds,1)
        (own.output/'before.json').write_text(json.dumps(before,indent=2)+'\n')
        np.savez_compressed(str(own.output/'before.npz'),**arrays)
        env.reset();runner.save(str(own.output/'run/model_0.pt'))
        runner.learn(own.updates,init_at_random_ep_len=True)
        checkpoint=own.output/('run/model_%d.pt'%runner.current_learning_iteration)
    reports=[]
    for seed in map(int,own.seeds.split(',')):
        report,arrays=evaluate(env,runner,own.seconds,seed);reports.append(report)
        np.savez_compressed(str(own.output/('rollout_seed%d.npz'%seed)),**arrays)
        print('EXPAND BEHAVIOR',seed,[(r['name'],r['kind'],r['gait_passed'],round(r['linear_error_m_s'],3),r['rejection_reasons']) for r in report['results']],flush=True)
    result=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        updates=own.updates,curriculum_stage=env.curriculum_stage,reports=reports,contract=runner.contract,
        source_sha256=runner.experiment['source_sha256'],long_training_started=False)
    (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINITE EXPANSION EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
