"""Frozen V3 checkpoint screening with pure public-history students.

No training, actuator/config changes, teacher commands or concealed resets.
Finalists are checked separately using evaluate_v3_policy.py.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner,V3StudentPolicy
from legged_gym.envs.urdf4.amp_reference import amp_features
from evaluate_v3_policy import CASES,FRONTIER,starts


class PolicyBatch:
    def __init__(self,policies,per):
        self.count=len(policies);self.per=per
        self.params,self.buffers=torch.func.stack_module_state(policies)
        base=copy.deepcopy(policies[0]).to('meta')
        def call(params,buffers,history):
            return torch.func.functional_call(base,(params,buffers),(history,),strict=True)
        self.forward=torch.vmap(call,in_dims=(0,0,0))
        device=next(policies[0].parameters()).device
        with torch.no_grad():
            probe=torch.randn(self.count,per,1020,device=device)*.1
            expected=torch.stack([p(probe[i]) for i,p in enumerate(policies)])
            self.parity_error=float((self.forward(self.params,self.buffers,probe)-expected).abs().max())
            assert self.parity_error<1e-5,self.parity_error

    def __call__(self,obs):
        return self.forward(self.params,self.buffers,obs[:,:1020].reshape(self.count,self.per,1020)).reshape(-1,12)


def assess(a,t,peak,mpeak,rms,fallen,commands,names,frequencies,seconds):
    moving=(t>=6)&(t<seconds);stopping=t>=seconds+4;rows=[]
    for i,name in enumerate(names):
        mask=moving&a['valid'][:,i];f=a['velocity'][:,i];ct=a['contact'][:,i];cc=ct[mask];n=cc.sum(1)
        diag=(cc[:,0]&cc[:,3])|(cc[:,1]&cc[:,2]);height=a['height'][:,i]
        row=dict(motion=name,command=commands[i].tolist(),frontier=name not in CASES,fallen_count=int(fallen[i]),
            raw_peak_Nm=float(peak[i]),motor_peak_Nm=float(mpeak[i]),motor_rms_Nm=float(rms[i].mean()),
            mean_velocity=f[mask].mean(0).tolist() if mask.any() else None,
            clearance_p95_mm=(np.quantile(height[mask],.95,axis=0)*1000).tolist() if mask.any() else None,
            flight_fraction=float((n==0).mean()) if len(n) else 1.,illegal_support_fraction=float(((n==1)|((n==2)&~diag)).mean()) if len(n) else 1.,
            cross_max_m=float(abs(a['cross'][:,i][mask]).max()) if mask.any() else 1e3,
            heading_mean_abs_deg=float(abs(a['heading'][:,i][mask]).mean()*180/np.pi) if mask.any() else 1e3)
        gaps=[]
        for active in [ct,~ct]:
            values=[];c=active[moving];times=t[moving]
            for aa,bb in [(0,3),(1,2)]:
                ea=starts(c[:,aa]);eb=starts(c[:,bb])
                if len(ea) and len(eb):values.extend(np.min(abs(times[eb]-times[e])) for e in ea)
            gaps.append(float(np.quantile(values,.95)*1000) if values else None)
        lifts=[len(starts((height[moving,j]>.012)&~ct[moving,j])) for j in range(4)]
        stop=f[stopping].mean(0);sc=float(ct[stopping].all(1).mean())
        row.update(diagonal_event_p95_ms=gaps,lift_events_per_foot=lifts,stop_mean_velocity=stop.tolist(),stop_four_foot_fraction=sc)
        reasons=[]
        if fallen[i]:reasons.append('fall')
        if peak[i]>17:reasons.append('raw_torque')
        if row['flight_fraction']>0:reasons.append('flight')
        if row['illegal_support_fraction']>.01:reasons.append('support')
        if name!='stand':
            if row['clearance_p95_mm'] is None or min(row['clearance_p95_mm'])<12:reasons.append('clearance')
            if min(lifts)<(seconds-6)*frequencies[i]*.65:reasons.append('alternation')
            if any(g is None or g>60.001 for g in gaps):reasons.append('diagonal_timing')
            if row['cross_max_m']>.05 or row['heading_mean_abs_deg']>2:reasons.append('direction')
            if row['mean_velocity'] is None or np.linalg.norm(np.array(row['mean_velocity'])[:2]-commands[i,:2])>.05 or abs(row['mean_velocity'][2]-commands[i,2])>.12:reasons.append('tracking')
        if np.linalg.norm(stop[:2])>.02 or abs(stop[2])>.04 or sc<.95:reasons.append('stop')
        row.update(passed=not reasons,rejection_reasons=reasons);rows.append(row)
    return rows


def summary(rows):
    core=[r for r in rows if not r['frontier']];front=[r for r in rows if r['frontier']]
    safety={'fall','raw_torque','flight','support'};gait={'clearance','alternation','diagonal_timing'}
    score=lambda r:r['cross_max_m']/.05+r['heading_mean_abs_deg']/2+(
        np.linalg.norm(np.array(r['mean_velocity'])[:2]-r['command'][:2])/.05+abs(r['mean_velocity'][2]-r['command'][2])/.12 if r['mean_velocity'] is not None else 1e3)
    return dict(core_passed=sum(r['passed'] for r in core),core_safety_failures=sum(bool(safety&set(r['rejection_reasons'])) for r in core),
        core_gait_failures=sum(bool(gait&set(r['rejection_reasons'])) for r in core),core_stop_failures=sum('stop' in r['rejection_reasons'] for r in core),
        core_direction_failures=sum('direction' in r['rejection_reasons'] for r in core),frontier_passed=sum(r['passed'] for r in front),
        core_tracking_score=float(np.mean([score(r) for r in core])),core_raw_peak_Nm=max(r['raw_peak_Nm'] for r in core))


def rank(item):
    s=item['summary']
    return (s['core_safety_failures'],s['core_gait_failures'],s['core_stop_failures'],-s['core_passed'],-s['frontier_passed'],s['core_tracking_score'])


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=24);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--seconds',type=float,default=14.);p.add_argument('--limit',type=int)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    assert 10<=own.seconds<=60 and 1<=own.batch<=32 and 0<=own.shard<own.shards
    args=get_args();args.task='urdf4_amp_cts_v3';args.headless=True
    own.output.mkdir(parents=True,exist_ok=False)
    paths=sorted(own.run.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
    paths=[ROOT/'artifacts/urdf4_amp_v3_training/selected/model_40.pt']+paths
    if own.limit:paths=paths[:own.limit]
    paths=paths[own.shard::own.shards];assert paths
    cases=dict(CASES,**FRONTIER,forward_035=(.35,0,0),forward_060=(.6,0,0),turn_left_080=(0,0,.8),turn_right_080=(0,0,-.8))
    names=list(cases);per=len(names);args.num_envs=own.batch*per
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device)
    command=torch.tensor(list(cases.values()),device=env.device,dtype=torch.float).repeat(own.batch,1)
    original_reset=env.reset_idx;original_torques=env._compute_torques;slots=[env.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
    results=[];started=time.time()
    for offset in range(0,len(paths),own.batch):
        group=paths[offset:offset+own.batch];padded=group+[group[0]]*(own.batch-len(group));policies=[];stages=[];hashes=[]
        for path in padded:
            saved=torch.load(path,map_location=env.device)
            assert saved['contract']==runner.contract,(path,'contract differs')
            runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
            policies.append(V3StudentPolicy(runner.alg.actor_critic).to(env.device).eval().requires_grad_(False))
            stages.append(saved['v3_state']['stage']);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        policy=PolicyBatch(policies,per);env.reset_idx=original_reset;env._compute_torques=original_torques
        env.set_command_override(command);env.reset()
        fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        peak=torch.zeros(env.num_envs,device=env.device);mpeak=peak.clone();squares=torch.zeros(env.num_envs,12,device=env.device);ticks=peak.clone()
        def no_reset(ids):fallen[ids]=True
        def torques(actions):
            value=original_torques(actions);alive=(~fallen).float()
            peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*alive));mpeak.copy_(torch.maximum(mpeak,env.motor_torques.abs().amax(1)*alive))
            squares.add_(env.motor_torques.square()*alive[:,None]);ticks.add_(alive);return value
        env.reset_idx=no_reset;env._compute_torques=torques;records=[]
        with torch.no_grad():
            for k in range(round((own.seconds+6)/env.dt)):
                if k==round(own.seconds/env.dt):env.set_command_override(torch.zeros_like(command))
                actions=policy(env.obs_buf);assert torch.isfinite(actions).all();actions[fallen]=0.;env.step(actions)
                records.append(dict(velocity=amp_features(env)[:,[36,37,41]].clone(),contact=env.get_foot_contact_mask()[:,slots].clone(),
                    height=(env.feet_pos[:,slots,2]-env.env_origins[:,None,2]-.02).clone(),cross=env.path.cross_error.clone(),
                    heading=env.path.heading_error.clone(),valid=(~fallen).clone()))
        arrays={key:torch.stack([r[key] for r in records]).cpu().numpy() for key in records[0]}
        t=np.arange(1,len(records)+1)*env.dt
        frequencies=env.reference_frequencies[env._select_reference(command)].cpu().numpy()
        values=[x.cpu().numpy() for x in [peak,mpeak,torch.sqrt(squares/ticks[:,None].clamp(min=1)),fallen]]
        for i,path in enumerate(group):
            sl=slice(i*per,(i+1)*per)
            rows=assess({k:v[:,sl] for k,v in arrays.items()},t,*[v[sl] for v in values],command[sl].cpu().numpy(),names,frequencies[sl],own.seconds)
            item=dict(checkpoint=str(path.resolve()),checkpoint_sha256=hashes[i],iteration=int(path.stem.split('_')[-1]),
                baseline=path==ROOT/'artifacts/urdf4_amp_v3_training/selected/model_40.pt',stage=stages[i],cases=rows,summary=summary(rows),batch_parity_error=policy.parity_error)
            results.append(item)
        receipt=dict(protocol=dict(seconds=own.seconds,stop_seconds=6,seed=args.seed,nominal=True,commands=cases,
            pure_student=True,teacher_controller_used=False,automatic_resets=False,root_or_joint_overwrite_during_rollout=False,
            shard=own.shard,shards=own.shards),contract=runner.contract,source_sha256=runner.experiment['source_sha256'],candidates=results)
        (own.output/'screen.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('SCREEN',len(results),'/',len(paths),'elapsed',round(time.time()-started,1),'top',[(v['iteration'],v['summary']) for v in sorted(results,key=rank)[:3]],flush=True)
        del policy,policies,records,arrays;torch.cuda.empty_cache()


if __name__=='__main__':main()
