"""Finite student-only behavior checks. Scores alone cannot approve long training."""
import sys,argparse,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat


def events(mask,min_duration=2):
 padded=np.r_[False,mask,False];start=np.flatnonzero(~padded[:-1]&padded[1:]);end=np.flatnonzero(padded[:-1]&~padded[1:])
 return start[end-start>=min_duration]


def evaluate(env,runner,seconds,seed,mode):
 torch.manual_seed(seed);np.random.seed(seed)
 ids=torch.arange(env.num_envs,device=env.device)%12;commands=env.reference_commands[ids]
 env.reference_reset_probability=1. if mode=='rsi' else 0.
 env.set_command_override(commands if mode=='rsi' else torch.zeros_like(commands),gait_enabled=mode=='rsi')
 old_length=env.max_episode_length;env.max_episode_length=math.ceil((seconds+5)/env.dt)
 obs,_=env.reset();runner.style.clear_history()
 if mode=='cold':
  with torch.no_grad():
   for _ in range(100):obs,_,_,_,_=env.step(torch.zeros(env.num_envs,12,device=env.device))
  env.set_command_override(commands);env.episode_length_buf[:]=0;env.clock_offset[:]=0
  env.heading_target[:]=yaw_from_quat(env.root_states[:,3:7]);env.heading_error[:]=0
  all_ids=torch.arange(env.num_envs,device=env.device);env.velocity_error.reset(all_ids);env.velocity_drift[:]=0
  env.compute_observations();obs=env.get_observations()
 record={k:[] for k in ['state','contact','clearance','valid','style','phase','done']}
 peak=torch.zeros(env.num_envs,device=env.device);sat=peak.clone();ticks=0
 original=env._compute_torques
 def torque(action):
  nonlocal ticks
  applied=original(action);peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)))
  sat.add_((env.raw_pd_torques.abs()>env.peak_torque).float().mean(1));ticks+=1
  return applied
 env._compute_torques=torque;runner.alg.actor_critic.eval()
 try:
  with torch.no_grad():
   for step in range(round(seconds/env.dt)):
    obs,_,_,done,info=env.step(runner.alg.actor_critic.act_inference(obs));a=info['amp'];style,_,valid=runner.style.rollout_reward(a)
    values=[a['next'],a['contact'],env.feet_pos[:,:,2]-.02,valid,style,a['reference_quality'],done]
    for key,value in zip(record,values):record[key].append(value[:48].cpu().numpy().copy())
 finally:env._compute_torques=original;env.max_episode_length=old_length
 arrays={k:np.stack(v) for k,v in record.items()};rows=[];cut=round(min(3.,seconds/3)/env.dt);cycle=25
 manifest=json.loads((Path(env.cfg.amp_cts.reference)/'manifest.json').read_text())
 for i,clip in enumerate(manifest['clips']):
  slots=np.arange(48)%12==i;x=arrays['state'][cut:,slots];c=arrays['contact'][cut:,slots];cmd=np.array(clip['command'])
  v=x[:,:,[36,37,41]]
  mean=np.stack([np.stack([np.convolve(v[:,j,k],np.ones(cycle)/cycle,'valid') for k in range(3)],1) for j in range(4)],1)
  error=float(np.linalg.norm(mean[:,:,:2]-cmd[:2],axis=2).mean());yaw_error=float(np.abs(mean[:,:,2]-cmd[2]).mean())
  foot=np.percentile(arrays['clearance'][cut:,slots].reshape(-1,4),95,axis=0)*1000
  n=c.sum(2);diag=(c[:,:,0]&c[:,:,3])|(c[:,:,1]&c[:,:,2]);illegal=(n==0)|(n==1)|((n==2)&~diag)
  td=[];lifts=[];alternations=[]
  for copy in range(4):
   cc=c[:,copy];lifts.append([len(events(~cc[:,leg])) for leg in range(4)])
   for aa,bb in [(0,3),(1,2)]:
    ea,eb=events(cc[:,aa]),events(cc[:,bb])
    if len(ea) and len(eb):td.extend(float(np.min(abs(eb-k)))*env.dt for k in ea)
   group=np.full(len(cc),-1);group[cc[:,0]&cc[:,3]&~cc[:,1]&~cc[:,2]]=0;group[cc[:,1]&cc[:,2]&~cc[:,0]&~cc[:,3]]=1
   seq=group[group>=0];alternations.append(int(np.sum(seq[1:]!=seq[:-1])))
  reset=int(arrays['done'][:,slots].sum());flight=float((n==0).mean());bad=float(illegal.mean())
  touchdown=float(np.percentile(td,95)) if td else None
  with np.load(Path(env.cfg.amp_cts.reference)/(clip['clip']+'.npz')) as ref:
   ref_height=np.percentile(ref['foot_height_m'],95,axis=0)*1000
  clearance_error=float(abs(foot-ref_height).max());elapsed=(len(x))*env.dt
  reasons=[]
  if reset:reasons.append('reset')
  if error>.03:reasons.append('linear_tracking')
  if yaw_error>.10:reasons.append('yaw_tracking')
  if flight>0:reasons.append('flight')
  if bad>.001:reasons.append('illegal_support')
  if clearance_error>10:reasons.append('foot_trajectory')
  if min(min(a) for a in lifts)<.7*elapsed*2:reasons.append('not_stepping')
  if min(alternations)<.7*elapsed*4:reasons.append('not_alternating_diagonals')
  if touchdown is None or touchdown>.040001:reasons.append('diagonal_timing')
  casepeak=float(peak[torch.arange(env.num_envs,device=env.device)%12==i].max())
  saturation=float((sat/max(ticks,1))[torch.arange(env.num_envs,device=env.device)%12==i].mean())
  if casepeak>17 or saturation>.01:reasons.append('torque')
  row=dict(clip=clip['clip'],command=clip['command'],mean_velocity=v.mean((0,1)).tolist(),cycle_linear_error=error,
           cycle_yaw_error=yaw_error,clearance_p95_mm=foot.tolist(),clearance_error_mm=clearance_error,
           AMP_style=float(arrays['style'][cut:,slots].mean()),phase_quality=float(arrays['phase'][cut:,slots].mean()),
           resets=reset,flight_fraction=flight,illegal_support_fraction=bad,lifts_per_foot=lifts,
           diagonal_alternations=alternations,diagonal_touchdown_p95_s=touchdown,
           raw_peak_Nm=casepeak,saturation_fraction=saturation,rejection_reasons=reasons,passed=not reasons)
  rows.append(row)
 return dict(seed=seed,mode=mode,seconds=seconds,warmup_s=cut*env.dt,results=rows),arrays


def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seconds',type=float,default=30.);p.add_argument('--seeds',default='1,2,3');p.add_argument('--modes',default='cold,rsi')
 p.add_argument('--updates',type=int,default=0);p.add_argument('--approve',action='store_true')
 own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
 if not 0<=own.updates<=500:raise ValueError('Behavior preflight is capped at 500 PPO updates')
 if own.seconds<5:raise ValueError('Need at least five seconds')
 args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True
 if args.num_envs is None:args.num_envs=1024 if own.updates else 48
 if args.num_envs<48:raise ValueError('Need four environments per V2 clip')
 cfg,train=task_registry.get_cfgs(args.task);env,cfg=task_registry.make_env(args.task,args=args,env_cfg=cfg)
 own.output.mkdir(parents=True,exist_ok=True)
 runner=Urdf4AMPCTSRunner(env,class_to_dict(train),str(own.output/'run') if own.updates else None,device=args.rl_device)
 runner.load(str(own.checkpoint));checkpoint=own.checkpoint
 if own.updates:
  runner.learn(own.updates,init_at_random_ep_len=True)
  checkpoint=own.output/'run'/('model_%d.pt'%runner.current_learning_iteration)
 reports=[]
 for seed in [int(x) for x in own.seeds.split(',')]:
  for mode in own.modes.split(','):
   if mode not in ['cold','rsi']:raise ValueError('Unknown reset mode')
   report,arrays=evaluate(env,runner,own.seconds,seed,mode);reports.append(report)
   np.savez_compressed(str(own.output/('rollout_%s_%d.npz'%(mode,seed))),**arrays)
   (own.output/'result.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),reports=reports),indent=2)+'\n')
   print('BEHAVIOR',seed,mode,[(r['clip'],r['passed'],[round(v,3) for v in r['mean_velocity']],r['rejection_reasons']) for r in report['results']],flush=True)
 passed=all(r['passed'] for report in reports for r in report['results'])
 approved=passed and own.seconds>=30 and len(set(r['seed'] for r in reports))>=3 and set(own.modes.split(','))=={'cold','rsi'}
 result=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             contract=runner.contract,reports=reports,passed=passed,approved=approved,
             source_sha256=runner.experiment['source_sha256'])
 (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 if own.approve:
  if not approved:raise RuntimeError('Behavior gate FAILED; long training is not approved')
  gate=ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/behavior_acceptance.json';gate.write_text(json.dumps(result,indent=2)+'\n')
 print('BEHAVIOR GATE',approved,flush=True)

if __name__=='__main__':main()
