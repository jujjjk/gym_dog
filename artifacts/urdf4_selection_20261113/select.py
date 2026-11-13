import sys,os,json,argparse
from pathlib import Path
sys.path.insert(0,'/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')
import play_urdf4_omni as play
import torch
from legged_gym.utils import get_args
p=argparse.ArgumentParser(add_help=False);p.add_argument('--models',default='all');p.add_argument('--output',required=True);p.add_argument('--stress',action='store_true'); own,remaining=p.parse_known_args();sys.argv=sys.argv[:1]+remaining
args=get_args();args.headless=True;args.task=play.TASK
run=Path('/home/nszb/gym/unitree_rl_gym/logs/urdf4_omni_v1/Nov12_20-20-27_long_gyroobs5000_20261112_202024')
files=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]));files=[f for f in files if own.models=='all' or int(f.stem.split('_')[-1]) in list(map(int,own.models.split(',')))]
cases=(('forward 0.1',.1,0,0),)+play.SWEEP
C=len(files); per=max(1,args.eval_envs);E=len(cases)*per
flags=type('Flags',(),{'disturbed':own.stress})();env,_=play.make_env(args,flags,C*E)
weights=[]
for f in files:weights.append(torch.load(str(f),map_location=env.device)['model_state_dict'])
layers=[]
for i in [0,2,4,6]:layers.append((torch.stack([w['actor.%d.weight'%i].T for w in weights]),torch.stack([w['actor.%d.bias'%i] for w in weights])[:,None,:]))
def policy(obs):
 x=obs.reshape(C,E,-1)
 for j,(w,b) in enumerate(layers):
  x=torch.bmm(x,w)+b
  if j<3:x=torch.nn.functional.elu(x)
 return x.reshape(C*E,12)
commands=torch.tensor([c[1:] for c in cases],device=env.device).repeat_interleave(per,dim=0).repeat(C,1)
N=C*E; rawpeak=torch.zeros(N,device=env.device);motorpeak=rawpeak.clone();appliedpeak=rawpeak.clone();raw17=rawpeak.clone();sat=rawpeak.clone();ticks=[0]
orig=env._compute_torques
def torque(a):
 v=orig(a);raw=env.raw_pd_torques.abs();rawpeak.copy_(torch.maximum(rawpeak,raw.amax(1)));motorpeak.copy_(torch.maximum(motorpeak,env.motor_torques.abs().amax(1)));appliedpeak.copy_(torch.maximum(appliedpeak,v.abs().amax(1)));raw17.add_((raw>17).float().mean(1));sat.add_((raw>env.peak_torque).float().mean(1));ticks[0]+=1
 return v
env._compute_torques=torque
contactsum={k:torch.zeros(N,device=env.device) for k in ['flight','single','non_diag_two','pair_mismatch','yaw_sq','vy_sq','roll_sq','pitch_sq']};steps=[0]; observed=[0]
origstep=env.step
pairidx=[env.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
def step(a):
 out=origstep(a); observed[0]+=1
 if observed[0]*env.dt<=2.:return out
 ct=env.get_foot_contact_mask()[:,pairidx];cnt=ct.sum(1);diag=(ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2]);
 for k,v in [('flight',cnt==0),('single',cnt==1),('non_diag_two',(cnt==2)&~diag),('pair_mismatch',(ct[:,0]!=ct[:,3])|(ct[:,1]!=ct[:,2]))]:contactsum[k].add_(v.float())
 contactsum['yaw_sq'].add_((env.base_ang_vel[:,2]-env.commands[:,2]).square());contactsum['vy_sq'].add_((env.base_lin_vel[:,1]-env.commands[:,1]).square());contactsum['roll_sq'].add_(env.rpy[:,0].square());contactsum['pitch_sq'].add_(env.rpy[:,1].square());steps[0]+=1
 return out
env.step=step
with torch.no_grad():stats=play.run(env,policy,commands,args.duration_s,False)
for k,v in contactsum.items():stats[k]=v/steps[0]
for k,v in [('raw_peak',rawpeak),('motor_peak_all',motorpeak),('applied_peak',appliedpeak),('raw_over17',raw17/ticks[0]),('saturation_all',sat/ticks[0])]:stats[k]=v
records=[]
for ci,f in enumerate(files):
 d={'checkpoint':str(f),'iteration':int(f.stem.split('_')[-1]),'cases':[]}
 for j,c in enumerate(cases):
  sl=slice(ci*E+j*per,ci*E+(j+1)*per);row={'command':c,'metrics':{k:float(v[sl].mean()) for k,v in stats.items()}}
  row['metrics']['fallen_count']=int(stats['fallen'][sl].sum());row['metrics']['raw_peak']=float(rawpeak[sl].max());row['metrics']['applied_peak']=float(appliedpeak[sl].max());d['cases'].append(row)
 moving=d['cases'][:1]+d['cases'][2:]
 d['falls']=sum(r['metrics']['fallen_count'] for r in d['cases']);d['flight']=max(r['metrics']['flight'] for r in moving);d['illegal']=sum(r['metrics']['single']+r['metrics']['non_diag_two'] for r in moving)/len(moving)
 avg=lambda k:sum(r['metrics'][k] for r in moving)/len(moving)
 d['score']=d['falls']*100+100*d['flight']+20*d['illegal']+avg('lin_err')*5+avg('yaw_err')+avg('yaw_drift')*2+avg('side_drift')*5+(1-avg('gait'))+avg('saturation_all')*10
 records.append(d)
records.sort(key=lambda r:r['score'])
result={'seed':args.seed,'duration_s':args.duration_s,'envs_per_case':per,'disturbed':own.stress,'contact_sample_hz':50,'torque_sample_hz':400,'records':records};Path(own.output).write_text(json.dumps(result,indent=2))
for r in records:print('RANK',r['iteration'],'score',round(r['score'],4),'falls',r['falls'],'flight',round(r['flight'],5),'illegal',round(r['illegal'],4),flush=True)
