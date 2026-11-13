"""Per-action, per-joint RS01 torque audit; samples every physics substep."""
import sys,json,argparse,hashlib
from pathlib import Path
sys.path.insert(0,'/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')
import play_urdf4_omni as play
import torch,numpy as np
from legged_gym.utils import get_args
p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',required=True);p.add_argument('--stress',action='store_true');own,remaining=p.parse_known_args();sys.argv=sys.argv[:1]+remaining
args=get_args();args.headless=True;args.task=play.TASK
cases=(('forward 0.1',.1,0,0),)+play.SWEEP;per=args.eval_envs;n=len(cases)*per
flags=type('Flags',(),{'disturbed':own.stress})();env,cfg=play.make_env(args,flags,n);runner,checkpoint=play.load_policy(env,args,cfg);policy=runner.get_inference_policy(device=env.device)
cmd=torch.tensor([c[1:] for c in cases],device=env.device).repeat_interleave(per,0);env.set_command_override(cmd);obs,_=env.reset()
steps=round(args.duration_s/env.dt);sub=env.cfg.control.decimation;dt=env.physics_dt
trace=torch.empty((steps*sub,n,12,3),device=env.device);idx=[0];original=env._compute_torques
# Record before the engine step, so a reset after a step cannot erase the torque.
def torque(a):
 applied=original(a);trace[idx[0]]=torch.stack((env.raw_pd_torques,env.motor_torques,applied),-1);idx[0]+=1;return applied
env._compute_torques=torque
fallen=torch.zeros(n,device=env.device,dtype=torch.int);velocity=torch.zeros(n,3,device=env.device);count=0
with torch.no_grad():
 for step in range(steps):
  obs,_,_,dones,_=env.step(policy(obs));fallen+=dones.int()
  if step*env.dt>=2.:
   velocity+=torch.cat((env.base_lin_vel[:,:2],env.base_ang_vel[:,2:3]),1);count+=1
  if (step+1)%500==0:print('[progress]',(step+1)*env.dt,'s',flush=True)
a=trace.cpu().numpy();f=fallen.cpu().tolist();v=(velocity/count).cpu().tolist();settle=round(2/dt);names=env.dof_names
out=Path(own.output);out.parent.mkdir(parents=True,exist_ok=True)
result={'checkpoint':checkpoint,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'seed':args.seed,'disturbed':own.stress,'envs_per_action':per,'duration_s':args.duration_s,'steady_window_s':[2.,args.duration_s],'sample_period_s':dt,'joint_order':names,'torque_channels':['raw_pd','motor','applied'],'motor_cap_Nm':14.,'continuous_reference_Nm':6.,'physics_dt_s':dt,'control_dt_s':env.dt,'kp':env.p_gains.cpu().tolist(),'kd':env.d_gains.cpu().tolist(),'cases':[]}
def stats(x):
 ab=np.abs(x);return {'mean_abs_Nm':float(ab.mean()),'rms_Nm':float(np.sqrt(np.mean(x*x))),'p95_abs_Nm':float(np.quantile(ab,.95)),'p99_abs_Nm':float(np.quantile(ab,.99)),'peak_abs_Nm':float(ab.max()),**{'over_%d_pct'%t:float((ab>t).mean()*100) for t in [6,12,14,17]}}
def longest_over(x,threshold):
 # Max run of consecutive substeps, computed separately per environment/joint.
 b=np.abs(x)>threshold;run=np.zeros(b.shape[1:],dtype=np.int32);longest=np.zeros_like(run)
 for row in b:
  run=np.where(row,run+1,0);np.maximum(longest,run,out=longest)
 return float(longest.max()*dt)
for ci,c in enumerate(cases):
 sl=slice(ci*per,(ci+1)*per);x=a[:,sl];steady=x[settle:];row={'label':c[0],'command_vx_vy_wz':c[1:],'resets':sum(f[sl]),'mean_actual_vx_vy_wz':np.mean(v[sl],axis=0).tolist(),'channels':{},'joints':[]}
 for k,ch in enumerate(result['torque_channels']):
  row['channels'][ch]={'steady':stats(steady[:,:,:,k]),'whole_run_peak_Nm':float(np.abs(x[:,:,:,k]).max())}
 for j,name in enumerate(names):
  jr={'name':name,'channels':{ch:stats(steady[:,:,j,k]) for k,ch in enumerate(result['torque_channels'])}}
  jr['motor_longest_over6_s']=longest_over(steady[:,:,j:j+1,1],6);jr['motor_longest_over12_s']=longest_over(steady[:,:,j:j+1,1],12);row['joints'].append(jr)
 worst=max(row['joints'],key=lambda j:j['channels']['motor']['rms_Nm']);row['highest_motor_rms_joint']=worst['name'];row['highest_motor_rms_Nm']=worst['channels']['motor']['rms_Nm'];result['cases'].append(row)
 print('[torque]',c[0],row['channels']['raw_pd']['steady']['peak_abs_Nm'],row['channels']['motor']['steady']['rms_Nm'],flush=True)
np.savez_compressed(out.with_suffix('.npz'),torques_Nm=a,sample_period_s=np.array(dt),joint_order=np.array(names),commands=cmd.cpu().numpy(),channels=np.array(result['torque_channels']))
out.write_text(json.dumps(result,indent=2));print('[saved]',out,flush=True)
