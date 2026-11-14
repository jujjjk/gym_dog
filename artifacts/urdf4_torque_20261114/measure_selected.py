"""Nominal student-only measurement, all 400 Hz actuator substeps included."""
import sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat

args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True;args.num_envs=52
cfg,train=task_registry.get_cfgs(args.task)
cfg.rewards.scales.command_tracking=0.
cfg.commands.resampling_time=1.e6
cfg.env.episode_length_s=40.
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner=Urdf4AMPCTSRunner(env,class_to_dict(train),None,device=args.rl_device)
selected=json.loads((ROOT/'artifacts/urdf4_selection_20261114/selected.json').read_text())
checkpoint=Path(selected['checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selected['checkpoint_sha256']
runner.load(str(checkpoint),load_optimizer=False);runner.alg.actor_critic.eval()
torch.manual_seed(1);np.random.seed(1)
manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text())
names=[c['clip'] for c in manifest['clips']]+['stand']
commands=torch.cat((env.reference_commands,torch.zeros(1,3,device=env.device)),0)
labels=torch.arange(52,device=env.device)%13
command=commands[labels]
env.reference_reset_probability=0.
env.set_command_override(torch.zeros_like(command),gait_enabled=False)
obs,_=env.reset()
with torch.no_grad():
    for _ in range(100):obs,_,_,_,_=env.step(torch.zeros(52,12,device=env.device))
env.set_command_override(command,gait_enabled=True);env.gait_enabled[labels==12]=False
env.episode_length_buf[:]=0;env.clock_offset[:]=0
env.heading_target[:]=yaw_from_quat(env.root_states[:,3:7]);env.heading_error[:]=0
env.velocity_error.reset(torch.arange(52,device=env.device));env.velocity_drift[:]=0
env.compute_observations();obs=env.get_observations()
steps=1500;cut=150;count=(steps-cut)*cfg.control.decimation
torque_samples=torch.empty((count,3,52,12),device=env.device)
velocity=[];done_count=torch.zeros(52,device=env.device)
contacts=[];tick=0;record=False;original=env._compute_torques

def torques(action):
    global tick
    applied=original(action)
    if record:
        torque_samples[tick,0]=env.raw_pd_torques
        torque_samples[tick,1]=env.motor_torques
        torque_samples[tick,2]=applied
        tick+=1
    return applied

env._compute_torques=torques
try:
    with torch.no_grad():
        for step in range(steps):
            record=step>=cut
            obs,_,_,done,info=env.step(runner.alg.actor_critic.act_inference(obs))
            done_count+=done
            if record:
                velocity.append(info['amp']['next'][:,[36,37,41]].cpu().numpy().copy())
                contacts.append(info['amp']['contact'].cpu().numpy().copy())
            if (step+1)%500==0:print('measured_seconds',(step+1)*env.dt,flush=True)
finally:env._compute_torques=original
assert tick==count and bool(torch.isfinite(torque_samples).all())
assert int(done_count.sum())==0,'Measurement contained resets'
v=np.stack(velocity);contact=np.stack(contacts);labels_cpu=labels.cpu().numpy();rows=[];joints=[]
for i,name in enumerate(names):
    slots=labels_cpu==i;t=torque_samples[:,:,labels==i,:]
    metrics={}
    for k,key in enumerate(['raw_pd','motor','applied']):
        x=t[:,k];absolute=x.abs()
        metrics[key]=dict(mean_abs_Nm=float(absolute.mean()),signed_mean_Nm=float(x.mean()),
            rms_Nm=float(x.square().mean().sqrt()),p95_abs_Nm=float(torch.quantile(absolute.flatten(),.95)),
            peak_abs_Nm=float(absolute.max()),fraction_abs_over_6=float((absolute>6).float().mean()),
            fraction_abs_over_14=float((absolute>14).float().mean()),fraction_abs_over_17=float((absolute>17).float().mean()))
        for j,joint in enumerate(env.dof_names):
            tj=x[:,:,j]
            joints.append(dict(action=name,kind=key,joint=joint,mean_abs_Nm=float(tj.abs().mean()),
                rms_Nm=float(tj.square().mean().sqrt()),peak_abs_Nm=float(tj.abs().max())))
    vv=v[:,slots];window=25
    cycle=np.stack([np.stack([np.convolve(vv[:,copy,k],np.ones(window)/window,'valid') for k in range(3)],1) for copy in range(4)],1)
    ranges=lambda values:dict(mean=values.mean((0,1)).tolist(),min=values.min((0,1)).tolist(),
        max=values.max((0,1)).tolist(),p05=np.percentile(values,5,axis=(0,1)).tolist(),p95=np.percentile(values,95,axis=(0,1)).tolist())
    planar=np.linalg.norm(cycle[:,:,:2],axis=2);cc=contact[:,slots];n=cc.sum(2)
    diagonal=(cc[:,:,0]&cc[:,:,3])|(cc[:,:,1]&cc[:,:,2])
    row=dict(action=name,command=commands[i].cpu().tolist(),torque=metrics,velocity_components=['vx_m_s','vy_m_s','wz_rad_s'],
        instantaneous_velocity=ranges(vv),cycle_mean_velocity=ranges(cycle),
        cycle_planar_speed_m_s=dict(mean=float(planar.mean()),p05=float(np.percentile(planar,5)),p95=float(np.percentile(planar,95))),
        resets=int(done_count[labels==i].sum()),flight_fraction=float((n==0).mean()),
        illegal_support_fraction=float(((n==0)|(n==1)|((n==2)&~diagonal)).mean()))
    rows.append(row)
    print(name,'motor_mean_abs',round(metrics['motor']['mean_abs_Nm'],3),'cycle_velocity',row['cycle_mean_velocity']['mean'],flush=True)
out=Path(__file__).resolve().parent
result=dict(checkpoint=str(checkpoint),checkpoint_sha256=selected['checkpoint_sha256'],seed=1,
    duration_s=30.,warmup_excluded_s=3.,default_pose_settle_s=2.,retained_duration_s=27.,
    policy_dt_s=env.dt,physics_dt_s=env.sim_params.dt,torque_substeps=count,envs_per_action=4,
    torque_mean_definition='mean of absolute torque over all 12 motors, 4 environments and retained 400 Hz substeps',
    speed_range_definition='P05-P95 of body-frame velocity averaged over a 25-sample (0.5 s) gait cycle',
    domain_randomization=False,noise=False,pushes=False,results=rows)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'per_joint_torque.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(joints[0]));writer.writeheader();writer.writerows(joints)
np.savez_compressed(str(out/'velocities.npz'),velocity=v,contact=contact,action_id=labels_cpu,dt_s=env.dt)
representatives=torch.stack([torque_samples[:,:,i,:] for i in range(13)],2).cpu().numpy()
np.savez_compressed(str(out/'representative_torques.npz'),torque=representatives,kinds=np.array(['raw_pd','motor','applied']),
    action_names=np.array(names),joint_names=np.array(env.dof_names),dt_s=env.sim_params.dt)
print('MEASUREMENT COMPLETE',flush=True)
