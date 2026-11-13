"""Isaac Gym validation: static FK OR free-base RS01 command replay and AMP smoke.

Pose writes belong only to the FK stage and the single rollout initialization.
During replay no root/joint states are imposed and automatic resets are disabled.
"""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import play_urdf4_omni as play
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch,numpy as np
from legged_gym.utils import get_args,task_registry
from legged_gym.algorithms.rs01_amp import StyleLearner
p=argparse.ArgumentParser(add_help=False);p.add_argument('--mode',choices=['fk','replay'],required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--commands-dir',type=Path);p.add_argument('--output',type=Path,required=True);own,remaining=p.parse_known_args();sys.argv=sys.argv[:1]+remaining
args=get_args();args.task=play.TASK;args.headless=True;own.output.mkdir(parents=True,exist_ok=False);manifest=json.loads((own.dataset/'manifest.json').read_text());clips=manifest['clips'];names=manifest['joint_order'];source=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf';assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['urdf_sha256'];manifest_hash=hashlib.sha256((own.dataset/'manifest.json').read_bytes()).hexdigest()
cfg,train=task_registry.get_cfgs(name=play.TASK);cfg.env.test=True;cfg.noise.add_noise=False;cfg.push.enabled=False;cfg.domain_rand.push_robots=False;cfg.env.episode_length_s=args.duration_s+10;cfg.commands.fixed_command=[0,0,0]
for flag in ['randomize_friction','randomize_restitution','randomize_base_mass','randomize_base_com','randomize_link_mass','randomize_armature']:setattr(cfg.domain_rand,flag,False)
if own.mode=='fk':cfg.env.num_envs=len(clips)*51;cfg.sim.gravity=[0.,0.,0.]
else:cfg.env.num_envs=len(clips)*args.eval_envs
n=cfg.env.num_envs;env,_=task_registry.make_env(name=play.TASK,args=args,env_cfg=cfg);assert list(env.dof_names)==names;device=env.device;foot_ids=[env.foot_legs.index(x) for x in manifest['foot_order']]
def update():
 env.gym.refresh_actor_root_state_tensor(env.sim);env.gym.refresh_dof_state_tensor(env.sim);env.gym.refresh_rigid_body_state_tensor(env.sim);env._refresh_feet()
def put_states():
 env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states));env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
def features():
 quat=env.root_states[:,3:7];rel=env.feet_pos[:,foot_ids]-env.root_states[:,None,:3];feet=quat_rotate_inverse(quat[:,None,:].expand(-1,4,-1).reshape(-1,4),rel.reshape(-1,3)).reshape(n,12);down=torch.tensor([0.,0.,-1.],device=device).repeat(n,1)
 return torch.cat((env.dof_pos,env.dof_vel,feet,quat_rotate_inverse(quat,env.root_states[:,7:10]),quat_rotate_inverse(quat,env.root_states[:,10:13]),quat_rotate_inverse(quat,down),(env.root_states[:,2]-env.env_origins[:,2])[:,None]),1)
common={'task':play.TASK,'mode':own.mode,'dataset_manifest_sha256':manifest_hash,'urdf_sha256':manifest['urdf_sha256'],'joint_order':names,'seed':args.seed,'physics_dt_s':env.physics_dt,'control_dt_s':env.dt,'amp_training_approved':False}
if own.mode=='fk':
 expected=[];positions=[];quats=[];labels=[]
 for c in clips:
  z=np.load(own.dataset/(c['clip']+'.npz'));ix=np.linspace(0,len(z['amp_features'])-1,51).astype(int);expected.extend(z['amp_features'][ix]);positions.extend(z['joint_pos_rad'][ix]);quats.extend(z['root_quat_wxyz'][ix]);labels.extend([c['clip']]*len(ix))
 exp=torch.tensor(np.array(expected),device=device,dtype=torch.float);env.dof_pos[:]=torch.tensor(np.array(positions),device=device,dtype=torch.float);env.dof_vel[:]=0;env.root_states[:,:3]=env.env_origins;env.root_states[:,2]+=exp[:,45]+2.;quat=np.array(quats);env.root_states[:,3:7]=torch.tensor(quat[:,[1,2,3,0]],device=device,dtype=torch.float);env.root_states[:,7:]=0;put_states();env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True);update();f=features();error=(f[:,24:36]-exp[:,24:36]).abs().amax(1);rows=[]
 for i,c in enumerate(clips):rows.append({'clip':c['clip'],'samples':51,'foot_fk_max_error_m':float(error[i*51:(i+1)*51].max())})
 common.update(passed=bool(error.max()<2e-5),tolerance_m=2e-5,samples=n,cases=rows,note='One zero-gravity step above the floor, without torques; pose writes are only a kinematic test.')
else:
 per=args.eval_envs;reference=[];targets=[];defaults=[];roots=[];cmds=[]
 for c in clips:
  z=np.load(own.commands_dir/c['clip']/'physical.npz');reference.append(z['features']);targets.extend([z['command_targets']]*per);defaults.extend([z['actuator_default']]*per);roots.extend([z['initial_root']]*per);cmds.extend([c['command']]*per)
 targets=torch.tensor(np.stack(targets,1),device=device,dtype=torch.float);qm=torch.tensor(np.array(defaults),device=device,dtype=torch.float);initial=torch.tensor(np.array(roots),device=device,dtype=torch.float);commands=torch.tensor(np.array(cmds),device=device,dtype=torch.float);env.set_command_override(commands);env.reset();env.dof_pos[:]=qm;env.dof_vel[:]=0;env.root_states[:,:3]=env.env_origins+initial[:,:3];env.root_states[:,3:7]=initial[:,[4,5,6,3]];env.root_states[:,7:]=0
 # Change only the command coordinates to preserve the MuJoCo equilibrium despite
 # the Gym policy's different default pose. The plant and action clip are unchanged.
 correction=(qm-env.default_dof_pos)*(1/.928-1);initial_target=qm+correction;env.position_target[:]=initial_target;env.response_target[:]=qm;env.target_rate[:]=0;env.delay_buffer[:]=initial_target;env.response_tick=0;env.raw_pd_torques[:]=0;env.motor_torques[:]=0;env.episode_length_buf[:]=0;put_states();update();env._reset_sensors(torch.arange(n,device=device));env.compute_observations()
 fallen=torch.zeros(n,device=device,dtype=torch.bool);firstfall=torch.full((n,),float('nan'),device=device);peak=torch.zeros(n,device=device);motorpeak=peak.clone();appliedpeak=peak.clone();raw17=peak.clone();sat=peak.clone();ticks=peak.clone();last_failure={}
 def no_reset(ids):
  if len(ids):
   fresh=ids[~fallen[ids]];firstfall[fresh]=(env.common_step_counter*.02)-2.;fallen[ids]=True
 env.reset_idx=no_reset
 original=env._compute_torques
 def torque(actions):
  v=original(actions);active=(~fallen).float();peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*active));motorpeak.copy_(torch.maximum(motorpeak,env.motor_torques.abs().amax(1)*active));appliedpeak.copy_(torch.maximum(appliedpeak,v.abs().amax(1)*active));raw17.add_((env.raw_pd_torques.abs()>17).float().mean(1)*active);sat.add_((env.raw_pd_torques.abs()>14).float().mean(1)*active);ticks.add_(active);return v
 env._compute_torques=torque;record=[];valid=[];contact=[];action_excess=[];alive_before=[]
 with torch.no_grad():
  for k in range(min(len(targets),round((args.duration_s+2)/.02))):
   desired=targets[k]+correction;actions=(desired-env.default_dof_pos)/env.action_scale;action_excess.append((actions.abs()>env.cfg.normalization.clip_actions).float().mean(1));actions=torch.where(fallen[:,None],torch.zeros_like(actions),actions);was=~fallen;env.step(actions);record.append(features().clone());valid.append(was&~fallen);contact.append(env.get_foot_contact_mask()[:,foot_ids].clone())
   if (k+1)%250==0:print('[replay]',round((k+1)*.02-2,2),'s; fallen',int(fallen.sum()),flush=True)
 F=torch.stack(record);V=torch.stack(valid);CT=torch.stack(contact);A=torch.stack(action_excess);t=np.arange(1,len(F)+1)*.02-2.;f=F.cpu().numpy();v=V.cpu().numpy();ct=CT.cpu().numpy();rows=[]
 for i,c in enumerate(clips):
  sl=slice(i*per,(i+1)*per);z=reference[i];steady=(t>=2)&(t<args.duration_s-.02);source_index=np.clip((t-.0025)/.02,0,len(z)-1.000001);lo=np.floor(source_index).astype(int);bl=(source_index-lo)[:,None];expected=(1-bl)*z[lo]+bl*z[np.minimum(lo+1,len(z)-1)];good=steady[:,None]&v[:,sl];obs=f[:,sl];contactcase=ct[:,sl];qerror=(obs[:,:,:12]-expected[:,None,:12])**2;selected=obs[good];sq=qerror[good]
  row={'clip':c['clip'],'command':c['command'],'fallen_count':int(fallen[sl].sum()),'first_fall_motion_s':[None if not np.isfinite(x) else float(x) for x in firstfall[sl].cpu().tolist()],'valid_steady_frames':int(good.sum()),'action_clip_fraction':float(A[:,sl].mean()),'raw_peak_Nm':float(peak[sl].max()),'motor_peak_Nm':float(motorpeak[sl].max()),'applied_peak_Nm':float(appliedpeak[sl].max()),'raw_over17_fraction':float((raw17[sl]/ticks[sl].clamp(min=1)).mean()),'saturation_fraction':float((sat[sl]/ticks[sl].clamp(min=1)).mean())}
  if len(selected):
   cc=contactcase[good];count=cc.sum(1);diag=(cc[:,0]&cc[:,3])|(cc[:,1]&cc[:,2]);height=obs[:,:,45:46]+obs[:,:,24:36].reshape(len(obs),per,4,3)[:,:,:,2]-.02
   row.update(joint_rmse_rad=float(np.sqrt(sq.mean())),mean_velocity=selected[:,[36,37,41]].mean(0).tolist(),foot_clearance_p95_mm=(np.quantile(height[good],.95,axis=0)*1000).tolist(),flight_fraction=float((count==0).mean()),single_fraction=float((count==1).mean()),illegal_two_fraction=float(((count==2)&~diag).mean()))
  else:row.update(joint_rmse_rad=None,mean_velocity=None,foot_clearance_p95_mm=None,flight_fraction=None,single_fraction=None,illegal_two_fraction=None)
  # Actual force-contact event timing, measured per robot at 50 Hz.
  def starts(mask):
   pad=np.r_[False,mask,False];ss=np.flatnonzero(~pad[:-1]&pad[1:]);ee=np.flatnonzero(pad[:-1]&~pad[1:]);return ss[ee-ss>=2]
  gaprows=[]
  for support in [True,False]:
   gaps=[]
   for copy in range(per):
    cvalid=contactcase[steady,copy];times=t[steady]
    if not support:cvalid=~cvalid
    for aa,bb in [(0,3),(1,2)]:
     ea=starts(cvalid[:,aa]);eb=starts(cvalid[:,bb])
     if len(ea) and len(eb):gaps.extend(float(np.min(abs(times[eb]-times[k]))) for k in ea)
   gaprows.append(float(np.quantile(gaps,.95)*1000) if gaps else None)
  row['diagonal_touchdown_p95_ms']=gaprows[0];row['diagonal_liftoff_p95_ms']=gaprows[1]
  reasons=[]
  if row['fallen_count']:reasons.append('fall_or_termination')
  if row['joint_rmse_rad'] is None or row['joint_rmse_rad']>.10:reasons.append('joint_tracking')
  if row['mean_velocity'] is None or np.linalg.norm(np.array(row['mean_velocity'])[:2]-np.array(c['command'])[:2])>.05 or abs(row['mean_velocity'][2]-c['command'][2])>.12:reasons.append('velocity_tracking')
  if row['foot_clearance_p95_mm'] is None or min(row['foot_clearance_p95_mm'])<20:reasons.append('clearance')
  if row['flight_fraction'] is None or row['flight_fraction']>0:reasons.append('flight')
  if row['raw_peak_Nm']>17:reasons.append('raw_over17')
  if row['action_clip_fraction']>.001:reasons.append('policy_action_clip')
  row['rejection_reasons']=reasons;row['passed']=not reasons;rows.append(row);print('[case]',row,flush=True)
 # This is a feature/gradient numerical smoke only, not PPO or AMP success.
 guard_ok=False
 try:StyleLearner(str(own.dataset),device='cpu')
 except ValueError:guard_ok=True
 style=StyleLearner(str(own.dataset),device=device,allow_kinematic=True,settings={'amp_disc_steps':1});pairmask=V[:-1]&V[1:];prev=F[:-1][pairmask];nxt=F[1:][pairmask];conditional=commands[None].expand(len(F)-1,-1,-1)[pairmask]
 if len(prev)>8192:prev=prev[:8192];nxt=nxt[:8192];conditional=conditional[:8192]
 with torch.no_grad():bonus,inputs=style.reward(prev,nxt,conditional);style.add(inputs)
 style.update();amp={'approval_guard_preserved':guard_ok,'live_feature_dim':int(F.shape[-1]),'expert_shape':list(style.expert.shape),'conditional_input_dim':int(inputs.shape[-1]),'valid_live_transitions':len(prev),'reward_finite':bool(torch.isfinite(bonus).all()),'one_discriminator_update':style.metrics,'ppo_training_performed':False,'legacy_amp_runner_compatible':False}
 np.savez_compressed(own.output/'live_rollout.npz',amp_features=f,valid=v,contact=ct,time_s=t,commands=commands.cpu().numpy(),action_clip_fraction=A.cpu().numpy())
 common.update(duration_s=args.duration_s,envs_per_clip=per,initialization='single source q0/root initialization; zero velocity; actuator equilibrium converted algebraically',root_or_joint_overwrite_during_rollout=False,automatic_resets=False,command_mode='open_loop_MuJoCo_teacher_commands_through_original_Gym_actuator',statistical_window_motion_s=[2,args.duration_s],state_alignment='MuJoCo saved states at time+2.5ms; linear interpolation to Gym 20ms endpoints',passed=all(x['passed'] for x in rows),cases=rows,amp_smoke=amp)
common['dataset_manifest_unchanged']=hashlib.sha256((own.dataset/'manifest.json').read_bytes()).hexdigest()==manifest_hash;(own.output/'result.json').write_text(json.dumps(common,indent=2,allow_nan=False)+'\n' if own.mode=='fk' else json.dumps(common,indent=2)+'\n');print('[result]',own.output/'result.json',flush=True)
