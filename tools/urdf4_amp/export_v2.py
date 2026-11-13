"""Validate V2 candidates for 60s, export recorded states, and preserve V1 byte-for-byte."""
from pathlib import Path
import json,hashlib,numpy as np
from batch_v2 import evaluate,MOVES
from design import NAMES,LEGS
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'amp_reference_v2';out.mkdir(exist_ok=False);validation=BASE/'validation60';validation.mkdir(exist_ok=False);accepted=[];rejected=[];all_attempts=[]
def candidates(name):
 pools=[]
 for stage in ['batch_screen','action_refinement','yaw_placement','left_final_trim']:
  p=BASE/stage/((name+'_ranking.json') if stage!='left_final_trim' else 'ranking.json')
  if stage=='left_final_trim' and name!='left':continue
  if p.exists():pools+=json.loads(p.read_text())
 if name=='forward':pools+=json.loads((BASE/'forward_finalists60/ranking.json').read_text())+json.loads((BASE/'forward_yaw_trim/ranking.json').read_text())
 valid=[r for r in pools if not r.get('rejection_reasons') and (abs(MOVES[name][2])>0 or abs(r['mean_velocity'][2])<=.03)]
 valid.sort(key=lambda r:r['score'])
 # Both are 60s-proven; favor the higher-lift variant at the same 20ms diagonal timing.
 if name=='forward':valid.sort(key=lambda r:0 if r.get('candidate_name')=='221' else 1)
 return valid
for name,cmd in MOVES.items():
 chosen=None;seen=set();attempts=[]
 for b in candidates(name):
  fingerprint=json.dumps([b['design'],b['lead'],b['feed'],b['height']],sort_keys=True)
  if fingerprint in seen:continue
  seen.add(fingerprint);folder=validation/(name+'_attempt'+str(len(attempts)));r=evaluate(folder,cmd,b,60);r['motion']=name;attempts.append(r);all_attempts.append(r)
  print('VALIDATE',name,len(attempts),r['rejection_reasons'],np.round(r['mean_velocity'],3),np.round(r['clearance_p95_mm'],1),flush=True)
  if not r['rejection_reasons']:chosen=r;break
  if len(attempts)>=6:break
 (BASE/'final_validation_attempts.json').write_text(json.dumps(all_attempts,indent=2)+'\n')
 if chosen is None:
  rejected.append({'clip':name,'reason':'no candidate passed 60s','attempts':attempts});continue
 r=chosen;a=np.load(Path(r['source_folder'])/'physical.npz');cut=slice(100,2600);feat=a['features'][cut];ct=a['contact'][cut];q=a['qpos'][cut];lift=a['foot_clearance'][cut];time=a['time'][cut];T=r['design']['period_s'];duty=r['design']['stance_ratio'];controller_phase=(time[:,None]/T+np.array([0,.5,.5,0]))%1;match,offset=max((float((ct==((controller_phase+x)%1<duty)).mean()),float(x)) for x in np.linspace(-.25,.25,101));phase=(controller_phase+offset)%1;valid=np.ones(len(feat),bool);valid[-1]=False
 assert feat.shape==(2500,46) and np.isfinite(feat).all()
 np.savez_compressed(out/(name+'.npz'),amp_features=feat.astype(np.float32),transition_valid=valid,command=np.tile(cmd,(len(feat),1)).astype(np.float32),joint_pos_rad=feat[:,:12],joint_vel_rad_s=feat[:,12:24],root_pos_world_m=q[:,:3],root_quat_wxyz=q[:,3:7],leg_phase=phase,controller_leg_phase=controller_phase,foot_height_m=lift,contact=ct,foot_normal_force_N=a['foot_normal_force_N'][cut],raw_torque_Nm=a['raw'][cut],motor_torque_Nm=a['motor'][cut],applied_torque_Nm=a['applied'][cut],time_s=time-time[0])
 r.update(clip=name,frames=len(feat),cadence_hz=1/T,stance_ratio_target=duty,phase_alignment_offset_cycles=offset,phase_aligned_contact_match=match,physically_executed=True,physical_screen_passed=True,amp_training_approved=False,sha256=hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest(),measured_command=r['mean_velocity']);accepted.append(r)
 manifest=dict(schema='rs01_style_reference_v1',dataset_version='V2 faster / higher clearance / measured diagonal coordination',kind='urdf4_designed_physical_demonstrations_v2',amp_training_approved=False,validation_status='mujoco_physical_screen_only',note='Independent V2; V1 retained unchanged. No learned policy loaded. Requires new-URDF Isaac Gym AMP validation before automatic training approval.',frame_dt_s=.02,joint_order=NAMES,foot_order=LEGS,quaternion_order='wxyz',feature_size=46,urdf_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf').read_bytes()).hexdigest(),amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12','base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3','projected_gravity:3','root_height_m:1'],settings={'robot':'urdf4_rs01','policy_dt_s':.02,'physics_dt_s':.0025,'kp':40,'kd':1,'delay_s':.04,'response_tau_s':.0319,'response_gain':.928,'motor_cap_Nm':14,'source':'fresh foot trajectories + IK + centroidal support and optional yaw step-placement feedback'},phase_annotation='Actual contact-aligned phase labels; controller_leg_phase preserves planned phase. Physical states unchanged.',clips=accepted,rejected_clips=rejected)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
manifest['rejected_clips']=rejected
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EXPORTED',len(accepted),'REJECTED',len(rejected),flush=True)
