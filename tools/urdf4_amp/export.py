"""Longer audit and immutable AMP-format export. No rollout reset is hidden."""
from pathlib import Path
import json,hashlib,numpy as np
from physical import run
from design import NAMES,LEGS
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113';rows=json.loads((BASE/'physical_candidates_v1/selected_candidates.json').read_text());left=json.loads((BASE/'left_refinement/ranking.json').read_text())[0];li=next(i for i,x in enumerate(rows) if x['motion']=='left');left['motion']='left';rows[li]=left
out=BASE/'recorded_v1';out.mkdir(exist_ok=False);recordings=BASE/'validation60';recordings.mkdir(exist_ok=False);accepted=[];rejected=[]
def events(mask):
 p=np.r_[False,mask,False];a=np.flatnonzero(~p[:-1]&p[1:]);b=np.flatnonzero(p[:-1]&~p[1:]);return a[b-a>=2]
for chosen in rows:
 name=chosen['motion'];name='march' if name=='stand' else name;cmd=np.array(chosen['command'],dtype=float);folder=recordings/name;r=run(BASE/'kinematic_v1',cmd,60,chosen['lead'],chosen['feed'],chosen['height'],folder,chosen['design']);a=np.load(folder/'physical.npz');start=100;cut=slice(start,start+2500);feat=a['features'][cut];ct=a['contact'][cut];lift=a['foot_clearance'][cut];time=a['time'][cut];T=chosen['design']['period_s'];duty=chosen['design']['stance_ratio'];phase=(time[:,None]/T+np.array([0,.5,.5,0]))%1;planned=phase<duty;liftoffs=[events((lift[:,j]>.008)&~ct[:,j]) for j in range(4)];pp=[events((ct==x).all(1)) for x in [np.array([1,0,0,1]),np.array([0,1,1,0])]];seq=sorted([(int(t),0) for t in pp[0]]+[(int(t),1) for t in pp[1]]);alt=sum(seq[i][1]!=seq[i-1][1] for i in range(1,len(seq)));lift95=np.quantile(lift,.95,axis=0)
 reasons=[]
 if r['stop']!='completed' or len(feat)!=2500:reasons.append('incomplete_or_fall')
 if not np.isfinite(feat).all():reasons.append('nonfinite')
 if r['flight_fraction']>0:reasons.append('flight')
 if r['single_fraction']+r['illegal_two_fraction']>.01:reasons.append('illegal_support')
 if r['raw_peak_Nm']>17:reasons.append('raw_torque_over17')
 if min(lift95)<.01 or max(lift95)>.035:reasons.append('clearance')
 if min(map(len,liftoffs))<60 or alt<100:reasons.append('repeated_lift_or_diagonal')
 if max(r['max_roll_pitch_deg'])>5:reasons.append('body_tilt')
 if np.linalg.norm(np.array(r['mean_velocity'])[:2]-cmd[:2])>.03 or abs(r['mean_velocity'][2]-cmd[2])>.10:reasons.append('tracking')
 r.update(clip=name,frames=len(feat),cadence_hz=1/T,stance_ratio_target=duty,liftoffs_per_foot=[len(x) for x in liftoffs],diagonal_alternations=int(alt),planned_contact_match=float((ct==planned).mean()),designed_controller_parameters=chosen,urdf_matched=True,physically_executed=True,rejection_reasons=reasons,physical_screen_passed=not reasons,amp_training_approved=False)
 (folder/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 if reasons:rejected.append(r);print('REJECT',name,reasons,flush=True);continue
 valid=np.ones(len(feat),bool);valid[-1]=False;q=a['qpos'][cut];np.savez_compressed(out/(name+'.npz'),amp_features=feat.astype(np.float32),transition_valid=valid,command=np.tile(cmd,(len(feat),1)).astype(np.float32),joint_pos_rad=feat[:,:12],joint_vel_rad_s=feat[:,12:24],root_pos_world_m=q[:,:3],root_quat_wxyz=q[:,3:7],leg_phase=phase,foot_height_m=lift,contact=ct,foot_normal_force_N=a['foot_normal_force_N'][cut],raw_torque_Nm=a['raw'][cut],motor_torque_Nm=a['motor'][cut],applied_torque_Nm=a['applied'][cut],time_s=time-time[0])
 r['sha256']=hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest();r['command']=cmd.tolist();r['measured_command']=r['mean_velocity'];accepted.append(r);print('ACCEPT_SCREEN',name,np.round(lift95*1000,1),r['raw_peak_Nm'],r['planned_contact_match'],flush=True)
 manifest=dict(schema='rs01_style_reference_v1',kind='urdf4_designed_physical_demonstrations',amp_training_approved=False,validation_status='mujoco_physical_screen_only',note='New gait design, no policy checkpoint used. Physical states executed through RS01 delay, lag, target rate/acceleration and PD torque cap. Still requires a matching new-URDF AMP environment and Isaac Gym playback validation before automatic AMP training approval. March is stepping in place, not standing.',frame_dt_s=.02,joint_order=NAMES,foot_order=LEGS,quaternion_order='wxyz',feature_size=46,urdf_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf').read_bytes()).hexdigest(),amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12','base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3','projected_gravity:3','root_height_m:1'],settings={'robot':'urdf4_rs01','policy_dt_s':.02,'physics_dt_s':.0025,'kp':40,'kd':1,'delay_s':.04,'response_tau_s':.0319,'response_gain':.928,'motor_cap_Nm':14,'self_collision':'MuJoCo default adjacency filtering','search_seed':20261114,'period_s':.5,'source':'fresh analytic foot trajectories + IK + physical support controller'},clips=accepted,rejected_clips=rejected)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
