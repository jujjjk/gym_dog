"""V3 physical teacher: ramped commands, fixed-frame route, true stand.

PD plus centroidal support-force feedforward, not a learned policy and not MPC.
No base state/force is overwritten during a rollout.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
import numpy as np
import mujoco as mj
from design_v2 import setup,NAMES,LEGS,MOVES
from design_v3 import design as template_design
from functools import partial

def rpy(q):
 w,x,y,z=q;return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])
def rot(q):
 a=np.empty(9);mj.mju_quat2Mat(a,q);return a.reshape(3,3)
def crossmat(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
def run(folder,cmd,seconds=8,lead=.07,feed=1.,height=None,save=None,overrides=None,viewer=False):
 model_meta=json.loads((folder/'manifest.json').read_text());urdf=Path(__file__).resolve().parents[2]/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
 if hashlib.sha256(urdf.read_bytes()).hexdigest()!=model_meta['urdf_sha256']:raise ValueError('V3 scene URDF is stale')
 if model_meta.get('scene_sha256') and hashlib.sha256((folder/'scene.xml').read_bytes()).hexdigest()!=model_meta['scene_sha256']:raise ValueError('V3 scene hash mismatch')
 meta=json.loads((folder/'manifest.json').read_text());s=dict(meta['settings']);s.update(overrides or {});design=partial(template_design,leg_stance_ratio=s.get('leg_stance_ratio'),stride_scale=s.get('stride_scale'));h=s['body_height_m'] if height is None else height;T=s['period_s'];duty=s['stance_ratio'];m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m);qa,va,fb,limits=setup(m);ref=design(m,cmd,T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'));q0=design(m,(0,0,0),T,duty,0,h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.))['q'][0];d.qpos[:7]=[0,0,h+.003,1,0,0,0];d.qpos[qa]=q0;mj.mj_forward(m,d)
 yaw_derivative=None;velocity_derivatives=[]
 timeline=s.get('command_schedule');scheduled_templates=[];march_ref=None
 if timeline:
  times=np.asarray([row['time_s'] for row in timeline],float)
  if times[0]!=0 or np.any(np.diff(times)<=0):raise ValueError('Schedule must start at zero and increase strictly')
  for row in timeline:
   command=np.asarray(row['command'],float)
   if np.linalg.norm(command)<1e-8:scheduled_templates.append(np.tile(q0,(len(ref['q']),1)))
   else:scheduled_templates.append(design(m,command,T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'))['q'])
  if s.get('gait_activation_threshold',0)>0:march_ref=design(m,(0,0,0),T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'))['q']
 if s.get('yaw_step_feedback',0)>0 or s.get('heading_step_gain',0)>0:
  templates=[]
  for delta in [-.2,.2]:
   yawcmd=np.array(cmd,dtype=float).copy();yawcmd[2]+=delta
   templates.append(design(m,yawcmd,T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'))['q'])
  yaw_derivative=(templates[1]-templates[0])/.4
 if s.get('velocity_step_gain',0)>0:
  for axis in [0,1]:
   templates=[]
   for delta in [-.05,.05]:
    vcmd=np.array(cmd,dtype=float).copy();vcmd[axis]+=delta
    templates.append(design(m,vcmd,T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'))['q'])
   velocity_derivatives.append((templates[1]-templates[0])/.1)
 yaw_mean=0.;yaw_integral=0.;heading_integral=0.;yaw_trim=0.;velocity_mean=np.zeros(2);velocity_integral=np.zeros(2);velocity_trim=np.zeros(2);reference_yaw=0.;reference_xy=d.qpos[:2].copy();schedule=[];previous_translation_speed=0.;previous_command_size=0.;anchor_events=[]
 initial_root=d.qpos[:7].copy();ramp_s=s.get('ramp_s',2.)
 target=q0.copy();rate=np.zeros(12);response=q0.copy();buf=np.tile(q0,(9,1));gain=.928;tau=.0319;kp=40;kd=1;fric=.156;dt=m.opt.timestep;mass=m.body_mass.sum();J=np.zeros((3,m.nv));ff=np.zeros(12);records=[];rawpeak=0;cntsat=0;ticks=0;stop='completed';idx=0;contact_state=np.zeros(4,bool)
 ratecap=np.array([2,2.6,3.2]*4);acc=ratecap*30
 handle=None;viewer_threads=[];wall_start=time.monotonic()
 if viewer:
  import mujoco.viewer,threading
  previous_threads=set(threading.enumerate());handle=mujoco.viewer.launch_passive(m,d);viewer_threads=[thread for thread in threading.enumerate() if thread not in previous_threads];handle.cam.distance=1.3;handle.cam.azimuth=135;handle.cam.elevation=-25
 for k in range(round((seconds+2)/dt)):
  t=k*dt-2;settle=t<0;phase=0 if settle else t/T%1
  u=np.clip(t/max(ramp_s,1e-9),0,1);amplitude=u*u*(3-2*u)
  if s.get('stop_ramp_s',0)>0:
   down=np.clip((s.get('stop_at_s',seconds)-t)/s['stop_ramp_s'],0,1);amplitude*=down*down*(3-2*down)
  envelope=amplitude;active_ref=ref['q']
  if timeline:
   ia=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1));ib=min(ia+1,len(times)-1)
   blend_time=np.clip((t-times[ia])/max(times[ib]-times[ia],1e-9),0,1) if ia!=ib else 0.
   blend_time=blend_time**2*(3-2*blend_time)
   ca=np.asarray(timeline[ia]['command']);cb=np.asarray(timeline[ib]['command']);planned=envelope*((1-blend_time)*ca+blend_time*cb)
   active_ref=(1-blend_time)*scheduled_templates[ia]+blend_time*scheduled_templates[ib]
   amplitude=envelope*((1-blend_time)*float(np.linalg.norm(ca)>1e-8)+blend_time*float(np.linalg.norm(cb)>1e-8))
  else:
   if np.linalg.norm(cmd)<1e-8:amplitude=0.
   planned=np.asarray(cmd)*amplitude
  linear_amplitude=amplitude
  if march_ref is not None:
   activation=np.clip(amplitude/s['gait_activation_threshold'],0,1);amplitude=activation**2*(3-2*activation)
  Rref=rot(np.array([np.cos(reference_yaw/2),0.,0.,np.sin(reference_yaw/2)]))
  route_velocity=(Rref@np.r_[planned[:2],0.])[:2]
  if k%8==0:
   translation_speed=np.linalg.norm(planned[:2])
   command_size=np.linalg.norm(planned)
   if s.get('stop_anchor',False) and translation_speed<=.01 and previous_translation_speed>.01:
    anchor_events.append(dict(time_s=t,kind='stop_position_hold',previous_reference_xy=reference_xy.tolist(),measured_xy=d.qpos[:2].tolist()));reference_xy=d.qpos[:2].copy()
   if s.get('stop_anchor',False) and command_size<=.01 and previous_command_size>.01:
    actual_yaw=rpy(d.qpos[3:7])[2];anchor_events.append(dict(time_s=t,kind='stand_heading_hold',previous_reference_yaw=float(reference_yaw),measured_yaw=float(actual_yaw)));reference_yaw=actual_yaw;heading_integral=0.
   previous_translation_speed=translation_speed
   previous_command_size=command_size
   R=rot(d.qpos[3:7]);worldv=d.qvel[:3];bodyv=R.T@worldv;omega=d.qvel[3:6];angles=rpy(d.qpos[3:7])
   if s.get('route_progress_projection',False) and translation_speed>.01 and abs(planned[2])<.01:
    tangent=route_velocity/max(np.linalg.norm(route_velocity),1e-9);reference_xy+=np.dot(d.qpos[:2]-reference_xy,tangent)*tangent
   error_xy=d.qpos[:2]-reference_xy
   if np.linalg.norm(planned[:2])>.01:
    tangent=route_velocity/max(np.linalg.norm(route_velocity),1e-9);normal=np.array([-tangent[1],tangent[0]])
    correction=-s.get('path_gain',1.)*np.dot(error_xy,normal)*normal
    if abs(planned[2])>.01:correction-=s.get('along_gain',.5)*np.dot(error_xy,tangent)*tangent
   else:correction=-s.get('path_gain',1.)*error_xy
   desired_world=np.r_[route_velocity+np.clip(correction,-.08,.08),0.]
   force=mass*(np.array([0,0,9.81])+np.r_[s.get('velocity_gain',2)*(desired_world-worldv)[:2],s.get('height_kp',100)*(h-d.qpos[2])-s.get('height_kd',15)*worldv[2]])
   heading_error=np.arctan2(np.sin(reference_yaw-angles[2]),np.cos(reference_yaw-angles[2]))
   if not settle and amplitude>.1:heading_integral=np.clip(heading_integral+.02*heading_error,-.3,.3)
   if not settle and yaw_derivative is not None:
    yaw_world=(R@omega)[2];yaw_mean+=.02/.27*(yaw_world-yaw_mean);yaw_integral=np.clip(yaw_integral+.02*(planned[2]-yaw_world),-.3,.3);yaw_trim=np.clip(s.get('yaw_step_feedback',0)*(planned[2]-yaw_mean)+s.get('yaw_step_integral',.5)*yaw_integral+s.get('heading_step_gain',0)*heading_error+s.get('heading_integral_gain',0)*heading_integral,-.3,.3)
   if not settle and velocity_derivatives:
    velocity_mean+=.02/.27*(bodyv[:2]-velocity_mean);desired_body=(R.T@desired_world)[:2]
    velocity_integral=np.clip(velocity_integral+.02*(desired_body-velocity_mean),-.2,.2)
    velocity_trim=np.clip(s['velocity_step_gain']*(desired_body-velocity_mean)+s.get('velocity_step_integral',0)*velocity_integral,-.12,.12)
   moment=R@np.array([-8*angles[0]-.8*omega[0],-8*angles[1]-.8*omega[1],s.get('yaw_velocity_gain',1.)*(planned[2]-omega[2])+s.get('heading_gain',.5)*heading_error])
   leads=np.array([lead+s.get('front_lead_offset_s',0.)]*2+[lead+s.get('rear_lead_offset_s',0.)]*2)+np.asarray(s.get('leg_lead_offsets_s',[0.,0.,0.,0.]))
   p=(phase+leads/T+np.array([0,.5,.5,0]))%1
   duties=np.asarray(s.get('leg_stance_ratio',[duty]*4))
   w=(1-amplitude)+amplitude*np.clip(np.minimum(p/s.get('support_ramp_phase',.08),(duties-p)/s.get('support_ramp_phase',.08)),0,1)
   A=np.zeros((6,12));com=d.subtree_com[1]
   for j,body in enumerate(fb):A[:3,j*3:j*3+3]=np.eye(3);A[3:,j*3:j*3+3]=crossmat(d.xpos[body]-com)
   W=np.repeat(w,3);F=W*(A.T@np.linalg.solve((A*W)@A.T+np.eye(6)*1e-5,np.r_[force,moment]));F=F.reshape(4,3);F[:,2]=np.clip(F[:,2],0,100);F[:,:2]=np.clip(F[:,:2],-.6*F[:,2:3],.6*F[:,2:3]);desiredff=d.qfrc_bias[va].copy()
   for j,body in enumerate(fb):mj.mj_jacBody(m,d,J,None,body);desiredff-=J[:,va].T@F[j]
   alpha=s.get('feedforward_alpha',.5);ff=(1-alpha)*ff+alpha*desiredff
   desired=q0.copy()
   if not settle:
    for j in range(4):
     fidx=((phase+leads[j]/T)%1)*len(ref['q']);lo=int(fidx);blend=fidx-lo;ix=slice(j*3,j*3+3);desired[ix]=(1-blend)*active_ref[lo,ix]+blend*active_ref[(lo+1)%len(ref['q']),ix]
     if march_ref is not None:desired[ix]+=(amplitude-linear_amplitude)*((1-blend)*march_ref[lo,ix]+blend*march_ref[(lo+1)%len(ref['q']),ix]-q0[ix])
     feedback_scale=amplitude/max(envelope,1e-9) if timeline else 1.
     if yaw_derivative is not None:desired[ix]+=feedback_scale*yaw_trim*((1-blend)*yaw_derivative[lo,ix]+blend*yaw_derivative[(lo+1)%len(ref['q']),ix])
     for axis,derivative in enumerate(velocity_derivatives):desired[ix]+=feedback_scale*velocity_trim[axis]*((1-blend)*derivative[lo,ix]+blend*derivative[(lo+1)%len(ref['q']),ix])
   desired=q0+((envelope if timeline else amplitude)*(desired-q0)+feed*np.clip(ff,-10,10)/kp)/gain
   desired=np.clip(desired,limits[:,0]+.01,limits[:,1]-.01);schedule.append(desired.copy())
   want=np.clip((desired-target)/.02,-ratecap,ratecap);rate=np.clip(rate+np.clip(want-rate,-acc*.02,acc*.02),-ratecap,ratecap);nxt=target+rate*.02;over=(desired-target)*(desired-nxt)<=0;target=np.where(over,desired,nxt);rate=np.where(over,0,rate)
  if k%2==0:
   buf=np.roll(buf,1,axis=0);buf[0]=target;equ=q0+gain*(buf[8]-q0);response+=(1-np.exp(-.005/tau))*(equ-response)
  raw=kp*(response-d.qpos[qa])-kd*d.qvel[va];motor=np.clip(raw,-14,14);applied=motor-fric*np.tanh(d.qvel[va]/.05);d.qfrc_applied[:]=0;d.qfrc_applied[va]=applied;mj.mj_step(m,d)
  if handle is not None and k%8==0:
   if not handle.is_running():stop='viewer_closed';break
   handle.cam.lookat[:]=d.qpos[:3];handle.sync();time.sleep(max(0.,(k+1)*dt-(time.monotonic()-wall_start)))
  if not settle:rawpeak=max(rawpeak,float(abs(raw).max()));cntsat+=int((abs(raw)>14).sum());ticks+=12
  if k%8==0 and not settle:
   mj.mj_forward(m,d);R=rot(d.qpos[3:7]);bodyv=R.T@d.qvel[:3];ang=d.qvel[3:6].copy();feet=(d.xpos[fb]-d.qpos[:3])@R;gravity=R.T@np.array([0,0,-1]);features=np.r_[d.qpos[qa],d.qvel[va],feet.reshape(-1),bodyv,ang,gravity,d.qpos[2]];contact=np.zeros(4,bool)
   loads=np.zeros(4)
   for ci,c in enumerate(d.contact):
    bodies=[m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]
    if 0 in bodies:
     force_contact=np.zeros(6);mj.mj_contactForce(m,d,ci,force_contact)
     for j,body in enumerate(fb):
      if body in bodies:loads[j]+=max(0.,force_contact[0])
   contact_state=np.where(contact_state,loads>1.5,loads>=3.);contact=contact_state.copy()
   records.append(dict(time=t,features=features,contact=contact,foot_normal_force_N=loads,foot_clearance=d.xpos[fb,2]-.02,qpos=d.qpos.copy(),qvel=d.qvel.copy(),raw=raw.copy(),motor=motor.copy(),applied=applied.copy(),rpy=rpy(d.qpos[3:7]),command=planned.copy(),reference_xy=reference_xy.copy(),reference_yaw=reference_yaw,leg_phase_commanded=p.copy(),desired_contact=np.where(amplitude>.01,p<duties,np.ones(4,bool)),limited_target=target.copy(),response_target=response.copy()))
   if d.qpos[2]<.14 or max(abs(rpy(d.qpos[3:7])[:2]))>.6 or not np.isfinite(d.qpos).all():stop='fallen';break
  if not settle:reference_xy+=route_velocity*dt;reference_yaw=np.arctan2(np.sin(reference_yaw+planned[2]*dt),np.cos(reference_yaw+planned[2]*dt))
 if handle is not None:
  # launch_passive starts a daemon render thread. Wait for its native resources
  # before model destruction/interpreter shutdown rather than racing GLFW exit.
  handle.close()
  for thread in viewer_threads:thread.join(timeout=10.)
 if not records:
  if viewer:return dict(stop=stop,frames=0,amp_training_approved=False)
  raise RuntimeError('no motion recorded')
 full={k:np.array([r[k] for r in records]) for k in records[0]};steady=(full['time']>=ramp_s+1)&(full['time']<s.get('stop_at_s',seconds)-s.get('stop_ramp_s',0)-.1)
 if not steady.any():steady=np.ones(len(full['time']),bool)
 a={k:v[steady] for k,v in full.items()};ct=a['contact'];vel=a['features'][:,36:39];yaw=a['features'][:,41];flight=float((ct.sum(1)==0).mean());single=float((ct.sum(1)==1).mean());illegal=float(((ct.sum(1)==2)&~((ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2]))).mean());rp=np.max(abs(a['rpy'][:,:2]),axis=0);err=np.mean((vel[:,:2]-cmd[:2])**2)+np.mean((yaw-cmd[2])**2)*.05;clear=np.quantile(a['foot_clearance'],.95,axis=0)
 loss=(1000 if stop!='completed' else 0)+err*100+float(np.sum(rp**2))*10+flight*500+(single+illegal)*200+np.mean(np.maximum(.012-clear,0)**2)*1e4+max(rawpeak-17,0)*.5
 # Coordination is measured from actual force-contact events, not the planned clock.
 def starts(mask):
  pp=np.r_[False,mask,False];st=np.flatnonzero(~pp[:-1]&pp[1:]);en=np.flatnonzero(pp[:-1]&~pp[1:]);return st[en-st>=2]
 def paired_gaps(active):
  values=[]
  for left,right in [(0,3),(1,2)]:
   a0=starts(active[:,left]);b0=starts(active[:,right]);a0=a0[a['time'][a0]>=1.];b0=b0[a['time'][b0]>=1.]
   if not len(a0) or not len(b0):values.append(T);continue
   for ea in a0:
    gap=float(np.min(abs(a['time'][b0]-a['time'][ea])));values.append(min(T,gap))
  return float(np.quantile(values,.95)*1000),float(max(values)*1000)
 touchdown95,touchdownmax=paired_gaps(ct);liftoff95,liftoffmax=paired_gaps(~ct)
 coordination=dict(diagonal_touchdown_p95_ms=touchdown95,diagonal_touchdown_max_ms=touchdownmax,diagonal_liftoff_p95_ms=liftoff95,diagonal_liftoff_max_ms=liftoffmax,body_z_std_mm=float(a['qpos'][:,2].std()*1000),yaw_rate_rms_rad_s=float(np.sqrt(np.mean((yaw-cmd[2])**2))),lift_events_per_foot=[len(starts((a['foot_clearance'][:,j]>.012)&~ct[:,j])) for j in range(4)])
 for label,active in [('touchdown',ct),('liftoff',~ct)]:
  event_phases=[]
  for j in range(4):
   events=starts(active[:,j]);events=events[events>0]
   z=np.exp(2j*np.pi*(a['time'][events]/T-np.array([0,.5,.5,0])[j]))
   event_phases.append(float(np.angle(z.mean())/(2*np.pi)) if len(z) else None)
  coordination[label+'_mean_phase_per_foot']=event_phases
 report=dict(controller_version='v2_body_frame_moment_and_per_leg_lead',coordination=coordination,command=np.asarray(cmd,dtype=float).tolist(),lead=lead,feed=feed,height=h,seconds=seconds,stop=stop,mean_velocity=np.mean(np.c_[vel[:,:2],yaw],axis=0).tolist(),max_roll_pitch_deg=np.degrees(rp).tolist(),clearance_p95_mm=(clear*1000).tolist(),flight_fraction=flight,single_fraction=single,illegal_two_fraction=illegal,raw_peak_Nm=rawpeak,saturation_fraction=cntsat/max(ticks,1),actual_body_height_mean_m=float(a['qpos'][:,2].mean()),joint_speed_max_rad_s=np.max(abs(a['features'][:,12:24]),axis=0).tolist(),loss=loss,amp_training_approved=False)
 delta_xy=a['qpos'][:,:2]-a['reference_xy'];angle=np.arctan2(cmd[1],cmd[0]) if np.linalg.norm(cmd[:2])>1e-8 else 0.
 normal=np.c_[-np.sin(a['reference_yaw']+angle),np.cos(a['reference_yaw']+angle)]
 cross_error=abs((delta_xy*normal).sum(1)) if np.linalg.norm(cmd[:2])>1e-8 else np.linalg.norm(delta_xy,axis=1)
 if timeline:
  angles=a['reference_yaw']+np.arctan2(a['command'][:,1],a['command'][:,0]);normal=np.c_[-np.sin(angles),np.cos(angles)]
  cross_error=np.where(np.linalg.norm(a['command'][:,:2],axis=1)>.01,abs((delta_xy*normal).sum(1)),np.linalg.norm(delta_xy,axis=1))
 heading_error=np.arctan2(np.sin(a['rpy'][:,2]-a['reference_yaw']),np.cos(a['rpy'][:,2]-a['reference_yaw']))
 report.update(controller_version='v3_ramped_fixed_frame',cross_max_m=float(cross_error.max()),heading_mean_abs_deg=float(abs(heading_error).mean()*180/np.pi),steady_from_s=ramp_s+1,steady_seconds=len(a['time'])*.02,total_mass_kg=float(mass))
 report.update(max_ik_error_m=ref['max_ik_error_m'],joint_motor_rms_Nm=np.sqrt(np.mean(full['motor']**2,axis=0)).tolist(),joint_applied_rms_Nm=np.sqrt(np.mean(full['applied']**2,axis=0)).tolist(),stop_final_velocity=full['features'][-50:, [36,37,41]].mean(0).tolist(),stop_final_contact_all_fraction=float(full['contact'][-50:].all(1).mean()))
 report['reference_anchor_events']=anchor_events
 report.update(urdf_sha256=model_meta['urdf_sha256'],scene_sha256=hashlib.sha256((folder/'scene.xml').read_bytes()).hexdigest())
 assert abs(mass-12.369864831733414)<1e-7
 if save is not None:save.mkdir(parents=True,exist_ok=True);np.savez_compressed(save/'physical.npz',**full,command_targets=np.asarray(schedule),actuator_default=q0,initial_root=initial_root,command_dt_s=np.array(.02));(save/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--design',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--motion',default='forward');p.add_argument('--seconds',type=float);p.add_argument('--lead',type=float);p.add_argument('--feed',type=float);p.add_argument('--spec',type=Path,help='Saved report or dataset manifest');p.add_argument('--viewer',action='store_true');a=p.parse_args()
 spec=json.loads(a.spec.read_text()) if a.spec else None
 if spec and 'clips' in spec:spec=next(c for c in spec['clips'] if c['clip']==a.motion)
 command=spec['command'] if spec else MOVES[a.motion];seconds=a.seconds if a.seconds is not None else (spec.get('seconds',8) if spec else 8)
 lead=a.lead if a.lead is not None else (spec['lead'] if spec else .07);feed=a.feed if a.feed is not None else (spec['feed'] if spec else 1.)
 print(json.dumps(run(a.design,command,seconds,lead,feed,height=spec.get('height') if spec else None,save=a.output,overrides=spec['design'] if spec else None,viewer=a.viewer),indent=2))
