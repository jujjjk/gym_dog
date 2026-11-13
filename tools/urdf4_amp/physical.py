"""Bounded reference tracking audit of a new design through the RS01 plant.

PD plus centroidal support-force feedforward, not a learned policy and not MPC.
No base state/force is overwritten during a rollout.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco as mj
from design import setup,design,feet_target,NAMES,LEGS,MOVES

def rpy(q):
 w,x,y,z=q;return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])
def rot(q):
 a=np.empty(9);mj.mju_quat2Mat(a,q);return a.reshape(3,3)
def crossmat(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
def run(folder,cmd,seconds=8,lead=.07,feed=1.,height=None,save=None,overrides=None):
 meta=json.loads((folder/'manifest.json').read_text());s=dict(meta['settings']);s.update(overrides or {});h=s['body_height_m'] if height is None else height;T=s['period_s'];duty=s['stance_ratio'];m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m);qa,va,fb,limits=setup(m);ref=design(m,cmd,T,duty,s['clearance_m'],h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),rear_clear=s.get('rear_clearance_m'));q0=design(m,(0,0,0),T,duty,0,h,s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.))['q'][0];d.qpos[:7]=[0,0,h+.003,1,0,0,0];d.qpos[qa]=q0;mj.mj_forward(m,d)
 target=q0.copy();rate=np.zeros(12);response=q0.copy();buf=np.tile(q0,(9,1));gain=.928;tau=.0319;kp=40;kd=1;fric=.156;dt=m.opt.timestep;mass=m.body_mass.sum();J=np.zeros((3,m.nv));ff=np.zeros(12);records=[];rawpeak=0;cntsat=0;ticks=0;stop='completed';idx=0;contact_state=np.zeros(4,bool)
 ratecap=np.array([2,2.6,3.2]*4);acc=ratecap*30
 for k in range(round((seconds+2)/dt)):
  t=k*dt-2;settle=t<0;phase=0 if settle else t/T%1
  if k%8==0:
   R=rot(d.qpos[3:7]);worldv=d.qvel[:3];bodyv=R.T@worldv;omega=d.qvel[3:6];angles=rpy(d.qpos[3:7]);vel=np.zeros(3) if settle else np.array([cmd[0],cmd[1],0.]);force=mass*(np.array([0,0,9.81])+np.r_[2*(R@vel-worldv)[:2],100*(h-d.qpos[2])-15*worldv[2]])
   moment=np.array([-8*angles[0]-.8*omega[0],-8*angles[1]-.8*omega[1],1.*((0 if settle else cmd[2])-omega[2])])
   p=(phase+lead/T+np.array([0,.5,.5,0]))%1
   w=np.ones(4) if settle else np.clip(np.minimum(p/.08,(duty-p)/.08),0,1)
   A=np.zeros((6,12));com=d.subtree_com[1]
   for j,body in enumerate(fb):A[:3,j*3:j*3+3]=np.eye(3);A[3:,j*3:j*3+3]=crossmat(d.xpos[body]-com)
   W=np.repeat(w,3);F=W*(A.T@np.linalg.solve((A*W)@A.T+np.eye(6)*1e-5,np.r_[force,moment]));F=F.reshape(4,3);F[:,2]=np.clip(F[:,2],0,100);F[:,:2]=np.clip(F[:,:2],-.6*F[:,2:3],.6*F[:,2:3]);desiredff=d.qfrc_bias[va].copy()
   for j,body in enumerate(fb):mj.mj_jacBody(m,d,J,None,body);desiredff-=J[:,va].T@F[j]
   ff=.5*ff+.5*desiredff
   fidx=((phase+lead/T)%1)*len(ref['q']);lo=int(fidx);blend=fidx-lo;desired=q0.copy() if settle else (1-blend)*ref['q'][lo]+blend*ref['q'][(lo+1)%len(ref['q'])]
   desired=q0+(desired-q0+feed*np.clip(ff,-10,10)/kp)/gain
   desired=np.clip(desired,limits[:,0]+.01,limits[:,1]-.01)
   want=np.clip((desired-target)/.02,-ratecap,ratecap);rate=np.clip(rate+np.clip(want-rate,-acc*.02,acc*.02),-ratecap,ratecap);nxt=target+rate*.02;over=(desired-target)*(desired-nxt)<=0;target=np.where(over,desired,nxt);rate=np.where(over,0,rate)
  if k%2==0:
   buf=np.roll(buf,1,axis=0);buf[0]=target;equ=q0+gain*(buf[8]-q0);response+=(1-np.exp(-.005/tau))*(equ-response)
  raw=kp*(response-d.qpos[qa])-kd*d.qvel[va];motor=np.clip(raw,-14,14);applied=motor-fric*np.tanh(d.qvel[va]/.05);d.qfrc_applied[:]=0;d.qfrc_applied[va]=applied;mj.mj_step(m,d)
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
   records.append(dict(time=t,features=features,contact=contact,foot_normal_force_N=loads,foot_clearance=d.xpos[fb,2]-.02,qpos=d.qpos.copy(),qvel=d.qvel.copy(),raw=raw.copy(),motor=motor.copy(),applied=applied.copy(),rpy=rpy(d.qpos[3:7])))
   if d.qpos[2]<.14 or max(abs(rpy(d.qpos[3:7])[:2]))>.6 or not np.isfinite(d.qpos).all():stop='fallen';break
 if not records:raise RuntimeError('no records')
 a={k:np.array([r[k] for r in records]) for k in records[0]};ct=a['contact'];vel=a['features'][:,36:39];yaw=a['features'][:,41];flight=float((ct.sum(1)==0).mean());single=float((ct.sum(1)==1).mean());illegal=float(((ct.sum(1)==2)&~((ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2]))).mean());rp=np.max(abs(a['rpy'][:,:2]),axis=0);err=np.mean((vel[:,:2]-cmd[:2])**2)+np.mean((yaw-cmd[2])**2)*.05;clear=np.quantile(a['foot_clearance'],.95,axis=0)
 loss=(1000 if stop!='completed' else 0)+err*100+float(np.sum(rp**2))*10+flight*500+(single+illegal)*200+np.mean(np.maximum(.012-clear,0)**2)*1e4+max(rawpeak-17,0)*.5
 report=dict(command=np.asarray(cmd,dtype=float).tolist(),lead=lead,feed=feed,height=h,seconds=seconds,stop=stop,mean_velocity=np.mean(np.c_[vel[:,:2],yaw],axis=0).tolist(),max_roll_pitch_deg=np.degrees(rp).tolist(),clearance_p95_mm=(clear*1000).tolist(),flight_fraction=flight,single_fraction=single,illegal_two_fraction=illegal,raw_peak_Nm=rawpeak,saturation_fraction=cntsat/max(ticks,1),actual_body_height_mean_m=float(a['qpos'][:,2].mean()),joint_speed_max_rad_s=np.max(abs(a['features'][:,12:24]),axis=0).tolist(),loss=loss,amp_training_approved=False)
 if save is not None:save.mkdir(parents=True,exist_ok=True);np.savez_compressed(save/'physical.npz',**a);(save/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--design',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--motion',default='forward');p.add_argument('--seconds',type=float,default=8);p.add_argument('--lead',type=float,default=.07);p.add_argument('--feed',type=float,default=1);a=p.parse_args();print(json.dumps(run(a.design,MOVES[a.motion],a.seconds,a.lead,a.feed,save=a.output),indent=2))
