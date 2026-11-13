"""Design new-URDF diagonal motion; kinematic references are NOT approved AMP experts."""
import argparse,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj
ROOT=Path(__file__).resolve().parents[2]
LEGS=['FL','FR','RL','RR'];NAMES=[l+'_'+j+'_joint' for l in LEGS for j in ['hip','thigh','calf']]
MOVES={'stand':(0,0,0),'forward_slow':(.08,0,0),'forward':(.10,0,0),'backward':(-.08,0,0),'left':(0,.05,0),'right':(0,-.05,0),'turn_left':(0,0,.2),'turn_right':(0,0,-.2),'forward_left':(.08,.035,0),'forward_right':(.08,-.035,0),'arc_left':(.08,0,.15),'arc_right':(.08,0,-.15)}
def make_model(out):
 src=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf';r=ET.parse(src).getroot()
 for mesh in r.iter('mesh'):mesh.set('filename',str((src.parent/mesh.get('filename')).resolve()))
 ex=ET.SubElement(r,'mujoco');ET.SubElement(ex,'compiler',{'discardvisual':'false','fusestatic':'false','balanceinertia':'false'})
 ET.ElementTree(r).write(out/'resolved.urdf');m=mj.MjModel.from_xml_path(str(out/'resolved.urdf'));mj.mj_saveLastXML(str(out/'imported.xml'),m)
 tree=ET.parse(out/'imported.xml');r=tree.getroot();wb=r.find('worldbody');ET.SubElement(wb.find('body'),'freejoint',{'name':'root'});ET.SubElement(wb,'geom',{'name':'ground','type':'plane','size':'5 5 .1','friction':'.8 .01 .001','solref':'.006 1'})
 ET.SubElement(wb,'light',{'pos':'0 -1 3','dir':'0 0 -1','diffuse':'.8 .8 .8'});ET.SubElement(r,'option',{'timestep':'.0025','gravity':'0 0 -9.81','integrator':'implicitfast'})
 ET.SubElement(r,'visual');tree.write(out/'scene.xml');return mj.MjModel.from_xml_path(str(out/'scene.xml'))
def setup(m):
 js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n) for n in NAMES]);return m.jnt_qposadr[js],m.jnt_dofadr[js],np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,l+'_foot') for l in LEGS]),m.jnt_range[js]
def feet_target(phase,cmd,T,duty,clear,height,width,xoffset=0.,front_clear=None,rear_clear=None):
 xy=np.array([[.215,width],[.215,-width],[-.215,width],[-.215,-width]])
 xy[:,0]+=xoffset
 u=np.array([cmd[0]-cmd[2]*xy[:,1],cmd[1]+cmd[2]*xy[:,0]]).T
 phases=(phase+np.array([0,.5,.5,0]))%1;result=np.c_[xy,np.full(4,.02-height)];stance=phases<duty
 for j,p in enumerate(phases):
  if stance[j]:delta=u[j]*T*(duty*.5-p);lift=0
  else:
   s=(p-duty)/(1-duty);b=T*duty*.5;D=T*(1-duty)
   delta=u[j]*((-b)*(2*s**3-3*s**2+1)+b*(-2*s**3+3*s**2)-D*((s**3-2*s**2+s)+(s**3-s**2)))
   c=(front_clear if j<2 else rear_clear);c=clear if c is None else c
   lift=c*64*s**3*(1-s)**3
  result[j,:2]+=delta;result[j,2]+=lift
 return result,phases,stance
class IK:
 def __init__(self,m):self.m=m;self.d=mj.MjData(m);self.qa,self.va,self.fb,self.limits=setup(m);self.q=np.array([0,-.3734,1.2011,0,.3734,-1.2011,0,-.3734,1.2011,0,.3734,-1.2011]);self.J=np.zeros((3,m.nv))
 def solve(self,target):
  d,m=self.d,self.m;d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[self.qa]=self.q
  for _ in range(12):
   mj.mj_forward(m,d);err=target-d.xpos[self.fb]
   if np.max(abs(err))<1e-9:break
   for j,body in enumerate(self.fb):
    mj.mj_jacBody(m,d,self.J,None,body);ix=slice(j*3,j*3+3);dq=np.linalg.lstsq(self.J[:,self.va[ix]],err[j],rcond=None)[0];d.qpos[self.qa[ix]]+=np.clip(dq,-.15,.15)
   d.qpos[self.qa]=np.clip(d.qpos[self.qa],self.limits[:,0]+1e-5,self.limits[:,1]-1e-5)
  mj.mj_forward(m,d);self.q=d.qpos[self.qa].copy();return self.q.copy(),float(np.max(abs(target-d.xpos[self.fb])))
def design(m,cmd,T,duty,clear,height,width,dt=.02,xoffset=0.,front_clear=None,rear_clear=None):
 ik=IK(m);n=round(T/dt);dt=T/n;Q=[];F=[];P=[];S=[];error=[]
 for k in range(n):
  f,p,s=feet_target(k/n,cmd,T,duty,clear,height,width,xoffset,front_clear,rear_clear);q,e=ik.solve(f);Q.append(q);F.append(f);P.append(p);S.append(s);error.append(e)
 q=np.array(Q);dq=(np.roll(q,-1,axis=0)-np.roll(q,1,axis=0))/(2*dt)
 return dict(q=q,dq=dq,feet=np.array(F),phase=np.array(P),stance=np.array(S),max_ik_error_m=max(error),dt=dt)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--period',type=float,default=.5);p.add_argument('--duty',type=float,default=.76);p.add_argument('--clearance',type=float,default=.018);p.add_argument('--height',type=float,default=.27);p.add_argument('--width',type=float,default=.1475);p.add_argument('--seconds',type=float,default=10);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);m=make_model(a.output)
 clips=[]
 for name,cmd in MOVES.items():
  stationary=name=='stand';ref=design(m,cmd,a.period,a.duty,0 if stationary else a.clearance,a.height,a.width);n=round(a.seconds/.02);idx=np.arange(n)%len(ref['q']);time=np.arange(n)*.02;quat=np.c_[np.cos(cmd[2]*time/2),np.zeros((n,2)),np.sin(cmd[2]*time/2)]
  if abs(cmd[2])<1e-8:xy=time[:,None]*np.array(cmd[:2])
  else:
   w=cmd[2];t=w*time;xy=np.c_[(cmd[0]*np.sin(t)+cmd[1]*(np.cos(t)-1))/w,(cmd[0]*(1-np.cos(t))+cmd[1]*np.sin(t))/w]
  root=np.c_[xy,np.full(n,a.height)];phase=ref['phase'][idx];contact=np.ones((n,4),bool) if stationary else ref['stance'][idx]
  feat=np.c_[ref['q'][idx],ref['dq'][idx],ref['feet'][idx].reshape(n,12),np.tile(cmd[:2]+(0,), (n,1)),np.tile((0,0,cmd[2]),(n,1)),np.tile((0,0,-1),(n,1)),np.full(n,a.height)]
  valid=np.ones(n,bool);valid[-1]=False;lift=ref['feet'][idx,:,2]+a.height-.02
  data=dict(amp_features=feat.astype(np.float32),transition_valid=valid,command=np.tile(cmd,(n,1)).astype(np.float32),joint_pos_rad=ref['q'][idx],joint_vel_rad_s=ref['dq'][idx],root_pos_world_m=root,root_quat_wxyz=quat,leg_phase=phase,foot_height_m=lift,desired_contact=contact,time_s=time)
  np.savez_compressed(a.output/(name+'.npz'),**data)
  audit={'clip':name,'command':list(cmd),'frames':n,'cadence_hz':0 if stationary else 1/a.period,'max_ik_error_m':ref['max_ik_error_m'],'max_joint_speed_rad_s':abs(ref['dq']).max(0).tolist(),'desired_flight_ratio':float((contact.sum(1)==0).mean()),'designed_clearance_m':0 if stationary else a.clearance,'physically_executed':False};clips.append(audit);print(name,ref['max_ik_error_m'],abs(ref['dq']).max(),flush=True)
 manifest=dict(schema='rs01_style_reference_v1',kind='urdf4_designed_kinematic_reference',amp_training_approved=False,frame_dt_s=.02,joint_order=NAMES,foot_order=LEGS,quaternion_order='wxyz',urdf_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf').read_bytes()).hexdigest(),amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12','base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3','projected_gravity:3','root_height_m:1'],settings={'period_s':a.period,'stance_ratio':a.duty,'body_height_m':a.height,'clearance_m':a.clearance,'half_width_m':a.width},clips=clips,note='New geometric design, no existing policy loaded. Requires actuator/contact audit before AMP expert use.')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
