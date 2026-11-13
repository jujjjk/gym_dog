import os, json, hashlib, xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
import mujoco as mj
ROOT=Path('/home/nszb/gym'); OUT=Path(__file__).resolve().parent
src=ROOT/'URDF4.0.SLDASM/urdf/URDF4.0.SLDASM.urdf'; r=E.parse(src).getroot()
report={'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'engine':mj.__version__,'links':len(r.findall('link')),'joints':[],'inertias':[]}
for l in r.findall('link'):
 i=l.find('inertial/inertia'); a=i.attrib; I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
 ev=np.linalg.eigvalsh(I); report['inertias'].append({'link':l.get('name'),'mass':float(l.find('inertial/mass').get('value')),'eigenvalues':ev.tolist(),'valid':bool(ev[0]>0 and ev[2]<=ev[0]+ev[1]+1e-12)})
for j in r.findall('joint'):
 report['joints'].append({'name':j.get('name'),'type':j.get('type'),'xyz':j.find('origin').get('xyz'),'axis':None if j.find('axis') is None else j.find('axis').get('xyz'),'limit':None if j.find('limit') is None else j.find('limit').attrib})
report['total_mass']=sum(i['mass'] for i in report['inertias'])
report['mesh_missing']=[]
for m in r.iter('mesh'):
 p=ROOT/m.get('filename').replace('package://','');
 if not p.exists(): report['mesh_missing'].append(str(p))
 m.set('filename',str(p))
ex=E.SubElement(r,'mujoco'); E.SubElement(ex,'compiler',{'discardvisual':'false','fusestatic':'false','balanceinertia':'false'})
converted=OUT/'resolved.urdf'; E.ElementTree(r).write(converted)
m=mj.MjModel.from_xml_path(str(converted)); mj.mj_saveLastXML(str(OUT/'imported.xml'),m)
s=E.parse(OUT/'imported.xml'); rt=s.getroot(); wb=rt.find('worldbody'); trunk=wb.find('body'); E.SubElement(trunk,'freejoint',{'name':'root'}); E.SubElement(wb,'geom',{'name':'ground','type':'plane','size':'3 3 .1','friction':'0.8 0.01 0.001'})
E.SubElement(rt,'option',{'timestep':'0.001','gravity':'0 0 -9.81','integrator':'implicitfast'})
E.SubElement(wb,'light',{'pos':'0 -1 2','dir':'0 0 -1','diffuse':'0.8 0.8 0.8'}); s.write(OUT/'scene.xml'); m=mj.MjModel.from_xml_path(str(OUT/'scene.xml')); d=mj.MjData(m)
jids=[i for i in range(m.njnt) if m.jnt_type[i]==mj.mjtJoint.mjJNT_HINGE]; qa=m.jnt_qposadr[jids]; va=m.jnt_dofadr[jids]
feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,p+'_foot') for p in ['FR','FL','RR','RL']]
# Solve each leg for a foot center 0.25 m below trunk, directly below hip in x.
d.qpos[2]=0
for leg,fb in zip(['FR','FL','RR','RL'],feet):
 ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,leg+'_'+t+'_joint') for t in ['thigh','calf']]; adr=m.jnt_qposadr[ids]; sign=-1 if leg in ['FR','RR'] else 1
 d.qpos[adr]=np.array([-.8,1.5])*sign
 target=np.array([.215 if leg[0]=='F' else -.215,-.25])
 for _ in range(60):
  mj.mj_forward(m,d); err=target-d.xpos[fb,[0,2]]
  J=np.zeros((3,m.nv)); mj.mj_jacBody(m,d,J,None,fb); delta=np.linalg.lstsq(J[[0,2]][:,m.jnt_dofadr[ids]],err,rcond=None)[0]
  d.qpos[adr]=np.clip(d.qpos[adr]+np.clip(delta,-.2,.2),m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001)
q=d.qpos.copy(); q[2]=.272; d.qpos[:]=q; mj.mj_forward(m,d)
report['stance_angles']={mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i):float(q[m.jnt_qposadr[i]]) for i in jids}
report['stance_feet']=d.xpos[feet].tolist(); report['stance_com']=d.subtree_com[1].tolist()
report['initial_self_contacts']=[{'a':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom1])),'b':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom2])),'distance':float(c.dist)} for c in d.contact if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]]
# Per-joint kinematic sweep; no claim that extreme combinations are collision-free.
report['sweep']={}
for i in jids:
 positions=[]
 for v in np.linspace(*m.jnt_range[i],21):
  d.qpos[:]=q; d.qpos[m.jnt_qposadr[i]]=v; mj.mj_forward(m,d); positions.append(d.xpos[feet].copy())
 report['sweep'][mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i)]=bool(np.isfinite(positions).all())
rows=[]
for mode in ['passive','pd_stand']:
 mj.mj_resetData(m,d); d.qpos[:]=q; rawmax=0.; appliedmax=0.; sat=0; trace=[]
 for k in range(10000):
  raw=45*(q[qa]-d.qpos[qa])-2*d.qvel[va] if mode=='pd_stand' else np.zeros(12)
  d.qfrc_applied[:]=0; d.qfrc_applied[va]=np.clip(raw,-17,17); rawmax=max(rawmax,float(abs(raw).max())); appliedmax=max(appliedmax,float(abs(d.qfrc_applied[va]).max())); sat+=int((abs(raw)>17).sum()); mj.mj_step(m,d)
  if k%10==0: trace.append([d.time,*d.qpos[:7],float(np.max(abs(d.qvel))),d.ncon])
 trace=np.array(trace); np.savetxt(OUT/(mode+'.csv'),trace,delimiter=',',header='t,x,y,z,qw,qx,qy,qz,max_abs_velocity,ncon')
 report[mode]={'duration_s':10,'finite':bool(np.isfinite(trace).all()),'final_xyz':d.qpos[:3].tolist(),'min_height_m':float(trace[:,3].min()),'raw_peak_Nm':rawmax,'applied_peak_Nm':appliedmax,'saturated_joint_fraction':sat/120000,'final_joint_error_rad':float(max(abs(q[qa]-d.qpos[qa]))),'warnings':[int(w.number) for w in d.warning],'final_contacts':[]}
 for c in d.contact:
  report[mode]['final_contacts'].append({'a':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom1])),'b':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom2])),'distance_m':float(c.dist)})
 if mode=='pd_stand':
  try:
   renderer=mj.Renderer(m,480,640); cam=mj.MjvCamera(); cam.lookat[:]=[0,0,.12];cam.distance=1.15;cam.azimuth=135;cam.elevation=-20;renderer.update_scene(d,camera=cam)
   from PIL import Image
   Image.fromarray(renderer.render()).save(OUT/'stand.png'); renderer.close()
  except Exception as e: report['render_error']=str(e)
(OUT/'results.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
