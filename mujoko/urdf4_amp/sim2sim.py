"""Closed-loop URDF4 AMP/CTS student in MuJoCo with the exported RS01 contract."""
import argparse,json,hashlib,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

class NumpyStudent:
    def __init__(self,path):
        self.w=dict(np.load(path,allow_pickle=False))
    def _mlp(self,x,prefix):
        layers=sorted(int(k.split('.')[1]) for k in self.w if k.startswith(prefix+'.') and k.endswith('.weight'))
        for i,k in enumerate(layers):
            x=x@self.w['%s.%d.weight'%(prefix,k)].T+self.w['%s.%d.bias'%(prefix,k)]
            if i<len(layers)-1:x=np.where(x>0,x,np.expm1(np.minimum(x,0))).astype(np.float32)
        return x
    def __call__(self,history):
        x=np.asarray(history,dtype=np.float32).reshape(-1,1020)
        return self._mlp(np.concatenate((x[:,:51]*self.w['frame_gain'],self._mlp(x,'encoder')),1),'actor')

def prepare_scene(bundle):
    import mujoco as mj
    c=json.loads((bundle/'contract.json').read_text());src=Path(c['urdf'])
    assert hashlib.sha256(src.read_bytes()).hexdigest()==c['urdf_sha256']
    tree=ET.parse(src);r=tree.getroot()
    for mesh in r.iter('mesh'):mesh.set('filename',str((src.parent/mesh.get('filename')).resolve()))
    ext=ET.SubElement(r,'mujoco');ET.SubElement(ext,'compiler',discardvisual='false',fusestatic='false',balanceinertia='false')
    tree.write(bundle/'resolved.urdf');m=mj.MjModel.from_xml_path(str(bundle/'resolved.urdf'))
    mj.mj_saveLastXML(str(bundle/'imported.xml'),m)
    tree=ET.parse(bundle/'imported.xml');r=tree.getroot();world=r.find('worldbody')
    # mj_saveLastXML uses compact decimal formatting. Preserve the compiled
    # URDF inertial values at full precision in the deployment scene.
    values=lambda x:' '.join(format(float(v),'.17g') for v in np.asarray(x).reshape(-1))
    for body in world.iter('body'):
        index=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,body.get('name'))
        inertia=body.find('inertial')
        if inertia is not None:
            inertia.attrib.update(pos=values(m.body_ipos[index]),quat=values(m.body_iquat[index]),
                mass=values([m.body_mass[index]]),diaginertia=values(m.body_inertia[index]))
    base=world.find("body[@name='trunk']");assert base is not None
    ET.SubElement(base,'freejoint',name='root')
    # Physics parameters are not adjusted to fit this policy. Keep the URDF
    # masses, inertia, primitives, joint limits and disabled self-collisions.
    for geom in r.iter('geom'):
        if geom.get('contype','1')=='0' and geom.get('conaffinity','1')=='0':continue
        geom.set('contype','2');geom.set('conaffinity','1');geom.set('friction','.8 0 0')
        geom.set('solref','.0065 1');geom.set('solimp','.99 .999 .0001 .5 2')
        geom.set('condim','3')
    for joint in r.iter('joint'):
        joint.set('damping','0');joint.set('armature','0');joint.set('frictionloss','0')
    asset=r.find('asset')
    if asset is None:asset=ET.SubElement(r,'asset')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.32 .34 .38',rgb2='.52 .54 .58',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='5 5',texuniform='true')
    ET.SubElement(world,'geom',name='ground',type='plane',size='0 0 .1',material='floor_mat',contype='1',conaffinity='2',friction='.8 0 0',
        condim='3',solref='.0065 1',solimp='.99 .999 .0001 .5 2')
    ET.SubElement(world,'light',pos='0 -1 4',dir='0 0 -1',diffuse='.8 .8 .8')
    opt=r.find('option')
    if opt is None:opt=ET.SubElement(r,'option')
    opt.attrib.update(timestep=str(c['physics_dt_s']),gravity='0 0 -9.81',integrator='implicitfast',iterations='100',tolerance='1e-10')
    ET.SubElement(r,'visual');ET.indent(tree);tree.write(bundle/'scene.xml')
    model=mj.MjModel.from_xml_path(str(bundle/'scene.xml'))
    assert abs(model.body_mass.sum()-11.569864831733414)<1.e-7
    return model

def wrap(x):return np.arctan2(np.sin(x),np.cos(x))

class StudentSim:
    def __init__(self,bundle):
        import mujoco as mj
        self.mj=mj;self.bundle=Path(bundle);self.c=json.loads((self.bundle/'contract.json').read_text())
        assert hashlib.sha256((self.bundle/'student.npz').read_bytes()).hexdigest()==self.c['numpy_weights_sha256']
        self.policy=NumpyStudent(self.bundle/'student.npz')
        self.m=mj.MjModel.from_xml_path(str(self.bundle/'scene.xml'));self.d=mj.MjData(self.m)
        self.names=self.c['joint_order'];j=np.array([mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_JOINT,n) for n in self.names]);assert min(j)>=0
        self.qa=self.m.jnt_qposadr[j];self.va=self.m.jnt_dofadr[j]
        self.trunk=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'trunk')
        self.feet=np.array([mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,l+'_foot') for l in self.c['foot_order']])
        self.ground=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'ground')
        self.default=np.array([self.c['default_joint_angles'][n] for n in self.names],np.float32)
        by_joint=lambda values:np.array([next(v for key,v in values.items() if key in n) for n in self.names],np.float32)
        self.scale=by_joint(self.c['control']['action_scale_by_joint']);self.kp=by_joint(self.c['control']['stiffness']);self.kd=by_joint(self.c['control']['damping'])
        act=self.c['actuator'];self.rate_limit=by_joint(act['target_rate_limit_rad_s']);self.accel_limit=by_joint(act['target_acceleration_limit_rad_s2'])
        self.policy_dt=np.float32(act['control_dt_s']);self.dt=self.m.opt.timestep
        self.response_every=round(act['response_update_dt_s']/self.dt);self.delay_steps=round(act['closed_loop_delay_s']/act['response_update_dt_s'])
        self.alpha=np.float32(1-np.exp(-act['response_update_dt_s']/act['time_constant_s']))
        self.decimation=self.c['control']['decimation'];assert abs(self.dt*self.decimation-float(self.policy_dt))<1e-8
        self.contacts=np.zeros(4,bool);self.load=np.zeros(4);self.raw=np.zeros(12);self.motor=self.raw.copy();self.applied=self.raw.copy()
        self.reset()

    def truth(self):
        # mjOBJ_XBODY uses the URDF link frame rather than the diagonalized
        # principal-inertia frame; convert world velocities into that frame.
        R=self.d.xmat[self.trunk].reshape(3,3);velocity=np.zeros(6)
        self.mj.mj_objectVelocity(self.m,self.d,self.mj.mjtObj.mjOBJ_XBODY,self.trunk,velocity,0)
        return (R.T@velocity[:3]).astype(np.float32),(R.T@velocity[3:]).astype(np.float32),(R.T@np.array([0,0,-1])).astype(np.float32)

    def yaw(self):
        w,x,y,z=self.d.qpos[3:7];return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

    def observe(self):
        gyro,_,gravity=self.truth()
        self.yaw_head=(self.yaw_head+1)%25;self.yaw_window[self.yaw_head]=self.command[2]-gyro[2]
        phase=np.remainder(np.float32(self.policy_tick)*np.float32(self.c['dt'])/np.float32(self.c['gait']['period_s']),np.float32(1))
        angle=np.float32(2*np.pi)*phase
        error=np.float32(wrap(self.heading_target-self.yaw()))
        frame=np.r_[gyro*.25,gravity,self.command*np.array([2,2,.25],np.float32),
            (self.d.qpos[self.qa].astype(np.float32)-self.default),self.d.qvel[self.va].astype(np.float32)*.05,
            self.action,np.sin(angle),np.cos(angle),self.yaw_window.mean()*4,np.sin(error),np.cos(error)-1,float(self.enabled)].astype(np.float32)
        assert frame.shape==(51,)
        self.history=np.roll(self.history,1,axis=0);self.history[0]=frame
        if self.stale:self.history[:]=frame;self.stale=False
        self.history=np.clip(self.history,-self.c['normalization']['clip_observations'],self.c['normalization']['clip_observations'])
        return self.history.reshape(1,-1)

    def reset(self,seed=1,noise=.02):
        self.mj.mj_resetData(self.m,self.d);self.d.qpos[:7]=self.c['initial_root']
        self.d.qpos[self.qa]=self.default+np.random.RandomState(seed).uniform(-noise,noise,12)
        self.d.qvel[:]=0;self.mj.mj_forward(self.m,self.d)
        self.target=self.default.copy();self.rate=np.zeros(12,np.float32);self.response=self.default.copy()
        self.delay=np.tile(self.default,(self.delay_steps+1,1));self.action=np.zeros(12,np.float32)
        self.response_tick=0;self.policy_tick=0;self.heading_target=self.yaw()
        self.command=np.zeros(3,np.float32);self.enabled=False
        self.yaw_window=np.zeros(25,np.float32);self.yaw_head=0
        self.history=np.zeros((20,51),np.float32);self.stale=True;self.contacts[:]=False
        self.observe()

    def start(self,command,enabled=True):
        self.command=np.array(command,np.float32);self.enabled=enabled
        self.policy_tick=0;self.heading_target=self.yaw();self.observe()

    def step(self,action=None):
        if action is None:action=self.policy(self.history.reshape(1,-1))[0]
        self.action=np.clip(action,-self.c['normalization']['clip_actions'],self.c['normalization']['clip_actions']).astype(np.float32)
        desired=self.default+self.scale*self.action;want=np.clip((desired-self.target)/self.policy_dt,-self.rate_limit,self.rate_limit)
        self.rate=np.clip(self.rate+np.clip(want-self.rate,-self.accel_limit*self.policy_dt,self.accel_limit*self.policy_dt),-self.rate_limit,self.rate_limit)
        nxt=self.target+self.rate*self.policy_dt;crossed=(desired-self.target)*(desired-nxt)<=0
        self.target=np.where(crossed,desired,nxt);self.rate=np.where(crossed,0,self.rate).astype(np.float32)
        samples=[];act=self.c['actuator']
        for _ in range(self.decimation):
            if self.response_tick%self.response_every==0:
                self.delay=np.roll(self.delay,1,axis=0);self.delay[0]=self.target
                equilibrium=self.default+np.float32(act['response_gain'])*(self.delay[self.delay_steps]-self.default)
                self.response+=self.alpha*(equilibrium-self.response)
            self.response_tick+=1
            self.raw=self.kp*(self.response-self.d.qpos[self.qa].astype(np.float32))-self.kd*self.d.qvel[self.va].astype(np.float32)
            self.motor=np.clip(self.raw,-act['peak_torque_limit_nm'],act['peak_torque_limit_nm'])
            self.applied=self.motor-act['coulomb_friction_nm']*np.tanh(self.d.qvel[self.va]/act['friction_smoothing_rad_s'])
            self.d.qfrc_applied[:]=0;self.d.qfrc_applied[self.va]=self.applied
            samples.append(np.stack((self.raw,self.motor,self.applied)))
            self.mj.mj_step(self.m,self.d)
        self.mj.mj_forward(self.m,self.d);self.policy_tick+=1
        self.heading_target=wrap(self.heading_target+float(self.command[2])*float(self.policy_dt))
        self.observe();self.refresh_contacts();return np.stack(samples)

    def refresh_contacts(self):
        self.load[:]=0
        self.trunk_contact=False
        for ci,c in enumerate(self.d.contact):
            g1,g2=c.geom1,c.geom2
            if self.ground not in (g1,g2):continue
            other=g2 if g1==self.ground else g1;body=self.m.geom_bodyid[other]
            if body==self.trunk:self.trunk_contact=True
            for j,foot in enumerate(self.feet):
                if body==foot:
                    force=np.zeros(6);self.mj.mj_contactForce(self.m,self.d,ci,force)
                    self.load[j]+=max(0.,force[0])
        self.contacts=np.where(self.contacts,self.load>1.5,self.load>=3.)

    def state(self):
        gyro,v,gravity=self.truth();R=self.d.xmat[self.trunk].reshape(3,3)
        feet=(self.d.xpos[self.feet]-self.d.xpos[self.trunk])@R
        return np.r_[self.d.qpos[self.qa],self.d.qvel[self.va],feet.ravel(),v,gyro,gravity,self.d.qpos[2]]

    def fallen(self):
        return bool(self.trunk_contact or self.d.qpos[2]<.16 or self.truth()[2][2]>-.6 or not np.isfinite(self.d.qpos).all())

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/urdf4_mujoco_20261114')
    p.add_argument('--prepare',action='store_true');p.add_argument('--sweep',action='store_true');p.add_argument('--motion',default='forward_slow')
    p.add_argument('--seconds',type=float,default=10);p.add_argument('--follow',action='store_true');a=p.parse_args()
    if a.prepare:
        m=prepare_scene(a.bundle);print('SCENE READY',m.nq,m.nv,m.body_mass.sum());return
    import mujoco.viewer
    sim=StudentSim(a.bundle);motions=sim.c['motions'];names=[m['name'] for m in motions]
    # Hide primitive collision envelopes in the viewer while keeping all
    # collision, mass and force calculations intact.
    for geom in range(sim.m.ngeom):
        if sim.m.geom_bodyid[geom]!=0 and (sim.m.geom_contype[geom] or sim.m.geom_conaffinity[geom]):
            sim.m.geom_group[geom]=3
    sim.m.vis.headlight.ambient[:]=.4
    if a.motion not in names:p.error('Unknown motion: '+', '.join(names))
    index=names.index(a.motion);requests=dict(delta=0,pause=False,restart=False)
    def key(code):
        if code==78:requests['delta']+=1
        if code==66:requests['delta']-=1
        if code==32:requests['pause']=not requests['pause']
        if code==82:requests['restart']=True
    with mujoco.viewer.launch_passive(sim.m,sim.d,key_callback=key) as viewer:
        viewer.opt.geomgroup[3]=0
        viewer.cam.distance=1.4;viewer.cam.azimuth=135;viewer.cam.elevation=-20;viewer.cam.lookat[:]=sim.d.qpos[:3]
        tick=0;settle=100;previous_xy=sim.d.qpos[:2].copy()
        print('Live student policy, N next / B previous / Space pause / R restart',flush=True)
        while viewer.is_running():
            start=time.monotonic()
            if requests['delta'] or requests['restart']:
                index=(index+requests['delta'])%len(motions);sim.reset(noise=0);tick=0;settle=100
                requests['delta']=0;requests['restart']=False
            if not requests['pause']:
                if settle>0:
                    sim.step(np.zeros(12,np.float32));settle-=1
                    if settle==0:
                        move=motions[index];sim.start(move['command'],move['gait_enabled']);print('MOTION',move,flush=True)
                else:
                    sim.step();tick+=1
                    if sim.fallen():requests['pause']=True;print('FALL: paused; press R to restart',flush=True)
                    if a.sweep and tick>=round(a.seconds/float(sim.policy_dt)):requests['delta']=1
            if a.follow:viewer.cam.lookat[:2]+=sim.d.qpos[:2]-previous_xy
            previous_xy=sim.d.qpos[:2].copy()
            move=motions[index]
            if hasattr(viewer,'set_texts'):
                viewer.set_texts((None,None,'URDF4 AMP+CTS 17800 | live student\n'+move['name']+' '+str(move['command'])+'\nN/B change | Space pause | R restart',
                    ('SETTLING' if settle else 'PAUSED' if requests['pause'] else 'RUNNING')))
            viewer.sync();time.sleep(max(0,float(sim.policy_dt)-(time.monotonic()-start)))

if __name__=='__main__':main()
