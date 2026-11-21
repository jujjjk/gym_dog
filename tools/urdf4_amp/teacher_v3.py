"""Portable closed-loop V3 teacher. The caller owns the physical simulator.

MuJoCo is used only for this URDF's FK/Jacobians/bias at the measured state.
command() never advances or writes the caller's root/joint state and returns
joint position commands, which must pass through the original RS01 actuator.
"""
from functools import partial
import json
import hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from design_v2 import setup
from design_v3 import design
from physical_v3 import rot, rpy, crossmat


class Teacher:
    def __init__(self, folder, spec):
        self.s=dict(json.loads((Path(folder)/'manifest.json').read_text())['settings'])
        meta=json.loads((Path(folder)/'manifest.json').read_text())
        urdf=Path(__file__).resolve().parents[2]/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!=meta['urdf_sha256']:raise ValueError('Teacher scene has a stale URDF')
        if meta.get('scene_sha256') and hashlib.sha256((Path(folder)/'scene.xml').read_bytes()).hexdigest()!=meta['scene_sha256']:raise ValueError('Teacher scene hash mismatch')
        self.s.update(spec['design']);s=self.s
        self.cmd=np.asarray(spec['command'],float);self.lead=spec['lead'];self.feed=spec['feed']
        self.h=s['body_height_m'];self.T=s['period_s'];self.duty=s['stance_ratio']
        self.m=mj.MjModel.from_xml_path(str(Path(folder)/'scene.xml'));self.d=mj.MjData(self.m)
        self.qa,self.va,self.fb,self.limits=setup(self.m)
        assert abs(self.m.body_mass.sum()-12.369864831733414)<1e-7
        template=partial(design,self.m,T=self.T,duty=self.duty,clear=s['clearance_m'],height=self.h,
            width=s['half_width_m'],xoffset=s.get('foot_x_offset_m',0.),front_clear=s.get('front_clearance_m'),
            rear_clear=s.get('rear_clearance_m'),leg_clearance=s.get('leg_clearance_m'),
            leg_stance_ratio=s.get('leg_stance_ratio'),stride_scale=s.get('stride_scale'))
        self.q=template(cmd=self.cmd)['q']
        self.q0=design(self.m,(0,0,0),self.T,self.duty,0,self.h,s['half_width_m'],
            xoffset=s.get('foot_x_offset_m',0.))['q'][0]
        self.derivatives={}
        for axis,delta,enabled in [(2,.2,s.get('yaw_step_feedback',0)>0 or s.get('heading_step_gain',0)>0),
                (0,.05,s.get('velocity_step_gain',0)>0),(1,.05,s.get('velocity_step_gain',0)>0)]:
            if enabled:
                minus=self.cmd.copy();plus=self.cmd.copy();minus[axis]-=delta;plus[axis]+=delta
                self.derivatives[axis]=(template(cmd=plus)['q']-template(cmd=minus)['q'])/(2*delta)
        self.timeline=s.get('command_schedule');self.scheduled=[];self.march=None
        if self.timeline:
            self.times=np.asarray([x['time_s'] for x in self.timeline])
            for row in self.timeline:
                c=np.asarray(row['command']);self.scheduled.append(template(cmd=c)['q'] if np.linalg.norm(c)>1e-8 else np.tile(self.q0,(len(self.q),1)))
            if s.get('gait_activation_threshold',0)>0:self.march=template(cmd=np.zeros(3))['q']
        self.ff=np.zeros(12);self.yaw_mean=0.;self.yaw_integral=0.;self.heading_integral=0.;self.velocity_mean=np.zeros(2);self.velocity_integral=np.zeros(2)
        self.reference_xy=np.zeros(2);self.reference_yaw=0.;self.J=np.zeros((3,self.m.nv))
        self.planned=np.zeros(3);self.amplitude=0.;self.previous_translation_speed=0.;self.previous_command_size=0.;self.reference_anchor_events=[]

    def command(self,t,qpos,qvel,dt=.02,phase_override=None,planned_override=None,amplitude_override=None):
        s=self.s;m=self.m;d=self.d;T=self.T;h=self.h;cmd=self.cmd;settle=t<0
        d.qpos[:]=qpos;d.qvel[:]=qvel;mj.mj_forward(m,d)
        phase=0. if settle else (t/T%1 if phase_override is None else float(phase_override)%1)
        u=np.clip(t/max(s.get('ramp_s',2.),1e-9),0.,1.);envelope=u*u*(3-2*u)
        if s.get('stop_ramp_s',0)>0:
            z=np.clip((s.get('stop_at_s',s.get('seconds',60.))-t)/s['stop_ramp_s'],0,1);envelope*=z*z*(3-2*z)
        active=self.q;amplitude=envelope
        if self.timeline:
            ia=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-1));ib=min(ia+1,len(self.times)-1)
            blend=np.clip((t-self.times[ia])/max(self.times[ib]-self.times[ia],1e-9),0,1) if ia!=ib else 0.
            blend=blend**2*(3-2*blend)
            ca=np.asarray(self.timeline[ia]['command']);cb=np.asarray(self.timeline[ib]['command'])
            planned=envelope*((1-blend)*ca+blend*cb);active=(1-blend)*self.scheduled[ia]+blend*self.scheduled[ib]
            amplitude=envelope*((1-blend)*float(np.linalg.norm(ca)>1e-8)+blend*float(np.linalg.norm(cb)>1e-8))
        else:
            if np.linalg.norm(cmd)<1e-8:amplitude=0.
            planned=cmd*amplitude
        if planned_override is not None:planned=np.asarray(planned_override,dtype=float)
        if amplitude_override is not None:amplitude=float(np.clip(amplitude_override,0,1));envelope=amplitude
        linear_amplitude=amplitude
        if self.march is not None:
            activation=np.clip(amplitude/s['gait_activation_threshold'],0,1);amplitude=activation**2*(3-2*activation)
        translation_speed=np.linalg.norm(planned[:2])
        if s.get('stop_anchor',False) and translation_speed<=.01 and self.previous_translation_speed>.01:self.reference_xy=d.qpos[:2].copy()
        command_size=np.linalg.norm(planned)
        if s.get('stop_anchor',False) and command_size<=.01 and self.previous_command_size>.01:self.reference_yaw=rpy(d.qpos[3:7])[2];self.heading_integral=0.
        self.previous_translation_speed=translation_speed
        self.previous_command_size=command_size
        R=rot(d.qpos[3:7]);worldv=d.qvel[:3];bodyv=R.T@worldv;omega=d.qvel[3:6];angles=rpy(d.qpos[3:7])
        Rref=rot(np.array([np.cos(self.reference_yaw/2),0.,0.,np.sin(self.reference_yaw/2)]))
        route_velocity=(Rref@np.r_[planned[:2],0.])[:2]
        if s.get('route_progress_projection',False) and translation_speed>.01 and abs(planned[2])<.01:
            tangent=route_velocity/max(np.linalg.norm(route_velocity),1e-9);self.reference_xy+=np.dot(d.qpos[:2]-self.reference_xy,tangent)*tangent
        error_xy=d.qpos[:2]-self.reference_xy
        if np.linalg.norm(planned[:2])>.01:
            tangent=route_velocity/max(np.linalg.norm(route_velocity),1e-9);normal=np.array([-tangent[1],tangent[0]])
            correction=-s.get('path_gain',1.)*np.dot(error_xy,normal)*normal
            if abs(planned[2])>.01:correction-=s.get('along_gain',.5)*np.dot(error_xy,tangent)*tangent
        else:correction=-s.get('path_gain',1.)*error_xy
        desired_world=np.r_[route_velocity+np.clip(correction,-.08,.08),0.]
        force=m.body_mass.sum()*(np.array([0,0,9.81])+np.r_[s.get('velocity_gain',2.)*(desired_world-worldv)[:2],s.get('height_kp',100)*(h-d.qpos[2])-s.get('height_kd',15)*worldv[2]])
        heading=np.arctan2(np.sin(self.reference_yaw-angles[2]),np.cos(self.reference_yaw-angles[2]))
        if s.get('cross_heading_gain',0)>0 and planned[0]>.02 and abs(planned[2])<.01:
            # Couple cross-track recovery to foot-placement yaw without
            # rotating the requested world route or adding a reward term.
            tangent=route_velocity/max(np.linalg.norm(route_velocity),1e-9)
            cross_error=np.dot(error_xy,np.array([-tangent[1],tangent[0]]))
            heading-=np.clip(s['cross_heading_gain']*cross_error,-.06,.06)
        if not settle and amplitude>.1:self.heading_integral=np.clip(self.heading_integral+dt*heading,-.3,.3)
        trim=np.zeros(3)
        if not settle and 2 in self.derivatives:
            yaw_world=(R@omega)[2];self.yaw_mean+=dt/.27*(yaw_world-self.yaw_mean)
            self.yaw_integral=np.clip(self.yaw_integral+dt*(planned[2]-yaw_world),-.3,.3)
            trim[2]=np.clip(s.get('yaw_step_feedback',0)*(planned[2]-self.yaw_mean)+s.get('yaw_step_integral',.5)*self.yaw_integral+s.get('heading_step_gain',0)*heading+s.get('heading_integral_gain',0)*self.heading_integral,-.3,.3)
        if not settle and 0 in self.derivatives:
            self.velocity_mean+=dt/.27*(bodyv[:2]-self.velocity_mean);desired_body=(R.T@desired_world)[:2]
            self.velocity_integral=np.clip(self.velocity_integral+dt*(desired_body-self.velocity_mean),-.2,.2)
            trim[:2]=np.clip(s['velocity_step_gain']*(desired_body-self.velocity_mean)+s.get('velocity_step_integral',0)*self.velocity_integral,-.12,.12)
        moment=R@np.array([-8*angles[0]-.8*omega[0],-8*angles[1]-.8*omega[1],s.get('yaw_velocity_gain',1.)*(planned[2]-omega[2])+s.get('heading_gain',.5)*heading])
        leads=np.array([self.lead+s.get('front_lead_offset_s',0.)]*2+[self.lead+s.get('rear_lead_offset_s',0.)]*2)+np.asarray(s.get('leg_lead_offsets_s',[0.]*4))
        p=(phase+leads/T+np.array([0,.5,.5,0]))%1;duties=np.asarray(s.get('leg_stance_ratio',[self.duty]*4));ramp=s.get('support_ramp_phase',.08)
        w=(1-amplitude)+amplitude*np.clip(np.minimum(p/ramp,(duties-p)/ramp),0,1)
        A=np.zeros((6,12));com=d.subtree_com[1]
        for j,body in enumerate(self.fb):A[:3,j*3:j*3+3]=np.eye(3);A[3:,j*3:j*3+3]=crossmat(d.xpos[body]-com)
        W=np.repeat(w,3);F=(W*(A.T@np.linalg.solve((A*W)@A.T+np.eye(6)*1e-5,np.r_[force,moment]))).reshape(4,3)
        F[:,2]=np.clip(F[:,2],0,100);F[:,:2]=np.clip(F[:,:2],-.6*F[:,2:3],.6*F[:,2:3]);ff=d.qfrc_bias[self.va].copy()
        for j,body in enumerate(self.fb):mj.mj_jacBody(m,d,self.J,None,body);ff-=self.J[:,self.va].T@F[j]
        alpha=s.get('feedforward_alpha',.5);self.ff=(1-alpha)*self.ff+alpha*ff
        desired=self.q0.copy()
        if not settle:
            for j in range(4):
                fidx=((phase+leads[j]/T)%1)*len(self.q);lo=int(fidx);blend=fidx-lo;ix=slice(j*3,j*3+3)
                desired[ix]=(1-blend)*active[lo,ix]+blend*active[(lo+1)%len(self.q),ix]
                if self.march is not None:desired[ix]+=(amplitude-linear_amplitude)*((1-blend)*self.march[lo,ix]+blend*self.march[(lo+1)%len(self.q),ix]-self.q0[ix])
                feedback_scale=amplitude/max(envelope,1e-9) if self.timeline else 1.
                for axis,derivative in self.derivatives.items():desired[ix]+=feedback_scale*trim[axis]*((1-blend)*derivative[lo,ix]+blend*derivative[(lo+1)%len(self.q),ix])
        desired=self.q0+((envelope if self.timeline else amplitude)*(desired-self.q0)+self.feed*np.clip(self.ff,-10,10)/40)/.928
        self.planned=planned;self.amplitude=amplitude;self.leg_phase=p
        # Explicit control reference integration. The physical plant is untouched.
        self.reference_xy+=route_velocity*dt
        self.reference_yaw=np.arctan2(np.sin(self.reference_yaw+planned[2]*dt),np.cos(self.reference_yaw+planned[2]*dt))
        return np.clip(desired,self.limits[:,0]+.01,self.limits[:,1]-.01)
