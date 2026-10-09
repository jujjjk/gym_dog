"""Independent RS01 reference prototype. No learned policy or ONNX is loaded.

Kinematic templates and actual nonideal-plant recordings are separate. A failed
physics recording is NEVER approved for AMP. The controller is PD/feedforward,
not MPC and not a deployment controller.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import isaacgym  # before shared Torch sensor model
import numpy as np
import mujoco
import torch
from reference_gait import Kinematics, make_clip, export_clip, ROOT, NAMES, LEGS
from sim2sim_v22 import V22Sim
from sim2sim import METADATA_KEY, limited_target_step, roll_pitch_yaw
from sim2sim_v20 import project_guard, update_guard

MOVES = dict(march=(0,0,0), forward=(.10,0,0), backward=(-.08,0,0),
             left=(0,.06,0), right=(0,-.06,0), turn_left=(0,0,.20),
             turn_right=(0,0,-.20), forward_left=(.08,.04,0),
             forward_right=(.08,-.04,0), combined=(.06,.03,.15))


def accepts_diagonal_load(force, outgoing, mass):
    """Both incoming feet must contact AND collectively accept the load."""
    incoming=~np.asarray(outgoing,dtype=bool)
    return bool(np.all(np.asarray(force)[incoming]>5.)
                and np.asarray(force)[incoming].sum()>.65*mass*9.81)


class ContractOnly:
    def __init__(self, path): self.text=Path(path).read_text()
    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={METADATA_KEY:self.text})
    def run(self, *args, **kwargs):
        raise RuntimeError('Fresh reference must never invoke a learned policy')


class ReferencePlant(V22Sim):
    def make_policy_session(self, path): return ContractOnly(path)

    def update_policy(self):
        self.limited_target,self.target_rate=limited_target_step(
            self.requested,self.limited_target,self.target_rate,self.rate_limit,
            self.acceleration_limit,self.policy_dt,reached_atol=1e-12)
        self.guard_target,self.guard_raw_pd,self.guard_safe_pd=project_guard(
            self.limited_target,self.sensor.q[0].numpy(),self.sensor.dq[0].numpy(),
            self.kp,self.kd,self.guard_limit,self.lower,self.upper)
        self.guard_rms_sq,self.guard_limit=update_guard(self.guard_rms_sq,
            np.maximum(abs(self.guard_safe_pd),abs(self.sensor.torque[0].numpy())),
            self.policy_dt,self.cfg['v20']['guard'])

    def physics_step(self):
        super().physics_step()
        self.substeps.append((self.last_raw_torque.copy(),self.last_motor_torque.copy(),
                              self.last_applied_torque.copy(),self.guard_limit.copy()))


def rate_audit(data, cfg, dt):
    # Include one-sided finite differences: central differences can hide peaks.
    q=data['joint_pos_rad']; v=np.diff(q,axis=0)/dt; a=np.diff(v,axis=0)/dt
    rate=np.array(cfg['control']['target_rate_limit_rad_s'])
    accel=np.array(cfg['control']['target_acceleration_limit_rad_s2'])
    return float(np.max(abs(v)/rate)),float(np.max(abs(a)/accel))


def phase_control_residual(phase, coefficients):
    """Smooth front/rear two-harmonic control correction; mirrored hip signs."""
    coefficients=np.asarray(coefficients).reshape(2,3,2)
    result=np.zeros((4,3))
    for leg in range(4):
        angle=2*np.pi*phase[leg]
        result[leg]=coefficients[leg//2]@np.array([np.sin(angle),np.sin(2*angle)])
        if leg%2:result[leg,0]*=-1
    return result.ravel()


def playback(k, path):
    import mujoco.viewer
    files=sorted(path.glob('*.npz')) if path.is_dir() else [path]
    if not files:raise ValueError('No reference clips found')
    with mujoco.viewer.launch_passive(k.model,k.data) as viewer:
        viewer.cam.distance=1.6;viewer.cam.elevation=-18;viewer.cam.azimuth=130
        while viewer.is_running():
            for file in files:
                print('RECORDED PHYSICS' if file.name=='physical.npz' else 'KINEMATIC TEMPLATE (not physical walking)',file,flush=True)
                with np.load(file,allow_pickle=False) as d:
                    for q,dq in zip(d['qpos'],d['qvel']):
                        if not viewer.is_running():return
                        begin=time.monotonic();k.data.qpos[:]=q;k.data.qvel[:]=dq
                        mujoco.mj_forward(k.model,k.data);viewer.cam.lookat[:]=q[:3];viewer.sync()
                        time.sleep(max(0,.02-(time.monotonic()-begin)))


def physical(k, data, args, out):
    sim=ReferencePlant(args.scene,args.contract,data['command'][0])
    assert sim.names==NAMES
    sim.data.qpos[:]=data['qpos'][0];sim.data.qvel[:]=0
    q=data['joint_pos_rad'][0].copy()
    sim.limited_target=q.copy();sim.response_target=q.copy();sim.guard_target=q.copy()
    sim.target_history[:]=q;sim.target_rate[:]=0;sim.requested=q.copy();sim.substeps=[]
    mujoco.mj_forward(sim.model,sim.data)
    sim.sensor.reset(torch.tensor([0]),0,*sim.sensor_truth());sim.sensor.read(0)
    mass=sim.model.body_mass.sum();frames=[];features=[];contacts=[];rpy_log=[]
    targets=[];unplanned=[];feet_log=[];reason='completed';desired_log=[];vel=[];dq=[];contact_force_log=[];clearance_log=[];reference_indices=[]
    filtered_offset=np.zeros(12);limited_log=[];response_log=[]
    # Gravity support expressed through target offset, never an external force.
    def request(index, settling=False):
        nonlocal filtered_offset
        desired=data['joint_pos_rad'][index].copy()
        support=np.ones(4,dtype=bool) if settling else data['desired_contact'][index].copy()
        support_index=index
        if getattr(args,'preview_support',False) and not settling:
            delay=np.mean(sim.cfg['control']['observed_closed_loop_delay_s'])
            support_index=min(len(data['time_s'])-1,index+round(delay/sim.policy_dt))
            support=data['desired_contact'][support_index].copy()
        lead=getattr(args,'rear_lead_s',0.)
        if lead>0 and not settling:
            future=min(len(data['time_s'])-1,index+round(lead/sim.policy_dt))
            desired[6:]=data['joint_pos_rad'][future,6:]
            support[2:]=data['desired_contact'][future,2:]
        if getattr(args,'rear_ground_feedback',False) and not settling:
            R=np.empty(9);mujoco.mju_quat2Mat(R,sim.data.qpos[3:7]);R=R.reshape(3,3)
            base=sim.data.qpos[:3].copy()
            world=data['foot_pos_body_m'][index]@R.T+base
            phase=data['leg_phase'][index]
            duty=getattr(args,'stance_ratio',.65)
            u=np.clip((phase-duty)/(1-duty),0,1)
            blend=np.minimum(np.clip(u/.20,0,1),np.clip((1-u)/.20,0,1))
            blend=blend*blend*(3-2*blend)
            world[2:,2]+=blend[2:]*(data['foot_pos_world_m'][index,2:,2]-world[2:,2])
            corrected,_=k.solve(base,R,world)
            desired[6:]=corrected[k.qids[6:]]
        weights=support.astype(float)
        if getattr(args,'preview_support',False) and not settling:
            phase=data['leg_phase'][support_index]
            u=np.clip(np.minimum(phase,getattr(args,'stance_ratio',.65)-phase)/.10,0,1)
            weights*=u*u*(3-2*u)
        if getattr(args,'contact_weighting',False) and not settling:
            measured=sim.contact_diagnostics()[0][[1,0,3,2]]
            confirmed=weights*np.clip(measured/10.,0,1)
            if confirmed.sum()>.5:weights=confirmed
        unload=getattr(args,'unload_phase',0.)
        if unload>0 and not settling:
            phase=data['leg_phase'][index]
            duty=float(getattr(args,'stance_ratio',.65))
            u=np.clip((duty-phase)/unload,0,1)
            weights*=u*u*(3-2*u)
        forces=np.zeros((4,3));forces[:,2]=mass*9.81*weights/max(1e-6,weights.sum())
        if args.feedback and not settling:
            R=np.empty(9);mujoco.mju_quat2Mat(R,sim.data.qpos[3:7]);R=R.reshape(3,3)
            linear,angular=sim.base_velocity_body()
            cmd=data['command'][index]
            force=R@np.r_[mass*3.*(cmd[:2]-linear[:2]),0.]
            force[2]=mass*(9.81+30*(args.body_height-sim.data.qpos[2])-4*linear[2])
            rpy=roll_pitch_yaw(sim.data.qpos[3:7])
            moment=R@np.array([-12*rpy[0]-2*angular[0],
                getattr(args,'pitch_gain',12.)*(getattr(args,'pitch_target',0.)-rpy[1])-2*angular[1],
                2*(cmd[2]-angular[2])])
            ids=np.flatnonzero(support);blocks=[]
            for leg in ids:
                r=sim.data.geom_xpos[sim.foot_geoms[LEGS[leg]]]-sim.data.qpos[:3]
                skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
                blocks.append(np.vstack((np.eye(3),skew)))
            A=np.hstack(blocks);prior=forces[ids].ravel()
            W=np.repeat(weights[ids],3)
            forces[ids]=(prior+(W[:,None]*A.T)@np.linalg.solve((A*W)@A.T+.005*np.eye(6),np.r_[force,moment]-A@prior)).reshape(-1,3)
            forces[:,2]=np.clip(forces[:,2],0,mass*9.81)
            forces[:,:2]=np.clip(forces[:,:2],-.5*forces[:,2:3],.5*forces[:,2:3])
        tau=np.zeros(12)
        for leg in np.flatnonzero(support):
            jp=np.zeros((3,sim.model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacGeom(sim.model,sim.data,jp,jr,sim.foot_geoms[LEGS[leg]])
            tau-=jp[:,sim.qvel_indices].T@forces[leg]
        offset=args.support_feedforward*tau/sim.kp
        smoothing=getattr(args,'support_smoothing',0.)
        if smoothing>0:
            alpha=1-np.exp(-sim.policy_dt/smoothing)
            filtered_offset+=alpha*(offset-filtered_offset)
            offset=filtered_offset
        compensation=getattr(args,'actuator_compensation',0.)
        if compensation>0 and not settling:
            delay=np.asarray(sim.cfg['control']['observed_closed_loop_delay_s'])
            times=data['time_s'];now=times[index]
            future=np.array([np.interp(now+compensation*delay[j],times,data['joint_pos_rad'][:,j]) for j in range(12)])
            velocity=np.array([np.interp(now+compensation*delay[j],times,data['joint_vel_rad_s'][:,j]) for j in range(12)])
            desired=future+compensation*(sim.time_constant+sim.kd/sim.kp)*velocity
        ground_gain=getattr(args,'swing_ground_gain',0.)
        if ground_gain>0 and not settling:
            R=np.empty(9);mujoco.mju_quat2Mat(R,sim.data.qpos[3:7]);R=R.reshape(3,3)
            base=sim.data.qpos[:3].copy()
            world=data['foot_pos_body_m'][index]@R.T+base
            duty=getattr(args,'stance_ratio',.65)
            u=np.clip((data['leg_phase'][index]-duty)/(1-duty),0,1)
            blend=np.minimum(np.clip(u/.20,0,1),np.clip((1-u)/.20,0,1))
            blend=blend*blend*(3-2*blend)
            world[:,2]+=blend*(data['foot_pos_world_m'][index,:,2]-world[:,2])
            corrected,_=k.solve(base,R,world)
            desired+=ground_gain*np.clip(corrected[k.qids]-data['joint_pos_rad'][index],-.15,.15)
        desired += offset
        if compensation>0:
            desired=sim.default+(desired-sim.default)/(1+compensation*(sim.response_gain-1))
        coefficients=getattr(args,'control_harmonics',None)
        if coefficients is not None and not settling:
            desired+=phase_control_residual(data['leg_phase'][index],coefficients)
        desired=np.clip(desired,sim.lower,sim.upper)
        sim.unplanned_target=desired.copy()
        return desired
    for _ in range(round(args.settle_seconds/sim.policy_dt)):
        sim.requested=request(0,True);sim.control_step();mujoco.mj_forward(sim.model,sim.data)
        rpy=roll_pitch_yaw(sim.data.qpos[3:7])
        if sim.step_overspeed or sim.data.qpos[2]<.16 or max(abs(np.array(rpy[:2])))>.6:
            reason='settle_failed';break
    sim.substeps=[]
    reference_index=0;handoff_wait=0.;handoff_wait_total=0.
    for step in range(len(data['time_s'])) if reason=='completed' else []:
        i=reference_index;reference_indices.append(i)
        sim.requested=request(i);targets.append(sim.requested.copy())
        unplanned.append(sim.unplanned_target.copy())
        sim.control_step();mujoco.mj_forward(sim.model,sim.data)
        limited_log.append(sim.limited_target.copy());response_log.append(sim.response_target.copy())
        frames.append(sim.data.qpos.copy());dq.append(sim.data.qvel.copy())
        R=np.empty(9);mujoco.mju_quat2Mat(R,sim.data.qpos[3:7]);R=R.reshape(3,3)
        feet=sim.data.geom_xpos[[sim.foot_geoms[l] for l in LEGS]].copy()
        clearance_log.append(feet[:,2]-.016)
        feet_log.append((feet-sim.data.qpos[:3])@R)
        linear,angular=sim.base_velocity_body();vel.append(np.r_[linear[:2],angular[2]])
        features.append(np.r_[sim.data.qpos[sim.qpos_indices],sim.data.qvel[sim.qvel_indices],
                            feet_log[-1].ravel(),linear,angular,R.T@np.array([0.,0.,-1.]),sim.data.qpos[2]])
        force,illegal,_=sim.contact_diagnostics()
        contact_force_log.append(force[[1,0,3,2]].copy())
        contacts.append(force[[1,0,3,2]]>=sim.gait_cfg['contact_threshold_n'])
        desired_log.append(data['desired_contact'][i]);rpy_log.append(roll_pitch_yaw(sim.data.qpos[3:7])[:2])
        if not np.isfinite(features[-1]).all():reason='nonfinite'
        elif sim.step_overspeed:reason='overspeed'
        elif sim.data.qpos[2]<.16 or max(abs(np.array(rpy_log[-1])))>.6:reason='unstable'
        elif illegal:reason='illegal_contact'
        advance=True
        if getattr(args,'contact_handoff',False) and i+1<len(data['time_s']):
            outgoing=data['desired_contact'][i]&~data['desired_contact'][i+1]
            if outgoing.any():
                measured=contact_force_log[-1]
                accepted=accepts_diagonal_load(measured,outgoing,mass)
                if not accepted:
                    advance=False;handoff_wait+=sim.policy_dt;handoff_wait_total+=sim.policy_dt
                    if handoff_wait>.3:reason='handoff_timeout'
            if advance:handoff_wait=0.
        if advance:reference_index=min(reference_index+1,len(data['time_s'])-1)
        if reason!='completed':break
    n=len(frames);report=dict(stop_reason=reason,frames=n,amp_training_approved=False,
                            control='joint PD + support wrench feedback through unchanged RS01 plant',
                            old_policy_loaded=False,external_forces=False,handoff_wait_total_s=handoff_wait_total)
    if n:
        c=np.array(contacts);f=np.array(feet_log);s=np.array(sim.substeps);r=np.array(rpy_log)
        report.update(duration_s=n*sim.policy_dt,mean_vx_vy_wz=np.mean(vel,axis=0).tolist(),
            command=data['command'][0].tolist(),velocity_rmse=np.sqrt(np.mean((np.array(vel)-data['command'][:n])**2,axis=0)).tolist(),
            roll_pitch_max_deg=(np.max(abs(r),axis=0)*180/np.pi).tolist(),
            flight_ratio=float(np.mean(c.sum(1)==0)),
            contact_mismatch_ratio=float(np.mean(c!=desired_log)),
            inward_support_ratio=float(np.sum((f[:,:,1]*[1,-1,1,-1]<.12)&c)/max(1,c.sum())),
            raw_peak_nm=float(np.max(abs(s[:,0]))),motor_peak_nm=float(np.max(abs(s[:,1]))),
            applied_peak_nm=float(np.max(abs(s[:,2]))),
            active_over_limit_ratio=float(np.mean(abs(s[:,0])>s[:,3]+1e-6)))
        swing=~np.array(desired_log,dtype=bool)
        report['foot_order']=list(LEGS)
        report['clearance_p95_mm']=(np.quantile(clearance_log,.95,axis=0)*1000).tolist()
        report['planned_swing_contact_ratio']=((c&swing).sum(0)/np.maximum(1,swing.sum(0))).tolist()
        diagonal=(c[:,0]&c[:,3]&~c[:,1]&~c[:,2])|(c[:,1]&c[:,2]&~c[:,0]&~c[:,3])
        report['illegal_two_foot_ratio']=float(np.mean((c.sum(1)==2)&~diagonal))
        report['single_support_ratio']=float(np.mean(c.sum(1)==1))
        report['target_limited_fraction']=float(np.mean(np.abs(np.asarray(targets)-limited_log)>.001))
        report['target_limiter_error_p95_rad']=float(np.quantile(np.abs(np.asarray(targets)-limited_log),.95))
        for label,values in [('final',targets),('unplanned',unplanned)]:
            if n>2:
                velocity=np.diff(values,axis=0)/sim.policy_dt
                report[label+'_target_rate_ratio']=float(np.max(abs(velocity)/sim.rate_limit))
                report[label+'_target_acceleration_ratio']=float(np.max(abs(np.diff(velocity,axis=0)/sim.policy_dt)/sim.acceleration_limit))
        # A diagnostic run alone does not prove multi-seed/transition/thermal safety.
        report['diagnostic_gates_passed']=bool(reason=='completed' and report['flight_ratio']==0
            and report['inward_support_ratio']<.01 and report['contact_mismatch_ratio']<.10
            and max(report['velocity_rmse'][:2])<.03 and report['velocity_rmse'][2]<.10
            and max(report['roll_pitch_max_deg'])<10 and report['active_over_limit_ratio']<.05
            and report['illegal_two_foot_ratio']==0 and report['single_support_ratio']==0
            and min(report['clearance_p95_mm'][2:])>=20
            and max(report['planned_swing_contact_ratio'][2:])<.15)
        np.savez_compressed(out/'physical.npz',qpos=frames,qvel=dq,amp_features=features,
            command=data['command'][:n],contact=c,desired_contact=desired_log,contact_force_n=contact_force_log,
            foot_clearance_m=clearance_log,
            requested_joint_target_rad=targets,unplanned_joint_target_rad=unplanned,substep_torques_and_limits=s,
            limited_joint_target_rad=limited_log,response_joint_target_rad=response_log,
            reference_index=reference_indices,
            transition_valid=np.r_[np.ones(n-1,dtype=bool),False],time_s=np.arange(n)*sim.policy_dt)
    (out/'physical_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--motion',choices=['all']+list(MOVES),default='all')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--forward-speed',type=float,help='Explicit physical command m/s; forward only, no automatic time scaling')
    p.add_argument('--frequency',type=float,default=1.25,help='Reference gait frequency Hz')
    p.add_argument('--stance-ratio',type=float,default=.65)
    p.add_argument('--front-x-offset',type=float,default=0.)
    p.add_argument('--rear-x-offset',type=float,default=0.)
    p.add_argument('--half-width',type=float,default=.175)
    p.add_argument('--front-lift',type=float,default=30.,help='Front clearance template mm')
    p.add_argument('--control-harmonics',type=float,nargs=12,help='Teacher-only front/rear x joint x two sine harmonics, rad')
    p.add_argument('--physics',action='store_true')
    p.add_argument('--view',type=Path,help='Saved .npz; displayed as recorded/kinematic, not live physics')
    p.add_argument('--scene',type=Path,default=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml')
    p.add_argument('--contract',type=Path,default=ROOT/'artifacts/rs01_v22_sim2sim/B23500.json')
    p.add_argument('--support-feedforward',type=float,default=1.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--feedback',action='store_true',help='Privileged-state support wrench feedback (teacher only)')
    p.add_argument('--body-height',type=float,default=.29)
    p.add_argument('--rear-lift',type=float,default=30.,help='Rear template clearance mm; front stays 30 mm')
    p.add_argument('--unload-phase',type=float,default=0.,help='Smooth diagonal unloading before liftoff, fraction of cycle')
    p.add_argument('--fixed-scale',type=float,help='Diagnostic fixed speed/clock; report rate excess, never change plant limits')
    p.add_argument('--rear-ground-feedback',action='store_true',help='Privileged rear swing IK accounts for actual base height/tilt')
    p.add_argument('--rear-lead-s',type=float,default=0.,help='Diagnostic rear target/support preview, seconds')
    p.add_argument('--pitch-gain',type=float,default=12.)
    p.add_argument('--pitch-target',type=float,default=0.,help='Diagnostic body pitch target rad')
    p.add_argument('--actuator-compensation',type=float,default=0.,help='0..1 identified delay/gain/response command compensation')
    p.add_argument('--support-smoothing',type=float,default=0.,help='Support-offset smoothing time constant seconds')
    p.add_argument('--contact-weighting',action='store_true',help='Weight planned support by measured contact force')
    p.add_argument('--preview-support',action='store_true',help='Smooth phase load envelope aligned with identified delay')
    p.add_argument('--contact-handoff',action='store_true',help='Hold lift-off boundary until opposite diagonal accepts measured load; fail after 300ms')
    p.add_argument('--sensor-seed',type=int,default=20260925)
    p.add_argument('--sensor-profile',choices=['normal','robust'],default='normal')
    p.add_argument('--swing-ground-gain',type=float,default=0.,help='0..1 all-leg swing clearance feedback, teacher truth only')
    args=p.parse_args();k=Kinematics(args.scene,args.contract)
    ReferencePlant.seed=args.sensor_seed;ReferencePlant.sensor_profile=args.sensor_profile
    if args.view:
        playback(k,args.view);return
    if args.output is None:p.error('--output required')
    if args.seconds<2:p.error('Require >=2 seconds')
    if args.control_harmonics is not None and (not np.isfinite(args.control_harmonics).all()
            or np.max(np.abs(args.control_harmonics))>.08):p.error('Invalid control harmonics')
    if not (abs(args.front_x_offset)<=.05 and abs(args.rear_x_offset)<=.05
            and .145<=args.half_width<=.19 and 20<=args.front_lift<=45):p.error('Invalid teacher geometry')
    if (args.front_x_offset or args.rear_x_offset) and args.motion!='forward':p.error('Anchor-offset prototype is forward only')
    if not np.isfinite(args.seconds) or not .5<=args.frequency<=3 or not .55<=args.stance_ratio<=.75:p.error('Invalid duration/frequency/stance ratio')
    if args.forward_speed is not None:
        if not 0<args.forward_speed<=.5 or args.motion!='forward' or args.fixed_scale is not None:
            p.error('--forward-speed requires --motion forward, speed (0,.5], and no --fixed-scale')
    if not 0<=args.swing_ground_gain<=1:p.error('Invalid swing ground gain')
    if not 0<=args.actuator_compensation<=1 or not 0<=args.support_smoothing<=.3:p.error('Invalid compensation/smoothing')
    if args.actuator_compensation and (args.rear_lead_s or args.rear_ground_feedback):p.error('Do not combine incompatible preview controllers')
    if not 0<=args.unload_phase<=.14 or not 10<=args.rear_lift<=70:p.error('Invalid rear lift/unload range')
    if not 0<=args.rear_lead_s<=.14 or not 0<args.pitch_gain<=100 or abs(args.pitch_target)>.1:p.error('Invalid preview/pitch parameters')
    if args.rear_lead_s and (args.rear_ground_feedback or args.unload_phase):p.error('Test rear preview separately from ground feedback/unloading')
    if args.fixed_scale is not None and not 0<args.fixed_scale<=1:p.error('Invalid fixed scale')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=SimpleNamespace(seconds=args.seconds,fps=50,frequency=args.frequency,duty=args.stance_ratio,
                        height=args.body_height,half_width=args.half_width,lift=args.front_lift,directional_lift=30.,rear_lift=args.rear_lift,
                        front_x_offset=args.front_x_offset,rear_x_offset=args.rear_x_offset)
    # One shared time scale for every command: the reference clock stays coherent.
    scale=args.fixed_scale or 1.
    candidates=MOVES if args.forward_speed is None else {'forward':(args.forward_speed,0.,0.)}
    if args.forward_speed is None and (args.front_x_offset or args.rear_x_offset):
        candidates={'forward':MOVES['forward']}
    for attempt in range(6):
        cfg.frequency=args.frequency*scale;clips={};ratio=0.
        for name,cmd in candidates.items():
            d=make_clip(k,np.array(cmd)*scale,cfg);v,a=rate_audit(d,k.cfg,.02)
            ratio=max(ratio,v,np.sqrt(a));clips[name]=d
        if ratio<=.90 or args.fixed_scale is not None or args.forward_speed is not None:break
        scale*=.88/ratio
    else:raise RuntimeError('Unable to meet kinematic rate/acceleration envelope')
    chosen=MOVES if args.motion=='all' else {args.motion:MOVES[args.motion]}
    manifest=dict(kind='fresh_RS01_kinematic_candidate',amp_training_approved=False,
        reason='Requires accepted physical trajectories, transitions and visual review before AMP',
        old_policy_loaded=False,joint_order=NAMES,frame_dt_s=.02,settings=vars(cfg),
        physical_controller_settings=dict(feedback=args.feedback,support_feedforward=args.support_feedforward,
            settle_seconds=args.settle_seconds,sensor_profile=ReferencePlant.sensor_profile,seed=ReferencePlant.seed,
            unload_phase=args.unload_phase,fixed_scale=args.fixed_scale,rear_ground_feedback=args.rear_ground_feedback,
            rear_lead_s=args.rear_lead_s,pitch_gain=args.pitch_gain,pitch_target=args.pitch_target,
            actuator_compensation=args.actuator_compensation,support_smoothing=args.support_smoothing,
            contact_weighting=args.contact_weighting,preview_support=args.preview_support,contact_handoff=args.contact_handoff,
            swing_ground_gain=args.swing_ground_gain),
        time_scale=scale,contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),clips=[])
    manifest['physical_controller_settings']['control_harmonics']=args.control_harmonics
    for name in chosen:
        d=clips[name];meta=export_clip(args.output,name,d,k,cfg)
        v,a=rate_audit(d,k.cfg,.02);meta.update(command=d['command'][0].tolist(),rate_ratio=v,acceleration_ratio=a)
        if args.physics:
            folder=args.output/(name+'_physics');folder.mkdir()
            meta['physics']=physical(k,d,args,folder)
        manifest['clips'].append(meta)
        print(name,json.dumps(meta),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved:',args.output,flush=True)


if __name__=='__main__':main()
