"""Procedural RS01 style clips, explicit kinematic preview and real-plant replay.

Not video motion capture, not an MPC controller, not a deployed policy.
Reference states are exported separately from physically executed states.
"""
import argparse
import csv
import hashlib
import json
import time
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
if '--physics' in sys.argv:
    import isaacgym  # Load PhysX/USD before MuJoCo/Torch used by the shared bridge.
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
LEGS = ('FL', 'FR', 'RL', 'RR')
NAMES = [leg+'_'+joint+'_joint' for leg in LEGS for joint in ('hip','thigh','calf')]
MOVES = {'march': (0,0,0), 'forward': (.25,0,0), 'backward': (-.18,0,0),
         'left': (0,.12,0), 'right': (0,-.12,0), 'turn_left': (0,0,.45),
         'turn_right': (0,0,-.45), 'forward_left': (.18,.08,0),
         'forward_right': (.18,-.08,0), 'combined': (.15,.06,.3)}


def pose(t, command, height):
    vx,vy,w = command; a = w*t; c,s = np.cos(a),np.sin(a)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    if abs(w)<1e-9: xy = np.array([vx*t,vy*t])
    else: xy = np.array([(vx*s+vy*(c-1))/w, (vx*(1-c)+vy*s)/w])
    return np.r_[xy,height], R


def trajectory(t, command, frequency, duty, height, width, lift):
    """Stance anchors fixed in WORLD; quintic swing XY, C2 vertical bump."""
    nominal = np.array([[.216,width,0],[.216,-width,0],[-.216,width,0],[-.216,-width,0]])
    base,R = pose(t,command,height)
    phase = np.remainder(t*frequency+np.array([0,.5,.5,0]),1.)
    feet=[]
    for i,p in enumerate(phase):
        touchdown = t-p/frequency
        def anchor(when):
            b,r = pose(when+duty/(2*frequency),command,height)
            point=b+r@nominal[i];point[2]=.016
            return point
        start=anchor(touchdown)
        if p<duty: point=start
        else:
            u=(p-duty)/(1-duty); blend=u**3*(10-15*u+6*u*u)
            point=start+blend*(anchor(touchdown+1/frequency)-start)
            point[2]+=(64*u**3*(1-u)**3)*lift
        feet.append(point)
    return base,R,np.asarray(feet),phase<duty,phase


class Kinematics:
    def __init__(self, scene, contract):
        self.model=mujoco.MjModel.from_xml_path(str(scene));self.data=mujoco.MjData(self.model)
        joints=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in NAMES])
        if np.any(joints<0): raise ValueError('Missing RS01 joints')
        self.qids=self.model.jnt_qposadr[joints];self.vids=self.model.jnt_dofadr[joints]
        self.limits=self.model.jnt_range[joints]
        self.feet=[]
        for leg in LEGS:
            body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,leg+'_calf_joint')
            ids=[g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==body
                 and self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE
                 and abs(self.model.geom_size[g,0]-.016)<1e-6]
            if len(ids)!=1: raise ValueError('Foot sphere mismatch: '+leg)
            self.feet.append(ids[0])
        cfg=json.loads(Path(contract).read_text())
        source=Path(cfg['simulator']['urdf'])
        expected=cfg['simulator'].get('urdf_sha256')
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
        custom=ET.parse(scene).getroot().find('custom')
        scene_hash=next((x.get('data') for x in custom if x.get('name')=='rs01_source_urdf_sha256'),None) if custom is not None else None
        if expected and (actual!=expected or scene_hash!=expected):
            raise ValueError('Scene/contract/actual URDF mismatch; regenerate scene with prepare_model.py')
        self.default=np.array([cfg['default_joint_angles_rad'][cfg['joint_names'].index(n)] for n in NAMES])
        self.data.qpos[:7]=[0,0,.29,1,0,0,0];self.data.qpos[self.qids]=self.default
        self.cfg=cfg

    def solve(self, base, R, targets):
        d,m=self.data,self.model
        d.qpos[:3]=base;quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel());d.qpos[3:7]=quat
        for _ in range(30):
            mujoco.mj_forward(m,d);error=targets-d.geom_xpos[self.feet]
            if np.max(np.linalg.norm(error,axis=1))<1e-6:break
            for leg,g in enumerate(self.feet):
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacGeom(m,d,jp,jr,g)
                ix=slice(3*leg,3*leg+3);J=jp[:,self.vids[ix]]
                step=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(3),error[leg])
                d.qpos[self.qids[ix]]+=np.clip(step,-.15,.15)
            d.qpos[self.qids]=np.clip(d.qpos[self.qids],self.limits[:,0]+1e-5,self.limits[:,1]-1e-5)
        mujoco.mj_forward(m,d)
        residual=float(np.max(np.linalg.norm(targets-d.geom_xpos[self.feet],axis=1)))
        if residual>.001:raise RuntimeError('Unreachable reference: max IK error %.4fm'%residual)
        return d.qpos.copy(),residual


def make_clip(k, command, args):
    if (getattr(args,'front_x_offset',0.) or getattr(args,'rear_x_offset',0.)) and abs(command[2])>1e-9:
        raise ValueError('Offset anchors currently support translation only; yaw needs touchdown-frame anchors')
    # Extra frame permits finite differences; retain N samples, N-1 valid transitions.
    t=np.arange(round(args.seconds*args.fps)+1)/args.fps
    qpos=[];feet=[];contacts=[];phases=[];errors=[]
    directional=abs(command[1])>1e-9 or abs(command[2])>1e-9
    lift=(args.directional_lift if directional else args.lift)/1000
    for ti in t:
        b,R,f,c,p=trajectory(ti,command,args.frequency,args.duty,args.height,args.half_width,lift)
        # Optional teacher geometry; offsets are constant world anchors in straight motion.
        shift=np.zeros((4,3))
        shift[:2,0]=getattr(args,'front_x_offset',0.)
        shift[2:,0]=getattr(args,'rear_x_offset',0.)
        f+=shift@R.T
        rear_lift=getattr(args,'rear_lift',None)
        if rear_lift is not None:
            u=np.clip((p[2:]-args.duty)/(1-args.duty),0,1)
            f[2:,2]+=(rear_lift/1000-lift)*64*u**3*(1-u)**3
        q,e=k.solve(b,R,f);qpos.append(q);feet.append(f);contacts.append(c);phases.append(p);errors.append(e)
    qpos=np.array(qpos);qvel=np.empty((len(t),k.model.nv))
    for i in range(len(t)-1):mujoco.mj_differentiatePos(k.model,qvel[i],1/args.fps,qpos[i],qpos[i+1])
    qvel[-1]=qvel[-2]
    # Central differences for smooth features, quaternion derivative via MuJoCo.
    for i in range(1,len(t)-1):mujoco.mj_differentiatePos(k.model,qvel[i],2/args.fps,qpos[i-1],qpos[i+1])
    feet=np.array(feet);body_feet=[];base_vel=[]
    for i,ti in enumerate(t):
        b,R=pose(ti,command,args.height);body_feet.append((feet[i]-b)@R);base_vel.append(R.T@qvel[i,:3])
    q=qpos[:,k.qids];dq=qvel[:,k.vids]
    gravity=np.tile([0.,0.,-1.],(len(t),1))  # This reference has yaw only.
    features=np.concatenate((q,dq,np.array(body_feet).reshape(len(t),12),np.array(base_vel),qvel[:,3:6],gravity,qpos[:,2:3]),axis=1)
    assert features.shape[1]==46
    data=dict(time_s=t[:-1],qpos=qpos[:-1],qvel=qvel[:-1],joint_pos_rad=q[:-1],joint_vel_rad_s=dq[:-1],
              root_pos_world_m=qpos[:-1,:3],root_quat_wxyz=qpos[:-1,3:7],
              base_lin_vel_body_m_s=np.array(base_vel)[:-1],base_ang_vel_body_rad_s=qvel[:-1,3:6],
              foot_pos_world_m=feet[:-1],foot_pos_body_m=np.array(body_feet)[:-1],
              desired_contact=np.array(contacts)[:-1],leg_phase=np.array(phases)[:-1],
              projected_gravity=gravity[:-1],command=np.tile(command,(len(t)-1,1)),amp_features=features[:-1],
              transition_valid=np.r_[np.ones(len(t)-2,dtype=bool),False],ik_error_m=np.array(errors)[:-1])
    if not all(np.isfinite(v).all() for v in data.values()):raise RuntimeError('Nonfinite reference')
    return data


def export_clip(out, name, data, k, args):
    np.savez_compressed(out/(name+'.npz'),**data)
    header=['time_s']+[n+'_rad' for n in NAMES]+[n+'_rad_s' for n in NAMES]
    header += ['root_'+s for s in ('x_m','y_m','z_m','qw','qx','qy','qz')]
    header += [leg+'_'+a+'_world_m' for leg in LEGS for a in 'xyz']+[leg+'_desired_contact' for leg in LEGS]
    matrix=np.column_stack((data['time_s'],data['joint_pos_rad'],data['joint_vel_rad_s'],
        data['root_pos_world_m'],data['root_quat_wxyz'],data['foot_pos_world_m'].reshape(-1,12),data['desired_contact']))
    with (out/(name+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(matrix)
    return dict(clip=name,frames=len(matrix),ik_error_max_mm=float(data['ik_error_m'].max()*1000),
                joint_speed_peak_rad_s=float(abs(data['joint_vel_rad_s']).max()),
                reference_rate_to_target_limit_max=float(np.max(abs(data['joint_vel_rad_s'])/np.array(k.cfg['control']['target_rate_limit_rad_s']))),
                clearance_peak_mm=float((data['foot_pos_world_m'][:,:,2]-.016).max()*1000),
                min_signed_foot_y_body_m=float((data['foot_pos_body_m'][:,:,1]*[1,-1,1,-1]).min()),
                reference_flight_frames=int((data['desired_contact'].sum(1)==0).sum()))


def view(k, data, fps):
    import mujoco.viewer
    print('KINEMATIC PREVIEW: prescribed base/joints; NOT physical walking.',flush=True)
    with mujoco.viewer.launch_passive(k.model,k.data) as viewer:
        viewer.cam.distance=1.6;viewer.cam.elevation=-18;viewer.cam.azimuth=130
        while viewer.is_running():
            for i in range(len(data['time_s'])):
                if not viewer.is_running():return
                start=time.monotonic();k.data.qpos[:]=data['qpos'][i];k.data.qvel[:]=data['qvel'][i]
                mujoco.mj_forward(k.model,k.data);viewer.cam.lookat[:]=k.data.qpos[:3];viewer.sync()
                time.sleep(max(0,1/fps-(time.monotonic()-start)))


def physics_replay(k, data, args, out):
    # Existing V22 bridge provides measured delay/FOPDT, rate/accel limiting,
    # sensor-snapshot torque guard and physics-rate PD. No policy inference.
    from sim2sim_v22 import V22Sim
    from sim2sim import limited_target_step, roll_pitch_yaw
    from sim2sim_v20 import project_guard, update_guard
    sim=V22Sim(args.scene,args.policy,data['command'][0])
    if NAMES!=sim.names:raise ValueError('Joint order mismatch')
    if abs(sim.policy_dt-1/args.fps)>1e-9:raise ValueError('Replay fps must match RS01 policy dt')
    sim.data.qpos[:]=data['qpos'][0];sim.data.qvel[:]=0
    q=data['joint_pos_rad'][0].copy()
    sim.limited_target=q.copy();sim.response_target=q.copy();sim.guard_target=q.copy();sim.target_history[:]=q
    mujoco.mj_forward(sim.model,sim.data)
    import torch
    sim.sensor.reset(torch.tensor([0]),0,*sim.sensor_truth());sim.sensor.read(0)
    requested=q.copy()
    def update():
        sim.limited_target,sim.target_rate=limited_target_step(requested,sim.limited_target,sim.target_rate,
            sim.rate_limit,sim.acceleration_limit,sim.policy_dt,reached_atol=1e-12)
        sim.guard_target,sim.guard_raw_pd,sim.guard_safe_pd=project_guard(sim.limited_target,
            sim.sensor.q[0].numpy(),sim.sensor.dq[0].numpy(),sim.kp,sim.kd,sim.guard_limit,sim.lower,sim.upper)
        sim.guard_rms_sq,sim.guard_limit=update_guard(sim.guard_rms_sq,
            np.maximum(abs(sim.guard_safe_pd),abs(sim.sensor.torque[0].numpy())),sim.policy_dt,sim.cfg['v20']['guard'])
    sim.update_policy=update
    rows=[];reason='completed';raw=[];applied=[];motor=[];contact=[];actual_feet=[];actual_features=[]
    requested_log=[];guard_log=[];limit_log=[];speed=[];rolls=[];illegal=[]
    for i in range(len(data['time_s'])):
        requested=data['joint_pos_rad'][i]
        sim.control_step()
        mujoco.mj_forward(sim.model,sim.data)
        rows.append(np.r_[sim.data.time,sim.data.qpos,sim.data.qvel])
        raw.append(sim.last_raw_torque.copy());applied.append(sim.last_applied_torque.copy());motor.append(sim.last_motor_torque.copy())
        forces=sim.contact_forces();contact.append(forces[[1,0,3,2]]>1)  # bridge FR,FL,RR,RL
        rpy=roll_pitch_yaw(sim.data.qpos[3:7])
        R=np.empty(9);mujoco.mju_quat2Mat(R,sim.data.qpos[3:7]);R=R.reshape(3,3)
        f=sim.data.geom_xpos[[sim.foot_geoms[leg] for leg in LEGS]].copy();actual_feet.append(f)
        v,w=sim.base_velocity_body()
        actual_features.append(np.r_[sim.data.qpos[sim.qpos_indices],sim.data.qvel[sim.qvel_indices],
            ((f-sim.data.qpos[:3])@R).ravel(),v,w,R.T@np.array([0.,0.,-1.]),sim.data.qpos[2]])
        requested_log.append(requested.copy());guard_log.append(sim.guard_target.copy());limit_log.append(sim.guard_limit.copy())
        speed.append(sim.step_max_speed);rolls.append(rpy[:2]);illegal.append(sim.contact_diagnostics()[1])
        if not np.isfinite(rows[-1]).all():reason='nonfinite';break
        if sim.step_overspeed:reason='speed_domain';break
        if sim.data.qpos[2]<.16 or max(abs(rpy[0]),abs(rpy[1]))>.6:reason='unstable';break
    np.savez_compressed(out/'physical_replay.npz',state=np.array(rows),raw_torque_nm=raw,
                        motor_torque_nm=motor,applied_torque_nm=applied,contact=contact,
                        foot_pos_world_m=actual_feet,amp_features=actual_features,
                        command=data['command'][:len(rows)],requested_joint_target_rad=requested_log,
                        guard_joint_target_rad=guard_log,active_torque_limit_nm=limit_log,
                        transition_valid=np.r_[np.ones(len(rows)-1,dtype=bool),False])
    report=dict(kind='dynamic_open_loop_RS01_replay',stop_reason=reason,frames=len(rows),
                duration_s=len(rows)*sim.policy_dt,raw_sampled_peak_nm=float(np.max(np.abs(raw))),
                flight_ratio=float(np.mean(np.sum(contact,axis=1)==0)),
                displacement_world_m=(np.array(rows)[-1,1:4]-data['root_pos_world_m'][0]).tolist(),
                reference_displacement_world_m=(data['root_pos_world_m'][len(rows)-1]-data['root_pos_world_m'][0]).tolist(),
                roll_pitch_max_deg=(np.max(np.abs(rolls),axis=0)*180/np.pi).tolist(),
                joint_speed_peak_rad_s=float(max(speed)),illegal_contact_samples=int(np.sum(np.array(illegal)>0)),
                joint_tracking_rmse_rad=float(np.sqrt(np.mean((np.array(rows)[:,1+k.qids]-np.array(requested_log))**2))),
                amp_training_approved=False,
                note='No base fixing or stabilizing external force. Replay is diagnostic, not an MPC teacher. Torque samples at policy boundaries only.')
    (out/'physical_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml')
    p.add_argument('--contract',type=Path,default=ROOT/'artifacts/rs01_v22_sim2sim/B23500.json')
    p.add_argument('--policy',type=Path,default=ROOT/'artifacts/rs01_v22_sim2sim/B23500.onnx')
    p.add_argument('--motion',choices=['all']+list(MOVES),default='all')
    p.add_argument('--seconds',type=float,default=4.);p.add_argument('--fps',type=int,default=50)
    p.add_argument('--frequency',type=float,default=2.);p.add_argument('--duty',type=float,default=.65)
    p.add_argument('--lift',type=float,default=60.,help='Straight/march clearance in mm')
    p.add_argument('--directional-lift',type=float,default=50.,help='Lateral/turn clearance in mm')
    p.add_argument('--height',type=float,default=.29);p.add_argument('--half-width',type=float,default=.18)
    p.add_argument('--output',type=Path);p.add_argument('--viewer',action='store_true');p.add_argument('--physics',action='store_true')
    args=p.parse_args()
    if not (args.seconds>=1 and args.fps>=20 and 0<args.frequency<=4 and .5<args.duty<.9
            and 0<args.lift<=100 and 0<args.directional_lift<=100):p.error('Invalid gait parameters')
    if (args.viewer or args.physics) and args.motion=='all':p.error('Choose one motion for viewer or physics')
    out=args.output or ROOT/'artifacts/rs01_reference'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    k=Kinematics(args.scene,args.contract);clips=[]
    for name,command in MOVES.items():
        if args.motion not in ('all',name):continue
        k.data.qpos[k.qids]=k.default
        data=make_clip(k,command,args);clips.append(export_clip(out,name,data,k,args))
    manifest=dict(schema='rs01_style_reference_v1',kind='kinematic_design',amp_training_approved=False,
        note='Designed style example, NOT reconstructed video or dynamically validated expert. Do not concatenate clip boundaries.',
        frame_dt_s=1/args.fps,joint_order=NAMES,foot_order=LEGS,quaternion_order='wxyz',
        amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12',
                            'base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3','projected_gravity:3','root_height_m:1'],
        intended_use='AMP style-reference prototype; custom 46D schema, not a drop-in 61D actor observation or stock AMP loader',
        settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),clips=clips)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(clips,indent=2));print('Output:',out)
    if args.physics:physics_replay(k,data,args,out)
    if args.viewer:view(k,data,args.fps)


if __name__=='__main__':main()
