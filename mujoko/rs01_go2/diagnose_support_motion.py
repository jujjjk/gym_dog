"""A6850 read-only dynamics probes: bounce, stance width, odometry error.

Truth contacts/velocities are diagnostics ONLY, never actor observations.
Run from gym root. No reset, no training, no controller parameter changes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from sim2sim_v15 import V15Sim
from sim2sim import quaternion_rotation_matrix, roll_pitch_yaw

LEGS = ('FL','FR','RL','RR')


def probe(scene, policy, command, gait, phase, duration, sim_class=V15Sim):
    sim=sim_class(scene,policy,command); sim.gait_enable=gait; sim.phase_value=phase
    cache={}; estimate=sim.leg_odometry.estimate
    def recorded(*args):
        result=estimate(*args); cache.update(result); return result
    sim.leg_odometry.estimate=recorded
    rows=[]; detailed=mujoco.MjData(sim.model)
    for tick in range(round(duration/sim.policy_dt)):
        sim.control_step()
        # Independent data: do not disturb solver/contacts of the actor rollout.
        detailed.qpos[:]=sim.data.qpos; detailed.qvel[:]=sim.data.qvel
        mujoco.mj_forward(sim.model,detailed)
        rot=quaternion_rotation_matrix(sim.data.qpos[3:7])
        linear,angular=sim.base_velocity_body()
        forces,illegal,_=sim.contact_diagnostics(); forces=forces[[1,0,3,2]]
        contacts=forces>=sim.gait_cfg['contact_threshold_n']
        center_v=[]; surface_v=[]; actual_p=[]
        for leg in LEGS:
            geom=sim.foot_geoms[leg]; body=sim.model.geom_bodyid[geom]
            p=detailed.geom_xpos[geom]
            jp=np.zeros((3,sim.model.nv)); jr=jp.copy()
            mujoco.mj_jac(sim.model,detailed,jp,jr,p,body)
            v=jp@detailed.qvel; w=jr@detailed.qvel
            center_v.append(rot.T@v)
            # Lowest point of spherical foot on a flat ground plane.
            surface_v.append(rot.T@(v+np.cross(w,[0,0,-.016])))
            actual_p.append(rot.T@(p-sim.data.qpos[:3]))
        row=dict(t=(tick+1)*sim.policy_dt,phase=sim.phase,
            q=sim.data.qpos[sim.qpos_indices].copy(),dq=sim.data.qvel[sim.qvel_indices].copy(),
            omega=angular,gravity=rot.T@np.array([0.,0.,-1.]),
            fresh_omega=sim.data.qvel[3:6].copy(),fresh_velocity=rot.T@sim.data.qvel[:3],
            z=sim.data.qpos[2],vz=sim.base_velocity_world()[0][2],
            rpy=np.array(roll_pitch_yaw(sim.data.qpos[3:7])),
            velocity=linear,estimate=cache['base_linear_velocity'].copy(),
            confidence=cache['confidence'],selected=cache['stance_mask'].copy(),
            inferred=cache['velocity_by_foot'].copy(),foot=cache['foot_position'].copy(),
            actual_foot=np.array(actual_p),center_v=np.array(center_v),surface_v=np.array(surface_v),
            contacts=contacts,forces=forces,illegal=illegal,
            raw=sim.last_raw_torque.copy(),motor=sim.last_motor_torque.copy(),
            speed=sim.step_max_speed)
        rows.append(row)
        if sim.step_overspeed or row['z']<.18 or max(abs(row['rpy'][:2]))>.8: break
    return {k:np.array([r[k] for r in rows]) for k in rows[0]},sim


def summarize(d,sim,duration):
    # Exclude initial placement and large-tilt fall trajectory for gait beauty.
    steady=(d['t']>2)&(np.max(np.abs(d['rpy'][:,:2]),axis=1)<.2)
    early=(d['t']>=.2)&(d['t']<1.5)
    out=dict(completed_s=float(d['t'][-1]),requested_s=duration,
             steady_samples=int(steady.sum()), max_speed_rad_s=float(d['speed'].max()))
    if steady.any():
        z=d['z'][steady]; feet=d['foot'][steady]
        width=feet[:,::2,1]-feet[:,1::2,1]
        out.update(height_p95_minus_p5_mm=float(np.diff(np.percentile(z,[5,95]))[0]*1000),
                   vz_rms_m_s=float(np.sqrt(np.mean(d['vz'][steady]**2))),
                   flight_ratio=float(np.mean(~d['contacts'][steady].any(1))),
                   four_support_ratio=float(np.mean(d['contacts'][steady].all(1))),
                   total_force_p95_bodyweights=float(np.percentile(d['forces'][steady].sum(1),95)/(sim.model.body_mass.sum()*9.81)),
                   front_rear_width_p5_mm=(np.percentile(width,5,axis=0)*1000).tolist(),
                   front_rear_width_mean_mm=(width.mean(0)*1000).tolist(),
                   cross_midline_ratio=float(np.mean(feet[:,:,1]*np.array([1,-1,1,-1])<0)),
                   raw_p95_nm=float(np.percentile(np.abs(d['raw'][steady]),95)))
    mask=d['selected'][early]; actual=d['contacts'][early]
    error=d['inferred'][early]-d['velocity'][early,None,:]
    identity=error+d['center_v'][early]
    out.update(early_vy_rmse=float(np.sqrt(np.mean((d['estimate'][early,1]-d['velocity'][early,1])**2))),
               selected_noncontact_ratio=float(np.sum(mask & ~actual)/max(mask.sum(),1)),
               fk_position_max_error_m=float(np.max(np.abs(d['foot']-d['actual_foot']))),
               velocity_identity_rms=float(np.sqrt(np.mean(identity**2))),
               selected_center_vy_rms=float(np.sqrt(np.mean(d['center_v'][early,:,1][mask]**2))) if mask.any() else None,
               selected_surface_vy_rms=float(np.sqrt(np.mean(d['surface_v'][early,:,1][mask]**2))) if mask.any() else None)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--sensor-sync',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scene=Path('artifacts/rs01_v14_actuator_parity/scene.xml')
    policy=Path('artifacts/rs01_v15_support/stand_only_6850.onnx')
    results=[]
    sim_class=V15Sim
    if args.sensor_sync:
        from sim2sim_sensor_sync import SensorSyncSim
        sim_class=SensorSyncSim
    for name,command,gait in [('stand',[0,0,0],0),('march',[0,0,0],1),
                               ('forward02',[.2,0,0],1),('backward02',[-.2,0,0],1)]:
        for phase in [0,.25,.5,.75]:
            d,sim=probe(scene,policy,command,gait,phase,args.duration,sim_class)
            np.savez_compressed(args.output/f'{name}_p{int(100*phase):03}.npz',**d)
            result=dict(case=name,command=command,phase=phase,**summarize(d,sim,args.duration))
            results.append(result);print(json.dumps(result),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
