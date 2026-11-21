"""Export only independently confirmed new-mass physical states, keeping V2 intact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from design_v2 import NAMES, LEGS
from complete_v3 import BASE, ROOT, assess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(output, confirmed, only=None):
    output.mkdir(parents=True,exist_ok=False)
    accepted=[]
    for file in sorted(confirmed.glob('*.json')):
        r=json.loads(file.read_text())
        if not isinstance(r,dict) or not r.get('passed') or (only and r['clip'] not in only):continue
        assert assess(r)['passed'] and r['seconds']>=60
        source=confirmed/r['clip']/'physical.npz'
        with np.load(source,allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        take=(a['time']>=5.)&(a['time']<55.)
        f=a['features'][take];ct=a['contact'][take];time=a['time'][take];q=a['qpos'][take]
        assert f.shape==(2500,46) and np.isfinite(f).all()
        T=r['design']['period_s'];duty=r['design']['stance_ratio']
        clock=(time[:,None]/T+np.array([0,.5,.5,0]))%1
        match,offset=max((float((ct==((clock+x)%1<duty)).mean()),float(x)) for x in np.linspace(-.35,.35,141))
        valid=np.ones(len(f),bool);valid[-1]=False
        extra={name:a[key][take] for name,key in dict(foot_normal_force_N='foot_normal_force_N',
            raw_torque_Nm='raw',motor_torque_Nm='motor',applied_torque_Nm='applied',
            reference_xy_world_m='reference_xy',reference_yaw_rad='reference_yaw',
            commanded_leg_phase='leg_phase_commanded',limited_target_rad='limited_target',
            response_target_rad='response_target').items()}
        np.savez_compressed(output/(r['clip']+'.npz'),amp_features=f.astype(np.float32),transition_valid=valid,
            command=np.tile(r['command'],(len(f),1)).astype(np.float32),
            joint_pos_rad=f[:,:12],joint_vel_rad_s=f[:,12:24],root_pos_world_m=q[:,:3],
            root_quat_wxyz=q[:,3:7],leg_phase=(clock+offset)%1,controller_leg_phase=clock,
            foot_height_m=a['foot_clearance'][take],contact=ct,time_s=time-time[0],**extra)
        entry=dict(r);entry.update(frames=len(f),cadence_hz=0. if r['family']=='stand' else 1/T,
            stance_ratio_target=duty,phase_alignment_offset_cycles=offset,phase_aligned_contact_match=match,
            physically_executed=True,sha256=sha(output/(r['clip']+'.npz')),
            source_npz_sha256=sha(source),source_folder=str(source.parent),
            measured_command=r['mean_velocity'],amp_training_approved=False)
        accepted.append(entry)
    source=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
    manifest=dict(schema='rs01_style_reference_v1',dataset_version='V3',
        kind='urdf4_new_mass_multispeed_physical_demonstrations',amp_training_approved=False,
        validation_status='60s_MuJoCo_screen_passed; independent_Gym_validation_pending',
        frame_dt_s=.02,joint_order=NAMES,foot_order=LEGS,quaternion_order='wxyz',feature_size=46,
        urdf_sha256=sha(source),total_mass_kg=12.369864831733414,
        battery_model='0.8kg point mass at original trunk CoM; battery dimensions/inertia pending measurement',
        zero_command_behavior='four_foot_stand',
        amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12',
            'base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3','projected_gravity:3','root_height_m:1'],
        settings=dict(robot='urdf4_rs01',policy_dt_s=.02,physics_dt_s=.0025,kp=40,kd=1,delay_s=.04,
            response_tau_s=.0319,response_gain=.928,motor_cap_Nm=14,
            target_rate_limit_rad_s=[2.,2.6,3.2],target_acceleration_limit_rad_s2=[60.,78.,96.],
            source='IK diagonal templates with centroidal joint-target feedforward and route feedback'),
        phase_annotation='Actual contact-aligned common diagonal clock; physical states are never synthesized or mirrored.',
        training_integration='Multi-cadence plus stand requires a dedicated training config; old fixed-cadence task is not changed.',
        clips=accepted)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('EXPORTED',len(accepted),'clips',len(accepted)*2500,'frames',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--confirmed',type=Path,default=BASE/'confirmation60');p.add_argument('--clips',default='')
    a=p.parse_args();export(a.output,a.confirmed,set(a.clips.split(',')) if a.clips else None)
