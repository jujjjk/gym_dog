"""Verify every exported frame against its physical source and this URDF's FK."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from complete_v3 import BASE, ROOT, V2
from design_v2 import setup
from physical_v3 import rot
from review_gym_v3 import starts


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=BASE/'amp_reference_v3');a=p.parse_args();root=a.dataset
    model=mj.MjModel.from_xml_path(str(BASE/'plant/scene.xml'));data=mj.MjData(model);qa,va,fb,limits=setup(model)
    urdf=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf';rows=[]
    for folder in [root,root/'transitions']:
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['urdf_sha256']==digest(urdf)
        for c in manifest['clips']:
            file=folder/(c['clip']+'.npz');assert digest(file)==c['sha256'];z=dict(np.load(file,allow_pickle=False));n=len(z['amp_features']);f=z['amp_features'];assert f.shape==(n,46) and np.isfinite(f).all()
            assert z['command'].shape==(n,3) and z['contact'].shape==(n,4) and np.isfinite(z['command']).all()
            assert len(z['transition_valid'])==n and not z['transition_valid'][-1] and np.all(z['transition_valid'][:-1])
            assert np.max(abs(np.diff(z['time_s'])-.02))<1e-8
            source=root/c['source_folder']/'physical.npz';assert digest(source)==c['source_npz_sha256'];physical=dict(np.load(source,allow_pickle=False))
            take=np.ones(len(physical['time']),bool) if c.get('variable_command') else (physical['time']>=5)&(physical['time']<55)
            assert np.array_equal(f,physical['features'][take].astype(np.float32))
            assert np.array_equal(z['root_pos_world_m'],physical['qpos'][take,:3]) and np.array_equal(z['root_quat_wxyz'],physical['qpos'][take,3:7])
            assert np.array_equal(z['joint_pos_rad'],physical['features'][take,:12]) and np.array_equal(z['joint_vel_rad_s'],physical['features'][take,12:24])
            assert np.max(abs(np.linalg.norm(z['root_quat_wxyz'],axis=1)-1))<1e-8
            violation=float(max(0.,np.max(limits[:,0]-z['joint_pos_rad']),np.max(z['joint_pos_rad']-limits[:,1])))
            assert violation<=.005
            fk=0.;gravity=0.
            for q,quat,features in zip(z['joint_pos_rad'],z['root_quat_wxyz'],f):
                data.qpos[:7]=np.r_[0.,0.,2.,quat];data.qpos[qa]=q;mj.mj_forward(model,data);R=rot(quat)
                fk=max(fk,float(abs(((data.xpos[fb]-data.qpos[:3])@R).reshape(-1)-features[24:36]).max()))
                gravity=max(gravity,float(abs(R.T@np.array([0,0,-1])-features[42:45]).max()))
            assert fk<1e-6 and gravity<1e-6
            raw=float(abs(z['raw_torque_Nm']).max());motor=float(abs(z['motor_torque_Nm']).max());assert raw<=17.+1e-8 and motor<=14.+1e-8
            row=dict(clip=c['clip'],frames=n,variable_command=bool(c.get('variable_command')),source_states_exactly_preserved=True,
                fk_max_error_m=fk,projected_gravity_error=gravity,actual_joint_limit_violation_rad=violation,
                raw_sample_peak_Nm=raw,raw_400Hz_episode_peak_Nm=c['raw_peak_Nm'],motor_peak_Nm=motor,
                motor_joint_rms_Nm=np.sqrt(np.mean(z['motor_torque_Nm']**2,axis=0)).tolist())
            if not c.get('variable_command'):
                ct=z['contact'];count=ct.sum(1);diag=(ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2]);illegal=float(((count==1)|((count==2)&~diag)).mean());assert illegal<=.001 and (count>0).all()
                row['illegal_support_fraction']=illegal;cadences=[];gap=[]
                if c['family']!='stand':
                    for j in range(4):
                        events=starts(ct[:,j]);cadences.extend(1/np.diff(z['time_s'][events]))
                    row['measured_cadence_hz']=float(np.median(cadences));assert abs(row['measured_cadence_hz']-c['cadence_hz'])<=c['cadence_hz']*.05+1e-6
                    for active in [ct,~ct]:
                        values=[]
                        for left,right in [(0,3),(1,2)]:
                            ea=starts(active[:,left]);eb=starts(active[:,right]);assert len(ea) and len(eb)
                            values.extend(float(np.min(abs(z['time_s'][eb]-z['time_s'][e]))) for e in ea)
                        gap.append(float(np.quantile(values,.95)*1000))
                    row['diagonal_event_p95_ms']=gap;assert max(gap)<=40.001
            rows.append(row);print('AUDIT',c['clip'],n,fk,flush=True)
    old=json.loads((V2/'manifest.json').read_text());preserved=all(digest(V2/(c['clip']+'.npz'))==c['sha256'] for c in old['clips'])
    expected='e5e1f0fff2f593b153cd2ac090861833d9319a2e0c1aec1847872aeed7bd7b86'
    old_receipt=dict(clips_unchanged=preserved,manifest_sha256=digest(V2/'manifest.json'),expected_manifest_sha256=expected,manifest_unchanged=digest(V2/'manifest.json')==expected)
    assert preserved and old_receipt['manifest_unchanged']
    result=dict(passed=True,urdf_sha256=digest(urdf),scene_sha256=digest(BASE/'plant/scene.xml'),total_mass_kg=float(model.body_mass.sum()),
        frames=sum(r['frames'] for r in rows),clips=len(rows),every_frame_FK_checked=True,every_exported_state_matches_physical_source=True,
        source_states_edited=False,physics_rate_Hz=400.,data_rate_Hz=50.,cases=rows,old_v2_preservation=old_receipt)
    (root/'dataset_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),result['frames'],flush=True)


if __name__=='__main__':main()
