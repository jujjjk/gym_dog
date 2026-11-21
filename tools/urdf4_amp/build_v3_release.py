"""Assemble the V3 data release from measured, accepted physical rollouts."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from complete_v3 import BASE, ROOT, assess, cases
from export_v3 import export, sha
from transitions_v3 import assess as assess_transition


def gym_evidence():
    good={}
    for index in [1,2,3]:
        run=BASE/('gym_closed%02d'%index);file=run/'v3_windowed_result.json'
        if not file.exists():continue
        receipt=json.loads(file.read_text());manifest=json.loads((BASE/('gym_batch%02d'%index)/'manifest.json').read_text())
        for row,spec in zip(receipt['cases'],manifest['clips']):
            if row['passed']:
                canonical=row['clip'].split('__')[0]
                quality=row['cross_max_m']+row['heading_mean_abs_deg']*.01
                if canonical not in good or quality<good[canonical][0]:good[canonical]=(quality,spec,row,str(file))
    return good


def export_transitions(output):
    out=output/'transitions';out.mkdir();rows=json.loads((BASE/'transition_search/ranking.json').read_text());best={}
    for r in sorted(rows,key=lambda r:r['score']):best.setdefault(r['clip'],r)
    clips=[]
    for name,r in best.items():
        if not r['passed']:continue
        folder=Path(r.get('source_folder',''))
        if not folder.is_dir() or not (folder/'physical.npz').exists():
            # Earlier reports predate the explicit source-folder field.
            for report in (BASE/'transition_search').glob('*/report.json'):
                candidate=json.loads(report.read_text())
                if candidate['clip']==name and candidate['design']==r['design'] and candidate['lead']==r['lead'] and candidate['feed']==r['feed']:
                    folder=report.parent;break
            else:raise ValueError('Missing physical transition source: '+name)
        a=dict(np.load(folder/'physical.npz',allow_pickle=False));r=assess_transition(copy.deepcopy(r),a)
        assert r['passed'],(name,r['rejection_reasons'])
        f=a['features'];q=a['qpos'];valid=np.ones(len(f),bool);valid[-1]=False;phase_valid=np.linalg.norm(a['command'],axis=1)>1e-4;T=r['design']['period_s']
        np.savez_compressed(out/(name+'.npz'),amp_features=f.astype(np.float32),transition_valid=valid,
            command=a['command'].astype(np.float32),joint_pos_rad=f[:,:12],joint_vel_rad_s=f[:,12:24],
            root_pos_world_m=q[:,:3],root_quat_wxyz=q[:,3:7],leg_phase=a['leg_phase_commanded'],
            controller_leg_phase=a['leg_phase_commanded'],phase_valid=phase_valid,cadence_hz=np.where(phase_valid,1/T,0.),
            foot_height_m=a['foot_clearance'],contact=a['contact'],foot_normal_force_N=a['foot_normal_force_N'],
            raw_torque_Nm=a['raw'],motor_torque_Nm=a['motor'],applied_torque_Nm=a['applied'],
            reference_xy_world_m=a['reference_xy'],reference_yaw_rad=a['reference_yaw'],time_s=a['time'])
        destination=output/'transition_commands'/name;destination.mkdir(parents=True);shutil.copy2(folder/'physical.npz',destination/'physical.npz')
        r.update(frames=len(f),cadence_hz=1/T,sha256=sha(out/(name+'.npz')),source_npz_sha256=sha(folder/'physical.npz'),
            source_folder='transition_commands/'+name,physically_executed=True,variable_command=True,
            amp_training_approved=False,phase_valid_field='phase_valid',simulator='MuJoCo 3.17.0')
        clips.append(r)
    parent=json.loads((output/'manifest.json').read_text());manifest={k:parent[k] for k in ['schema','dataset_version','frame_dt_s','joint_order','foot_order','quaternion_order','feature_size','urdf_sha256','total_mass_kg','settings','amp_features_order']}
    manifest.update(kind='V3_variable_command_physical_transitions',amp_training_approved=False,clips=clips,
        validation_status='MuJoCo full physical trajectory and actual stop/steady-window checks passed; Gym scope recorded separately',
        control_semantics='Velocity commands: preserve cross-track; project uncontrolled along-track progress. On stopping, hold the measured stopping position and heading. Anchor events remain in each report.',
        contact_gates=dict(full_illegal_fraction_max=.004,longest_illegal_burst_s_max=.02,flight_fraction_max=0.),
        variable_commands_require_V3_adapter=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return clips


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'amp_reference_v3');a=p.parse_args()
    source=BASE/'release_sources';source.mkdir(exist_ok=False);evidence=gym_evidence();selected={}
    for file in (BASE/'confirmation60').glob('*.json'):
        r=json.loads(file.read_text())
        if isinstance(r,dict) and r.get('passed'):
            assert assess(r)['passed'];selected[r['clip']]=(r, file.parent/r['clip'])
    # Prefer a teacher independently accepted by both simulators at the same command.
    for name,(_,r,row,receipt) in evidence.items():
        selected[name]=(copy.deepcopy(r),Path(r['source_folder']))
    for name,(r,folder) in selected.items():
        original=r['clip'];r['clip']=name;r['original_candidate_id']=original;r['original_source_folder']=str(folder)
        if name in evidence:r['isaac_gym_validation']=dict(receipt=evidence[name][3],case=evidence[name][2],scope='50s expert interval; full 60s fall/torque gates')
        r['isaac_gym_verified']=name in evidence
        (source/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n');os.symlink(folder.resolve(),source/name,target_is_directory=True)
    export(a.output,source)
    m=json.loads((a.output/'manifest.json').read_text())
    for c in m['clips']:
        original=Path(c['source_folder'])/'physical.npz';target=a.output/'commands'/c['clip'];target.mkdir(parents=True);shutil.copy2(original,target/'physical.npz');c['source_folder']='commands/'+c['clip'];c['data_interval_motion_s']=[5.,55.]
    m.update(validation_status='MuJoCo 60s qualified; per-clip Isaac Gym closed-loop results retained',
        amp_training_approved=False,physical_expert_verified=True,requires_new_training_task=True,
        approval_note='Top-level legacy AMP guard remains closed: V3 contains stand, several cadences and a larger action envelope. Use the V3 adapter and the simulator-qualified subset.',
        target_action_space='URDF-bounded joint position targets through the unchanged RS01 actuator; the old actor action envelope is insufficient for some larger strides.',
        transition_dataset='transitions/manifest.json',kinematic_validation_pending=True)
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');transitions=export_transitions(a.output)
    coverage=dict(steady_clips=len(m['clips']),steady_frames=sum(c['frames'] for c in m['clips']),transition_clips=len(transitions),
        transition_frames=sum(c['frames'] for c in transitions),gym_closed_loop_qualified_steady_clips=sum(c['isaac_gym_verified'] for c in m['clips']),
        qualified_commands=[dict(clip=c['clip'],family=c['family'],requested=c['command'],measured=c['mean_velocity'],gym_verified=c['isaac_gym_verified']) for c in m['clips']],
        unmet_requested_bins=[c for c in cases() if c['clip'] not in selected],
        note='Requested bins are design goals, not demonstrated capabilities. Failed candidates are not expert data; 0.6 m/s has not qualified.')
    (a.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    print(json.dumps({k:v for k,v in coverage.items() if k not in ['qualified_commands','unmet_requested_bins']},indent=2))


if __name__=='__main__':main()
