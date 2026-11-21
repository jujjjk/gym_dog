"""Apply the same full-transition criteria to actual Gym recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from physical_v3 import rpy
from transitions_v3 import assess


def review(run,dataset):
    manifest=json.loads((dataset/'manifest.json').read_text());parent=json.loads((run/'result.json').read_text());z=np.load(run/'live_rollout.npz');rows=[]
    assert parent['envs_per_clip']==1 and parent['root_or_joint_overwrite_during_rollout'] is False
    time=z['time_s']-.02 # command timestamp; the measured state is at the end of its 20ms Gym step.
    for i,(spec,old) in enumerate(zip(manifest['clips'],parent['cases'])):
        take=(time>=-1e-8)&(time<spec['seconds']-1e-8);root=z['root_state_xyzw_world'][take,i];q=np.c_[root[:,:3],root[:,[6,3,4,5]],z['amp_features'][take,i,:12]]
        a=dict(time=time[take],features=z['amp_features'][take,i],qpos=q,contact=z['contact'][take,i],
            foot_clearance=z['feet_pos_world_m'][take,i,:,2]-.02,rpy=np.array([rpy(x) for x in q[:,3:7]]),
            command=z['frame_commands'][take,i],reference_xy=z['reference_xy_world_m'][take,i],reference_yaw=z['reference_yaw_rad'][take,i])
        r=dict(spec);r.update(stop='fallen' if old['fallen_count'] else 'completed',raw_peak_Nm=old['raw_peak_Nm'],simulator='Isaac Gym',seed=parent['seed'],source_mujoco_npz_sha256=spec['source_npz_sha256'])
        r=assess(r,a);r.update(isaac_gym_verified=r['passed'],automatic_resets=False,root_or_joint_overwrite_during_rollout=False)
        rows.append(r)
    result=dict(simulator='Isaac Gym',passed=all(r['passed'] for r in rows),cases=rows,seed=parent['seed'],
        source_result_sha256=hashlib.sha256((run/'result.json').read_bytes()).hexdigest(),
        live_rollout_sha256=hashlib.sha256((run/'live_rollout.npz').read_bytes()).hexdigest(),
        statistical_window='Each original transition duration; explicitly padded final stand is excluded',
        torque_and_fall_window='Entire 30s validation episode',new_training_performed=False)
    (run/'transition_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GYM TRANSITIONS',sum(r['passed'] for r in rows),'/',len(rows))
    for r in rows:print(r['clip'],r['passed'],r['rejection_reasons'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();review(a.run,a.dataset)
