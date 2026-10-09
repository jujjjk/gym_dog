"""Offline CSV alignment and B18000 ONNX replay check; no simulator needed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def check(folder, onnx):
    meta = json.loads((folder / 'metadata.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    assert not summary['failed'], summary
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    scale = meta['config']['normalization']['obs_scales']
    default = np.array([meta['config']['init_state']['default_joint_angles'][name] for name in meta['joint_order']])
    count = 0; max_error = 0.
    for path in sorted(folder.glob('*.csv')):
        rows = list(csv.DictReader(path.open()))
        assert rows, path
        def array(names):
            return np.array([[float(r[k]) for k in names] for r in rows], dtype=np.float32)
        obs = array([f'obs_{i:02d}' for i in range(61)])
        raw = array([f'raw_action_{i:02d}' for i in range(1, 13)])
        action = array([f'action_{i:02d}' for i in range(1, 13)])
        np.testing.assert_allclose(obs[:, :3], array(['est_vx','est_vy','est_vz'])*scale['lin_vel'], atol=1e-6)
        np.testing.assert_allclose(obs[:, 3:6], array(['gyro_x','gyro_y','gyro_z'])*scale['ang_vel'], atol=1e-6)
        np.testing.assert_allclose(obs[:, 6:9], array([f'projected_gravity_{x}' for x in 'xyz']), atol=1e-6)
        np.testing.assert_allclose(obs[:, 9:12], array(['effective_target_vx','effective_target_vy','effective_target_wz'])*[2,2,.25], atol=1e-6)
        np.testing.assert_allclose(obs[:, 12:24], (array([f'q_{i:02d}' for i in range(1,13)])-default)*scale['dof_pos'], atol=2e-7)
        np.testing.assert_allclose(obs[:, 24:36], array([f'dq_{i:02d}' for i in range(1,13)])*scale['dof_vel'], atol=1e-6)
        np.testing.assert_allclose(obs[1:,36:48], action[:-1], atol=1e-7)
        np.testing.assert_allclose(action, raw.clip(-1,1), atol=1e-7)
        q = array([f'q_{i:02d}' for i in range(1,13)])
        dq = array([f'dq_{i:02d}' for i in range(1,13)])
        limited = array([f'limited_target_q_{i:02d}' for i in range(1,13)])
        guard_raw = array([f'guard_raw_pd_{i:02d}' for i in range(1,13)])
        guard_safe = array([f'guard_safe_pd_{i:02d}' for i in range(1,13)])
        guard_limit = array([f'guard_limit_used_{i:02d}' for i in range(1,13)])
        # This fixed B18000 contract has Kp=40, Kd=1 on all twelve joints.
        np.testing.assert_allclose(guard_raw,40*(limited-q)-dq,atol=1e-5)
        np.testing.assert_allclose(guard_safe,np.minimum(np.maximum(guard_raw,-guard_limit),guard_limit),atol=1e-6)
        np.testing.assert_allclose(np.diff(array(['timestamp_policy'])[:,0]), meta['dt_s'], atol=2e-5)
        assert all(r['done_after_step']=='0' and r['finite_after_step']=='1' for r in rows)
        assert all(r['motor_current_01']=='' and r['motor_temperature_01']=='' for r in rows)
        for index in range(len(rows)):
            predicted = session.run(None, {input_name:obs[index:index+1]})[0]
            error = float(np.max(np.abs(predicted[0]-raw[index])))
            max_error = max(max_error,error)
            assert error < 3e-5, (path,index,error)
        count += len(rows)
    assert count == summary['rows']
    print(json.dumps(dict(rows=count, onnx_max_abs_error=max_error, aligned=True, folder=str(folder)), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--onnx',type=Path,default=Path('/home/nszb/gym/artifacts/rs01_v20_sim2sim/B18000.onnx'))
    args=parser.parse_args()
    check(args.folder,args.onnx)
