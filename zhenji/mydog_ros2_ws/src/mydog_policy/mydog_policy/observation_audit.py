"""Offline mounting proposal and measured-distance comparison. No device access."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .observation_pipeline import rotation_rpy


def propose_mount(rows):
    if len(rows) < 400:
        raise ValueError('Need at least 400 supported, static samples over 10 seconds')
    t = np.array([float(r['timestamp_policy']) for r in rows])
    if np.any(np.diff(t) <= 0) or t[-1]-t[0] < 10. or np.max(np.diff(t)) > .08:
        raise ValueError('Need a continuous >=10s static window')
    if any(r['mode'] != 'ready' or r['observation_temporal_ok'] != 'True' for r in rows):
        raise ValueError('Window must be ready with healthy timestamp estimates')
    gravity = np.array([[float(r['raw_gravity_'+a]) for a in 'xyz'] for r in rows])
    gyro = np.array([[float(r['raw_gyro_'+a]) for a in 'xyz'] for r in rows])
    joints = ['FR_hip','FR_thigh','FR_calf','FL_hip','FL_thigh','FL_calf',
              'RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
    q = np.array([[float(r['q_'+j]) for j in joints] for r in rows])
    if not np.isfinite(np.r_[gravity.ravel(), gyro.ravel(), q.ravel()]).all():
        raise ValueError('Non-finite calibration data')
    if np.max(np.ptp(q, axis=0)) > .02 or np.max(np.linalg.norm(gyro, axis=1)) > .08 or np.max(np.std(gravity,axis=0)) > .01:
        raise ValueError('Motion/noise detected; do not use this window for mounting')
    a = gravity.mean(axis=0); a /= np.linalg.norm(a)
    b = np.array([0.,0.,-1.]); dot=float(a@b)
    if dot < np.cos(np.radians(15.)):
        raise ValueError('Mount tilt exceeds 15 degrees; verify mechanical axes manually')
    v=np.cross(a,b); x,y,z=v
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    rotation=np.eye(3)+k+k@k/(1+dot)
    return rotation_rpy(rotation), rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--start',type=float,required=True,help='policy monotonic start time')
    parser.add_argument('--end',type=float,required=True)
    sub=parser.add_subparsers(dest='mode',required=True)
    mount=sub.add_parser('mount')
    mount.add_argument('--level-confirmed',action='store_true',required=True,
                       help='Operator confirms mechanically level platform and static support')
    mount.add_argument('--output',required=True)
    gt=sub.add_parser('ground-truth')
    gt.add_argument('--distance-m',type=float,required=True,help='signed forward displacement measured externally')
    args=parser.parse_args()
    if not np.isfinite([args.start,args.end]).all() or args.end <= args.start:
        parser.error('Require finite end > start')
    with Path(args.csv).open() as f:
        rows=[r for r in csv.DictReader(f) if args.start <= float(r['timestamp_policy']) <= args.end]
    if not rows: parser.error('No rows in selected interval')
    if args.mode == 'mount':
        angles, rotation=propose_mount(rows)
        with Path(args.output).open('x') as f:
            f.write('# Gravity constrains tilt only. Yaw is a minimum-rotation assumption, not a measured heading calibration.\n/**:\n  ros__parameters:\n')
            for axis,value in zip(('roll','pitch','yaw'),angles):
                f.write(f'    imu_mount_{axis}_deg: {float(value):.9f}\n')
        print(json.dumps(dict(rotation_base_from_imu=rotation.tolist(), output=args.output,
                              applied=False, yaw_observable=False),indent=2))
    else:
        if not np.isfinite(args.distance_m): parser.error('Distance must be finite')
        velocities=np.array([float(r['est_vx']) for r in rows])
        times=np.array([float(r['timestamp_policy']) for r in rows])
        if len(times)<2 or not np.isfinite(velocities).all() or np.any(np.diff(times)<=0) or np.max(np.diff(times))>.08:
            parser.error('Need continuous finite estimator samples')
        if times[0]-args.start>.04 or args.end-times[-1]>.04:
            parser.error('CSV must cover the measured interval')
        estimate=float(np.trapz(velocities,times)/(times[-1]-times[0]))
        measured=args.distance_m/(args.end-args.start)
        print(json.dumps(dict(ground_truth_vx_mps=measured, estimator_mean_vx_mps=estimate,
                              error_mps=estimate-measured, rows=len(rows)),indent=2))


if __name__ == '__main__':
    main()
