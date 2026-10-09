"""Build an AMP style reference from MEASURED plant recordings, not from a kinematic template.

Why: a template that needs more joint rate than the identified RS01 target-rate limits
(hip/thigh/calf 2.0/2.6/3.2 rad/s) cannot be followed; open-loop replay of the 60 mm/2 Hz
design lags 0.20 rad and walks backwards. Recorded plant states are feasible by construction.
"""
import argparse
import json
from pathlib import Path
import numpy as np

CLIPS = ['march','forward','backward','left','right','turn_left','turn_right',
         'forward_left','forward_right','combined']
JOINT_NAMES = [leg+'_'+j+'_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
FOOT_RADIUS_M = 0.016


def measured_command(requested, mean_velocity):
    """Keep only the axes the clip commands, at the speed the plant actually produced."""
    requested = np.asarray(requested, dtype=float)
    measured = np.asarray(mean_velocity, dtype=float)
    # Rounded so the stored clip command, the manifest and the env task config are the same numbers.
    return np.round(np.where(np.abs(requested) > 1e-9, measured, 0.0), 4)


def clip_record(source, name):
    with np.load(source/(name+'_physics/physical.npz'), allow_pickle=False) as phys, \
         np.load(source/(name+'.npz'), allow_pickle=False) as template:
        report = json.loads((source/(name+'_physics/physical_report.json')).read_text())
        if report['stop_reason'] != 'completed':
            raise ValueError((name, report['stop_reason']))
        features = phys['amp_features']
        if features.shape != (500, 46):
            raise ValueError((name, features.shape))
        command = measured_command(template['command'][0], report['mean_vx_vy_wz'])
        feet_z = features[:, 24:36].reshape(-1, 4, 3)[:, :, 2]
        lift = features[:, 45:46] + feet_z - FOOT_RADIUS_M
        # amp_features already uses the env leg order FL,FR,RL,RR; MuJoCo qpos joints do not.
        out = dict(amp_features=features,
                   transition_valid=phys['transition_valid'],
                   command=np.repeat(command[None], len(features), 0),
                   joint_pos_rad=features[:, 0:12], joint_vel_rad_s=features[:, 12:24],
                   root_pos_world_m=phys['qpos'][:, 0:3], root_quat_wxyz=phys['qpos'][:, 3:7],
                   leg_phase=template['leg_phase'], foot_height_m=lift)
        rate_ratio = np.abs(out['joint_vel_rad_s']) / np.array([2.0, 2.6, 3.2]*4)
        audit = dict(clip=name, frames=int(len(features)), physically_executed=True,
                     command=np.round(command, 4).tolist(),
                     requested_command=np.round(template['command'][0], 4).tolist(),
                     foot_height_p95_mm=np.round(np.quantile(lift, .95, axis=0)*1000, 1).tolist(),
                     rate_ratio_max=round(float(rate_ratio.max()), 3),
                     template_tracking_rmse_rad=round(float(np.sqrt(np.mean(
                         (out['joint_pos_rad']-template['joint_pos_rad'])**2))), 4),
                     raw_peak_nm=round(report['raw_peak_nm'], 2),
                     flight_ratio=report['flight_ratio'],
                     contact_mismatch_ratio=report['contact_mismatch_ratio'],
                     inward_support_ratio=report['inward_support_ratio'],
                     roll_pitch_max_deg=np.round(report['roll_pitch_max_deg'], 2).tolist(),
                     diagnostic_gates_passed=report['diagnostic_gates_passed'])
        return out, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Directory with <clip>.npz templates and <clip>_physics/ recordings')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or args.source.parent/(args.source.name+'_amp')
    if out.exists():
        raise SystemExit('Reference datasets are immutable; choose a new --output')
    out.mkdir(parents=True)
    clips = []
    for name in CLIPS:
        data, audit = clip_record(args.source, name)
        np.savez_compressed(out/(name+'.npz'), **data)
        clips.append(audit)
    commands = [c['command'] for c in clips]
    settings = dict(json.loads((args.source/'manifest.json').read_text())['settings'])
    settings.update(source=str(args.source), provenance='measured plant recordings, 50 Hz policy frames')
    manifest = dict(schema='rs01_style_reference_v1', kind='recorded_nonideal_plant',
                    amp_training_approved=False,
                    note=('States actually executed through the unchanged RS01 plant (rate/accel limits, '
                          'identified delay, torque guard). Commands are the measured speeds along each '
                          'commanded axis, so the velocity task target is attainable. Not a deployment '
                          'expert: contact handoff and directional tracking gates are still failing.'),
                    frame_dt_s=0.02, joint_order=JOINT_NAMES, foot_order=['FL','FR','RL','RR'],
                    quaternion_order='wxyz',
                    amp_features_order=['joint_pos_rad:12','joint_vel_rad_s:12','foot_pos_body_m:12',
                                        'base_lin_vel_body_m_s:3','base_ang_vel_body_rad_s:3',
                                        'projected_gravity:3','root_height_m:1'],
                    intended_use='AMP style reference from recorded physics; commands are measured speeds',
                    settings=settings, clips=clips)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for c in clips:
        print('%-14s cmd %s p95lift %s rate %.2f raw %s Nm mismatch %.2f' %
              (c['clip'], c['command'], c['foot_height_p95_mm'], c['rate_ratio_max'],
               c['raw_peak_nm'], c['contact_mismatch_ratio']))
    print('commands for env cfg:', json.dumps(commands))
    print('wrote', out)


if __name__ == '__main__':
    main()
