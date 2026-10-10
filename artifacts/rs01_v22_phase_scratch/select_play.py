"""Read-only checkpoint evaluation/viewer; does not alter training configs on disk."""
import argparse
import json
import math
import sys
from pathlib import Path
import isaacgym
import torch
from isaacgym.torch_utils import quat_rotate_inverse

sys.path.insert(0, str(Path('/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')))
import evaluate_rs01_go2_omni as audit
from legged_gym.utils import get_args, task_registry

CASES = (
    ('march', 0., 0., 0., True),
    ('forward', .2, 0., 0., True), ('backward', -.2, 0., 0., True),
    ('left', 0., .2, 0., True), ('right', 0., -.2, 0., True),
    ('turn_left', 0., 0., .3, True), ('turn_right', 0., 0., -.3, True),
    ('forward_left', .2, .1, 0., True), ('forward_right', .2, -.1, 0., True),
    ('backward_left', -.2, .1, 0., True), ('backward_right', -.2, -.1, 0., True),
    ('combined', .2, .08, .25, True), ('combined_reverse', -.2, -.08, -.25, True),
)


def scaled_cases(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('speed_scale must be finite and positive')
    return tuple((name, vx*scale, vy*scale, wz*scale, gait)
                 for name, vx, vy, wz, gait in CASES)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--speed_scale', type=float, default=1.,
                        help='Scale velocity commands, NOT playback time')
    parser.add_argument('--sensor_profile', choices=('clean', 'normal'), default='normal')
    own, rest = parser.parse_known_args()
    cases = scaled_cases(own.speed_scale)
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    if args.task not in ('rs01_v22_phase_lift', 'rs01_v22_phase_guard5', 'rs01_v22_phase_guard3', 'rs01_v22_phase_lateral'):
        raise ValueError('Use a V22 phase lift/guard task')
    if args.duration_s <= 0:
        raise ValueError('duration_s must be positive')
    cfg, train = task_registry.get_cfgs(args.task)
    cfg.sensor.enabled = own.sensor_profile != 'clean'
    cfg.sensor.clean_fraction = 0.
    cfg.sensor.robust_fraction = 0.
    print('SENSOR PROFILE', own.sensor_profile, flush=True)
    print('COMMAND SPEED SCALE', own.speed_scale, flush=True)
    if not own.sequence:
        audit.SUPPORTED_TASKS.add(args.task)
        audit.COMMAND_CASES = cases
        args.eval_suite = 'nominal'
        audit.evaluate(args)
        return
    audit._set_nominal_eval_cfg(cfg, args.duration_s * len(cases), args.num_envs or 1)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                             train_cfg=train, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    stages = []
    legs = ('FL', 'FR', 'RL', 'RR')
    hips = [env.dof_names.index(leg+'_hip_joint') for leg in legs]
    feet_ids = [env.foot_slot_by_leg[leg] for leg in legs]
    sides = torch.tensor([1., -1., 1., -1.], device=env.device)
    with torch.no_grad():
        for name, vx, vy, wz, _ in cases:
            print(name, (vx, vy, wz), args.duration_s, 's', flush=True)
            command = torch.tensor([vx, vy, wz], device=env.device)
            rows, diagnostics, resets = [], [], 0
            for _ in range(round(args.duration_s/env.dt)):
                env.set_evaluation_command(command, 1.)
                env.compute_observations()
                obs, _, reward, done, _ = env.step(policy(env.get_observations()))
                assert bool(torch.isfinite(obs).all() and torch.isfinite(reward).all())
                resets += int(done.sum())
                rows.append(torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1).clone())
                rel = env.feet_pos-env.root_states[:, None, :3]
                quat = env.base_quat[:, None, :].expand(-1,4,-1).reshape(-1,4)
                feet = quat_rotate_inverse(quat, rel.reshape(-1,3)).reshape(-1,4,3)[:, feet_ids]
                contact = env.get_foot_contact_mask()[:, feet_ids]
                straight = (1.-abs(vy)/.08-abs(wz)/.20)
                minimum = env.cfg.rewards.footprint_omni_min_width_m + max(0.,straight)*(
                    env.cfg.rewards.footprint_straight_min_width_m-env.cfg.rewards.footprint_omni_min_width_m)
                # Per-foot safe-side proxy, not a dynamic support-polygon margin.
                inside = (feet[:,:,1]*sides < minimum/2.) & contact
                signed_hip = env.dof_pos[:, hips]*sides
                diagnostics.append(torch.stack((
                    env.rpy[:,0].square(), env.rpy[:,1].square(),
                    inside.float().sum(1), contact.float().sum(1),
                    (-signed_hip).clamp(min=0.).max(1).values,
                    env.raw_pd_torques.abs().max(1).values,
                    env.motor_electromagnetic_torques.abs().max(1).values,
                    (~contact.any(1)).float()),1).clone())
            data = torch.stack(diagnostics)
            stages.append(dict(name=name, command=[vx, vy, wz], resets=resets,
                mean_vx_vy_wz=torch.stack(rows).mean((0, 1)).tolist(),
                roll_rms_deg=float(data[:,:,0].mean().sqrt()*180/math.pi),
                pitch_rms_deg=float(data[:,:,1].mean().sqrt()*180/math.pi),
                stance_inside_fraction=float(data[:,:,2].sum()/data[:,:,3].sum().clamp(min=1)),
                actual_inward_hip_peak_deg=float(data[:,:,4].max()*180/math.pi),
                raw_torque_peak_nm=float(data[:,:,5].max()),
                motor_torque_peak_nm=float(data[:,:,6].max()),
                flight_fraction=float(data[:,:,7].mean())))
    print(json.dumps(dict(task=args.task, checkpoint=args.checkpoint, run=args.load_run,
        sensor=own.sensor_profile, speed_scale=own.speed_scale,
        duration_per_action_s=args.duration_s, seed=args.seed,
        finite=True, resets=sum(s['resets'] for s in stages), stages=stages), indent=2))


if __name__ == '__main__':
    main()
