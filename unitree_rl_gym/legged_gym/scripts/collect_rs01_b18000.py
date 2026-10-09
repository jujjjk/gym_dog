"""50 Hz B18000 sensor -> estimator -> actual actor input -> sent target audit."""
import argparse
import csv
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import isaacgym  # Must precede torch.
import torch
from legged_gym.envs.base import legged_robot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg

BASE = '/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v20_bounded_hip/Sep13_18-35-41_v20_gpu1_seed16_20260913_183537'
LEGS = ('FL', 'FR', 'RL', 'RR')
CASES = {
    'march': (0., 0., 0., 1.), 'stand': (0., 0., 0., 0.),
    'forward': (.15, 0., 0., 1.), 'backward': (-.15, 0., 0., 1.),
    'left': (0., .10, 0., 1.), 'right': (0., -.10, 0., 1.),
    'turn_left': (0., 0., .20, 1.), 'turn_right': (0., 0., -.20, 1.),
    'combined': (.20, .08, .25, 1.),
}


def vector(row, names, tensor):
    values = tensor.detach().cpu().reshape(-1).tolist()
    if len(names) != len(values):
        raise ValueError('Column shape mismatch')
    row.update(zip(names, values))


def numbered(prefix):
    return [f'{prefix}_{i:02d}' for i in range(1, 13)]


def joints(prefix):
    return [f'{prefix}_{leg}_{joint}' for leg in LEGS for joint in ('hip', 'thigh', 'calf')]


def install_taps(env):
    """Read-only return-value taps; never run estimator/limiter a second time."""
    cache = {}
    estimate = env.rs01_leg_odometry.estimate

    def tapped_estimate(*args, **kwargs):
        old_stance = env.rs01_leg_odometry.last_stance.clone()
        result = estimate(*args, **kwargs)
        cache['estimator'] = {k: v.clone() for k, v in result.items()}
        odom = env.rs01_leg_odometry
        heights = result['base_height_proxy']
        cache['score'] = ((heights.max(1).values[:, None] - heights) / odom.height_margin
                          + result['foot_velocity'][:, :, 2].abs() / odom.vertical_speed_threshold
                          - odom.previous_stance_score_bonus * old_stance.float())
        return result

    env.rs01_leg_odometry.estimate = tapped_estimate
    compute = env._compute_torques

    def tapped_torques(actions):
        first = getattr(env, '_v16_project_pending', False)
        limit_used = getattr(env, 'guard_active_limit', env.peak_torque_limit_nm).clone() if first else None
        result = compute(actions)
        if first:
            cache['targets'] = {k: getattr(env, k).clone() for k in (
                'rs01_limited_position_target_rad', 'rs01_target_rate_rad_s',
                'guard_target', 'guard_raw_pd', 'guard_safe_pd', 'guard_active_limit')}
            cache['targets']['guard_limit_used'] = limit_used
        return result

    env._compute_torques = tapped_torques
    return cache


def snapshot(env, cache, obs, step, case, repeat):
    stamp = float(env.gym.get_sim_time(env.sim))
    row = dict(timestamp_policy=stamp, timestamp_imu=stamp, timestamp_motor=stamp,
               control_step=step, case=case, repeat=repeat, imu_age_ms=0., motor_age_ms=0.,
               sensor_skew_ms=0., sensor_sample_valid=int('estimator' in cache),
               wall_timestamp_unix_s=time.time())
    vector(row, ['cmd_vx', 'cmd_vy', 'cmd_wz'], env.commands[0, :3])
    vector(row, ['gyro_x', 'gyro_y', 'gyro_z'], env.base_ang_vel[0])
    vector(row, ['roll', 'pitch', 'yaw'], env.rpy[0])
    vector(row, ['quat_x', 'quat_y', 'quat_z', 'quat_w'], env.base_quat[0])
    vector(row, [f'projected_gravity_{x}' for x in 'xyz'], env.projected_gravity[0])
    row['heading_target'] = float(env.straight_heading_target_rad[0])
    row['heading_error'] = float(env._straight_heading_error()[0])
    vector(row, ['effective_target_vx', 'effective_target_vy', 'effective_target_wz'], env._direction_velocity_target()[0])
    vector(row, ['est_vx', 'est_vy', 'est_vz'], env.estimated_base_lin_vel[0])
    row['odom_confidence'] = float(env.estimated_odom_confidence[0])
    for prefix, value in [('q', env.dof_pos), ('dq', env.dof_vel)]:
        vector(row, joints(prefix), value[0])
        vector(row, numbered(prefix), value[0])
    vector(row, [f'obs_{i:02d}' for i in range(61)], obs[0])
    vector(row, ['gt_x', 'gt_y', 'gt_z'], env.root_states[0, :3])
    vector(row, ['gt_vx', 'gt_vy', 'gt_vz'], env.base_lin_vel[0])
    vector(row, ['gt_world_vx', 'gt_world_vy', 'gt_world_vz'], env.root_states[0, 7:10])
    row['gt_yaw'] = row['yaw']; row['gt_wz'] = row['gyro_z']
    for axis in 'xyz':
        row[f'velocity_error_{axis}'] = row[f'est_v{axis}'] - row[f'gt_v{axis}']
    row['yaw_error'] = 0.  # Shared simulator orientation, NOT independent external IMU validation.
    phase = float(env._gait_phase()[0]); row['phase'] = phase
    row['phase_sin'] = float(obs[0, 48]); row['phase_cos'] = float(obs[0, 49])
    row['gait_enable'] = float(env.gait_enable[0])
    contact = env.get_foot_contact_mask()[0]
    expected = env._desired_contact_mask()[0]
    est = cache.get('estimator')
    for i, leg in enumerate(LEGS):
        slot = env.foot_slot_by_leg[leg]
        row[f'{leg}_phase'] = (phase + (0. if leg in ('FL', 'RR') else .5)) % 1.
        row[f'{leg}_expected_stance'] = int(expected[slot])
        row[f'{leg}_contact'] = int(contact[slot])
        row[f'{leg}_force_z_n'] = float(env.contact_forces[0, env.feet_indices[slot], 2])
        row[f'gt_foot_clearance_{leg}'] = float(env.feet_pos[0, slot, 2]) - env.cfg.rewards.foot_collision_radius_m
        row[f'{leg}_stance'] = int(env.estimated_odom_stance_mask[0, i])
        row[f'{leg}_stance_score'] = float(cache['score'][0, i]) if est else ''
        vector(row, [f'{leg}_est_v{x}' for x in 'xyz'], env.estimated_velocity_by_foot_m_s[0, i])
        row[f'foot_height_{leg}'] = float(est['foot_position'][0, i, 2]) if est else ''
        row[f'foot_vertical_velocity_{leg}'] = float(est['foot_velocity'][0, i, 2]) if est else ''
        residual_slot = (0 if leg in ('FL', 'RR') else 1) if env.rs01_leg_odometry.strict_diagonal_pairs else i
        row[f'velocity_residual_{leg}'] = float(est['velocity_residual'][0, residual_slot]) if est else ''
    for prefix, value in [('motor_torque', env.motor_electromagnetic_torques),
                          ('applied_torque', env.applied_joint_torques), ('raw_pd_feedback', env.raw_pd_torques)]:
        vector(row, numbered(prefix), value[0])
    vector(row, numbered('guard_rms_nm'), getattr(env, 'guard_rms_sq', torch.zeros_like(env.dof_pos))[0].sqrt())
    for prefix in ('motor_current', 'motor_temperature', 'motor_error_code'):
        row.update({name: '' for name in numbered(prefix)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASES) + ('sequence', 'suite'), default='sequence')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', default='/home/nszb/gym/artifacts/rs01_b18000_capture')
    options, rest = parser.parse_known_args()
    if options.seconds <= 0 or options.repeats < 1:
        parser.error('seconds and repeats must be positive')
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    args.task = 'rs01_omni_v20_bounded_hip'
    args.load_run = BASE; args.checkpoint = 18000; args.num_envs = 1
    if args.headless:
        legged_robot.time = types.SimpleNamespace(sleep=lambda seconds: None)
    segments = [(options.case, CASES[options.case])] if options.case in CASES else list(CASES.items())
    if options.case == 'sequence':
        segments = [('march', CASES['march'])]
        for name, command in CASES.items():
            if name not in ('stand', 'march'):
                segments.extend(((name, command), ('march', CASES['march'])))
    cfg, train = task_registry.get_cfgs(args.task)
    total = options.seconds * len(segments) * options.repeats
    _set_nominal_eval_cfg(cfg, total, 1)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if env.num_obs != 61 or env.num_actions != 12:
        raise ValueError('B18000 must have 61 observations and 12 actions')
    train.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    cache = install_taps(env)
    out = Path(options.output) / (time.strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns()))
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(BASE) / 'model_18000.pt'
    meta = dict(task=args.task, checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                seed=args.seed, dt_s=env.dt, leg_order=LEGS, joint_order=list(env.dof_names),
                options=vars(options), config=class_to_dict(cfg),
                observation_slots={'0:3':'estimated velocity * lin_vel scale','3:6':'body gyro * ang_vel scale',
                  '6:9':'projected gravity','9:12':'effective command * command scale',
                  '12:24':'q-default * q scale','24:36':'dq * dq scale','36:48':'previous clipped action',
                  '48:50':'phase sin/cos','50:52':'heading error sin/cos','52':'zero',
                  '53':'clipped 2*(est_vy-effective_vy)','54':'gait enable','55':'zero',
                  '56':'clipped 2*(est_vx-effective_vx)','57:60':'raw command * command scale','60':'odom confidence'},
                semantics=[
                  'One row: pre-step sensors/actual clipped actor input; same-action targets captured at first torque call; done_after_step refers to the following interval.',
                  'target_q is final guard-projected SENT target; limited_target_q is rate/acceleration-limited target BEFORE guard.',
                  'motor_torque/applied_torque/raw_pd_feedback are previous interval final substep feedback, not current action response or interval averages.',
                  'Timestamps are seconds on synchronous simulator clock, not real sensor acquisition or transport clocks. Zero age/skew is simulation construction, not hardware evidence.',
                  'No independent raw 100 Hz IMU, current, degrees-C temperature or error codes. Unavailable fields are blank, never fabricated zeros.',
                  'gt_v* is body frame; gt_world_v* and gt_xyz are world frame. IMU attitude equals simulator attitude; yaw_error=0 by construction.',
                  'foot_height is FK foot-center body Z, not ground clearance. gt_foot_clearance is world foot center Z minus radius on flat terrain.',
                  'stance_score lower is preferred; score is reconstructed from actual estimator return and PRE-update stance memory. No second estimator update.',
                  'With strict_diagonal_pairs, velocity_residual per leg is its diagonal PAIR planar velocity disagreement (FL/RR share pair 0; FR/RL share pair 1), not four independent residuals.',
                  'est_vz is clamped to zero by this estimator; it cannot measure vertical bounce.',
                  'suite resets posture per segment; sequence preserves continuous transitions. Stops on any environment reset/nonfinite.' ])
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('Recording to', out, flush=True)
    step = 0; failed = False
    try:
        with torch.no_grad():
            for repeat in range(1, options.repeats + 1):
                for segment, (name, command) in enumerate(segments):
                    if options.case == 'suite':
                        env.reset_idx(torch.arange(1, device=env.device)); cache.pop('estimator', None)
                    env.set_evaluation_command(torch.tensor(command[:3], device=env.device), command[3], reset_reference=options.case=='suite')
                    path = out / f'{repeat:02d}_{segment:02d}_{name}.csv'
                    print(name, command, options.seconds, 's', flush=True)
                    with path.open('x', newline='') as stream:
                        writer = None
                        for _ in range(max(1, round(options.seconds / env.dt))):
                            env.compute_observations()
                            obs = env.get_observations().clamp(-cfg.normalization.clip_observations, cfg.normalization.clip_observations).clone()
                            action = policy(obs).clone()
                            if not bool(torch.isfinite(obs).all() & torch.isfinite(action).all()):
                                raise RuntimeError('Nonfinite actor input/output; stopped')
                            row = snapshot(env, cache, obs, step, name, repeat)
                            vector(row, numbered('raw_action'), action[0])
                            vector(row, numbered('action'), action[0].clamp(-cfg.normalization.clip_actions, cfg.normalization.clip_actions))
                            _, _, reward, done, _ = env.step(action)
                            target = cache['targets']
                            for prefix, key in [('target_q', 'guard_target'), ('limited_target_q', 'rs01_limited_position_target_rad')]:
                                vector(row, joints(prefix), target[key][0]); vector(row, numbered(prefix), target[key][0])
                            for prefix, key in [('target_rate', 'rs01_target_rate_rad_s'), ('guard_limit_used', 'guard_limit_used'),
                                                ('guard_limit_next', 'guard_active_limit'), ('guard_raw_pd', 'guard_raw_pd'), ('guard_safe_pd', 'guard_safe_pd')]:
                                vector(row, numbered(prefix), target[key][0])
                            row['done_after_step'] = int(done[0])
                            row['finite_after_step'] = int(bool(torch.isfinite(reward).all() & torch.isfinite(env.get_observations()).all()))
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row); step += 1
                            if step % 50 == 0: stream.flush()
                            if row['done_after_step'] or not row['finite_after_step']:
                                raise RuntimeError('Reset/nonfinite: recording stopped, terminal row retained')
    except BaseException:
        failed = True
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(dict(rows=step, failed=failed, output=str(out)), indent=2))
        print('Saved', step, 'rows:', out, flush=True)


if __name__ == '__main__':
    main()
