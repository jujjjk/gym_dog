"""Measure the same cold start, acceleration, braking and reversal in Gym."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import numpy as np
import torch

import legged_gym.envs
from legged_gym.algorithms.urdf4_amp_rough import Urdf4AMPRoughRunner
from legged_gym.envs.urdf4.urdf4_omni_env import Urdf4OmniRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--axis', choices=('forward', 'lateral', 'yaw'), default='forward')
    parser.add_argument('--peak-speed', type=float, default=.50)
    parser.add_argument('--spawn-offsets', type=float, nargs='+', metavar='METERS',
                        help='Optional x/y spawn offsets for each nominal environment')
    options, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    args = get_args()
    args.task = 'urdf4_amp_cts_rough' if options.original else 'urdf4_amp_cts_transition'
    args.num_envs = args.num_envs or 1
    if not .35 < options.peak_speed <= .50:
        parser.error('--peak-speed must be within the fast-forward envelope (0.35, 0.50]')
    if options.spawn_offsets and len(options.spawn_offsets) != 2*args.num_envs:
        parser.error('--spawn-offsets needs one x/y pair per environment')
    args.headless = True
    checkpoint = torch.load(options.model, map_location='cpu')
    runner_type = Urdf4AMPRoughRunner
    if 'transition_training' in checkpoint['contract']:
        if options.original:
            parser.error('--original requires a rough model saved with the old controller')
        from legged_gym.algorithms.urdf4_amp_transition_training import Urdf4AMPTransitionTrainingRunner
        args.task = 'urdf4_amp_cts_transition_training'
        runner_type = Urdf4AMPTransitionTrainingRunner
    cfg, train = task_registry.get_cfgs(args.task)
    cfg.env.test = True
    cfg.env.episode_length_s = 40.
    for key, value in checkpoint['contract']['rough_ground'].items():
        setattr(cfg.rough, key, value)
    if 'transition_training' in checkpoint['contract']:
        from legged_gym.envs.urdf4.amp_transition_training import restore_transition_contract
        restore_transition_contract(cfg, train, checkpoint)
    if options.flat:
        # Use an exact PhysX plane with the same policy and controller. These
        # overrides are local to this evaluation process only.
        robot = task_registry.get_task_class(args.task)
        robot._create_ground_plane = Urdf4OmniRobot._create_ground_plane
        robot.ground_height_at = lambda self, xy: xy[..., 0].new_zeros(xy.shape[:-1])
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if options.spawn_offsets:
        env.env_origins[:, :2] += env.env_origins.new_tensor(options.spawn_offsets).reshape(-1, 2)
        env.env_origins[:, 2] = env.ground_height_at(env.env_origins[:, :2])
    runner = runner_type(env, class_to_dict(train), None, device=args.rl_device)
    runner.load(options.model, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    peak = torch.zeros(env.num_envs, device=env.device)
    original_torques = env._compute_torques

    def measure_torques(action):
        result = original_torques(action)
        peak.copy_(torch.maximum(peak, env.raw_pd_torques.abs().amax(1)))
        return result

    env._compute_torques = measure_torques
    targets = {
        'forward': [(0.18, 0., 0.), (options.peak_speed, 0., 0.), (.25, 0., 0.), (-.15, 0., 0.), (0., 0., 0.)],
        'lateral': [(0., .10, 0.), (0., .20, 0.), (0., -.10, 0.), (0., -.20, 0.), (0., 0., 0.)],
        'yaw': [(0., 0., .40), (0., 0., .60), (0., 0., -.40), (0., 0., -.60), (0., 0., 0.)],
    }[options.axis]
    steps = round(6. / env.dt)
    columns = ('external_vx,servo_vx,actual_vx,roll_deg,pitch_deg,contact_count,raw_substep_peak_Nm,'
               'cross_m,reset,actual_vy,actual_wz,illegal_support,swing_FL,swing_FR,swing_RL,swing_RR,'
               'support_FL_RR,support_FR_RL')
    slots = [env.foot_legs.index(leg) for leg in ('FL', 'FR', 'RL', 'RR')]
    records = []
    with torch.no_grad():
        for index, target in enumerate(targets):
            command = env.root_states.new_tensor(target).repeat(env.num_envs, 1)
            env.set_command_override(command)
            if index == 0:
                obs, _ = env.reset()
            for _ in range(steps):
                peak.zero_()
                obs, _, _, done, _ = env.step(policy(obs))
                gx, gy, gz = env.projected_gravity.unbind(1)
                contact = env.get_foot_contact_mask()[:, slots]
                feet = env.feet_pos[:, slots]
                clear = feet[:, :, 2] - env.ground_height_at(feet[:, :, :2]) - .02
                count = contact.sum(1)
                first = (count == 2) & contact[:, 0] & contact[:, 3]
                other = (count == 2) & contact[:, 1] & contact[:, 2]
                illegal = (count < 2) | ((count == 2) & ~(first | other))
                swing = (~contact & (clear > .012)).unbind(1)
                values = (env.external_commands[:, 0], env.commands[:, 0], env.base_lin_vel[:, 0],
                          torch.rad2deg(torch.atan2(-gy, -gz)), torch.rad2deg(torch.asin(gx.clamp(-1, 1))),
                          count, peak, env.path.cross_error, done, env.base_lin_vel[:, 1],
                          env.base_ang_vel[:, 2], illegal, *swing, first, other)
                records.append(torch.stack(values, dim=1).cpu().numpy())
    array = np.stack(records)
    cases = []
    for index, target in enumerate(targets):
        segment = array[index*steps:(index+1)*steps]
        cases.append(dict(command=target, vx_last_second_m_s=segment[-round(1/env.dt):, :, 2].mean(0).tolist(),
            twist_last_second=segment[-round(1/env.dt):, :, [2, 9, 10]].mean(0).tolist(),
            peak_roll_pitch_deg=np.abs(segment[:, :, 3:5]).max((0, 1)).tolist(),
            single_support_s_per_env=((segment[:, :, 5] == 1).sum(0)*env.dt).tolist(),
            flight_s_per_env=((segment[:, :, 5] == 0).sum(0)*env.dt).tolist(),
            illegal_support_s_per_env=(segment[:, :, 11].sum(0)*env.dt).tolist(),
            all_four_feet_lift=segment[-round(2/env.dt):, :, 12:16].any(0).all(1).tolist(),
            both_diagonals_support=segment[-round(2/env.dt):, :, 16:18].any(0).all(1).tolist(),
            raw_peak_Nm=float(segment[:, :, 6].max()), cross_peak_m=float(abs(segment[:, :, 7]).max()),
            resets=int(segment[:, :, 8].sum())))
    if options.axis == 'forward':
        for case in cases[1:3]:
            checks = {
                'fall': case['resets'] == 0,
                'speed': all(abs(v-case['command'][0]) <= .05 for v in case['vx_last_second_m_s']),
                'direction': case['cross_peak_m'] <= .05,
                'body_tilt': all(x <= limit for x, limit in zip(case['peak_roll_pitch_deg'], (5., 4.))),
                'torque': case['raw_peak_Nm'] <= 17.,
                'support': max(case['illegal_support_s_per_env']) <= .04 and max(case['flight_s_per_env']) == 0.,
                'swing': all(case['all_four_feet_lift']),
                'diagonal': all(case['both_diagonals_support']),
            }
            case['failed_gates'] = [name for name, passed in checks.items() if not passed]
    window = round(.4 / env.dt)
    startup = []
    velocity_column = {'forward': 2, 'lateral': 9, 'yaw': 10}[options.axis]
    first_target = targets[0][{'forward': 0, 'lateral': 1, 'yaw': 2}[options.axis]]
    for robot in range(env.num_envs):
        averaged = np.convolve(array[:steps, robot, velocity_column], np.ones(window) / window, mode='valid')
        # Start timing after the body has first reached a standing support.
        hits = np.flatnonzero((averaged >= first_target*.9) & (np.arange(len(averaged))*env.dt >= .5))
        startup.append(float((hits[0]+window-1)*env.dt) if len(hits) else None)
    report = dict(task=args.task, flat=options.flat, axis=options.axis, seed=args.seed, copies=env.num_envs, dt_s=env.dt,
                  spawn_offsets_m=options.spawn_offsets, runtime_source_sha256=runner.experiment['source_sha256'],
                  checkpoint_sha256=hashlib.sha256(options.model.read_bytes()).hexdigest(),
                  segment_s=steps*env.dt, startup_90pct_s=startup, cases=cases)
    if options.axis == 'forward':
        report['fast_windows_passed'] = all(not case['failed_gates'] for case in cases[1:3])
        report['peak_speed_m_s'] = options.peak_speed
    options.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(options.output / 'trace.npz', samples=array, columns=columns, dt_s=env.dt)
    (options.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
