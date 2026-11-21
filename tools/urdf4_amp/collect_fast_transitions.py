"""Generate and qualify real Gym AMP pairs for acceleration and braking."""
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
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict


def assess(features, clearance, contact, raw, cross, failed, dt, boundaries):
    gravity = features[:, 42:45]
    roll = np.degrees(np.arctan2(-gravity[:, 1], -gravity[:, 2]))
    pitch = np.degrees(np.arcsin(gravity[:, 0].clip(-1, 1)))
    rollouts = []
    for name, (start, end, target) in zip(('accelerate', 'brake'), boundaries):
        part = slice(start, end)
        last = slice(end-round(1/dt), end)
        contacts = contact[part]
        count = contacts.sum(1)
        diagonal = (contacts[:, 0] & contacts[:, 3]) | (contacts[:, 1] & contacts[:, 2])
        illegal = (count < 2) | ((count == 2) & ~diagonal)
        speed = float(features[last, 36].mean())
        foot_lift = ((clearance[end-round(2/dt):end] > .012) & ~contact[end-round(2/dt):end]).any(0)
        row = dict(window=name, target=float(target), speed_m_s=speed, cross_m=float(abs(cross[part]).max()),
                   roll_deg=float(abs(roll[part]).max()), pitch_deg=float(abs(pitch[part]).max()),
                   raw_peak_Nm=float(raw[part].max()), illegal_support_s=float(illegal.sum()*dt),
                   flight_s=float((count == 0).sum()*dt), all_four_feet_lift=bool(foot_lift.all()))
        reasons = []
        if failed[part].any(): reasons.append('fall')
        if abs(speed-target) > .05: reasons.append('tracking')
        if row['cross_m'] > .05: reasons.append('direction')
        if row['roll_deg'] > 5. or row['pitch_deg'] > 4.: reasons.append('tilt')
        if row['raw_peak_Nm'] > 17.: reasons.append('torque')
        if row['illegal_support_s'] > .04 or row['flight_s'] > 0.: reasons.append('support')
        if not row['all_four_feet_lift']: reasons.append('swing')
        row.update(passed=not reasons, rejection_reasons=reasons)
        rollouts.append(row)
    return rollouts


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rise-start-s', type=float, default=5.8)
    parser.add_argument('--rise-spacing-ticks', type=int, default=3)
    parser.add_argument('--bias-caps', type=float, nargs='+', default=[.06, .08, .10, .12])
    own, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1]+rest
    args = get_args()
    args.headless = True
    args.task = 'urdf4_amp_cts_transition'
    args.num_envs = args.num_envs or 32
    if not 4 <= args.num_envs <= 96:
        parser.error('Reference search uses 4–96 trials')
    if own.rise_start_s < 4. or own.rise_spacing_ticks < 1 or any(not 0 < cap <= .20 for cap in own.bias_caps):
        parser.error('Use a settled initial gait and bias caps inside the original .20 m/s limit')
    saved = torch.load(own.model, map_location='cpu')
    cfg, train = task_registry.get_cfgs(args.task)
    cfg.env.test = True
    cfg.env.episode_length_s = 35.
    for key, value in saved['contract']['rough_ground'].items():
        setattr(cfg.rough, key, value)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner = Urdf4AMPRoughRunner(env, class_to_dict(train), None, device=args.rl_device)
    runner.load(own.model, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    ids = torch.arange(env.num_envs, device=env.device)
    caps = env.commands.new_tensor(own.bias_caps)
    env.acceleration_bias_limit = caps[ids % len(caps)]
    dt = env.dt
    rise = round(own.rise_start_s/dt) + (ids // len(caps)) * own.rise_spacing_ticks
    brake = rise + round(8./dt)
    end = brake + round(7./dt)
    command = torch.zeros(env.num_envs, 3, device=env.device)
    command[:, 0] = .18
    env.set_command_override(command)
    observation, _ = env.reset()
    peak = torch.zeros(env.num_envs, device=env.device)
    fallen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    base_torques = env._compute_torques

    def measure(action):
        result = base_torques(action)
        peak.copy_(torch.maximum(peak, env.raw_pd_torques.abs().amax(1)))
        return result

    env._compute_torques = measure
    env.reset_idx = lambda reset_ids: fallen.__setitem__(reset_ids, True)
    slots = [env.foot_legs.index(leg) for leg in ('FL', 'FR', 'RL', 'RR')]
    records = []
    with torch.no_grad():
        for k in range(int(end.max())+1):
            planned = torch.where(k >= brake, .25, torch.where(k >= rise, .50, .18))
            command[:, 0] = planned
            env.set_command_override(command)
            peak.zero_()
            previous_command = env.external_commands.clone()
            action = policy(observation)
            action[fallen] = 0.
            observation, _, _, _, info = env.step(action)
            amp = info['amp']
            feet = env.feet_pos[:, slots]
            height = feet[:, :, 2] - env.ground_height_at(feet[:, :, :2]) - .02
            records.append((amp['previous'].clone(), amp['next'].clone(), previous_command,
                            env.external_commands.clone(), env.get_foot_contact_mask()[:, slots].clone(),
                            height.clone(), peak.clone(), env.path.cross_error.clone(), fallen.clone()))
            if (k+1) % 250 == 0:
                print('TRANSITION SEARCH', round((k+1)*dt, 2), int(fallen.sum()), flush=True)
    before, after, start_command, next_command, contact, height, raw, cross, failed = [
        torch.stack(fields).cpu().numpy() for fields in zip(*records)]
    trials = []
    accepted_pairs, accepted_commands, accepted_next, clip_ids = [], [], [], []
    for i, (s1, s2, s3) in enumerate(zip(rise.tolist(), brake.tolist(), end.tolist())):
        rows = assess(after[:, i], height[:, i], contact[:, i], raw[:, i], cross[:, i], failed[:, i], dt,
                      [(s1, s2, .50), (s2, s3, .25)])
        passed = all(row['passed'] for row in rows) and not failed[:s3, i].any()
        trials.append(dict(trial=i, rise_s=s1*dt, brake_s=s2*dt, speed_bias_cap_m_s=float(env.acceleration_bias_limit[i]),
                           passed=passed, windows=rows))
        if passed:
            # Keep only actual changing-command frame pairs. File includes the
            # ordered commands and source trial, so learning can condition on
            # acceleration or braking rather than mislabel these as steady.
            changed = np.linalg.norm(next_command[:, i]-start_command[:, i], axis=1) > 1.e-5
            take = np.flatnonzero(changed & (np.arange(len(changed)) >= s1) & (np.arange(len(changed)) < s3))
            accepted_pairs.append(np.c_[before[take, i], after[take, i]])
            accepted_commands.append(start_command[take, i]); accepted_next.append(next_command[take, i]); clip_ids.append(np.full(len(take), i))
    own.output.mkdir(exist_ok=False, parents=True)
    sha = hashlib.sha256(own.model.read_bytes()).hexdigest()
    report = dict(seed=args.seed, dt_s=dt, urdf_sha256=env.urdf_sha256, checkpoint_sha256=sha,
                  qualified_trials=sum(x['passed'] for x in trials), trials=trials,
                  true_gym_pairs=True, auto_reset=False, height_reference='local_terrain')
    if accepted_pairs:
        np.savez_compressed(own.output / 'qualified_pairs.npz', pairs=np.concatenate(accepted_pairs),
                            command=np.concatenate(accepted_commands), next_command=np.concatenate(accepted_next),
                            source_trial=np.concatenate(clip_ids))
    (own.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('QUALIFIED TRANSITION TRIALS', report['qualified_trials'], '/', len(trials), flush=True)


if __name__ == '__main__':
    main()
