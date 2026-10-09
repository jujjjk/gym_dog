"""Record an AMP style reference from states the IsaacGym RS01 plant itself produces.

Why this exists: an AMP expert must be reachable by the plant the policy trains in.
Two references were tried and both failed that test.
  * The kinematic design asked for 1.8-2.7x the identified target-rate limits.
  * The MuJoCo plant recordings lift all four feet ~20 mm, but replayed through the
    IsaacGym plant the rear legs land 0.15-0.19 rad short and never lift: at kp=40 the
    body settles rear-heavy (measured 64% of weight on the rear feet) and the rear knees
    collapse. The MPC teacher fails the same way from inside the sim (rear p95 lift 0 mm,
    17-27 Nm raw torque at 1.09 Hz).
So the demonstrator is a policy already trained in this plant: every recorded frame is
reachable by construction. --policy loads one or more "dir:checkpoint" demonstrators and
keeps, per command band, the clip that lifts its worst foot highest. Without --policy the
diagonal-VMC teacher drives the same loop.

The demonstrated tempo is command dependent (about 1.7Hz straight, 3Hz while mincing sideways
or turning), and pinning one cadence for every band costs the rear feet their lift, so the
reference carries a per-clip cadence and the training env has to reproduce the same
command-to-clock map (amp.command_dependent_clock).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse
import legged_gym.envs
from legged_gym.envs.rs01_omni_v2.rs01_mpc_teacher import Rs01MpcTeacher
from legged_gym.utils import get_args, task_registry

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from evaluate_rs01_amp_core import sustained_events

JOINT_NAMES = [l + '_' + j + '_joint' for l in ('FL', 'FR', 'RL', 'RR') for j in ('hip', 'thigh', 'calf')]
TARGET_RATE_LIMIT_RAD_S = np.array([2.0, 2.6, 3.2] * 4)
BANDS = ['march', 'forward', 'backward', 'left', 'right', 'turn_left', 'turn_right',
         'forward_left', 'forward_right', 'combined']
KINEMATIC_BANDS = [[0, 0, 0], [.25, 0, 0], [-.18, 0, 0], [0, .12, 0], [0, -.12, 0],
                   [0, 0, .45], [0, 0, -.45], [.18, .08, 0], [.18, -.08, 0], [.15, .06, .3]]
# A learned mincing gait only lifts a few centimetres, so the reference must not be held to
# template heights -- it is held to "every foot demonstrably leaves the ground".
MIN_LIFT_MM = 5.
# Acceptance asks whether every foot *repeatedly* leaves the ground, so one fluke lift is not
# enough evidence for an expert clip.
MIN_LIFTOFFS = 3
MIN_ALTERNATIONS = 4
MAX_SUSTAINED_RATE_RATIO = 1.15
MAX_RAW_TORQUE_NM = 17.
# The AMP runner replays the reference through the env's own gait clock, so a clip is only
# usable if it was stepped at the tempo that clock commands for its speed.
MAX_TEMPO_ERROR = .05
MAX_DUTY_ERROR = .10


def style_features(env):
    """Same 46D layout as Rs01AMPRobot.amp_features, so any RS01 env can be a recorder."""
    ids = [env.foot_slot_by_leg[l] for l in ('FL', 'FR', 'RL', 'RR')]
    relative = env.feet_pos[:, ids] - env.root_states[:, None, :3]
    quat = env.base_quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    feet = quat_rotate_inverse(quat, relative.reshape(-1, 3)).reshape(env.num_envs, 12)
    height = (env.root_states[:, 2] - env.env_origins[:, 2])[:, None]
    return torch.cat((env.dof_pos, env.dof_vel, feet, env.base_lin_vel,
                      env.base_ang_vel, env.projected_gravity, height), dim=1)


def band_commands(spec):
    """Command bands either from an existing reference manifest or the kinematic set."""
    if spec == 'kinematic':
        return torch.tensor(KINEMATIC_BANDS, dtype=torch.float32)
    manifest = json.loads((Path(spec)/'manifest.json').read_text())
    return torch.tensor([c['command'] for c in manifest['clips']], dtype=torch.float32)


def audit_clip(height, contact, velocity, command, duty):
    """Physical gait evidence for one clip: lift, alternation, measured speed."""
    swing = (height > .008) & ~contact.astype(bool)
    pairs = [(contact == np.array(a)).all(-1) for a in ([1, 0, 0, 1], [0, 1, 1, 0])]
    events = sorted([(int(t), 0) for t in sustained_events(pairs[0], 2)]
                    + [(int(t), 1) for t in sustained_events(pairs[1], 2)])
    return dict(
        command=[round(float(v), 4) for v in command],
        measured_velocity=[round(float(velocity[:, 0].mean()), 4), round(float(velocity[:, 1].mean()), 4),
                           round(float(velocity[:, 2].mean()), 4)],
        foot_height_p95_mm=[round(float(v)*1000, 1) for v in np.quantile(height.reshape(-1, 4), .95, axis=0)],
        liftoffs_per_foot=[len(sustained_events(swing[:, leg])) for leg in range(4)],
        diagonal_alternations=sum(events[i][1] != events[i-1][1] for i in range(1, len(events))),
        four_contact_ratio=round(float((contact.sum(-1) == 4).mean()), 4),
        stance_ratio=round(float(contact.astype(float).mean()), 4),
        stance_ratio_target=duty)


def build_clips(features, height, contact, root, root_quat, phase, torque, velocity,
                commands, tempos, per_command, dt, duty, source):
    """Candidate clip per command-band environment; the stored command is the speed actually driven."""
    window = slice(int(2/dt), features.shape[0])
    clips, audits, rejects = [], [], []
    for band, name in enumerate(BANDS):
        for slot in range(per_command):
            env_id = band*per_command + slot
            x, lift = features[window, env_id], height[window, env_id]
            cycle = np.diff(phase[window, env_id], axis=0)
            cadence = float(np.where(cycle < -.5, cycle + 1., cycle).mean()/dt)
            results = audit_clip(lift, contact[window, env_id], velocity[window, env_id],
                                 commands[band], duty)
            measured = np.round(np.array(results['measured_velocity'], dtype=np.float32), 4)
            commanded = np.asarray(commands[band], dtype=np.float32)
            stored = np.where(np.abs(commanded) > 1e-9, measured, 0.0).round(4)
            # Only sustained joint rates have to fit the target-rate limits; one frame can spike
            # while a foot slaps down.
            ratio = np.abs(x[:, 12:24]) / TARGET_RATE_LIMIT_RAD_S
            results.update(clip=name + ('' if per_command == 1 else '_' + str(slot)),
                           band=name, source=source, frames=int(len(x)), physically_executed=True,
                           cadence_hz=round(cadence, 4), tempo_hz=round(float(tempos[band]), 4),
                           rate_ratio_max=round(float(ratio.max()), 3),
                           rate_ratio_p95=round(float(np.quantile(ratio, .95)), 3),
                           raw_peak_nm=round(float(np.abs(torque[window, env_id]).max()), 2),
                           command=stored.tolist())
            results['rejected'] = rejection(results)
            if results['rejected']:
                rejects.append(results)
                continue
            clips.append(dict(clip=results['clip'], amp_features=x,
                              transition_valid=np.ones(len(x), bool),
                              command=np.tile(stored, (len(x), 1)), requested_command=commanded,
                              joint_pos_rad=x[:, 0:12], joint_vel_rad_s=x[:, 12:24], foot_height_m=lift,
                              root_pos_world_m=root[window, env_id], root_quat_wxyz=root_quat[window, env_id],
                              # The env integrates one clock and the second diagonal runs half a cycle
                              # behind it, so the stored per-leg phases stay in the FL,FR,RL,RR order
                              # the reference consumers read (column 0 is the clock itself).
                              leg_phase=np.mod(phase[window, env_id][:, None]
                                               + np.array([0., .5, .5, 0.]), 1.)))
            audits.append(results)
    return clips, audits, rejects


def rejection(audit):
    """Name the reason a demonstration clip cannot be an expert, or None if it can."""
    if abs(audit['cadence_hz'] - audit['tempo_hz']) > MAX_TEMPO_ERROR*audit['tempo_hz']:
        return 'stepped at %.2fHz while this env clock commands %.2fHz' % (
            audit['cadence_hz'], audit['tempo_hz'])
    if abs(audit['stance_ratio'] - audit['stance_ratio_target']) > MAX_DUTY_ERROR:
        return 'stance %.2f against the %.2f template duty' % (
            audit['stance_ratio'], audit['stance_ratio_target'])
    if audit['rate_ratio_p95'] > MAX_SUSTAINED_RATE_RATIO:
        return 'sustained joint rate %.2fx the target-rate limit' % audit['rate_ratio_p95']
    if audit['raw_peak_nm'] >= MAX_RAW_TORQUE_NM:
        return 'raw torque peak %.1f Nm' % audit['raw_peak_nm']
    if audit['clip'].startswith('march'):
        return None
    if min(audit['foot_height_p95_mm']) < MIN_LIFT_MM:
        return 'lowest foot p95 lift %.1f mm' % min(audit['foot_height_p95_mm'])
    if min(audit['liftoffs_per_foot']) < MIN_LIFTOFFS:
        return 'a foot never left the ground (%s)' % audit['liftoffs_per_foot']
    if audit['diagonal_alternations'] < MIN_ALTERNATIONS:
        return 'only %d diagonal alternations' % audit['diagonal_alternations']
    return None


def record(env, drive, commands, duration_s):
    ids = [env.foot_slot_by_leg[l] for l in ('FL', 'FR', 'RL', 'RR')]
    feats, heights, contacts, roots, quats, phases, torques, vels = ([] for _ in range(8))
    with torch.no_grad():
        for _ in range(round(duration_s/env.dt)):
            env.commands[:, :3] = commands.to(env.device)
            feats.append(style_features(env).cpu().numpy())
            heights.append((env.feet_pos[:, ids, 2] - env.env_origins[:, None, 2]
                            - env.cfg.rewards.foot_collision_radius_m).cpu().numpy())
            contacts.append(env.get_foot_contact_mask()[:, ids].cpu().numpy())
            roots.append(env.root_states[:, 0:3].cpu().numpy())
            quats.append(env.root_states[:, 3:7].cpu().numpy())
            phases.append(env._gait_phase().cpu().numpy())
            torques.append(env.raw_pd_torques.cpu().numpy())
            vels.append(torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1).cpu().numpy())
            env.step(drive())
    return [np.stack(x) for x in (feats, heights, contacts, roots, quats, phases, torques, vels)]


def commanded_tempos(env, commands):
    """Tempo the env's gait clock settles at for each constant command row."""
    saved = env.commands[:, :3].clone()
    out = []
    for row in commands:
        env.commands[:, :3] = row.to(env.device)
        out.append(float(env._command_gait_frequency()[0]))
    env.commands[:, :3] = saved
    return np.array(out)


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--record', type=str, default=None)
    own.add_argument('--commands_from', type=str, default='kinematic')
    own.add_argument('--duration_s', type=float, default=12.)
    own.add_argument('--pin_cadence', type=float, default=None,
                     help='force one gait clock for every band instead of the env command map')
    own.add_argument('--per_command', type=int, default=1,
                     help='parallel copies per band, so a band keeps its best lifting clip')
    own.add_argument('--policy', type=str, default=None,
                     help='comma separated dir:checkpoint demonstrators, e.g. logs/run/model:37000')
    known, rest = own.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()

    cfg, train = task_registry.get_cfgs(args.task)
    commands = band_commands(known.commands_from)
    n = len(commands)*known.per_command
    _set_nominal_eval_cfg(cfg, known.duration_s + 4, n)
    cfg.noise.noise_level = 0.
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)

    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(round(3./env.dt)):
        env.step(zero)
    env.gait_enable[:] = 1.
    env._resample_commands = lambda env_ids: None
    if known.pin_cadence:
        env._command_gait_frequency = lambda: torch.full(
            (env.num_envs,), known.pin_cadence, device=env.device)
        if hasattr(env, 'v25_frequency_hz'):
            env.v25_frequency_hz[:] = known.pin_cadence

    sources = []
    if known.policy:
        for spec in known.policy.split(','):
            path, _, checkpoint = spec.rpartition(':')
            train.runner.resume = True
            train.runner.load_run = path
            train.runner.checkpoint = int(checkpoint)
            train.runner.amp_enabled = False
            runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                      train_cfg=train, log_root=None)
            policy = runner.get_inference_policy(device=env.device)
            tag = '%s@%s' % (Path(path).name, checkpoint)
            sources.append((tag, lambda policy=policy: policy(env.get_observations())))
    else:
        teacher = Rs01MpcTeacher(env, gains=dict(duty=cfg.rewards.gait_stance_ratio))
        sources.append(('diagonal-VMC teacher', teacher.actions))

    kept, best_reject = {}, {}
    tempos = commanded_tempos(env, commands)
    print('env gait clock for these commands: %s' % np.round(tempos, 3))
    per_env_commands = commands.repeat_interleave(known.per_command, 0)
    for tag, drive in sources:
        stacked = record(env, drive, per_env_commands, known.duration_s)
        clips, audits, rejects = build_clips(*stacked, commands.numpy(), tempos,
                                             known.per_command, env.dt,
                                             cfg.rewards.gait_stance_ratio, tag)
        for clip, audit in zip(clips, audits):
            score = min(audit['foot_height_p95_mm']) if audit['band'] != 'march' else 1e6
            if audit['band'] not in kept or score > kept[audit['band']][1]:
                kept[audit['band']] = (clip, score, audit)
        for audit in rejects:
            band = audit['band']
            if band not in best_reject or min(audit['foot_height_p95_mm']) > min(
                    best_reject[band]['foot_height_p95_mm']):
                best_reject[band] = audit
        tempo = [a['cadence_hz'] for a in audits]
        print('%s: %d/%d bands feasible, clip tempo %.2f-%.2fHz, duty %.3f'
              % (tag, len(audits), len(BANDS),
                 min(tempo) if tempo else 0., max(tempo) if tempo else 0.,
                 np.median([a['stance_ratio'] for a in audits])))
        for audit in audits:
            print('  %-14s cmd %s -> measured %s | p95 lift mm %s | liftoffs %s | alternations %d'
                  ' | raw peak %.1f Nm | sustained rate %.2f (peak %.2f) | %.2fHz duty %.2f'
                  % (audit['clip'], np.round(audit['command'], 4),
                     np.round(audit['measured_velocity'], 4), np.round(audit['foot_height_p95_mm'], 1),
                     audit['liftoffs_per_foot'], audit['diagonal_alternations'], audit['raw_peak_nm'],
                     audit['rate_ratio_p95'], audit['rate_ratio_max'], audit['cadence_hz'],
                     audit['stance_ratio']))

    missing = [name for name in BANDS if name not in kept]
    for band in missing:
        audit = best_reject.get(band)
        if audit:
            print('  %-14s NO CLIP: %s (cmd %s, measured %s, p95 lift mm %s, liftoffs %s,'
                  ' alternations %d, raw peak %.1f Nm, sustained rate %.2f)'
                  % (band, audit['rejected'], np.round(audit['command'], 4),
                     np.round(audit['measured_velocity'], 4), np.round(audit['foot_height_p95_mm'], 1),
                     audit['liftoffs_per_foot'], audit['diagonal_alternations'], audit['raw_peak_nm'],
                     audit['rate_ratio_p95']))
    gate = bool(not missing)
    print('reference feasibility gate: every command band has a physically executed clip at this'
          ' env clock, with all four feet lifting >=%dmm >=%d times, diagonals alternating,'
          ' sustained rates inside the target-rate limits and torque <%dNm: %s%s'
          % (MIN_LIFT_MM, MIN_LIFTOFFS, MAX_RAW_TORQUE_NM, gate,
             '' if gate else ' (missing: %s)' % ','.join(missing)))

    if known.record and gate:
        out = Path(known.record)
        out.mkdir(parents=True, exist_ok=True)
        for name in BANDS:
            clip = kept[name][0]
            np.savez_compressed(out/(name + '.npz'),
                                **{k: v for k, v in clip.items() if k != 'clip'})
        manifest = dict(kind='recorded_isaacgym_plant', source=args.task,
                        # Passing the feasibility gate IS the approval: these frames were
                        # executed by this plant, not drawn from a kinematic template.
                        amp_training_approved=True, frame_dt_s=.02, joint_order=JOINT_NAMES,
                        settings=dict(seconds=known.duration_s-2, fps=round(1/env.dt),
                                      frequency=float(tempos[0]), period_s=cfg.rewards.gait_period_s,
                                      duty=cfg.rewards.gait_stance_ratio,
                                      command_dependent_clock=True,
                                      demonstrator=','.join(tag for tag, _ in sources),
                                      minimum_foot_lift_mm=MIN_LIFT_MM),
                        clips=[dict(clip=name, frames=kept[name][0]['amp_features'].shape[0],
                                    command=kept[name][2]['command'],
                                    measured_velocity_m_s=kept[name][2]['measured_velocity'],
                                    cadence_hz=kept[name][2]['cadence_hz'],
                                    tempo_hz=kept[name][2]['tempo_hz'],
                                    stance_ratio=kept[name][2]['stance_ratio'],
                                    foot_height_p95_mm=kept[name][2]['foot_height_p95_mm'],
                                    liftoffs_per_foot=kept[name][2]['liftoffs_per_foot'],
                                    diagonal_alternations=kept[name][2]['diagonal_alternations'],
                                    rate_ratio_max=kept[name][2]['rate_ratio_max'],
                                    rate_ratio_p95=kept[name][2]['rate_ratio_p95'],
                                    raw_peak_nm=kept[name][2]['raw_peak_nm'],
                                    demonstrated_by=kept[name][2]['source'],
                                    physically_executed=True) for name in BANDS],
                        feasibility_gate_passed=gate, created=time.strftime('%Y-%m-%dT%H:%M:%S'))
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('wrote', out, len(BANDS), 'clips; amp.commands for the task config:',
              json.dumps([kept[name][2]['command'] for name in BANDS]))
        print('the task config must keep this recording env clock: gait_period_s = %r,'
              ' gait_stance_ratio = %r, amp.command_dependent_clock = True'
              % (cfg.rewards.gait_period_s, cfg.rewards.gait_stance_ratio))
    elif known.record:
        print('not written: feasibility gate failed')


if __name__ == '__main__':
    main()
