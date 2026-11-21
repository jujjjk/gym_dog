"""Smoke or continue AMP+CTS for scheduled variable-speed locomotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import torch

import legged_gym.envs
from legged_gym.algorithms.urdf4_amp_transition_training import Urdf4AMPTransitionTrainingRunner
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.urdf4.amp_transition_training import restore_transition_contract

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / 'artifacts/urdf4_rough_selection_20261119/best/model_2500.pt'


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--transition-reference', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--prepare', action='store_true', help='Write a warm-start checkpoint for behavior verification')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--verification', type=Path, nargs=2, metavar=('ROUGH', 'FLAT'))
    parser.add_argument('--mujoco-verification', type=Path, nargs=4,
                        metavar=('ROUGH_A', 'ROUGH_B', 'FLAT_A', 'FLAT_B'))
    own, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    args = get_args()
    args.task, args.headless = 'urdf4_amp_cts_transition_training', True
    continuing = own.resume_from is not None
    args.num_envs = args.num_envs or (4096 if continuing else 64)
    iterations = args.max_iterations or (30000 if continuing else 72)
    if continuing != bool(own.verification):
        parser.error('--resume-from and --verification must be supplied together')
    if own.mujoco_verification and not continuing:
        parser.error('--mujoco-verification applies to checkpoint continuation')
    if continuing and (own.prepare or own.initialize_from):
        parser.error('Preparation and warm starts apply before checkpoint verification')
    if not continuing and (not 60 <= iterations <= 120 or args.num_envs < 16):
        parser.error('Use 60–120 updates and at least 16 environments for full-sequence smoke')
    cfg, train = task_registry.get_cfgs(args.task)
    seed = (own.resume_from if continuing else (own.initialize_from or SEED)).resolve()
    saved = torch.load(seed, map_location='cpu')
    for key, value in saved['contract']['rough_ground'].items():
        setattr(cfg.rough, key, value)
    train.runner.save_interval = 12
    if own.control:
        cfg.transition_control = json.loads(own.control.read_text())
        train.algorithm.learning_rate = 3.e-6
        train.runner.encoder_learning_rate = 2.5e-5
        train.runner.seed_trust_coef = 4.
        train.runner.seed_action_deadband = .02
        train.runner.transition_sampling_std_max = .035
    if own.transition_reference:
        train.runner.transition_reference = str(own.transition_reference.resolve())
    if continuing:
        if own.control or own.transition_reference or 'transition_control' not in saved['contract']:
            parser.error('Long training inherits the verified transition checkpoint configuration')
        restore_transition_contract(cfg, train, saved)
        initial_speed = cfg.transition_control.get('training_start_forward_m_s', .50)
        train.algorithm.learning_rate = saved['experiment']['train']['algorithm']['learning_rate']
        train.runner.encoder_learning_rate = saved['experiment']['train']['runner']['encoder_learning_rate']
        train.runner.save_interval = 250
        train.runner.v3_preflight = str((own.output / 'preflight.json').resolve())
        checkpoint_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
        verified_sources = []
        for surface_flat, file in zip((False, True), own.verification):
            measured = json.loads(file.read_text())
            if not (measured.get('fast_windows_passed') and measured.get('axis') == 'forward'
                    and measured.get('flat') is surface_flat and measured.get('copies', 0) >= 2
                    and measured.get('segment_s', 0) >= 5.99
                    and abs(measured.get('peak_speed_m_s', .50)-initial_speed) < 1.e-6
                    and measured.get('checkpoint_sha256') == checkpoint_sha
                    and all(case['resets'] == 0 for case in measured['cases'])):
                parser.error(f'Verification does not qualify the selected checkpoint: {file}')
            verified_sources.append(measured.get('runtime_source_sha256'))
        if 'coordination' in cfg.transition_control and not own.mujoco_verification:
            parser.error('The coordinated checkpoint requires two MuJoCo origins per terrain')
        verified_origins = {'rough': set(), 'flat': set()}
        for terrain, file in zip(('rough', 'rough', 'flat', 'flat'), own.mujoco_verification or ()):
            measured = json.loads(file.read_text())
            origin = tuple(measured.get('offset_m', ()))
            if not (measured.get('fast_windows_passed') and measured.get('completed_all')
                    and measured.get('suite') == 'forward' and measured.get('terrain') == terrain
                    and measured.get('requested_segment_s', 0) >= 5.99
                    and abs(measured.get('peak_speed_m_s', 0.)-initial_speed) < 1.e-6
                    and measured.get('sha256') == checkpoint_sha
                    and len(origin) == 2 and origin not in verified_origins[terrain]):
                parser.error(f'MuJoCo verification does not qualify the selected checkpoint: {file}')
            verified_origins[terrain].add(origin)
    own.output.mkdir(parents=True, exist_ok=False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner = Urdf4AMPTransitionTrainingRunner(env, class_to_dict(train), str(own.output), device=args.rl_device)
    if continuing:
        if any(source != runner.experiment['source_sha256'] for source in verified_sources):
            parser.error('Runtime sources changed since behavior verification')
        runner.load(seed)
        remaining = iterations - runner.current_learning_iteration
        if remaining <= 0:
            parser.error('Target iteration must be above the verified checkpoint')
        receipt = dict(approved_for_training=True, checkpoint_sha256=checkpoint_sha, contract=runner.contract,
                       source_sha256=runner.experiment['source_sha256'],
                       verification=[str(file.resolve()) for file in own.verification])
        if own.mujoco_verification:
            receipt['mujoco_verification'] = [str(file.resolve()) for file in own.mujoco_verification]
        (own.output / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
        launch = dict(checkpoint=str(seed), start_iteration=runner.current_learning_iteration,
                      target_iteration=iterations, num_envs=env.num_envs, device=str(env.device),
                      initial_forward_limit_m_s=initial_speed, final_forward_limit_m_s=.50)
        (own.output / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        print('TRANSITION LONG TRAIN', own.output, flush=True)
        runner.learn(remaining, init_at_random_ep_len=False)
        return
    runner.warm_start(seed)
    if own.control:
        runner.alg.actor_critic.std.data.clamp_(max=train.runner.transition_sampling_std_max)
    runner.save(str(own.output / 'model_0.pt'))
    if own.prepare:
        return
    actor = runner.alg.actor_critic
    before = actor.actor[-1].weight.detach().clone()
    initial_transition_updates = getattr(runner.style, 'transition_updates', 0)
    runner.learn(iterations, init_at_random_ep_len=False)
    rows = [json.loads(line) for line in (own.output / 'metrics.jsonl').read_text().splitlines()]
    samples = sum(row['transition_samples'] for row in rows)
    fast_samples = sum(row['fast_transition_samples'] for row in rows)
    assert samples > 0 and fast_samples > 0
    reference_added = runner.style.transition_sha is not None
    updates = getattr(runner.style, 'transition_updates', 0)
    forward_limit = env.training_forward_limit
    if reference_added and forward_limit > .485:
        assert updates > initial_transition_updates, 'The fast transition discriminator received no updates'
        assert min(rows[-1]['transition_accel_replay'], rows[-1]['transition_brake_replay']) > 0, (
            'The smoke must include accelerating and braking policy examples')
    assert not torch.equal(before, actor.actor[-1].weight)
    with torch.no_grad():
        obs = env.obs_buf.clone()
        action = actor.act_inference(obs)
        assert torch.isfinite(action).all() and torch.isfinite(env.rew_buf).all()
        obs[:, actor.public_size:] = 0.
        assert torch.equal(action, actor.act_inference(obs))
    report = dict(integration_passed=True, behavior_accepted=False, iterations=iterations, num_envs=env.num_envs,
                  seed=str(seed), final_checkpoint=str((own.output / f'model_{iterations}.pt').resolve()),
                  student_transition_samples=samples, student_fast_transition_samples=fast_samples,
                  student_transition_falls=sum(row['transition_falls'] for row in rows),
                  max_transition_roll_deg=max(row['transition_roll_peak_deg'] for row in rows),
                  max_transition_pitch_deg=max(row['transition_pitch_peak_deg'] for row in rows),
                  forward_training_limit_m_s=forward_limit,
                  fast_transition_amp_reference_added=reference_added, transition_reference_updates=updates,
                  new_transition_reference_updates=updates-initial_transition_updates,
                  transition_reference_rows=runner.style.transition_rows, final_metrics=rows[-1])
    if env.torque_burst is not None:
        exposed = sum(row['transition_burst_exposed_samples'] for row in rows)
        assert exposed > 0, 'No accelerated or braking exchanges used the configured burst window'
        report.update(transition_burst_exposed_samples=exposed,
                      transition_burst_used_samples=sum(row['transition_burst_used_samples'] for row in rows),
                      transition_measured_motor_peak_nm=max(row['transition_measured_motor_peak_nm'] for row in rows))
    (own.output / 'integration.json').write_text(json.dumps(report, indent=2) + '\n')
    print('TRANSITION INTEGRATION PASSED', own.output, flush=True)


if __name__ == '__main__':
    main()
