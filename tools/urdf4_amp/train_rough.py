"""Train the rough-ground continuation and validate the chosen warm start."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/urdf4_rough_20261118'
SEED = ROOT / 'artifacts/urdf4_robust050_selection_20261118/best/model_100.pt'
sys.path.insert(0, str(ROOT / 'unitree_rl_gym/legged_gym/scripts'))

import isaacgym
import torch
import legged_gym.envs  # Registers the rough-ground task.
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_rough import Urdf4AMPRoughRunner


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--amplitude', type=float, help='Rough height envelope in metres for the smoke run')
    parser.add_argument('--cross-yaw-gain', type=float)
    parser.add_argument('--cross-yaw-damping', type=float)
    own, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    args = get_args()
    args.task, args.headless = 'urdf4_amp_cts_rough', True
    args.num_envs = args.num_envs or (64 if own.smoke else 4096)
    total = args.max_iterations or (32 if own.smoke else 30000)
    if own.smoke and not 20 <= total <= 100:
        parser.error('Smoke run requires 20–100 PPO updates')
    cfg, train = task_registry.get_cfgs(args.task)
    if own.smoke:
        output = own.output or BASE / 'smoke'
        train.runner.save_interval = total
        if own.amplitude is not None:
            if not 0 < own.amplitude <= .02:
                parser.error('Rough amplitude must be between 0 and 0.02 m')
            cfg.rough.amplitude_m = own.amplitude
        if own.cross_yaw_gain is not None:
            cfg.rough.fast_cross_yaw_gain = own.cross_yaw_gain
        if own.cross_yaw_damping is not None:
            cfg.rough.fast_cross_yaw_damping = own.cross_yaw_damping
    else:
        # Gym's Python 3.8 environment does not provide zoneinfo.
        from datetime import timedelta, timezone
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%b%d_%H-%M-%S')
        output = own.output or ROOT / 'unitree_rl_gym/logs/urdf4_amp_cts_rough' / (stamp + '_rough')
        receipt = json.loads(Path(train.runner.v3_preflight).read_text())
        for key, value in receipt['contract']['rough_ground'].items():
            setattr(cfg.rough, key, value)
    output.mkdir(parents=True, exist_ok=False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner = Urdf4AMPRoughRunner(env, class_to_dict(train), str(output), device=args.rl_device)
    if own.smoke:
        runner.warm_start(SEED)
        runner.save(str(output / 'model_0.pt'))
        actor = runner.alg.actor_critic
        before = actor.actor[-1].weight.detach().clone()
        runner.learn(total, init_at_random_ep_len=False)
        with torch.no_grad():
            public = env.obs_buf.clone()
            action = actor.act_inference(public)
            public[:, actor.public_size:] = 0.
            assert torch.equal(action, actor.act_inference(public))
            assert torch.isfinite(action).all() and torch.isfinite(env.rew_buf).all()
        assert not torch.equal(before, actor.actor[-1].weight)
        assert 0. < float(env.rough_env.float().mean()) < 1.
        assert float(env.terrain_heights.std()) > .0005
        assert runner.last_metrics['amp_valid_fraction'] > .95
        assert float(env.push_count.sum()) > 0 and env.curriculum_stage == 1
        assert float(env.motor_torques.abs().max()) <= 14.0001
        report = dict(passed=True, iteration=total, num_envs=env.num_envs, phase='training_integration',
                      seed=str(SEED), terrain=runner.contract['rough_ground'],
                      terrain_height_bounds_m=[float(env.terrain_heights.min()), float(env.terrain_heights.max())],
                      rough_fraction=float(env.rough_env.float().mean()), push_count=float(env.push_count.sum()),
                      metrics=runner.last_metrics)
        (output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print('ROUGH INTEGRATION PASSED', output, flush=True)
        return
    checkpoint = Path(receipt['checkpoint'])
    runner.load(checkpoint)
    if not receipt.get('approved_for_training') or not receipt.get('rough_behavior_passed') or not receipt.get('flat_regression_passed'):
        raise ValueError('Rough and flat behavior checks must pass before launch')
    runner._validate_training_request(total - runner.current_learning_iteration)
    (output / 'launch.json').write_text(json.dumps(dict(task=args.task, checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), start_iteration=runner.current_learning_iteration,
        target_iteration=total, num_envs=env.num_envs), indent=2) + '\n')
    print('ROUGH LONG TRAIN', output, flush=True)
    runner.learn(total - runner.current_learning_iteration, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
