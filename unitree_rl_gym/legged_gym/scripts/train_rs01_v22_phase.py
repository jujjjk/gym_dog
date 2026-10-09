"""Fresh single-task V22 training; never load a checkpoint or seed a gait actor."""
import json
from pathlib import Path
import isaacgym  # Must precede torch.
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg


def main():
    args = get_args()
    if args.task not in ('rs01_v22_phase_scratch', 'rs01_v22_phase_lift') or args.resume:
        raise ValueError('Use a V22 scratch/phase_lift task without --resume')
    cfg, train = task_registry.get_cfgs(args.task)
    original = class_to_dict(Rs01OmniV22Cfg())
    original['seed'] = train.seed  # task_registry.get_cfgs injects this field.
    actual = class_to_dict(cfg)
    assert {k: v for k, v in actual.items() if k not in ('phase_exchange', 'phase_acquisition')} == original
    assert not train.runner.resume
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train)
    assert type(runner).__name__ == 'OnPolicyRunner'
    assert env.num_obs == 61 and env.num_actions == 12
    directory = Path(runner.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'experiment_contract.json').write_text(json.dumps(dict(
        env=class_to_dict(cfg), train=class_to_dict(train), args=vars(args),
        baseline='B23000 V22 config only; NO checkpoint loaded'), indent=2, default=str) + '\n')
    peak_motor = 0.
    with torch.no_grad():
        for _ in range(10):
            obs, _, reward, _, _ = env.step(torch.zeros((env.num_envs, 12), device=env.device))
            assert bool(torch.isfinite(obs).all() and torch.isfinite(reward).all())
            peak_motor = max(peak_motor, float(env.motor_electromagnetic_torques.abs().max()))
            assert peak_motor <= cfg.rs01_actuator.peak_torque_limit_nm + .001
    env.reset()
    print(f'STARTUP PASS: fresh V22, original reward weights, 61D/12D, real RS01; '
          f'motor peak={peak_motor:.3f} Nm', flush=True)
    runner.learn(num_learning_iterations=train.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
