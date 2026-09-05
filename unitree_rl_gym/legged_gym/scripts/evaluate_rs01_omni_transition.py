"""Short, deterministic march/motion transition test; does not train."""

import json

import isaacgym  # noqa: F401
import torch

from evaluate_rs01_go2_omni import _set_nominal_eval_cfg, SUPPORTED_TASKS
from legged_gym.utils import get_args, task_registry


def evaluate(args):
    if args.task not in SUPPORTED_TASKS or args.eval_envs <= 0:
        raise ValueError("Use a supported omni task and positive eval_envs")
    stages = (
        ("march", 0.0, 0.0, 0.0),
        ("forward", 0.1, 0.0, 0.0),
        ("march", 0.0, 0.0, 0.0),
        ("left", 0.0, 0.08, 0.0),
        ("right", 0.0, -0.08, 0.0),
        ("yaw_left", 0.0, 0.0, 0.3),
        ("yaw_right", 0.0, 0.0, -0.3),
        ("backward", -0.1, 0.0, 0.0),
        ("march", 0.0, 0.0, 0.0),
    )
    stage_seconds = 4.0
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg, stage_seconds * len(stages), args.eval_envs)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = runner.get_inference_policy(device=env.device)
    env.reset()
    results = []
    finite = True
    with torch.no_grad():
        for name, vx, vy, wz in stages:
            command = torch.tensor([vx, vy, wz], device=env.device)
            samples = []
            resets = 0
            for step in range(round(stage_seconds / env.dt)):
                if hasattr(env, "set_evaluation_command"):
                    env.set_evaluation_command(command, 1.0)
                else:
                    env.commands[:, :3] = command
                    env.gait_enable[:] = 1.0
                env.compute_observations()
                obs = env.get_observations()
                actions = policy(obs)
                obs, _, rewards, dones, _ = env.step(actions)
                resets += int(dones.sum().item())
                finite = finite and all(bool(torch.isfinite(x).all())
                    for x in (obs, actions, rewards))
                if step >= round(1.0 / env.dt):
                    samples.append(torch.cat(
                        (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1
                    ).clone())
            velocity = torch.stack(samples)
            results.append({
                "stage": name,
                "command_vx_vy_wz": command.tolist(),
                "mean_vx_vy_wz": velocity.mean(dim=(0, 1)).tolist(),
                "rmse_vx_vy_wz": ((velocity-command).square().mean(dim=(0, 1))).sqrt().tolist(),
                "resets_total": resets,
            })
    print(json.dumps({
        "task": args.task, "load_run": args.load_run,
        "checkpoint": args.checkpoint, "seed": args.seed,
        "num_envs": args.eval_envs, "stage_duration_s": stage_seconds,
        "measurement_skip_per_stage_s": 1.0, "domain_randomization": False,
        "finite": finite, "stages": results,
    }, indent=2))


if __name__ == "__main__":
    evaluate(get_args())
