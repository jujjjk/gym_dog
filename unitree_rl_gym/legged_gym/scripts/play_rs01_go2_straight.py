"""Play a checkpoint from the independent RS01 Go2-style straight task."""

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - performs task registration
from legged_gym.utils import get_args, task_registry


TASK_NAME = "rs01_go2_straight"


def play(args):
    if args.task != TASK_NAME:
        raise ValueError(f"This player only supports --task {TASK_NAME}")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [0.4, 0.4]
    env_cfg.commands.resampling_time = 1000.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    obs, _ = env.reset()

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)

    for _ in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play(get_args())
