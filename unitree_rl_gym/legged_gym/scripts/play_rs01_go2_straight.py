"""Play a checkpoint from the independent RS01 Go2-style straight task."""

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - performs task registration
from legged_gym.utils import get_args, task_registry


SUPPORTED_TASKS = {
    "rs01_go2_straight",
    "rs01_go2_straight_kp40",
    "rs01_go2_straight_kp40_polish",
    "rs01_go2_straight_path_polish",
    "rs01_go2_straight_rear_coord",
    "rs01_go2_sim2sim_adapt",
    "rs01_go2_sim2sim_calf_repair",
    "rs01_go2_sim2sim_kd050",
    "rs01_go2_sim2sim_robust",
    "rs01_go2_sim2sim_matched_transfer",
    "rs01_go2_sim2sim_heading52",
}


def play(args):
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            "This player only supports --task "
            + " or ".join(sorted(SUPPORTED_TASKS))
        )

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    playback_speed = float(
        getattr(env_cfg.commands, "playback_speed_mps", 0.4)
    )
    env_cfg.commands.ranges.lin_vel_x = [
        playback_speed,
        playback_speed,
    ]
    env_cfg.commands.resampling_time = 1000.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
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
