import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np, torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=(args.task))
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if args.task in (
        "dog_rs01_trot",
        "dog_rs01_balance",
        "dog_rs01_body_stable",
        "dog_rs01_low_twist",
        "dog_rs01_hip_torque",
        "dog_rs01_straight_balance",
        "dog_rs01_compact_hip",
        "dog_rs01_safe6nm",
        "dog_rs01_safe6nm_v2",
        "dog_rs01_smooth_straight",
        "dog_rs01_smooth_straight_v2",
        "dog_rs01_straight_guarded",
    ):
        env_cfg.env.num_envs = 1
        env_cfg.commands.ranges.lin_vel_x = [0.15, 0.15]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.stand_probability = 0.0
        env_cfg.commands.pure_sagittal_probability = 1.0
        env_cfg.commands.resampling_time = 1000.0
        env_cfg.domain_rand.randomize_gait_phase_on_reset = False
        if args.task in ("dog_rs01_safe6nm", "dog_rs01_safe6nm_v2"):
            env_cfg.commands.ranges.lin_vel_x = [0.12, 0.12]
        elif args.task in (
            "dog_rs01_smooth_straight",
            "dog_rs01_smooth_straight_v2",
            "dog_rs01_straight_guarded",
        ):
            env_cfg.commands.ranges.lin_vel_x = [0.16, 0.16]
        env_cfg.env.test = True
        env, _ = task_registry.make_env(name=(args.task), args=args, env_cfg=env_cfg)
        obs = env.get_observations()
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=(args.task), args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=(env.device))
        if EXPORT_POLICY:
            path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "policies",
            )
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print("Exported policy as jit script to: ", path)
        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
