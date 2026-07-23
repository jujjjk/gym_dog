"""Preview the deterministic RS01 CPG with an exactly zero policy residual."""

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def play_cpg(args):
    if args.task != "dog_rs01_trot":
        raise ValueError("play_cpg.py currently supports only dog_rs01_trot")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    # Preview the continuous physical CPG instead of restarting the viewer on
    # a brief threshold-level contact mismatch. Full flight remains a hard
    # one-sample termination, as do body collision and joint safety limits.
    env_cfg.env.test = False
    env_cfg.rewards.enable_non_diagonal_swing_termination = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gait_phase_on_reset = False
    env_cfg.commands.ranges.lin_vel_x = [0.15, 0.15]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.stand_probability = 0.0
    env_cfg.commands.pure_sagittal_probability = 1.0
    env_cfg.commands.resampling_time = 1000.0

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    zero_residual = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    for _ in range(10 * int(env.max_episode_length)):
        env.step(zero_residual)


if __name__ == "__main__":
    play_cpg(get_args())
