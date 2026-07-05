"""Show the omni policy on 18 robots covering nine command modes."""
import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


COMMANDS = (
    (0.20, 0.00, 0.00), (-0.08, 0.00, 0.00),
    (0.00, 0.05, 0.00), (0.00, -0.05, 0.00),
    (0.00, 0.00, 0.50), (0.00, 0.00, -0.50),
    (0.15, 0.00, 0.35), (-0.06, 0.00, 0.25),
    (0.12, 0.05, 0.00),
)

FAST_COMMANDS = (
    (0.35, 0.00, 0.00), (-0.10, 0.00, 0.00),
    (0.00, 0.07, 0.00), (0.00, -0.07, 0.00),
    (0.00, 0.00, 0.70), (0.00, 0.00, -0.70),
    (0.25, 0.00, 0.50), (-0.10, 0.00, 0.35),
    (0.20, 0.07, 0.00),
)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs or 18
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 6
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    matrix = FAST_COMMANDS if args.task == "fanfan_omni_fast" else COMMANDS
    commands = torch.tensor(matrix, device=env.device).repeat_interleave(2, 0)
    commands = commands[:env.num_envs]
    if len(commands) < env.num_envs:
        commands = commands.repeat((env.num_envs + len(commands) - 1) // len(commands), 1)[:env.num_envs]
    env.commands[:, 3] = env.rpy[:, 2]
    while True:
        env.commands[:, :3] = commands
        env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)


if __name__ == "__main__":
    play(get_args())
