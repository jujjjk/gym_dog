"""View an RS01 omni checkpoint under an explicit velocity command."""

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


SUPPORTED_TASKS = {
    "rs01_omni_v10_recovery",
    "rs01_omni_v10_recovery_strong",
    "rs01_go2_omni_diagonal",
    "rs01_omni_v2",
    "rs01_omni_v3_contact1",
    "rs01_omni_v3_contact15",
    "rs01_omni_v4_contact1",
    "rs01_omni_v4_contact15",
    "rs01_omni_v5_odd05",
    "rs01_omni_v5_odd10",
    "rs01_omni_v6_seed08",
    "rs01_omni_v6_seed11",
    "rs01_omni_v7_seed14",
    "rs01_omni_v7_seed18",
    "rs01_omni_v8_clearance2",
    "rs01_omni_v8_clearance4",
    "rs01_omni_v9_speed10",
    "rs01_omni_v9_speed14",
}


def _set_nominal_playback_cfg(env_cfg, duration_s):
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = duration_s + 10.0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_rs01_actuator = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = duration_s + 10.0


def play(args):
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"This player requires one of --task={sorted(SUPPORTED_TASKS)}"
        )
    if args.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    _set_nominal_playback_cfg(env_cfg, args.duration_s)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = runner.get_inference_policy(device=env.device)
    command = torch.tensor(
        [args.vx, args.vy, args.wz], device=env.device, dtype=torch.float
    )
    print(
        "Fixed command: "
        f"vx={args.vx:.3f} m/s, vy={args.vy:.3f} m/s, "
        f"wz={args.wz:.3f} rad/s, march={args.march}, "
        f"duration={args.duration_s:.1f} s"
    )

    with torch.no_grad():
        for _ in range(max(1, int(round(args.duration_s / env.dt)))):
            # reset_idx and the base callback may resample a command. Override
            # it before every observation so playback is truly deterministic.
            moving = bool(torch.linalg.vector_norm(command).item() > 0.0)
            if hasattr(env, "set_evaluation_command"):
                env.set_evaluation_command(command, float(args.march or moving))
            else:
                env.commands[:, :3] = command
                if hasattr(env, "gait_enable"):
                    env.gait_enable[:] = float(args.march or moving)
            env.compute_observations()
            observations = env.get_observations()
            env.step(policy(observations))


if __name__ == "__main__":
    play(get_args())
