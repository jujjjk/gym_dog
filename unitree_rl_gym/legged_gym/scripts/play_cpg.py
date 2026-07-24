"""Check direct-12 zero action through the measured RS01 motor chain."""

import isaacgym, torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def play_cpg(args):
    if args.task == "dog_rs01_tro":
        args.task = "dog_rs01_trot"
    if args.task != "dog_rs01_trot":
        raise ValueError(
            "play_cpg.py supports dog_rs01_trot (dog_rs01_tro is accepted as an alias)"
        )
    env_cfg, _ = task_registry.get_cfgs(name=(args.task))
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.rewards.enable_non_diagonal_swing_termination = False
    env_cfg.rewards.enable_flight_termination = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gait_phase_on_reset = False
    env_cfg.control.use_real_actuator_model = True
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.stand_probability = 1.0
    env_cfg.commands.resampling_time = 1000.0
    env, _ = task_registry.make_env(name=(args.task), args=args, env_cfg=env_cfg)
    print(
        f"[dog_rs01_trot] mode: direct 12-output policy (use_rs01_diagonal_cpg={env.cfg.control.use_rs01_diagonal_cpg}, period={env.cfg.rewards.gait_period:.3f}s, duty={env.cfg.rewards.gait_stance_ratio:.2f}). Zero action = stand; measured RS01 motor model enabled."
    )
    zero = torch.zeros((env.num_envs), (env.num_actions), device=(env.device))
    for _ in range(5 * int(env.max_episode_length)):
        env.step(zero)


if __name__ == "__main__":
    play_cpg(get_args())
