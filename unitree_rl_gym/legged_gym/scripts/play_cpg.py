"""Preview the deterministic, URDF-derived RS01 CPG with zero residual."""

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def play_cpg(args):
    # Keep compatibility with the shortened task spelling used in shell notes.
    if args.task == "dog_rs01_tro":
        args.task = "dog_rs01_trot"
    if args.task != "dog_rs01_trot":
        raise ValueError(
            "play_cpg.py supports dog_rs01_trot "
            "(dog_rs01_tro is accepted as an alias)"
        )

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
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

    # This viewer validates URDF geometry, phase order and trajectory
    # feasibility.  Random per-motor delay/gain/friction makes each preview
    # different and hides geometry errors, so it remains enabled in training
    # but is disabled here.
    env_cfg.control.use_real_actuator_model = False
    env_cfg.control.compensate_identified_position_gain_in_gait = False

    env_cfg.commands.ranges.lin_vel_x = [0.15, 0.15]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.stand_probability = 0.0
    env_cfg.commands.pure_sagittal_probability = 1.0
    env_cfg.commands.resampling_time = 1000.0

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    trajectory = env.rs01_foot_trajectory
    source = (
        str(trajectory.urdf_path)
        if trajectory.urdf_path is not None
        else "checked fallback vectors"
    )
    print(
        "[RS01 CPG] geometry source:", source,
        "| thigh:", f"{trajectory.thigh_length:.6f} m",
        "| calf:", f"{trajectory.calf_length:.6f} m",
    )

    zero_residual = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    for _ in range(10 * int(env.max_episode_length)):
        env.step(zero_residual)


if __name__ == "__main__":
    play_cpg(get_args())
