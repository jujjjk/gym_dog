import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np, torch


def play_urdf4(args):
    """Native viewer: leave the camera under mouse control throughout playback."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 1
    env_cfg.env.test = True
    cts = args.task == "urdf4_amp_cts_v1"
    if cts:
        env_cfg.amp_cts.student_only = True
    env_cfg.env.episode_length_s = args.duration_s + 10.0
    env_cfg.noise.add_noise = False
    env_cfg.push.enabled = False
    env_cfg.domain_rand.push_robots = False
    for flag in (
        "randomize_friction", "randomize_restitution", "randomize_base_mass",
        "randomize_base_com", "randomize_link_mass", "randomize_armature",
    ):
        setattr(env_cfg.domain_rand, flag, False)
    commands = (
        ("forward", 0.6, 0.0, 0.0),
        ("backward", -0.4, 0.0, 0.0),
        ("left", 0.0, 0.4 if cts else 0.3, 0.0),
        ("right", 0.0, -0.4 if cts else -0.3, 0.0),
        ("turn left", 0.0, 0.0, 1.2 if cts else 1.0),
        ("turn right", 0.0, 0.0, -1.2 if cts else -1.0),
        ("diagonal", 0.4, 0.2, 0.0),
        ("arc", 0.3, 0.0, 0.6),
        ("mixed", 0.3, -0.2, -0.5),
        ("stand", 0.0, 0.0, 0.0),
    ) if args.sweep else (("fixed", args.vx, args.vy, args.wz),)
    if cts and args.sweep and env_cfg.amp_cts.stage == 'v2_style':
        import json
        from pathlib import Path
        manifest=json.loads((Path(env_cfg.amp_cts.reference)/'manifest.json').read_text())
        commands=tuple((c['clip'],*c['command']) for c in manifest['clips'])+(("stand",0.,0.,0.),)
    env_cfg.commands.fixed_command = list(commands[0][1:])
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()
    last_case = None
    with torch.no_grad():
        for step in range(int(round(args.duration_s / env.dt))):
            case = int(step * env.dt / 10.0) % len(commands)
            if case != last_case:
                label, vx, vy, wz = commands[case]
                env.set_command_override(
                    torch.tensor([vx, vy, wz], device=env.device).repeat(env.num_envs, 1),
                    **({"gait_enabled": abs(vx)+abs(vy)+abs(wz)>0 or label=='march' or args.march} if cts else {})
                )
                print(f"[native play] {label}: vx={vx}, vy={vy}, wz={wz}", flush=True)
                last_case = case
            obs, _, _, _, _ = env.step(policy(obs))


def play(args):
    if args.task in ("urdf4_omni_v1", "urdf4_amp_cts_v1"):
        return play_urdf4(args)
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
        "dog_rs01_torque_straight_v5",
        "dog_rs01_stage2_actuator_a",
        "dog_rs01_stage2_actuator_b",
        "dog_rs01_straight_stand",
        "dog_rs01_straight_walk",
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
        elif args.task == "dog_rs01_straight_stand":
            # Stage 1 only ever learned to hold still.
            env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
            env_cfg.commands.stand_probability = 1.0
        elif args.task == "dog_rs01_straight_walk":
            # The single speed this stage is trained on.
            env_cfg.commands.ranges.lin_vel_x = [0.10, 0.10]
            env_cfg.commands.stand_probability = 0.0
        elif args.task in (
            "dog_rs01_stage2_actuator_a",
            "dog_rs01_stage2_actuator_b",
        ):
            env_cfg.commands.ranges.lin_vel_x = [0.10, 0.10]
        elif args.task in (
            "dog_rs01_smooth_straight",
            "dog_rs01_smooth_straight_v2",
            "dog_rs01_straight_guarded",
            "dog_rs01_torque_straight_v5",
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
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep", action="store_true", help="URDF4: change direction every 10 seconds")
    parser.add_argument("--march", action="store_true", help="CTS: step in place on a zero velocity command")
    flags, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining
    args = get_args()
    args.sweep = flags.sweep
    args.march = flags.march
    play(args)
