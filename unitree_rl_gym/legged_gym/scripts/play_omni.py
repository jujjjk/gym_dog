"""Show the omni policy and log play data to CSV for analysis."""
import isaacgym
import torch
from legged_gym.algorithms.symmetry_ppo import (
    mirror_fanfan_actions,
    mirror_fanfan_observations,
)
import os
import csv
from datetime import datetime

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


COMMANDS = (
    (0.35, 0.00, 0.00),
    (-0.10, 0.00, 0.00),
    (0.00, 0.07, 0.00),
    (0.00, -0.07, 0.00),
    (0.00, 0.00, 0.50),
    (0.00, 0.00, -0.50),
    (0.25, 0.00, 0.50),
    (-0.10, 0.00, 0.35),
    (0.20, 0.07, 0.00),
)

FAST_COMMANDS = (
    (0.35, 0.00, 0.00),
    (-0.10, 0.00, 0.00),
    (0.00, 0.07, 0.00),
    (0.00, -0.07, 0.00),
    (0.00, 0.00, 0.50),
    (0.00, 0.00, -0.50),
    (0.25, 0.00, 0.50),
    (-0.10, 0.00, 0.35),
    (0.20, 0.07, 0.00),
)

HIGH_SPEED_COMMANDS = (
    (0.45, 0.00, 0.00),
    (-0.18, 0.00, 0.00),
    (0.00, 0.15, 0.00),
    (0.00, -0.15, 0.00),
    (0.00, 0.00, 0.90),
    (0.00, 0.00, -0.90),
    (0.35, 0.00, 0.80),
    (-0.15, 0.00, 0.60),
    (0.35, 0.15, 0.00),
)


def quat_xyzw_to_yaw(quat):
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )
    return yaw


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = args.num_envs or 18
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 6
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_base_mass = False
    if hasattr(env_cfg.domain_rand, "randomize_base_com"):
        env_cfg.domain_rand.randomize_base_com = False
    if hasattr(env_cfg.rewards, "terminate_straight_heading_error"):
        # Training-only guard: evaluation must reveal accumulated heading
        # error instead of hiding it behind an automatic environment reset.
        env_cfg.rewards.terminate_straight_heading_error = None
    if hasattr(env_cfg.rewards, "terminate_translation_heading_error"):
        env_cfg.rewards.terminate_translation_heading_error = None
    env_cfg.env.test = True

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    if (
        getattr(env_cfg.control, "enforce_policy_symmetry", False)
        or os.environ.get("FANFAN_SYMMETRY_ENSEMBLE", "0") == "1"
    ):
        base_policy = policy

        def policy(observations):
            direct = base_policy(observations)
            mirrored = mirror_fanfan_actions(
                base_policy(mirror_fanfan_observations(observations))
            )
            return 0.5 * (direct + mirrored)

    if any(tag in args.task for tag in (
        "high_speed_transition", "high_authority_transition"
        , "high_authority_direction"
        , "high_authority_closed_loop"
        , "high_cadence"
        , "symmetric_transition"
    )):
        matrix = HIGH_SPEED_COMMANDS
    elif any(tag in args.task for tag in ("fast", "smooth", "filtered")):
        matrix = FAST_COMMANDS
    else:
        matrix = COMMANDS

    commands = torch.tensor(matrix, device=env.device).repeat_interleave(2, 0)
    commands = commands[:env.num_envs]

    if len(commands) < env.num_envs:
        commands = commands.repeat(
            (env.num_envs + len(commands) - 1) // len(commands), 1
        )[:env.num_envs]

    env.commands[:, 3] = env.rpy[:, 2]

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "play_debug")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_name = str(getattr(args, "checkpoint", "none"))
    load_run_name = str(getattr(args, "load_run", "none")).replace("/", "_")

    log_path = os.path.join(
        log_dir,
        f"play_{args.task}_{load_run_name}_ckpt{checkpoint_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)

    joint_names = list(env.dof_names)
    log_writer.writerow([
        "step", "time_s", "env_id",
        "cmd_vx", "cmd_vy", "cmd_yaw",
        "x", "y", "z", "yaw",
        "vx_body", "vy_body", "vz_body",
        "roll_rate", "pitch_rate", "yaw_rate",
        "action_abs_mean", "action_abs_max", "action_sat75_ratio",
        "torque_abs_mean", "torque_abs_max",
        "gait_phase",
    ] + [f"foot_z_{leg}" for leg in ("FL", "FR", "RL", "RR")]
      + [f"foot_contact_{leg}" for leg in ("FL", "FR", "RL", "RR")]
      + [f"policy_{name}" for name in joint_names]
      + [f"target_delta_{name}" for name in joint_names]
      + [f"raw_torque_{name}" for name in joint_names])

    print(f"[play_omni] logging to: {log_path}")
    print("[play_omni] press Ctrl+C to stop and save CSV.")

    step = 0
    segment_duration_s = float(os.environ.get("FANFAN_PLAY_SEGMENT_S", "15"))
    duration_s = float(os.environ.get("FANFAN_PLAY_DURATION_S", "0"))
    transition_interval_steps = max(1, int(segment_duration_s / env.dt))
    max_steps = int(duration_s / env.dt) if duration_s > 0.0 else None
    transition_eval = os.environ.get("FANFAN_PLAY_TRANSITIONS", "1") != "0"

    try:
        while True:
            if (
                transition_eval
                and any(tag in args.task for tag in (
                    "high_speed_transition", "high_authority_transition"
                    , "high_authority_direction"
                    , "high_authority_closed_loop"
                    , "high_cadence"
                    , "symmetric_transition"
                    , "hardware_balance_5530"
                    , "realdata_curriculum"
                    , "realdata_coordinated"
                    , "realdata_clearance_polish"
                    , "realdata_performance_recovery"
                ))
                and step > 0
                and step % transition_interval_steps == 0
            ):
                commands = torch.roll(commands, shifts=2, dims=0)
            env.commands[:, :3] = commands
            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            env.step(actions)

            with torch.no_grad():
                dt = getattr(env, "dt", 0.02)
                time_s = step * dt

                root_states = env.root_states
                pos = root_states[:, 0:3]
                quat = root_states[:, 3:7]
                yaw = quat_xyzw_to_yaw(quat)

                base_lin_vel = env.base_lin_vel
                base_ang_vel = env.base_ang_vel

                action_abs = actions.abs()
                action_abs_mean = action_abs.mean(dim=1)
                action_abs_max = action_abs.max(dim=1).values
                action_sat75_ratio = (action_abs > 0.75).float().mean(dim=1)

                if hasattr(env, "torques"):
                    torque_abs = env.torques.abs()
                    torque_abs_mean = torque_abs.mean(dim=1)
                    torque_abs_max = torque_abs.max(dim=1).values
                else:
                    torque_abs_mean = torch.zeros(env.num_envs, device=env.device)
                    torque_abs_max = torch.zeros(env.num_envs, device=env.device)

                policy_actions = getattr(env, "policy_actions", env.actions)
                target_delta = (
                    getattr(env, "target_dof_pos_rl", env.dof_pos)
                    - env.default_dof_pos
                )
                raw_torques = getattr(env, "raw_torques", env.torques)
                foot_slots = [
                    env.foot_slot_by_leg[leg]
                    for leg in ("FL", "FR", "RL", "RR")
                ]
                foot_z = env.feet_pos[:, foot_slots, 2]
                foot_contact = (
                    env.contact_forces[:, env.feet_indices[foot_slots], 2]
                    > 1.0
                )

                for eid in range(env.num_envs):
                    log_writer.writerow([
                        int(step),
                        float(time_s),
                        int(eid),

                        float(env.commands[eid, 0].item()),
                        float(env.commands[eid, 1].item()),
                        float(env.commands[eid, 2].item()),

                        float(pos[eid, 0].item()),
                        float(pos[eid, 1].item()),
                        float(pos[eid, 2].item()),
                        float(yaw[eid].item()),

                        float(base_lin_vel[eid, 0].item()),
                        float(base_lin_vel[eid, 1].item()),
                        float(base_lin_vel[eid, 2].item()),

                        float(base_ang_vel[eid, 0].item()),
                        float(base_ang_vel[eid, 1].item()),
                        float(base_ang_vel[eid, 2].item()),

                        float(action_abs_mean[eid].item()),
                        float(action_abs_max[eid].item()),
                        float(action_sat75_ratio[eid].item()),

                        float(torque_abs_mean[eid].item()),
                        float(torque_abs_max[eid].item()),
                        float(env.gait_phase[eid].item()),
                    ] + foot_z[eid].detach().cpu().tolist()
                      + foot_contact[eid].int().detach().cpu().tolist()
                      + policy_actions[eid].detach().cpu().tolist()
                      + target_delta[eid].detach().cpu().tolist()
                      + raw_torques[eid].detach().cpu().tolist())

                if step % 100 == 0:
                    log_file.flush()

                step += 1
                if max_steps is not None and step >= max_steps:
                    break

    except KeyboardInterrupt:
        print("\n[play_omni] stopped by user.")

    finally:
        log_file.flush()
        log_file.close()
        print(f"[play_omni] saved CSV: {log_path}")


if __name__ == "__main__":
    play(get_args())
