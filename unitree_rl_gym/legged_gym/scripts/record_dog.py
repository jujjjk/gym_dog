"""Record deterministic RS01 dog policy playback telemetry to CSV."""

import csv
import os
from datetime import datetime

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


LEGS = ("FL", "FR", "RL", "RR")


def quaternion_xyzw_to_rpy(quaternion):
    qx, qy, qz, qw = quaternion.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch = torch.asin(
        torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    )
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def tensor_row(tensor):
    return tensor[0].detach().cpu().tolist()


def play_and_record(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_base_mass = False
    if hasattr(env_cfg.domain_rand, "randomize_base_com"):
        env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_gait_phase_on_reset = False

    command_vx = float(os.environ.get("DOG_RECORD_VX", "0.16"))
    duration_s = float(os.environ.get("DOG_RECORD_DURATION_S", "30"))
    env_cfg.commands.ranges.lin_vel_x = [command_vx, command_vx]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.stand_probability = 0.0
    env_cfg.commands.pure_sagittal_probability = 1.0
    env_cfg.commands.resampling_time = duration_s + 10.0

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)
    observations = env.get_observations()

    output_path = os.environ.get("DOG_RECORD_OUTPUT")
    if not output_path:
        output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "dog_csv")
        os.makedirs(output_dir, exist_ok=True)
        run_name = str(args.load_run or "configured").replace("/", "_")
        checkpoint = str(args.checkpoint if args.checkpoint is not None else "configured")
        output_path = os.path.join(
            output_dir,
            f"{args.task}_{run_name}_model_{checkpoint}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    joint_names = list(env.dof_names)
    foot_slots = [env.foot_slot_by_leg[leg] for leg in LEGS]
    headers = [
        "step",
        "time_s",
        "episode",
        "reset",
        "reward",
        "cmd_vx_m_s",
        "cmd_vy_m_s",
        "cmd_yaw_rad_s",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_roll_rad",
        "base_pitch_rad",
        "base_yaw_rad",
        "body_vx_m_s",
        "body_vy_m_s",
        "body_vz_m_s",
        "body_roll_rate_rad_s",
        "body_pitch_rate_rad_s",
        "body_yaw_rate_rad_s",
        "gait_phase",
        "flight",
        "all_feet_contact",
    ]
    headers += [f"foot_force_z_{leg}_n" for leg in LEGS]
    headers += [f"foot_contact_{leg}" for leg in LEGS]
    headers += [f"foot_z_{leg}_m" for leg in LEGS]
    for signal in (
        "policy_action",
        "dof_pos_rad",
        "dof_vel_rad_s",
        "target_pos_rad",
        "raw_torque_nm",
        "motor_torque_nm",
        "applied_torque_nm",
        "thermal_rms_ratio",
        "motor_temperature_c",
        "active_torque_limit_nm",
    ):
        headers += [f"{signal}_{name}" for name in joint_names]

    max_steps = max(1, int(round(duration_s / float(env.dt))))
    episode = 0
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        with torch.no_grad():
            for step in range(max_steps):
                env.commands[:, 0] = command_vx
                env.commands[:, 1:3] = 0.0
                actions = policy(observations)
                observations, _, rewards, dones, _ = env.step(actions)

                root = env.root_states
                roll, pitch, yaw = quaternion_xyzw_to_rpy(root[:, 3:7])
                foot_force_z = env.contact_forces[
                    :, env.feet_indices[foot_slots], 2
                ].clip(min=0.0)
                foot_contact = foot_force_z > float(
                    getattr(env.cfg.rewards, "contact_force_threshold", 1.0)
                )
                flight = ~torch.any(foot_contact, dim=1)
                all_contact = torch.all(foot_contact, dim=1)
                target_pos = getattr(env, "target_dof_pos_rl", env.dof_pos)
                raw_torque = getattr(env, "raw_torques", env.torques)
                motor_torque = getattr(env, "motor_torques", env.torques)
                thermal_rms = torch.sqrt(
                    getattr(
                        env,
                        "thermal_torque_sq_ema",
                        torch.zeros_like(env.torques),
                    ).clip(min=0.0)
                )
                motor_temperature = getattr(
                    env,
                    "motor_temperature_c",
                    torch.zeros_like(env.torques),
                )
                active_limit = getattr(
                    env,
                    "active_motor_torque_limits",
                    env.torque_limits.unsqueeze(0),
                )
                policy_actions = getattr(env, "policy_actions", actions)
                gait_phase = getattr(
                    env,
                    "gait_phase",
                    torch.zeros(env.num_envs, device=env.device),
                )

                row = [
                    step,
                    step * float(env.dt),
                    episode,
                    int(dones[0].item()),
                    float(rewards[0].item()),
                    float(env.commands[0, 0].item()),
                    float(env.commands[0, 1].item()),
                    float(env.commands[0, 2].item()),
                    float(root[0, 0].item()),
                    float(root[0, 1].item()),
                    float(root[0, 2].item()),
                    float(roll[0].item()),
                    float(pitch[0].item()),
                    float(yaw[0].item()),
                    *tensor_row(env.base_lin_vel),
                    *tensor_row(env.base_ang_vel),
                    float(gait_phase[0].item()),
                    int(flight[0].item()),
                    int(all_contact[0].item()),
                    *tensor_row(foot_force_z),
                    *foot_contact[0].int().detach().cpu().tolist(),
                    *tensor_row(env.feet_pos[:, foot_slots, 2]),
                    *tensor_row(policy_actions),
                    *tensor_row(env.dof_pos),
                    *tensor_row(env.dof_vel),
                    *tensor_row(target_pos),
                    *tensor_row(raw_torque),
                    *tensor_row(motor_torque),
                    *tensor_row(env.torques),
                    *tensor_row(thermal_rms),
                    *tensor_row(motor_temperature),
                    *tensor_row(active_limit),
                ]
                writer.writerow(row)
                if bool(dones[0].item()):
                    episode += 1
                if step % 100 == 0:
                    stream.flush()

    print(f"[record_dog] rows: {max_steps}")
    print(f"[record_dog] dt: {float(env.dt):.6f} s")
    print(f"[record_dog] saved CSV: {output_path}")


if __name__ == "__main__":
    play_and_record(get_args())
