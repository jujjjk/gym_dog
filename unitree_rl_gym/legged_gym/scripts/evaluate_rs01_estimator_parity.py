"""Deterministic parity evaluation for the RS01 estimator-observation task."""

import json
import os

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


def evaluate(args):
    if args.task != "rs01_go2_estimator_parity":
        raise ValueError(
            "This evaluator requires --task=rs01_go2_estimator_parity"
        )

    duration_s = float(os.environ.get("RS01_EVAL_DURATION_S", "30"))
    command_vx = float(os.environ.get("RS01_EVAL_VX", "0.23"))
    num_envs = int(os.environ.get("RS01_EVAL_NUM_ENVS", "16"))

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = duration_s + 10.0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [command_vx, command_vx]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.resampling_time = duration_s + 10.0

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
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

    step_count = max(1, int(round(duration_s / float(env.dt))))
    true_displacements = []
    position_errors = []
    velocity_errors = []
    forward_velocities = []
    yaw_rates = []
    exact_contacts = []
    flights = []
    raw_torques = []
    motor_over_6 = []
    peak_saturation = []
    confidences = []
    reset_count = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )

    with torch.no_grad():
        for _ in range(step_count):
            env.commands[:, 0] = command_vx
            env.commands[:, 1:3] = 0.0
            actions = policy(observations)
            observations, _, _, dones, _ = env.step(actions)
            contact = env.get_foot_contact_mask()
            desired = env._desired_contact_mask()
            true_y, _ = env._straight_path_state()
            true_displacements.append(true_y.clone())
            position_errors.append(
                env.estimated_path_position_error_m.clone()
            )
            velocity_errors.append(
                env.estimated_path_velocity_error_m_s.clone()
            )
            forward_velocities.append(env.base_lin_vel[:, 0].clone())
            yaw_rates.append(env.base_ang_vel[:, 2].clone())
            exact_contacts.append(torch.all(contact == desired, dim=1))
            flights.append(~torch.any(contact, dim=1))
            raw_torques.append(env.raw_pd_torques.clone())
            motor_over_6.append(
                torch.abs(env.motor_electromagnetic_torques) > 6.0
            )
            peak_saturation.append(
                torch.abs(env.raw_pd_torques)
                >= env.peak_torque_limit_nm
            )
            confidences.append(env.estimated_odom_confidence.clone())
            reset_count += dones.to(dtype=torch.long)

    true_y = torch.stack(true_displacements)
    position_error = torch.stack(position_errors)
    velocity_error = torch.stack(velocity_errors)
    raw_torque = torch.stack(raw_torques)
    report = {
        "task": args.task,
        "duration_s": duration_s,
        "num_envs": env.num_envs,
        "command_vx_m_s": command_vx,
        "resets_total": int(reset_count.sum().item()),
        "resets_max_per_env": int(reset_count.max().item()),
        "true_path_rms_m": float(
            torch.sqrt(torch.mean(true_y.square())).item()
        ),
        "true_final_abs_lateral_m": float(
            torch.mean(torch.abs(true_y[-1])).item()
        ),
        "estimator_position_error_rms_m": float(
            torch.sqrt(torch.mean(position_error.square())).item()
        ),
        "estimator_velocity_error_rms_m_s": float(
            torch.sqrt(torch.mean(velocity_error.square())).item()
        ),
        "mean_forward_velocity_m_s": float(
            torch.stack(forward_velocities).mean().item()
        ),
        "yaw_rate_rms_rad_s": float(
            torch.sqrt(
                torch.mean(torch.stack(yaw_rates).square())
            ).item()
        ),
        "exact_diagonal_contact_ratio": float(
            torch.stack(exact_contacts).float().mean().item()
        ),
        "flight_ratio": float(
            torch.stack(flights).float().mean().item()
        ),
        "raw_torque_p95_nm": float(
            torch.quantile(torch.abs(raw_torque), 0.95).item()
        ),
        "raw_torque_peak_nm": float(
            torch.max(torch.abs(raw_torque)).item()
        ),
        "peak_saturation_ratio": float(
            torch.stack(peak_saturation).float().mean().item()
        ),
        "motor_over_6nm_ratio": float(
            torch.stack(motor_over_6).float().mean().item()
        ),
        "mean_odometry_confidence": float(
            torch.stack(confidences).mean().item()
        ),
        "finite_observations": bool(
            torch.all(torch.isfinite(observations)).item()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(get_args())
