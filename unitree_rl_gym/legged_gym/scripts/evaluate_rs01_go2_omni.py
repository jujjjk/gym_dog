"""Nominal fixed-command evaluation suite for RS01 omni checkpoints."""

import json

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
COMMAND_CASES = (
    ("stand", 0.00, 0.00, 0.00, False),
    ("march", 0.00, 0.00, 0.00, True),
    ("forward_0p10", 0.10, 0.00, 0.00, True),
    ("forward_0p20", 0.20, 0.00, 0.00, True),
    ("backward_0p10", -0.10, 0.00, 0.00, True),
    ("left_0p08", 0.00, 0.08, 0.00, True),
    ("right_0p08", 0.00, -0.08, 0.00, True),
    ("yaw_left_0p30", 0.00, 0.00, 0.30, True),
    ("yaw_right_0p30", 0.00, 0.00, -0.30, True),
    ("combined", 0.15, 0.06, 0.25, True),
)


def _set_nominal_eval_cfg(env_cfg, duration_s, num_envs):
    env_cfg.env.num_envs = num_envs
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


def _rms(value):
    return float(torch.sqrt(torch.mean(torch.square(value))).item())


def evaluate(args):
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"This evaluator requires one of --task={sorted(SUPPORTED_TASKS)}"
        )
    if args.duration_s <= 0.0 or args.eval_envs <= 0:
        raise ValueError("--duration_s and --eval_envs must be positive")

    case_count = len(COMMAND_CASES)
    total_envs = case_count * args.eval_envs
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(env_cfg, args.duration_s, total_envs)
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

    commands = torch.tensor(
        [[vx, vy, wz] for _, vx, vy, wz, _ in COMMAND_CASES],
        device=env.device,
        dtype=torch.float,
    ).repeat_interleave(args.eval_envs, dim=0)
    gait_enable = torch.tensor(
        [gait for _, _, _, _, gait in COMMAND_CASES],
        device=env.device,
        dtype=torch.float,
    ).repeat_interleave(args.eval_envs)
    steps = max(1, int(round(args.duration_s / env.dt)))
    warmup_steps = min(steps - 1, int(round(2.0 / env.dt)))
    reset_count = torch.zeros(total_envs, device=env.device, dtype=torch.long)
    samples = {
        key: []
        for key in (
            "base_lin_vel",
            "yaw_rate",
            "lateral_error",
            "position_distance",
            "heading_error",
            "roll",
            "phase_contact_match",
            "exact_diagonal_contact",
            "contact_count",
            "flight",
            "same_axle_flight",
            "raw_torque",
            "motor_torque",
        )
    }
    finite = True

    with torch.no_grad():
        for step in range(steps):
            if hasattr(env, "set_evaluation_command"):
                env.set_evaluation_command(commands, gait_enable)
            else:
                env.commands[:, :3] = commands
                if hasattr(env, "gait_enable"):
                    env.gait_enable[:] = gait_enable
            env.compute_observations()
            observations = env.get_observations()
            observations, _, _, dones, _ = env.step(policy(observations))
            reset_count += dones.to(dtype=torch.long)
            finite = finite and bool(torch.isfinite(observations).all())
            finite = finite and bool(torch.isfinite(env.rew_buf).all())
            if step < warmup_steps:
                continue

            contact = env.get_foot_contact_mask()
            desired = env._desired_contact_mask()
            fl = env.foot_slot_by_leg["FL"]
            fr = env.foot_slot_by_leg["FR"]
            rl = env.foot_slot_by_leg["RL"]
            rr = env.foot_slot_by_leg["RR"]
            same_axle = (~contact[:, fl] & ~contact[:, fr]) | (
                ~contact[:, rl] & ~contact[:, rr]
            )
            lateral_error, _ = env._straight_path_state()
            samples["base_lin_vel"].append(env.base_lin_vel.clone())
            samples["yaw_rate"].append(env.base_ang_vel[:, 2].clone())
            samples["lateral_error"].append(lateral_error.clone())
            samples["position_distance"].append(torch.linalg.vector_norm(
                env.root_states[:, :2] - env.omni_desired_position_xy, dim=1
            ).clone())
            samples["heading_error"].append(env._straight_heading_error().clone())
            samples["roll"].append(env.rpy[:, 0].clone())
            phase_match = torch.all(contact == desired, dim=1)
            desired_two = torch.sum(desired, dim=1) == 2
            samples["phase_contact_match"].append(phase_match)
            samples["exact_diagonal_contact"].append(
                phase_match & desired_two
            )
            samples["contact_count"].append(torch.sum(contact, dim=1))
            samples["flight"].append(~torch.any(contact, dim=1))
            samples["same_axle_flight"].append(same_axle)
            samples["raw_torque"].append(env.raw_pd_torques.clone())
            samples["motor_torque"].append(
                env.motor_electromagnetic_torques.clone()
            )

    stacked = {key: torch.stack(value) for key, value in samples.items()}
    results = []
    for case_index, (name, vx, vy, wz, gait) in enumerate(COMMAND_CASES):
        start = case_index * args.eval_envs
        stop = start + args.eval_envs
        linear = stacked["base_lin_vel"][:, start:stop]
        yaw_rate = stacked["yaw_rate"][:, start:stop]
        raw_torque = stacked["raw_torque"][:, start:stop]
        motor_torque = stacked["motor_torque"][:, start:stop]
        results.append(
            {
                "case": name,
                "command_vx_m_s": vx,
                "command_vy_m_s": vy,
                "command_wz_rad_s": wz,
                "gait_enable": gait,
                "mean_vx_m_s": float(linear[:, :, 0].mean().item()),
                "mean_vy_m_s": float(linear[:, :, 1].mean().item()),
                "mean_wz_rad_s": float(yaw_rate.mean().item()),
                "vx_rmse_m_s": _rms(linear[:, :, 0] - vx),
                "vy_rmse_m_s": _rms(linear[:, :, 1] - vy),
                "wz_rmse_rad_s": _rms(yaw_rate - wz),
                "planar_speed_rms_m_s": _rms(torch.linalg.vector_norm(linear[:, :, :2], dim=2)),
                "position_error_rms_m": _rms(stacked["position_distance"][:, start:stop]),
                "lateral_path_rms_m": _rms(
                    stacked["lateral_error"][:, start:stop]
                ),
                "heading_error_rms_rad": _rms(
                    stacked["heading_error"][:, start:stop]
                ),
                "roll_rms_rad": _rms(stacked["roll"][:, start:stop]),
                "phase_contact_match_ratio": float(
                    stacked["phase_contact_match"][:, start:stop]
                    .float()
                    .mean()
                    .item()
                ),
                "exact_diagonal_contact_ratio": float(
                    stacked["exact_diagonal_contact"][:, start:stop]
                    .float()
                    .mean()
                    .item()
                ),
                "two_feet_contact_ratio": float(
                    (stacked["contact_count"][:, start:stop] == 2)
                    .float()
                    .mean()
                    .item()
                ),
                "three_feet_contact_ratio": float(
                    (stacked["contact_count"][:, start:stop] == 3)
                    .float()
                    .mean()
                    .item()
                ),
                "all_feet_contact_ratio": float(
                    (stacked["contact_count"][:, start:stop] == 4)
                    .float()
                    .mean()
                    .item()
                ),
                "flight_ratio": float(
                    stacked["flight"][:, start:stop].float().mean().item()
                ),
                "same_axle_flight_ratio": float(
                    stacked["same_axle_flight"][:, start:stop]
                    .float()
                    .mean()
                    .item()
                ),
                "raw_torque_p95_nm": float(
                    torch.quantile(torch.abs(raw_torque), 0.95).item()
                ),
                "raw_torque_peak_nm": float(torch.abs(raw_torque).max().item()),
                "motor_over_6nm_ratio": float(
                    (torch.abs(motor_torque) > 6.0).float().mean().item()
                ),
                "peak_saturation_ratio": float(
                    (torch.abs(motor_torque) >= 17.0 - 1.0e-4)
                    .float()
                    .mean()
                    .item()
                ),
                "resets_total": int(reset_count[start:stop].sum().item()),
                "resets_max_per_env": int(reset_count[start:stop].max().item()),
            }
        )

    report = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "duration_s": args.duration_s,
        "steady_state_window_s": max(0.0, args.duration_s - 2.0),
        "eval_envs_per_case": args.eval_envs,
        "policy_frequency_hz": 1.0 / env.dt,
        "domain_randomization": False,
        "finite_observations_and_rewards": finite,
        "cases": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(get_args())
