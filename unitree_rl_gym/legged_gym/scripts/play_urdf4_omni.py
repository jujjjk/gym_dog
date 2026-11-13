"""Play, evaluate and export a urdf4_omni_v1 checkpoint.

Watch one fixed command in the viewer (latest run and checkpoint by default):

    python legged_gym/scripts/play_urdf4_omni.py --vx 0.3 --vy 0.0 --wz 0.0

Score the whole command envelope without a viewer:

    python legged_gym/scripts/play_urdf4_omni.py --sweep --headless

Options of this script (the usual --load_run, --checkpoint, --duration_s,
--eval_envs, --sim_device, --rl_device also apply):

    --sweep      run the fixed-command suite, --eval_envs robots per command
    --disturbed  keep the training randomization, sensor errors and full-strength
                 pushes (default: nominal plant, ideal sensors, no pushes)
    --export     write policy.pt, policy.onnx and contract.json next to the run
"""

import argparse
import json
import math
import os
import sys

import isaacgym  # noqa: F401  (must precede torch)
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.urdf4.gait_clock import LEG_PHASE_OFFSETS
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path

TASK = "urdf4_omni_v1"
SETTLE_S = 2.0  # Excluded from the statistics: start-up from the default stance.

# (label, vx m/s, vy m/s, wz rad/s): the corners of the trained envelope.
SWEEP = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward 0.3", 0.3, 0.0, 0.0),
    ("forward 0.6", 0.6, 0.0, 0.0),
    ("backward 0.4", -0.4, 0.0, 0.0),
    ("left 0.3", 0.0, 0.3, 0.0),
    ("right 0.3", 0.0, -0.3, 0.0),
    ("turn left 1.0", 0.0, 0.0, 1.0),
    ("turn right 1.0", 0.0, 0.0, -1.0),
    ("diagonal", 0.4, 0.2, 0.0),
    ("arc left", 0.3, 0.0, 0.6),
    ("mixed", 0.3, -0.2, -0.5),
)


def parse_script_flags():
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("--sweep", "--disturbed", "--export"):
        parser.add_argument(flag, action="store_true")
    flags, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining
    return flags


def make_env(args, flags, num_envs):
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = args.duration_s + 10.0
    env_cfg.commands.fixed_command = [0.0, 0.0, 0.0]
    if flags.disturbed:
        env_cfg.push.ramp_s = 0.0  # Full-strength pushes from the first second.
    else:
        env_cfg.env.test = True  # Nominal controller/actuator, no reset noise.
        env_cfg.noise.add_noise = False
        env_cfg.push.enabled = False
        for flag in (
            "randomize_friction", "randomize_restitution", "randomize_base_mass",
            "randomize_base_com", "randomize_link_mass", "randomize_armature",
        ):
            setattr(env_cfg.domain_rand, flag, False)
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    return env, train_cfg


def load_policy(env, args, train_cfg):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    path = get_load_path(
        log_root,
        load_run=-1 if args.load_run is None else args.load_run,
        checkpoint=-1 if args.checkpoint is None else args.checkpoint,
    )
    runner, _ = task_registry.make_alg_runner(env=env, name=TASK, args=args, train_cfg=train_cfg, log_root=None)
    runner.load(path, load_optimizer=False)
    print(f"[play] checkpoint {path}")
    return runner, path


def export(env, runner, checkpoint_path):
    """Deployment bundle: the actor network and everything needed to feed it."""
    cfg = env.cfg
    iteration = os.path.splitext(os.path.basename(checkpoint_path))[0].split("_")[-1]
    # The latest-checkpoint lookup matches any entry containing "model": keep that word out.
    out_dir = os.path.join(os.path.dirname(checkpoint_path), f"exported_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    actor = runner.alg.actor_critic.actor
    device = next(actor.parameters()).device
    actor.cpu().eval()
    example = torch.zeros(1, cfg.env.num_observations)
    torch.jit.trace(actor, example).save(os.path.join(out_dir, "policy.pt"))
    torch.onnx.export(
        actor, example, os.path.join(out_dir, "policy.onnx"),
        input_names=["obs"], output_names=["action"], opset_version=13,
    )
    actor.to(device)
    scales, act = env.obs_scales, cfg.rs01_actuator
    contract = {
        "task": TASK,
        "checkpoint": checkpoint_path,
        "control_period_s": env.dt,
        "joint_order": list(env.dof_names),
        "default_joint_position_rad": env.default_dof_pos[0].tolist(),
        "action": {
            "network_output": "unbounded mean, one value per joint in joint_order",
            "clip": cfg.normalization.clip_actions,
            "scale_rad": env.action_scale.tolist(),
            "target": "q_target = default + scale * clip(action, -clip, clip), then the 50 Hz rate/acceleration limiter",
            "target_rate_limit_rad_s": env.target_rate_limit.tolist(),
            "target_acceleration_limit_rad_s2": env.target_acceleration_limit.tolist(),
        },
        "pd": {
            "kp_nm_rad": env.p_gains.tolist(),
            "kd_nm_s_rad": env.d_gains.tolist(),
            "torque_limit_nm": act.peak_torque_limit_nm,
        },
        "observation": {
            "size": cfg.env.num_observations,
            "history_frames": cfg.env.history_length,
            "history_order": "newest frame first; at start-up fill every frame with the first one",
            "clip": cfg.normalization.clip_observations,
            "frame": [
                {"name": "gyro_body_rad_s", "size": 3, "scale": scales.ang_vel},
                {"name": "gravity_direction_body", "size": 3, "scale": 1.0, "note": "unit vector, (0, 0, -1) when level"},
                {"name": "command_vx_vy_wz", "size": 3, "scale": [scales.lin_vel, scales.lin_vel, scales.ang_vel]},
                {"name": "joint_position_minus_default_rad", "size": 12, "scale": scales.dof_pos},
                {"name": "joint_velocity_rad_s", "size": 12, "scale": scales.dof_vel},
                {"name": "previous_action", "size": 12, "scale": 1.0, "note": "clipped network output of the previous step"},
                {"name": "gait_clock_sin_cos", "size": 2, "scale": 1.0},
                {
                    "name": "yaw_rate_error_stride_mean", "size": 1, "scale": scales.yaw_rate_error_mean,
                    "note": f"mean of (command wz - gyro z) over the last {env.measured_yaw_error.window.shape[1]} "
                            "control steps (one gait period), buffer zero-filled at start-up",
                },
            ],
        },
        "gait_clock": {
            "period_s": cfg.gait.period_s,
            "stance_ratio": cfg.gait.stance_ratio,
            "phase": "phase = (t / period_s) mod 1; observe sin(2 pi phase), cos(2 pi phase); keeps running while standing",
            "leg_phase_offset": LEG_PHASE_OFFSETS,
            "stance_when": "(phase + leg offset) mod 1 < stance_ratio",
        },
        "command_range": {
            "vx_m_s": cfg.commands.ranges.lin_vel_x,
            "vy_m_s": cfg.commands.ranges.lin_vel_y,
            "wz_rad_s": cfg.commands.ranges.ang_vel_yaw,
            "standing": "send exactly zero; |v| < 0.05 m/s with |wz| < 0.1 rad/s was never trained",
        },
    }
    with open(os.path.join(out_dir, "contract.json"), "w") as handle:
        json.dump(contract, handle, indent=2)
    print(f"[export] {out_dir}: policy.pt, policy.onnx, contract.json")


def run(env, policy, commands, duration_s, follow):
    """Roll out fixed commands; return per-robot statistics after the settle time."""
    n, gait = env.num_envs, env.cfg.gait
    env.set_command_override(commands)
    obs, _ = env.reset()
    names = ("vx", "vy", "wz", "lin_err", "yaw_err", "gait", "height", "over_cont", "over_peak")
    sums = {name: torch.zeros(n, device=env.device) for name in names}
    mid_swing_height = torch.zeros(n, device=env.device)
    mid_swing_count = torch.zeros(n, device=env.device)
    peak_torque = torch.zeros(n, device=env.device)
    fallen = torch.zeros(n, device=env.device, dtype=torch.bool)
    steps, counted = int(round(duration_s / env.dt)), 0
    with torch.no_grad():
        for step in range(steps):
            obs, _, _, dones, _ = env.step(policy(obs))
            fallen |= dones.bool()  # The episode is longer than the run: every reset is a fall.
            if follow:
                base = env.root_states[0, :3].tolist()
                env.set_camera([base[0] + 1.6, base[1] - 1.8, 0.9], [base[0], base[1], 0.25])
            if step * env.dt < SETTLE_S:
                continue
            counted += 1
            velocity, yaw_rate = env.base_lin_vel, env.base_ang_vel[:, 2]
            sums["vx"] += velocity[:, 0]
            sums["vy"] += velocity[:, 1]
            sums["wz"] += yaw_rate
            sums["lin_err"] += (env.commands[:, :2] - velocity[:, :2]).norm(dim=1)
            sums["yaw_err"] += (env.commands[:, 2] - yaw_rate).abs()
            sums["gait"] += (env.get_foot_contact_mask() == env._scheduled_stance()).float().mean(dim=1)
            sums["height"] += env.root_states[:, 2]
            torque = env.motor_torques.abs()
            sums["over_cont"] += (torque > env.cfg.rs01_actuator.continuous_torque_nm).float().mean(dim=1)
            sums["over_peak"] += (env.raw_pd_torques.abs() > env.peak_torque).float().mean(dim=1)
            peak_torque = torch.maximum(peak_torque, torque.amax(dim=1))
            # Foot clearance around the top of the scheduled swing.
            progress = (env._leg_phase() - gait.stance_ratio) / (1.0 - gait.stance_ratio)
            mid = (progress > 0.4) & (progress < 0.6) & ~env._scheduled_stance()
            mid_swing_height += ((env.feet_pos[:, :, 2] - gait.foot_radius_m) * mid).sum(dim=1)
            mid_swing_count += mid.sum(dim=1)
    stats = {name: total / max(counted, 1) for name, total in sums.items()}
    stats["clearance"] = mid_swing_height / mid_swing_count.clamp(min=1.0)
    stats["peak_torque"] = peak_torque
    stats["fallen"] = fallen.float()
    # Drift: each robot's time-mean velocity error, i.e. what bends or shifts its path.
    # Averaging the signed error over robots instead would let opposite drifts cancel.
    error = torch.stack((stats["vx"], stats["vy"]), dim=1) - commands[:, :2]
    speed = commands[:, :2].norm(dim=1, keepdim=True)
    heading = commands[:, :2] / speed.clamp(min=1.0e-6)
    across = (heading[:, 0] * error[:, 1] - heading[:, 1] * error[:, 0]).abs()
    # Sideways of the commanded direction; any direction when no translation is commanded.
    stats["side_drift"] = torch.where(speed[:, 0] > 0.0, across, error.norm(dim=1))
    stats["yaw_drift"] = (stats["wz"] - commands[:, 2]).abs()
    return stats


def report(labels, commands, stats, per_case):
    print("\nside drift [cm/s] and yaw drift [deg/s]: each robot's |time-mean velocity error|, averaged over robots")
    print(
        f"{'command':15s} {'vx':>12s} {'vy':>12s} {'wz':>12s} {'side':>6s} {'yaw':>6s} {'lin err':>8s} {'yaw err':>8s} "
        f"{'gait':>5s} {'clear':>6s} {'height':>6s} {'peak Nm':>7s} {'>6 Nm':>6s} {'>cap':>6s} {'falls':>6s}"
    )
    print(
        f"{'':15s} {'cmd / real':>12s} {'cmd / real':>12s} {'cmd / real':>12s} {'drift':>6s} {'drift':>6s} {'m/s':>8s} {'rad/s':>8s} "
        f"{'match':>5s} {'mm':>6s} {'m':>6s} {'':>7s} {'share':>6s} {'share':>6s} {'':>6s}"
    )
    for index, label in enumerate(labels):
        rows = slice(index * per_case, (index + 1) * per_case)
        mean = {name: value[rows].mean().item() for name, value in stats.items()}
        vx, vy, wz = commands[index * per_case].tolist()
        print(
            f"{label:15s} {vx:+5.2f}/{mean['vx']:+5.2f}  {vy:+5.2f}/{mean['vy']:+5.2f}  {wz:+5.2f}/{mean['wz']:+5.2f} "
            f"{100.0 * mean['side_drift']:6.1f} {math.degrees(mean['yaw_drift']):6.1f} "
            f"{mean['lin_err']:8.3f} {mean['yaw_err']:8.3f} {mean['gait']:5.2f} {1000.0 * mean['clearance']:6.1f} "
            f"{mean['height']:6.3f} {stats['peak_torque'][rows].max().item():7.1f} {mean['over_cont']:6.3f} "
            f"{mean['over_peak']:6.3f} {int(stats['fallen'][rows].sum().item()):3d}/{per_case}"
        )


def main():
    flags = parse_script_flags()
    args = get_args()
    args.task = TASK
    per_case = max(1, args.eval_envs)
    cases = SWEEP if flags.sweep else ((f"vx{args.vx:+.2f} vy{args.vy:+.2f} wz{args.wz:+.2f}"[:15], args.vx, args.vy, args.wz),)
    env, train_cfg = make_env(args, flags, per_case * len(cases))
    runner, path = load_policy(env, args, train_cfg)
    if flags.export:
        export(env, runner, path)
    policy = runner.get_inference_policy(device=env.device)
    commands = torch.tensor([case[1:] for case in cases], device=env.device).repeat_interleave(per_case, dim=0)
    mode = "training randomization + sensor errors + pushes" if flags.disturbed else "nominal plant, ideal sensors"
    print(f"[play] {len(cases)} command(s) x {per_case} robots, {args.duration_s:.0f} s, {mode}")
    stats = run(env, policy, commands, args.duration_s, follow=not args.headless)
    report([case[0] for case in cases], commands, stats, per_case)


if __name__ == "__main__":
    main()
