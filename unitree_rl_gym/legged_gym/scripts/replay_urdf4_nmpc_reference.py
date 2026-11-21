"""Track the recorded URDF4 NMPC trot with the nominal RS01 actuator.

The Gazebo clip is only a position reference. This script keeps it as an expert
candidate when the Isaac Gym plant itself produces the gait.
"""

import os

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

TASK = "urdf4_omni_v1"
REFERENCE = "/home/nszb/legged_noetic_ws/logs_urdf4_record/urdf4_nmpc_trot_0p10.npz"
RATE_LIMIT = np.array([2.0, 2.6, 3.2] * 4)
DUTY = 0.75


def sustained_events(mask, minimum=2):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts[(ends - starts) >= minimum].tolist()


def make_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=TASK)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 30
    env_cfg.noise.add_noise = False
    env_cfg.push.enabled = False
    env_cfg.commands.fixed_command = [0.10, 0.0, 0.0]
    for flag in (
        "randomize_friction", "randomize_restitution", "randomize_base_mass",
        "randomize_base_com", "randomize_link_mass", "randomize_armature",
        "randomize_gains", "randomize_motor_strength", "randomize_rs01_actuator",
    ):
        setattr(env_cfg.domain_rand, flag, False)
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    return env


def actions_for(env, joint_pos, stand):
    """Command the reference relative to the measured loaded stance.

    Kp=40 sags about 0.1 rad under body weight. Tracking the Gazebo joint
    angles directly spends the whole swing on that sag, so the target is
    shifted by the measured stance error.
    """
    scale = env.action_scale.detach().cpu().numpy()
    return np.clip((joint_pos - stand) / scale, -2.0, 2.0).astype(np.float32)


def audit(height, contact, velocity, joint_vel, torque, root_z):
    swing = (height > 0.008) & ~contact.astype(bool)
    pairs = [(contact == np.array(pattern)).all(-1) for pattern in ([1, 0, 0, 1], [0, 1, 1, 0])]
    events = sorted([(int(t), 0) for t in sustained_events(pairs[0], 2)]
                    + [(int(t), 1) for t in sustained_events(pairs[1], 2)])
    alternations = sum(events[i][1] != events[i - 1][1] for i in range(1, len(events)))
    lift_mm = np.quantile(height, 0.95, axis=0) * 1000.0
    rate = np.quantile(np.abs(joint_vel) / RATE_LIMIT, 0.95)
    reasons = []
    if root_z.min() < 0.22:
        reasons.append("fell (height %.3f m)" % root_z.min())
    if velocity[:, 0].mean() < 0.05:
        reasons.append("forward speed %.3f m/s" % velocity[:, 0].mean())
    if abs(contact.mean() - DUTY) > 0.10:
        reasons.append("stance %.2f against duty %.2f" % (contact.mean(), DUTY))
    if (contact.sum(-1) == 0).any():
        reasons.append("flight phase")
    if lift_mm.min() < 12.0 or lift_mm.max() > 30.0:
        reasons.append("p95 lift mm %s" % np.round(lift_mm, 1).tolist())
    if min(len(sustained_events(swing[:, leg])) for leg in range(4)) < 3:
        reasons.append("liftoffs %s" % [len(sustained_events(swing[:, leg])) for leg in range(4)])
    if alternations < 4:
        reasons.append("only %d diagonal alternations" % alternations)
    if rate > 1.15:
        reasons.append("joint-rate p95 %.2fx limit" % rate)
    if np.abs(torque).max() >= 17.0:
        reasons.append("raw torque peak %.1f Nm" % np.abs(torque).max())
    return dict(
        speed=float(velocity[:, 0].mean()),
        height=float(np.median(root_z)),
        stance=float(contact.mean()),
        four_contact=float((contact.sum(-1) == 4).mean()),
        flight=float((contact.sum(-1) == 0).mean()),
        lift_mm=np.round(lift_mm, 1).tolist(),
        liftoffs=[len(sustained_events(swing[:, leg])) for leg in range(4)],
        alternations=alternations,
        rate_p95=float(rate),
        raw_peak=float(np.abs(torque).max()),
        reasons=reasons,
    )


def main():
    args = get_args()
    reference = np.load(REFERENCE)
    target = reference["joint_pos_rad"].astype(np.float32)
    stretch = float(os.environ.get("URDF4_REPLAY_STRETCH", "1"))
    if stretch != 1.0:
        # Time-stretch probe: same joint path, slower, to see whether Kp=40 can
        # follow this shape when the 0.12 s swing is not the limiting factor.
        grid = np.linspace(0.0, 1.0, int(round((len(target) - 1) * stretch)) + 1)
        source = np.linspace(0.0, 1.0, len(target))
        target = np.stack([np.interp(grid, source, target[:, joint]) for joint in range(12)], axis=1)
        target = target.astype(np.float32)
    env = make_env(args)
    expected = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    if list(env.dof_names) != expected or list(reference["joint_names"]) != expected:
        raise SystemExit("joint order mismatch")
    foot_ids = [env.foot_legs.index(leg) for leg in ("FL", "FR", "RL", "RR")]
    env.reset()
    zero = torch.zeros(1, 12, device=env.device)
    with torch.no_grad():
        for _ in range(int(2.0 / env.dt)):
            env.step(zero)
        stand = env.dof_pos[0].detach().cpu().numpy()
        sag = env.default_dof_pos[0].detach().cpu().numpy() - stand
        print("stance sag rad %s" % np.round(sag, 3).tolist(), flush=True)
        for alpha in np.linspace(0.0, 1.0, 20, endpoint=False):
            pose = (1.0 - alpha) * stand + alpha * target[0]
            action = torch.tensor(actions_for(env, pose, stand), device=env.device)
            env.step(action.unsqueeze(0))
        height, contact, velocity, joint_vel, torque, root_z, features = ([] for _ in range(7))
        for frame in target:
            _, _, _, done, _ = env.step(torch.tensor(actions_for(env, frame, stand), device=env.device).unsqueeze(0))
            if bool(done.any()):
                raise SystemExit("Isaac replay reset before the clip finished")
            sole = (env.feet_pos[0, foot_ids, 2] - env.env_origins[0, 2] - 0.02).detach().cpu().numpy()
            height.append(sole)
            contact.append(env.get_foot_contact_mask()[0, foot_ids].detach().cpu().numpy())
            velocity.append(torch.stack((
                env.base_lin_vel[0, 0], env.base_lin_vel[0, 1], env.base_ang_vel[0, 2]
            )).detach().cpu().numpy())
            joint_vel.append(env.dof_vel[0].detach().cpu().numpy())
            torque.append(env.raw_pd_torques[0].detach().cpu().numpy())
            root_z.append(float(env.root_states[0, 2] - env.env_origins[0, 2]))
            relative = env.feet_pos[0, foot_ids] - env.root_states[0, :3]
            quat = env.base_quat[0:1].expand(4, 4)
            feet = quat_rotate_inverse(quat, relative).reshape(-1)
            feature = torch.cat((
                env.dof_pos[0], env.dof_vel[0], feet, env.base_lin_vel[0],
                env.base_ang_vel[0], env.projected_gravity[0],
                (env.root_states[0, 2] - env.env_origins[0, 2]).unsqueeze(0),
            ))
            features.append(feature.detach().cpu().numpy())
    skip = int(0.5 / env.dt)
    achieved = np.stack(features)[:, 0:12]
    print(
        "dof_span %s ref_span %s track_p95 %s sole_max_mm %s" % (
            np.round(achieved.max(0) - achieved.min(0), 3).tolist(),
            np.round(target.max(0) - target.min(0), 3).tolist(),
            np.round(np.quantile(np.abs(achieved - target), 0.95, axis=0), 3).tolist(),
            np.round(np.stack(height).max(0) * 1000.0, 1).tolist(),
        ),
        flush=True,
    )
    result = audit(
        np.stack(height)[skip:], np.stack(contact)[skip:], np.stack(velocity)[skip:],
        np.stack(joint_vel)[skip:], np.stack(torque)[skip:], np.asarray(root_z)[skip:],
    )
    print(
        "speed=%.3f height=%.3f stance=%.2f four_contact=%.2f flight=%.3f lift_mm=%s "
        "liftoffs=%s alternations=%d rate_p95=%.2f raw_peak=%.1f" % (
            result["speed"], result["height"], result["stance"], result["four_contact"],
            result["flight"], result["lift_mm"], result["liftoffs"], result["alternations"],
            result["rate_p95"], result["raw_peak"],
        ),
        flush=True,
    )
    if stretch != 1.0:
        print("URDF4_REPLAY_PROBE stretch=%.2f" % stretch, flush=True)
        return
    if result["reasons"]:
        print("URDF4_REPLAY_REJECT " + "; ".join(result["reasons"]), flush=True)
        raise SystemExit(1)
    out = "/home/nszb/gym/unitree_rl_gym/datasets/urdf4_nmpc/forward_0p10.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = np.stack(features)[skip:]
    np.savez_compressed(
        out,
        amp_features=kept.astype(np.float32),
        joint_pos_rad=kept[:, 0:12],
        joint_vel_rad_s=kept[:, 12:24],
        command=np.tile(np.array([0.10, 0.0, 0.0], np.float32), (len(kept), 1)),
        frame_dt_s=np.float32(env.dt),
    )
    print("URDF4_REPLAY_OK %s frames=%d" % (out, len(kept)), flush=True)


if __name__ == "__main__":
    main()
