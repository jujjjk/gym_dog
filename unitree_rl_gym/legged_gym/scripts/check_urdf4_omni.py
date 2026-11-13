"""Pre-training checks for urdf4_omni_v1.  Run before every long training:

    python legged_gym/scripts/check_urdf4_omni.py --headless

1. Asset contract: DOF/body order, total mass, default pose inside the limits.
2. Nominal stand (no randomization, zero action, 4 s): settled height, tilt,
   foot contacts, torques.
3. Randomized rollout with random actions (4 s): finite observations and
   rewards, dimensions, every reward term exercised.
"""

import isaacgym  # noqa: F401  (must precede torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

TASK = "urdf4_omni_v1"


def make_env(args, test):
    env_cfg, _ = task_registry.get_cfgs(name=TASK)
    env_cfg.env.num_envs = 64
    env_cfg.env.test = test
    if test:
        env_cfg.noise.add_noise = False
        env_cfg.push.enabled = False
        env_cfg.commands.fixed_command = [0.0, 0.0, 0.0]
        for flag in (
            "randomize_friction", "randomize_restitution", "randomize_base_mass",
            "randomize_base_com", "randomize_link_mass", "randomize_armature",
        ):
            setattr(env_cfg.domain_rand, flag, False)
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    return env


def check_stand(env):
    env.reset()
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    peak_motor = torch.zeros(env.num_dof, device=env.device)
    for _ in range(int(4.0 / env.dt)):
        _, _, _, dones, _ = env.step(zero)
        peak_motor = torch.maximum(peak_motor, env.motor_torques.abs().amax(dim=0))
        if bool(dones.any()):
            raise RuntimeError("Nominal zero-action stand terminated")
    tilt_deg = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1.0, 1.0))).mean().item()
    sag = (env.dof_pos - env.default_dof_pos).mean(dim=0)
    print(f"[stand] trunk height {env.root_states[:, 2].mean().item():.4f} m, tilt {tilt_deg:.2f} deg")
    print(f"[stand] foot contact share {env.get_foot_contact_mask().float().mean(dim=0).tolist()}")
    print(f"[stand] foot centre height {env.feet_pos[:, :, 2].mean(dim=0).tolist()}")
    for index, name in enumerate(env.dof_names):
        print(
            f"[stand]   {name:15s} q-default {sag[index].item():+.4f} rad, "
            f"torque {env.motor_torques[:, index].mean().item():+.2f} Nm, 4 s peak {peak_motor[index].item():.2f} Nm"
        )
    if not bool(env.get_foot_contact_mask().all()):
        raise RuntimeError("Not all feet are loaded in the nominal stand")


def check_rollout(env):
    obs, privileged = env.reset()
    print(f"[rollout] actor obs {tuple(obs.shape)}, critic obs {tuple(privileged.shape)}")
    resets = 0
    for _ in range(int(4.0 / env.dt)):
        actions = 0.5 * torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, privileged, rewards, dones, _ = env.step(actions)
        resets += int(dones.sum())
        for name, value in (("obs", obs), ("critic obs", privileged), ("reward", rewards)):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Non-finite {name}")
    print(f"[rollout] finite for 4 s with random actions, {resets} resets, reward terms: {sorted(env.reward_names)}")
    print(f"[rollout] max |raw PD torque| {env.raw_pd_torques.abs().max().item():.1f} Nm, "
          f"max |motor torque| {env.motor_torques.abs().max().item():.1f} Nm")
    print(f"[rollout] pushes so far per robot {env.push_count.mean().item():.2f}, "
          f"IMU latency steps {env.imu_latency_steps.min().item()}-{env.imu_latency_steps.max().item()}")


def main():
    args = get_args()
    # Isaac Gym supports one simulation per process: nominal stand by default,
    # randomized rollout when a run name is given.
    randomized = args.run_name == "randomized"
    env = make_env(args, test=not randomized)
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    masses = [body.mass for body in env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])]
    print(f"[asset] DOF order {env.dof_names}")
    print(f"[asset] bodies {body_names}")
    print(f"[asset] foot columns {env.foot_legs}, total mass {sum(masses):.4f} kg (robot 0)")
    print(f"[asset] default pose {env.default_dof_pos[0].tolist()}")
    inside = (env.default_dof_pos[0] > env.dof_pos_limits[:, 0]) & (env.default_dof_pos[0] < env.dof_pos_limits[:, 1])
    if not bool(inside.all()):
        raise RuntimeError("Default pose violates the soft joint limits")
    if randomized:
        check_rollout(env)
    else:
        check_stand(env)
    print("[ok]")


if __name__ == "__main__":
    main()
