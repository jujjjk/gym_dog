"""Fixed forward command: old/new V22 checkpoint regression, never training."""
import json
import argparse
import sys
import time
from pathlib import Path
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_env import Rs01OmniV22Robot
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--action_noise_std', type=float, default=0.)
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    if args.task not in ('rs01_omni_v22_sensor', 'rs01_v22_phase_scratch', 'rs01_v22_phase_lift'):
        raise ValueError('V22 baseline or phase-scratch only')
    cfg, train = task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg, args.duration_s, args.num_envs or 1)
    cfg.sensor.enabled = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train.runner.resume = True  # Evaluation ONLY. The training launcher refuses resume.
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                             train_cfg=train, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    command = torch.tensor([.2, 0., 0.], device=env.device)
    samples, reset_count, rewards = [], 0, {}
    # Capture each reward ONCE; V21's coordination reward advances history.
    def record(name, func):
        def wrapped():
            value = func()
            rewards[name] = rewards.get(name, 0.) + float((value * env.reward_scales[name]).mean())
            return value
        return wrapped
    env.reward_functions = [record(name, func) for name, func in zip(env.reward_names, env.reward_functions)]
    max_contact_s = torch.zeros((env.num_envs, 4), device=env.device)
    contact_s = torch.zeros_like(max_contact_s)
    swing_total = torch.zeros_like(contact_s)
    swing_air = torch.zeros_like(contact_s)
    stance_total = torch.zeros_like(contact_s)
    stance_contact = torch.zeros_like(contact_s)
    liftoffs = torch.zeros_like(contact_s)
    previous_contact = env.get_foot_contact_mask().clone()
    phase_distance = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for _ in range(round(args.duration_s / env.dt)):
            env.set_evaluation_command(command, 1.)
            env.compute_observations()
            phase = env._gait_phase().clone()
            torch.testing.assert_close(env.obs_buf[:, 48:50], torch.stack((
                torch.sin(2*torch.pi*phase), torch.cos(2*torch.pi*phase)), 1))
            actions = policy(env.get_observations())
            if own.action_noise_std:
                actions = actions + own.action_noise_std * torch.randn_like(actions)
            obs, _, reward, done, _ = env.step(actions)
            assert bool(torch.isfinite(obs).all() and torch.isfinite(reward).all())
            reset_count += int(done.sum())
            contact = env.get_foot_contact_mask()
            desired_swing = ~env._desired_contact_mask()
            swing_total += desired_swing
            swing_air += desired_swing & ~contact
            stance_total += ~desired_swing
            stance_contact += ~desired_swing & contact
            liftoffs += previous_contact & ~contact & ~done.bool()[:, None]
            previous_contact = contact.clone()
            phase_distance += torch.where(done.bool(), 0., (env._gait_phase()-phase).remainder(1.))
            contact_s = torch.where(contact & ~done.bool()[:, None], contact_s + env.dt, 0.)
            max_contact_s = torch.maximum(max_contact_s, contact_s)
            extra = getattr(env, 'exchange_penalty', torch.zeros(env.num_envs, device=env.device))
            old = Rs01OmniV22Robot._reward_prolonged_all_feet_contact(env)
            new = env._reward_prolonged_all_feet_contact()
            samples.append(torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3],
                env.rpy[:, :2], env.root_states[:, 2:3],
                env.feet_pos[:, :, 2] - cfg.rewards.foot_collision_radius_m,
                env.raw_pd_torques.abs().amax(1, keepdim=True),
                env.motor_electromagnetic_torques.abs().amax(1, keepdim=True),
                (~contact.any(1)).float()[:, None], old[:, None], new[:, None], extra[:, None]), 1).clone())
    values = torch.stack(samples)
    result = dict(task=args.task, load_run=args.load_run, checkpoint=args.checkpoint,
        seed=args.seed, duration_s=args.duration_s, command=[.2, 0., 0.],
        num_envs=env.num_envs, sensor='clean', finite=True, resets=reset_count,
        action_noise_std=own.action_noise_std,
        phase_cycles=phase_distance.tolist(),
        swing_air_fraction_by_foot=(swing_air/swing_total.clamp(min=1.)).mean(0).tolist(),
        stance_contact_fraction_by_foot=(stance_contact/stance_total.clamp(min=1.)).mean(0).tolist(),
        liftoffs_by_env_foot=liftoffs.tolist(),
        mean_vx_vy_wz=values[:, :, :3].mean((0, 1)).tolist(),
        roll_pitch_rms_deg=(values[:, :, 3:5].square().mean((0, 1)).sqrt()*180/torch.pi).tolist(),
        world_z_peak_to_peak_mm=float((values[:, :, 5].amax(0)-values[:, :, 5].amin(0)).mean()*1000),
        foot_order=sorted(env.foot_slot_by_leg, key=env.foot_slot_by_leg.get),
        foot_clearance_max_mm=(values[:, :, 6:10].amax((0, 1))*1000).tolist(),
        maximum_continuous_contact_s=max_contact_s.amax(0).tolist(),
        raw_pd_peak_nm=float(values[:, :, 10].max()), motor_peak_nm=float(values[:, :, 11].max()),
        flight_fraction=float(values[:, :, 12].mean()),
        original_stall_mean=float(values[:, :, 13].mean()), new_stall_mean=float(values[:, :, 14].mean()),
        added_stall_mean=float((values[:, :, 14]-values[:, :, 13]).mean()),
        exchange_mean=float(values[:, :, 15].mean()),
        mean_reward_contribution_per_step={k: v/len(samples) for k, v in rewards.items()})
    output = Path('/home/nszb/gym/artifacts/rs01_v22_phase_scratch')
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{args.task}_{args.checkpoint}_{time.time_ns()}.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    print(path, flush=True)


if __name__ == '__main__':
    main()
