"""Standalone V16 viewer/diagnostic: every command lasts 5 simulated seconds."""
import sys
import json
import time
from pathlib import Path
import isaacgym  # Must precede torch.
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'unitree_rl_gym/legged_gym/scripts'))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args, task_registry

MOVEMENTS = (
    ('forward', (.20, 0., 0.)), ('backward', (-.20, 0., 0.)),
    ('left', (0., .20, 0.)), ('right', (0., -.20, 0.)),
    ('forward_left', (.20, .10, 0.)), ('forward_right', (.20, -.10, 0.)),
    ('backward_left', (-.20, .10, 0.)), ('backward_right', (-.20, -.10, 0.)),
    ('turn_left', (0., 0., .30)), ('turn_right', (0., 0., -.30)),
    ('combined', (.20, .08, .25)), ('combined_reverse', (-.20, -.08, -.25)),
)


def main():
    args = get_args()
    if args.task not in ('rs01_omni_v20_geometry', 'rs01_omni_v20_bounded_hip', 'rs01_omni_v16_guarded_support', 'rs01_omni_v17_balance', 'rs01_omni_v18_balance18', 'rs01_omni_v18_balance14', 'rs01_omni_v18_balance_soft', 'rs01_omni_v19_placement', 'rs01_omni_v19_placement_soft'):
        raise ValueError('This diagnostic is for V16/V17/V18 RS01 tasks')
    sequence = [('march', (0., 0., 0.))]
    for movement in MOVEMENTS:
        sequence.extend((movement, ('march', (0., 0., 0.))))
    cfg, train = task_registry.get_cfgs(args.task)
    count = args.num_envs or args.eval_envs
    _set_nominal_eval_cfg(cfg, 5. * len(sequence), count)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                             train_cfg=train, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    results, finite, max_speed = [], True, 0.
    with torch.no_grad():
        for name, values in sequence:
            command = torch.tensor(values, device=env.device)
            print(f'{name}: vx/vy/wz={values}, 5 s, gait=march/move', flush=True)
            samples, resets = [], 0
            for step in range(round(5. / env.dt)):
                # Explicit gait=1 even at zero: never random stand. Preserve
                # phase and the existing direction-reference transition logic.
                env.set_evaluation_command(command, 1.)
                env.compute_observations()
                obs, _, reward, done, _ = env.step(policy(env.get_observations()))
                resets += int(done.sum())
                finite &= bool(torch.isfinite(obs).all() & torch.isfinite(reward).all())
                max_speed = max(max_speed, float(env.rs01_step_max_speed.max()))
                if step >= round(1. / env.dt):
                    samples.append(torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1).clone())
                if not finite:
                    raise RuntimeError('Non-finite rollout; aborting')
            velocity = torch.stack(samples)
            results.append(dict(stage=name, command=list(values), resets=resets,
                mean_vx_vy_wz=velocity.mean((0, 1)).tolist(),
                rmse_vx_vy_wz=(velocity-command).square().mean((0, 1)).sqrt().tolist()))
    result = dict(task=args.task, run=args.load_run, checkpoint=args.checkpoint,
                  seed=args.seed, num_envs=env.num_envs, duration_s=5.*len(sequence),
                  interval_s=5., policy_dt_s=env.dt, finite=finite,
                  resets=sum(x['resets'] for x in results),
                  max_joint_speed_rad_s=max_speed, stages=results)
    output = Path(__file__).with_name(f'sequence_{args.checkpoint}_{args.seed}_{time.time_ns()}.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Finished: resets={result["resets"]}, finite={finite}; {output}')


if __name__ == '__main__':
    main()
