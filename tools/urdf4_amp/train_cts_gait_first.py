"""Explicit gait-first continuation; --max_iterations is the total target count.

This entry point implements the user's requested training stage. It does not
create or change a behavior-acceptance receipt.
"""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym  # must precede torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry


def main():
    args=get_args()
    args.task='urdf4_amp_cts_v1'
    args.resume=True
    if args.max_iterations is None:args.max_iterations=30000
    if args.num_envs is None:args.num_envs=4096
    if args.load_run is None:
        args.load_run=str(ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/trust300/run')
        if args.checkpoint is None:args.checkpoint=100
    if args.run_name is None:args.run_name='gait_first_30000'
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.rewards.scales.command_tracking=0.
    train.runner.training_objective='gait_first'
    train.runner.experiment_name='urdf4_amp_cts_gait_first'
    train.runner.load_optimizer=False  # New objective, fresh optimizer moments.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,_=task_registry.make_alg_runner(env,args.task,args=args,train_cfg=train)
    remaining=args.max_iterations-runner.current_learning_iteration
    if remaining<=0:
        raise ValueError('Target iteration must exceed the loaded checkpoint iteration')
    assert 'command_tracking' not in env.reward_scales
    print('GAIT FIRST: start=%d target=%d updates=%d active_rewards=%s'%(
        runner.current_learning_iteration,args.max_iterations,remaining,env.reward_scales),flush=True)
    runner.learn(remaining,init_at_random_ep_len=True)


if __name__=='__main__':main()
