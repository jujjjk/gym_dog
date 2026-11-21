"""User-launched expansion training; target updates count from this new stage."""
import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--target_stage',type=int,choices=[0,1,2],default=None)
    parser.add_argument('--x_only',action='store_true')
    own,rest=parser.parse_known_args();sys.argv=sys.argv[:1]+rest
    if own.target_stage is None:own.target_stage=2 if own.x_only else 1
    args=get_args();args.task='urdf4_amp_cts_expand_x_v1' if own.x_only else 'urdf4_amp_cts_expand_v1';args.resume=True
    if args.max_iterations is None:args.max_iterations=30000
    if args.num_envs is None:args.num_envs=4096
    if args.run_name is None:args.run_name=('expand_x_robust_stage%d' if own.x_only else 'expand_robust_stage%d')%own.target_stage
    if args.load_run is None:
        args.load_run=str(ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000')
        if args.checkpoint is None:args.checkpoint=17800
    cfg,train=task_registry.get_cfgs(args.task);cfg.expand.max_stage=own.target_stage
    low,high,side,yaw=cfg.expand.stages[own.target_stage]
    cfg.commands.ranges.lin_vel_x=[low,high];cfg.commands.ranges.lin_vel_y=[-side,side];cfg.commands.ranges.ang_vel_yaw=[-yaw,yaw]
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,_=task_registry.make_alg_runner(env,args.task,args=args,train_cfg=train)
    remaining=args.max_iterations-runner.current_learning_iteration
    if remaining<=0:raise ValueError('Target must exceed this stage checkpoint iteration')
    print('EXPANSION: stage target %d, iteration %d -> %d, %d updates'%(
        own.target_stage,runner.current_learning_iteration,args.max_iterations,remaining),flush=True)
    runner.learn(remaining,init_at_random_ep_len=True)

if __name__=='__main__':main()
