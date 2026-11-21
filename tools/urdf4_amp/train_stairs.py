"""Train or verify a continuous stair course using an existing URDF4 AMP+CTS seed."""
import argparse
import json
from pathlib import Path
import sys

import isaacgym
import torch
import legged_gym.envs

from legged_gym.algorithms.urdf4_amp_stairs import Urdf4AMPStairsRunner
from legged_gym.envs.urdf4.amp_transition_training import restore_transition_contract
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

TASK = 'urdf4_amp_cts_stairs'
HEIGHTS = {'foundation': [.04], 'low': [.04,.05,.06], 'middle': [.08,.09,.10], 'full': [.12,.13,.14]}


def configure(args, checkpoint, heights, evaluation=False):
    cfg,train=task_registry.get_cfgs(TASK)
    restore_transition_contract(cfg,train,checkpoint)
    for name,value in checkpoint['contract']['rough_ground'].items():
        setattr(cfg.rough,name,value)
    # Preserve each checkpoint's geometry when verifying; training selects a height band.
    if 'stairs' in checkpoint['contract']:
        for name,value in checkpoint['contract']['stairs'].items():setattr(cfg.stairs,name,value)
    if not evaluation:
        cfg.stairs.heights_m=heights
        cfg.stairs.top_start_fraction=.25
        cfg.stairs.lift_gain=.75
        cfg.stairs.leveling_gain=.75
    cfg.env.test=evaluation
    train.runner.seed_weight_index=None
    train.runner.anchor_coefficient=20.
    train.runner.seed_trust_coef=.35
    train.runner.seed_action_deadband=.14
    train.runner.transition_sampling_std_max=.055
    train.runner.save_interval=24
    train.algorithm.learning_rate=3.e-5
    train.runner.encoder_learning_rate=2.5e-5
    return task_registry.make_env(TASK,args=args,env_cfg=cfg)[0],class_to_dict(train)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False,add_help=False,description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--level',choices=HEIGHTS,default='low')
    parser.add_argument('--evaluate',action='store_true')
    own,rest=parser.parse_known_args()
    sys.argv=sys.argv[:1]+rest
    args=get_args()
    args.task,args.headless=TASK,True
    args.num_envs=args.num_envs or (6 if own.evaluate else 96)
    options=torch.load(own.model,map_location='cpu')
    env,train=configure(args,options,HEIGHTS[own.level],own.evaluate)
    own.output.mkdir(parents=True,exist_ok=False)
    runner=Urdf4AMPStairsRunner(env,train,None if own.evaluate else str(own.output),device=args.rl_device)
    if own.evaluate:
        runner.load(own.model,load_optimizer=False)
        from evaluate_stairs import gym_evaluate
        gym_evaluate(env,runner.get_inference_policy(device=env.device),own.model,own.output)
        return
    runner.warm_start(own.model)
    runner.save(str(own.output/'model_0.pt'))
    runner.learn(args.max_iterations or 96,init_at_random_ep_len=False)
    rows=[json.loads(s) for s in (own.output/'metrics.jsonl').read_text().splitlines()]
    counts={name:sum(row[name] for row in rows) for name in
            ('stairs_up','stairs_through','stairs_falls','stairs_incomplete','stairs_episodes',
             'stairs_down','stairs_down_episodes')}
    if not counts['stairs_episodes']:
        raise RuntimeError('The stair smoke did not reach any course episode termination')
    report=dict(iterations=rows[-1]['iteration'],model=str(own.model.resolve()),level=own.level,counts=counts)
    (own.output/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    main()
