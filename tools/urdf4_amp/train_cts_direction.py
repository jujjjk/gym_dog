"""Finite preflight or an explicitly authorized direction training run."""
import argparse,hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_direction import Urdf4AMPDirectionRunner


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--receipt',type=Path,default=ROOT/'artifacts/urdf4_direction_20261115/preflight/receipt.json')
    own,rest=parser.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_direction_v1';args.headless=True
    target=args.max_iterations or (100 if own.preflight else 30000)
    if own.preflight and not 1<=target<=500:raise ValueError('Preflight is limited to 500 updates')
    if args.num_envs is None:args.num_envs=1024 if own.preflight else 4096
    if own.preflight:
        checkpoint=ROOT/'artifacts/urdf4_selection_x_20261115/best/model_0.pt'
    else:
        receipt=json.loads(own.receipt.read_text())
        if not receipt.get('approved_for_training'):raise ValueError('Direction preflight has not passed')
        checkpoint=Path(receipt['checkpoint'])
    if args.load_run is not None:
        if args.checkpoint is None:raise ValueError('Specify an exact checkpoint number with load_run')
        checkpoint=Path(args.load_run)/('model_%d.pt'%args.checkpoint)
    output=own.output or ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_direction'/(
        datetime.now().strftime('%b%d_%H-%M-%S')+'_omni_direction_30000')
    output.mkdir(parents=True,exist_ok=True)
    cfg,train=task_registry.get_cfgs(args.task)
    train.runner.direction_preflight=str(own.receipt)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPDirectionRunner(env,class_to_dict(train),str(output),device=args.rl_device)
    runner.load(str(checkpoint),load_optimizer=not own.preflight)
    assert env.obs_buf.shape==(env.num_envs,1121)
    assert env.cfg.rewards.heading_gain==0 and len(env.reward_scales)==6
    with torch.no_grad():
        public=env.obs_buf[:,:1020].clone();expected=runner.alg.actor_critic.act_inference(public)
        altered=env.obs_buf.clone();altered[:,1020:]=torch.randn_like(altered[:,1020:])
        assert torch.equal(expected,runner.alg.actor_critic.act_inference(altered))
        if own.preflight:
            baseline=torch.jit.load(str(ROOT/'artifacts/urdf4_selection_x_20261115/best/student_0.pt'),map_location=env.device)
            error=float((baseline(public)-expected).abs().max());assert error<1.e-6,error
            (output/'technical_checks.json').write_text(json.dumps(dict(warm_start_error=error,
                public_size=1020,packed_size=1121,private_fields_do_not_affect_student=True,
                duplicate_heading_reward=False,rewards=env.reward_scales),indent=2)+'\n')
    remaining=target-runner.current_learning_iteration
    if remaining<=0:raise ValueError('Target must exceed the current direction-stage iteration')
    runner.save(str(output/('model_%d.pt'%runner.current_learning_iteration)))
    (output/'launch.json').write_text(json.dumps(dict(task=args.task,pid=__import__('os').getpid(),
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        start_iteration=runner.current_learning_iteration,target_iteration=target,num_envs=env.num_envs,
        gpu=args.sim_device,preflight=own.preflight,external_pose_assumption=class_to_dict(cfg.direction)),indent=2)+'\n')
    print('DIRECTION TRAIN: start=%d target=%d updates=%d output=%s'%(runner.current_learning_iteration,target,remaining,output),flush=True)
    initial_actor=[value.detach().clone() for value in runner.alg.actor_critic.actor.parameters()] if own.preflight else None
    runner.learn(remaining,init_at_random_ep_len=True)
    if own.preflight:
        changed=any(not torch.equal(a,b) for a,b in zip(initial_actor,runner.alg.actor_critic.actor.parameters()))
        assert changed and runner.last_metrics['amp_valid_fraction']>.5 and runner.last_metrics['amp_bonus_step']>0
        (output/'learning_checks.json').write_text(json.dumps(dict(actor_updated=changed,
            iterations=runner.current_learning_iteration,amp_valid_fraction=runner.last_metrics['amp_valid_fraction'],
            amp_bonus_step=runner.last_metrics['amp_bonus_step'],metrics=runner.last_metrics),indent=2)+'\n')


if __name__=='__main__':main()
