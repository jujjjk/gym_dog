"""User-authorized robustness continuation, capped at 0.50 m/s.

--prepare runs 50 finite updates and binds a training-only launch receipt.
--run starts the requested long run; it never expands the command envelope.
"""
import argparse,hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_speed_fix_20261117'
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--output',type=Path)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task='urdf4_amp_cts_v4';args.headless=True
    args.num_envs=args.num_envs or (32 if own.prepare else 4096)
    folder=BASE/'robust050';receipt_path=folder/'training_receipt.json';initial=folder/'initial/model_0.pt'
    seed=BASE/'selected/model_50.pt';saved=torch.load(seed,map_location='cpu')
    cfg,train=task_registry.get_cfgs(args.task)
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=True
    cfg.commands.ranges.lin_vel_x=[-.30,.50];cfg.commands.ranges.lin_vel_y=[-.20,.20];cfg.commands.ranges.ang_vel_yaw=[-.60,.60]
    cfg.expand.stages=[[-.20,.30,.10,.40],[-.30,.50,.20,.60]];cfg.expand.initial_stage=1;cfg.expand.max_stage=1
    train.runner.v3_preflight=str(receipt_path);train.runner.save_interval=10 if own.prepare else 100
    output=own.output or (folder/'smoke50' if own.prepare else ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_v4_feedback'/(datetime.now().strftime('%b%d_%H-%M-%S')+'_robust050'))
    output.mkdir(parents=True,exist_ok=False)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV4Runner(env,class_to_dict(train),str(output),device=args.rl_device)
    file=Path(__file__).resolve();runner.experiment['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    if own.prepare:
        nominal=json.load(open(BASE/'selected_nominal60/result.json'))
        assert nominal['core_passed'] and next(r['passed'] for r in nominal['cases'] if r['motion']=='fast_forward')
        assert saved['contract']['urdf_sha256']==env.urdf_sha256
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        runner.style.load_state_dict(saved['amp_state']);runner.style.clear_history()
        runner._freeze_seed_policy();runner.env.curriculum_stage=1
        runner.seed_migration=dict(checkpoint=str(seed),sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),optimizers_fresh=True,reason='user requested disturbance training capped at .50')
        initial.parent.mkdir(parents=True,exist_ok=False);runner.save(str(initial))
        before=runner.alg.actor_critic.actor[-1].weight.detach().clone()
        runner.learn(50,init_at_random_ep_len=False)
        assert torch.isfinite(env.obs_buf).all() and torch.isfinite(env.rew_buf).all()
        assert not torch.equal(before,runner.alg.actor_critic.actor[-1].weight)
        assert runner.last_metrics['amp_valid_fraction']>.5
        assert env.curriculum_stage<=1 and env.cfg.push.enabled and float(env.push_count.sum())>0
        result=dict(approved_for_training=True,approval_scope='robustness optimization only, not deployment or successful robustness validation',
            authorization='user explicitly requested disturbance training with forward maximum .50 m/s',
            checkpoint=str(initial),checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),contract=runner.contract,source_sha256=runner.experiment['source_sha256'],
            nominal_core_passed=True,nominal_forward_050_passed=True,robust_behavior_passed=False,
            smoke_updates=50,smoke_pushes=float(env.push_count.sum()),smoke_nominal_fraction=float(env.nominal_mask.float().mean()),
            smoke_actor_updated=True,smoke_finite=True,metrics=runner.last_metrics,long_training_started=False)
        receipt_path.write_text(json.dumps(result,indent=2)+'\n');print('ROBUST050 READY',str(receipt_path),flush=True)
    else:
        receipt=json.load(open(receipt_path));assert receipt['approved_for_training'] and receipt['smoke_finite']
        runner.load(initial);runner._validate_training_request(args.max_iterations or 30000)
        print('ROBUST050 LONG TRAIN',str(output),flush=True)
        runner.learn((args.max_iterations or 30000)-runner.current_learning_iteration,init_at_random_ep_len=False)


if __name__=='__main__':main()
