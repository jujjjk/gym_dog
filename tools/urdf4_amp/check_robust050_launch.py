"""Validate the exact long-training profile without starting PPO."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_speed_fix_20261117/robust050'
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner


def restore(target,values):
    if isinstance(target,dict):
        target.update(values);return
    for key,value in values.items():
        if isinstance(value,dict):restore(getattr(target,key),value)
        else:setattr(target,key,value)


def main():
    args=get_args();args.task='urdf4_amp_cts_v4';args.headless=True;args.num_envs=args.num_envs or 4096
    seed=BASE/'initial/model_0.pt';saved=torch.load(seed,map_location='cpu');cfg,train=task_registry.get_cfgs(args.task)
    for name in ('commands','expand','direction'):restore(getattr(cfg,name),saved['experiment']['env'][name])
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    train.runner.v3_preflight=str(BASE/'training_receipt.json')
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV4Runner(env,class_to_dict(train),None,device=args.rl_device)
    entry=ROOT/'tools/urdf4_amp/train_robust050.py';runner.experiment['source_sha256'][str(entry)]=hashlib.sha256(entry.read_bytes()).hexdigest()
    runner.load(seed);runner._validate_training_request(30000)
    with torch.no_grad():
        env.step(runner.alg.actor_critic.act_inference(env.obs_buf));assert torch.isfinite(env.obs_buf).all()
    result=dict(passed=True,num_envs=env.num_envs,PPO_updates=0,max_iterations=30000,stage=env.curriculum_stage,max_stage=cfg.expand.max_stage,
        stages=cfg.expand.stages,push_enabled=cfg.push.enabled,nominal_fraction=float(env.nominal_mask.float().mean()),checkpoint=str(seed))
    (BASE/'launch_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
