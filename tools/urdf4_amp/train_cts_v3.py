"""User starts long training. Agent smoke mode is explicitly limited to 200 updates."""
import argparse,hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--smoke',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--seed-checkpoint',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--preflight',type=Path,default=ROOT/'artifacts/urdf4_amp_v3_training/preflight.json')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task='urdf4_amp_cts_v3';args.headless=True
    target=args.max_iterations or (20 if own.smoke else 30000)
    if own.smoke and not 1<=target<=200:raise ValueError('Smoke is capped at 200 PPO updates')
    if args.num_envs is None:args.num_envs=256 if own.smoke else 4096
    if own.smoke:
        seed=own.seed_checkpoint or ROOT/'artifacts/urdf4_amp_v3_training/reference_init/run/model_0.pt'
    else:
        receipt=json.loads(own.preflight.read_text())
        if not receipt.get('approved_for_training'):raise ValueError('V3 smoke/behavior preflight has not passed')
        seed=own.seed_checkpoint or Path(receipt['checkpoint'])
    output=own.output or ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_v3'/(datetime.now().strftime('%b%d_%H-%M-%S')+'_newmass_omni_v3')
    if not own.check_only:output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs(args.task);train.runner.v3_preflight=str(own.preflight)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV3Runner(env,class_to_dict(train),None if own.check_only else str(output),device=args.rl_device)
    runner.load(seed);before=[p.detach().clone() for p in runner.alg.actor_critic.actor.parameters()]
    with torch.no_grad():
        obs=env.obs_buf.clone();expected=runner.alg.actor_critic.act_inference(obs);obs[:,1020:]=torch.randn_like(obs[:,1020:]);assert torch.equal(expected,runner.alg.actor_critic.act_inference(obs))
    if target<=runner.current_learning_iteration:raise ValueError('Target must exceed checkpoint iteration')
    if own.check_only:
        runner._validate_training_request(target-runner.current_learning_iteration)
        with torch.no_grad():
            env.step(expected)
            assert torch.isfinite(env.obs_buf).all() and torch.isfinite(env.rew_buf).all()
        result=dict(passed=True,training_updates=0,long_training_started=False,envs=env.num_envs,
            target_iterations=target,checkpoint=str(seed),checkpoint_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
            source_sha256=runner.experiment['source_sha256'],contract=runner.contract,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(args.rl_device))
        (own.preflight.parent/'launch_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print('V3 launch check passed; no training updates.',flush=True);return
    (output/'launch.json').write_text(json.dumps(dict(seed=str(seed),seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),smoke=own.smoke,target=target,envs=env.num_envs,contract=runner.contract),indent=2)+'\n')
    runner.learn(target-runner.current_learning_iteration,init_at_random_ep_len=False)
    if own.smoke:
        assert any(not torch.equal(a,b) for a,b in zip(before,runner.alg.actor_critic.actor.parameters()))
        assert runner.last_metrics['amp_valid_fraction']>.5 and runner.last_metrics['amp_bonus_step']>0
        (output/'smoke.json').write_text(json.dumps(dict(passed=True,actor_updated=True,student_private_input_independent=True,metrics=runner.last_metrics),indent=2)+'\n')


if __name__=='__main__':main()
