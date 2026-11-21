"""Manual long-training entry; agent smoke mode is bounded to 200 PPO updates."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone,timedelta
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
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--v5',action='store_true',help='Command-conditioned feedback and the qualified extra Gym reference')
    p.add_argument('--smoke',action='store_true');p.add_argument('--focus-fast',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--seed-checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight',type=Path,default=BASE/'preflight.json')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task='urdf4_amp_cts_v5' if own.v5 else 'urdf4_amp_cts_v4';args.headless=True
    if own.v5 and own.preflight==BASE/'preflight.json':own.preflight=BASE/'v5_preflight.json'
    total=args.max_iterations or (50 if own.smoke else 30000)
    assert not own.smoke or 1<=total<=200
    assert not own.focus_fast or own.smoke,'Focused override is restricted to finite smoke'
    if own.smoke:
        if not own.seed_checkpoint:raise ValueError('Specify a V4 smoke seed')
        seed=own.seed_checkpoint
    else:
        receipt=json.loads(own.preflight.read_text())
        if not receipt.get('approved_for_training'):
            raise ValueError('V4 is not ready for long training: '+receipt.get('reason','behavior preflight has not passed'))
        seed=own.seed_checkpoint or Path(receipt['checkpoint'])
    saved=torch.load(seed,map_location='cpu');cfg,train=task_registry.get_cfgs(args.task)
    for key in ['balance_archive','balance_variant']:setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
    if own.v5:
        cfg.amp_cts.extra_reference=saved['experiment']['env']['amp_cts']['extra_reference']
        cfg.amp_cts.adaptive_feedback_phase=saved['experiment']['env']['amp_cts']['adaptive_feedback_phase']
        cfg.amp_cts.use_public_prior=saved['experiment']['env']['amp_cts'].get('use_public_prior',False)
        cfg.amp_cts.reward_revision=saved['experiment']['env']['amp_cts']['reward_revision']
        cfg.amp_cts.public_prior_archive=saved['experiment']['env']['amp_cts'].get('public_prior_archive',cfg.amp_cts.public_prior_archive)
        cfg.direction.mid_cross_integral_gain=saved['experiment']['env']['direction'].get('mid_cross_integral_gain',0.)
        cfg.amp_cts.public_feedback_history=saved['experiment']['env']['amp_cts'].get('public_feedback_history',False)
        for key in ('public_obs_size','num_observations','num_privileged_obs'):
            setattr(cfg.env,key,saved['experiment']['env']['env'][key])
    train.runner.v3_preflight=str(own.preflight);train.runner.save_interval=10 if own.smoke else 100
    args.num_envs=args.num_envs or (256 if own.smoke else 4096)
    name=datetime.now(timezone(timedelta(hours=8))).strftime('%b%d_%H-%M-%S')+'_native_feedback_omni'
    output=own.output or ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_v4_feedback'/name
    if not own.check_only:output.mkdir(parents=True,exist_ok=False)
    runner_class=Urdf4AMPV4Runner
    if own.v5:
        from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner
        runner_class=Urdf4AMPV5Runner
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=runner_class(env,class_to_dict(train),None if own.check_only else str(output),device=args.rl_device);runner.load(seed)
    if own.focus_fast:
        env.curriculum_stage=1
        if own.v5:
            commands=torch.tensor([[.4,0,0],[.45,0,0],[.5,0,0]],device=env.device)[torch.arange(env.num_envs,device=env.device)%3]
        else:commands=torch.tensor([.5,0,0],device=env.device).repeat(env.num_envs,1)
        env.set_command_override(commands);env.reset()
    model=runner.alg.actor_critic;before=[v.detach().clone() for v in model.actor.parameters()]
    feedback_before=(model.balance_delta if own.v5 else model.balance_gain).detach().clone()
    with torch.no_grad():
        obs=env.obs_buf.clone();expected=model.act_inference(obs);obs[:,model.public_size:]=torch.randn_like(obs[:,model.public_size:]);assert torch.equal(expected,model.act_inference(obs))
    assert total>runner.current_learning_iteration
    if own.check_only:
        runner._validate_training_request(total-runner.current_learning_iteration)
        with torch.no_grad():env.step(expected);assert torch.isfinite(env.obs_buf).all()
        (BASE/'launch_check.json').write_text(json.dumps(dict(passed=True,envs=env.num_envs,PPO_updates=0,checkpoint=str(seed)),indent=2)+'\n');return
    (output/'launch.json').write_text(json.dumps(dict(smoke=own.smoke,focus_fast=own.focus_fast,seed=str(seed),seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),target=total,contract=runner.contract),indent=2)+'\n')
    runner.learn(total-runner.current_learning_iteration,init_at_random_ep_len=False)
    if own.smoke:
        assert any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters()))
        assert not any(p.requires_grad for p in model.teacher_encoder.parameters())
        assert runner.last_metrics['amp_valid_fraction']>.5
        feedback_updated=not torch.equal(feedback_before,model.balance_delta if own.v5 else model.balance_gain)
        if own.v5 and not model.use_public_prior:assert feedback_updated
        (output/'smoke.json').write_text(json.dumps(dict(passed=True,actor_updated=True,feedback_matrix_updated=feedback_updated,
            teacher_context_frozen=True,private_input_independent=True,metrics=runner.last_metrics),indent=2)+'\n')


if __name__=='__main__':main()
