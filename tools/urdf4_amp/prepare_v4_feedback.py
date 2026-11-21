"""Finite supervised feedback initialization; no long PPO run."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner
from prepare_v3_seed import collect_samples


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--data-only',action='store_true')
    p.add_argument('--balance-variant',type=int,default=1)
    p.add_argument('--balance-archive',type=Path)
    p.add_argument('--speed-servo',action='store_true')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;assert 0<=own.steps<=3000
    args=get_args();args.task='urdf4_amp_cts_v4';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.amp_cts.balance_variant=own.balance_variant
    if own.balance_archive:cfg.amp_cts.balance_archive=str(own.balance_archive.resolve())
    cfg.direction.speed_feedback_enabled=own.speed_servo
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    file=Path(cfg.amp_cts.feedback_samples)
    if not file.exists():file.parent.mkdir(parents=True,exist_ok=False);collect_samples(env,file.parent,manifest_clock=True,include_transitions=False)
    if own.data_only:return
    own.output.mkdir(parents=True,exist_ok=False);runner=Urdf4AMPV4Runner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
    runner.warm_start(ROOT/'artifacts/urdf4_v3_selection_20261117/best/model_700.pt');model=runner.alg.actor_critic
    params=list(model.actor.parameters())+list(model.student_encoder.parameters());optimizer=torch.optim.Adam(params,lr=1e-4)
    valid=runner.anchor_holdout.nonzero().flatten()
    def error(ids):
        obs=runner.anchor_observations[ids];public,state,_=model.split(obs);target=runner.anchor_actions[ids]
        student=model.public_mean(public);teacher=model.executed_mean(public,model.actor(torch.cat((model.control_frame(public),model.teacher_context(state)),1)))
        context=model.student_encoder(public)
        with torch.no_grad():latent=model.teacher_context(state)
        bc=.5*((student-target).square().mean()+(teacher-target).square().mean())
        return bc+.05*(context-latent).square().mean(),(student-target).square().mean()
    with torch.no_grad():before=float(error(valid[:4096])[1])
    for k in range(own.steps):
        ids=torch.multinomial(runner.anchor_weights,2048,replacement=True);loss,mse=error(ids)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (k+1)%100==0:
            with torch.no_grad():print('V4 FEEDBACK',k+1,'holdout',float(error(valid[:4096])[1]),flush=True)
    runner._freeze_seed_policy();runner.save(str(own.output/'run/model_0.pt'))
    with torch.no_grad():after=float(error(valid[:4096])[1])
    (own.output/'result.json').write_text(json.dumps(dict(steps=own.steps,samples=len(runner.anchor_actions),before_mse=before,after_mse=after,teacher_encoder_frozen=True,behavior_accepted=False),indent=2)+'\n')


if __name__=='__main__':main()
