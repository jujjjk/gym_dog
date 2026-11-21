"""Bounded supervised command-head initialization; physical evaluation is separate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seed-model',type=Path,default=ROOT/'artifacts/urdf4_speed_fix_20261117/selected/model_50.pt')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;assert 0<=own.steps<=1000
    own.output.mkdir(parents=True,exist_ok=False);(own.output/'run').mkdir()
    args=get_args();args.task='urdf4_amp_cts_v5';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV5Runner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
    (own.output/'run/experiment_contract.json').write_text(json.dumps(runner.experiment,indent=2,default=str)+'\n')
    runner.warm_start(own.seed_model);model=runner.alg.actor_critic
    data=torch.load(cfg.amp_cts.extra_reference,map_location=env.device);observations=data['observations'][:,:1029];targets=data['actions'];held=data['holdout']
    training=(~held).nonzero().flatten();validation=held.nonzero().flatten()
    requires={name:p.requires_grad for name,p in model.named_parameters()}
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.speed_gain.requires_grad_(True);optimizer=torch.optim.Adam([model.speed_gain],lr=.01)
    def error(ids):return (model.public_mean(observations[ids],detach_context=True)-targets[ids]).square().mean()
    with torch.no_grad():before=float(error(validation))
    for k in range(own.steps):
        ids=training[torch.randint(len(training),(512,),device=env.device)];loss=error(ids)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([model.speed_gain],1.);optimizer.step()
        if (k+1)%100==0:
            with torch.no_grad():print('V5 COMMAND FIT',k+1,float(error(validation)),flush=True)
    with torch.no_grad():after=float(error(validation))
    for name,parameter in model.named_parameters():parameter.requires_grad_(requires[name])
    model.teacher_encoder.requires_grad_(False);runner._freeze_seed_policy();runner.save(str(own.output/'run/model_0.pt'))
    model.zero_grad();runner._anchor_loss().backward()
    assert model.balance_delta.grad is not None and model.speed_gain.grad is not None
    assert all(p.grad is None for p in model.teacher_encoder.parameters()) and all(p.grad is None for p in model.student_encoder.parameters())
    with torch.no_grad():
        packed=env.obs_buf.clone();expected=model.act_inference(packed);packed[:,1029:]=torch.randn_like(packed[:,1029:])
        assert torch.equal(expected,model.act_inference(packed))
        jit=torch.jit.load(str(own.output/'run/exported/student_0.pt'),map_location=env.device)
        jit_error=float((jit(packed[:,:1029])-expected).abs().max());assert jit_error<1.e-5
    result=dict(supervised_steps=own.steps,PPO_updates=0,parameters_fitted='108 command-head coefficients only',
        before_MSE=before,after_MSE=after,private_input_independent=True,encoder_anchor_gradients_absent=True,
        trainable_feedback_gradient=True,JIT_max_error=jit_error,policy_accepted=False)
    (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
