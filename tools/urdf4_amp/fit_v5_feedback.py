"""Fit local command feedback from true visited states and teacher labels.

Assisted states train the feedback function only. They are never appended to
the AMP real-state distribution. The accepted V4 policy is embedded/frozen.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner,command_features,feedback_components
from legged_gym.algorithms.urdf4_phase_feedback import balance_actions


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corrective',type=Path,action='append',required=True)
    p.add_argument('--ridges',default='.001,.01,.1');p.add_argument('--seed-model',type=Path,default=ROOT/'artifacts/urdf4_speed_fix_20261117/selected/model_50.pt')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;own.output.mkdir(parents=True,exist_ok=False)
    args=get_args();args.task='urdf4_amp_cts_v5';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV5Runner(env,class_to_dict(train),None,device=args.rl_device)
    runner.warm_start(own.seed_model);model=runner.alg.actor_critic
    files=[Path(cfg.amp_cts.extra_reference),*own.corrective];blocks=[]
    for file in files:
        data=torch.load(file,map_location=env.device);obs=data['observations'];action=data['actions']
        keep=(obs[:,1020]>.35)&(obs[:,1020]<.5)&(obs[:,1028]>.55)
        pub=obs[keep,:1029];action=action[keep];assert len(pub)>100 and torch.isfinite(pub).all()
        blocks.append((pub,action))
    features=[];targets=[];weights=[];publics=[];actions=[]
    with torch.no_grad():
        for pub,action in blocks:
            template=pub.clone();template[:,1020]=.5
            base=torch.cat([model.protected_policy(row) for row in template.split(512)])
            original=balance_actions(pub,model.balance_reference,model.balance_stand,model.balance_scales,model.balance_offset,model.balance_gain)
            f=command_features(pub,model.balance_reference,model.balance_stand,model.command_reference,model.command_stand,model.balance_scales,model.balance_offset)
            desired=(action-base+original)/pub[:,1028:1029]
            base,deviation,coefficient=feedback_components(pub,f)
            y=torch.atanh(desired.clamp(-.985,.985))-base@model.balance_gain
            phase=torch.remainder(torch.atan2(pub[:,45],pub[:,46])/(2*math.pi)-model.balance_offset,1.)
            basis=[torch.ones_like(phase)]
            for k in range(1,5):basis.extend([torch.sin(2*math.pi*k*phase),torch.cos(2*math.pi*k*phase)])
            observer_features=model.observer.feedback_features(pub,model.balance_offset)
            features.append(coefficient*torch.cat((deviation,torch.stack(basis,1),observer_features),1));targets.append(y)
            weights.append(torch.full((len(pub),),1./len(pub),device=env.device));publics.append(pub);actions.append(action)
    x=torch.cat(features).double();y=torch.cat(targets).double();weight=torch.cat(weights).double();weight/=weight.sum()
    public=torch.cat(publics);target=torch.cat(actions);held=torch.arange(len(x),device=env.device)%5==4;training=~held
    w=weight[training];w/=w.sum();xx=x[training].T@(x[training]*w[:,None]);xy=x[training].T@(y[training]*w[:,None])
    diagonal=torch.ones(423,device=env.device,dtype=torch.double);diagonal[396:405]=.1;diagonal[405:]=.01
    reports=[]
    for index,ridge in enumerate(map(float,own.ridges.split(','))):
        fit=torch.linalg.solve(xx+torch.diag(diagonal*ridge),xy).float()
        with torch.no_grad():
            model.balance_delta.copy_(fit[:396]);model.speed_gain.copy_(fit[396:405]);model.observer_gain.copy_(fit[405:])
        with torch.no_grad():
            prediction=torch.cat([model.public_mean(row) for row in public[held].split(512)])
            error=float((prediction-target[held]).square().mean())
            example=public[held][:32].clone();example[:,1020]=.5
            assert torch.equal(model.public_mean(example),model.protected_policy(example))
        folder=own.output/('variant%d'%index);folder.mkdir();runner.seed_migration.update(
            fit_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},ridge=ridge,training_labels_not_AMP=True)
        runner.save(str(folder/'model_0.pt'))
        with torch.no_grad():
            jit=torch.jit.load(str(folder/'exported/student_0.pt'),map_location=env.device)
            sample=public[held][:64];jit_error=float((jit(sample)-model.public_mean(sample)).abs().max());assert jit_error<1.e-5
        report=dict(variant=index,ridge=ridge,heldout_action_MSE=error,JIT_max_error=jit_error,
            fitted_samples=len(x),PPO_updates=0,protected_policy_unchanged=True,policy_accepted=False)
        reports.append(report);print('V5 FIT',json.dumps(report),flush=True)
    (own.output/'result.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
