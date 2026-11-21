"""Read-only coarse screen of explicit capped V4 checkpoints."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner,V4StudentPolicy
from legged_gym.envs.urdf4.amp_reference import amp_features
from select_v3_models import assess,summary,rank
from evaluate_v3_policy import CASES,FRONTIER
from check_robust050_launch import restore


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--iterations',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=16.)
    p.add_argument('--rough',action='store_true');p.add_argument('--baseline',type=Path)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task='urdf4_amp_cts_rough' if own.rough else 'urdf4_amp_cts_v4';args.headless=True
    baseline=own.baseline or ROOT/'artifacts/urdf4_speed_fix_20261117/robust050/initial/model_0.pt'
    paths=[baseline]+[own.run/('model_%s.pt'%i) for i in own.iterations.split(',')]
    cases=dict(CASES,**FRONTIER);names=list(cases);per=len(names);args.num_envs=len(paths)*per
    saved=torch.load(paths[0],map_location='cpu');cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10
    for section in ('commands','expand','direction'):restore(getattr(cfg,section),saved['experiment']['env'][section])
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    runner_type=Urdf4AMPV4Runner
    if own.rough:
        from legged_gym.algorithms.urdf4_amp_rough import Urdf4AMPRoughRunner
        restore(cfg.rough,saved['contract']['rough_ground']);runner_type=Urdf4AMPRoughRunner
    own.output.mkdir(parents=True,exist_ok=False);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=runner_type(env,class_to_dict(train),None,device=args.rl_device);policies=[];hashes=[]
    for path in paths:
        s=torch.load(path,map_location=env.device);assert s['contract']==runner.contract
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest());runner.alg.actor_critic.load_state_dict(s['model_state_dict'],strict=True)
        policies.append(V4StudentPolicy(runner.alg.actor_critic).to(env.device).eval().requires_grad_(False))
    command=torch.tensor([cases[n] for _ in paths for n in names],device=env.device);env.set_command_override(command);env.reset()
    fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool);peak=torch.zeros(env.num_envs,device=env.device);motor=peak.clone();squares=torch.zeros(env.num_envs,12,device=env.device);ticks=peak.clone()
    original=env._compute_torques
    def torques(action):
        value=original(action);alive=(~fallen).float();peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*alive));motor.copy_(torch.maximum(motor,env.motor_torques.abs().amax(1)*alive));squares.add_(env.motor_torques.square()*alive[:,None]);ticks.add_(alive);return value
    env._compute_torques=torques;env.reset_idx=lambda ids:fallen.__setitem__(ids,True)
    slots=[env.foot_legs.index(l) for l in ('FL','FR','RL','RR')];records=[]
    with torch.no_grad():
        for k in range(round((own.seconds+6)/env.dt)):
            if k==round(own.seconds/env.dt):env.set_command_override(torch.zeros_like(command))
            action=torch.cat([policy(env.obs_buf[i*per:(i+1)*per,:1029]) for i,policy in enumerate(policies)])
            action[fallen]=0.;env.step(action)
            feet=env.feet_pos[:,slots]
            ground=env.ground_height_at(feet[:,:,:2]) if own.rough else env.env_origins[:,None,2]
            records.append(dict(velocity=amp_features(env)[:,[36,37,41]].clone(),contact=env.get_foot_contact_mask()[:,slots].clone(),height=(feet[:,:,2]-ground-.02).clone(),cross=env.path.cross_error.clone(),heading=env.path.heading_error.clone(),valid=(~fallen).clone()))
            if (k+1)%250==0:print('SCREEN',round((k+1)*env.dt,1),flush=True)
    arrays={key:torch.stack([r[key] for r in records]).cpu().numpy() for key in records[0]};t=np.arange(1,len(records)+1)*env.dt
    frequencies=env.reference_frequencies[env._select_reference(command)].cpu().numpy();values=[x.cpu().numpy() for x in (peak,motor,torch.sqrt(squares/ticks[:,None].clamp(min=1)),fallen)]
    results=[]
    for i,path in enumerate(paths):
        sl=slice(i*per,(i+1)*per);rows=assess({k:v[:,sl] for k,v in arrays.items()},t,*[v[sl] for v in values],command[sl].cpu().numpy(),names,frequencies[sl],own.seconds)
        results.append(dict(checkpoint=str(path.resolve()),checkpoint_sha256=hashes[i],iteration=int(path.stem.split('_')[-1]),cases=rows,summary=summary(rows)))
    results.sort(key=rank);(own.output/'screen.json').write_text(json.dumps(dict(seconds=own.seconds,seed=args.seed,rough=own.rough,screen_only=True,teacher_controller_used=False,automatic_resets=False,candidates=results),indent=2)+'\n')
    for r in results:print('RANK',r['iteration'],r['summary'],flush=True)


if __name__=='__main__':main()
