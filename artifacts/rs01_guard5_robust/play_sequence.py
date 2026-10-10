"""Read-only robust-model viewer: direct switching, optional full push/payload."""
import argparse
import json
import math
import sys
import time
import isaacgym
import torch
sys.path.insert(0,'/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')
sys.path.insert(0,'/home/nszb/gym/artifacts/rs01_v22_phase_scratch')
import evaluate_rs01_go2_omni as audit
from select_play import scaled_cases
from legged_gym.utils import get_args, task_registry


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--disturbances',action='store_true')
    parser.add_argument('--speed_scale',type=float,default=1.)
    own,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
    cases=scaled_cases(own.speed_scale);args=get_args()
    if args.task!='rs01_v22_guard5_robust' or not math.isfinite(args.duration_s) or args.duration_s<=0:
        raise ValueError('Use robust task and finite positive duration_s per action')
    cfg,train=task_registry.get_cfgs(args.task)
    audit._set_nominal_eval_cfg(cfg,args.duration_s*len(cases),args.num_envs or 1)
    cfg.domain_rand.randomize_base_mass=own.disturbances
    cfg.robust_push.enabled=own.disturbances
    cfg.robust_push.ramp_s=0.
    cfg.robust_push.nominal_fraction=0. # Every displayed robot is eligible in stress mode.
    cfg.sensor.clean_fraction=cfg.sensor.robust_fraction=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    print('PAYLOAD kg',env.payload_kg.tolist(),'FULL PUSHES',own.disturbances,flush=True)
    hips=[env.dof_names.index(x+'_hip_joint') for x in ('FL','FR','RL','RR')]
    sides=torch.tensor([1.,-1.,1.,-1.],device=env.device)
    stages=[]
    with torch.no_grad():
        for name,vx,vy,wz,_ in cases:
            print(name,(vx,vy,wz),args.duration_s,'seconds',flush=True)
            command=torch.tensor([vx,vy,wz],device=env.device)
            rows=[];resets=0;before=int(env.push_count.sum())
            for _ in range(max(1,round(args.duration_s/env.dt))):
                started=time.monotonic()
                env.set_evaluation_command(command,1.);env.compute_observations()
                obs,_,rew,done,_=env.step(policy(env.get_observations()))
                assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
                resets+=int(done.sum())
                rows.append(torch.cat((env.base_lin_vel[:,:2],env.base_ang_vel[:,2:3],env.rpy[:,:2],
                    (-env.dof_pos[:,hips]*sides).clamp(min=0.).max(1,keepdim=True).values,
                    env.raw_pd_torques.abs().max(1,keepdim=True).values,
                    (~env.get_foot_contact_mask().any(1)).float()[:,None]),1).clone())
                if not args.headless:
                    time.sleep(max(0.,env.dt-(time.monotonic()-started)))
            data=torch.stack(rows)
            stages.append(dict(name=name,command=[vx,vy,wz],resets=resets,
                mean_vx_vy_wz=data[:,:,:3].mean((0,1)).tolist(),
                roll_pitch_rms_deg=(data[:,:,3:5].square().mean((0,1)).sqrt()*180/torch.pi).tolist(),
                inward_hip_peak_deg=float(data[:,:,5].max()*180/torch.pi),
                raw_torque_peak_nm=float(data[:,:,6].max()),flight_fraction=float(data[:,:,7].mean()),
                pushes=int(env.push_count.sum())-before))
    print(json.dumps(dict(task=args.task,run=args.load_run,checkpoint=args.checkpoint,seed=args.seed,
        speed_scale=own.speed_scale,seconds_per_action=args.duration_s,sensor_profile='normal',
        disturbances=own.disturbances,payload_kg=env.payload_kg.tolist(),finite=True,
        resets=sum(s['resets'] for s in stages),pushes_per_env=env.push_count.tolist(),stages=stages),indent=2))


if __name__=='__main__':main()
