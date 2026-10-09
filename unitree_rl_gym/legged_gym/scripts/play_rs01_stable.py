"""Non-AMP viewer/audit. Direct omni changes every duration_s (default CLI=5s recommended)."""
import json
import argparse
import sys
import time
from pathlib import Path
import isaacgym
import torch
import numpy as np
import legged_gym.envs
from legged_gym.utils import get_args,task_registry
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fixed_command',action='store_true')
    local,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
    args=get_args()
    if args.task not in ('rs01_stable_phase','rs01_stable_phase_v2','rs01_stable_phase_v3'):raise ValueError('Use an RS01 stable phase task')
    cfg,train=task_registry.get_cfgs(args.task)
    duration=args.duration_s
    if duration<=0:raise ValueError('Positive duration_s required')
    n=args.num_envs or 4
    _set_nominal_eval_cfg(cfg,duration*10+5,n)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    stages=[];finite=True;trace=[]
    names=['march','forward','backward','left','right','turn_left','turn_right','forward_left','forward_right','combined']
    commands=cfg.stability.commands
    if local.fixed_command:
        names=['fixed'];commands=[[args.vx,args.vy,args.wz]]
    with torch.no_grad():
        for index,(name,command) in enumerate(zip(names,commands)):
            samples=[];resets=stalled=narrow=0
            print(name,command,'seconds',duration,flush=True)
            for tick in range(round(duration/env.dt)):
                env.set_evaluation_command(env.commands.new_tensor(command),1.)
                env.compute_observations()
                action=policy(env.get_observations())
                obs,_,reward,done,_=env.step(action)
                finite &= bool(torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(action).all())
                resets+=int(done.sum());stalled+=int(env.stalled_foot.sum());narrow+=int(env.narrow_failure.sum())
                row=torch.cat((env.base_lin_vel[:,:2],env.base_ang_vel[:,2:3],env.rpy[:,:2],
                    env.stable_y,env.stable_height,env.stable_contact.float(),env.stable_last_age_s,
                    env.stable_last_contact_s,env.raw_pd_torques.abs().amax(1)[:,None],
                    env.motor_electromagnetic_torques.abs().amax(1)[:,None],done[:,None]),1).cpu().numpy()
                samples.append(row);trace.append(row)
            x=np.stack(samples);valid=x[:,:,-1]==0;contact=x[:,:,13:17]>.5
            stages.append(dict(name=name,command=command,resets=resets,gait_timeout=stalled,narrow_failure=narrow,
                mean_vx_vy_wz=np.mean(x[:,:,:3][valid],axis=0).tolist() if valid.any() else None,
                roll_pitch_rms_deg=(np.sqrt(np.mean(x[:,:,3:5][valid]**2,axis=0))*180/np.pi).tolist() if valid.any() else None,
                support_inside_13cm_fraction=float(((x[:,:,5:9]<.13)&contact).sum()/max(1,contact.sum())),
                foot_height_p95_mm=(np.quantile(x[:,:,9:13].reshape(-1,4),.95,axis=0)*1000).tolist(),
                longest_no_lift_s=x[:,:,17:21].max((0,1)).tolist(),
                longest_contact_s=x[:,:,21:25].max((0,1)).tolist(),
                raw_pd_peak_nm=float(x[:,:,25].max()),motor_peak_nm=float(x[:,:,26].max())))
    out=Path('/home/nszb/gym/artifacts/rs01_stable_phase');out.mkdir(parents=True,exist_ok=True)
    stem=out/('sequence_'+str(args.checkpoint)+'_'+str(time.time_ns()))
    result=dict(task=args.task,run=args.load_run,checkpoint=args.checkpoint,seed=args.seed,
        seconds_per_command=duration,num_envs=n,finite=finite,resets=sum(s['resets'] for s in stages),
        note='Continuous switches; per-step data can contain reset frames. No deployment approval.',stages=stages)
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(stem.with_suffix('.npz'),samples=np.stack(trace))
    print('RESULT',stem.with_suffix('.json'),json.dumps(result),flush=True)
    if not finite:raise RuntimeError('Nonfinite audit')


if __name__=='__main__':main()
