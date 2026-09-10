"""PhysX 4-environment, 55-second transition check; stop at first reset."""
import json
from pathlib import Path
import isaacgym
import torch
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args, task_registry


def main():
    args=get_args(); cfg,train=task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg,55.,4);cfg.env.test=False
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    sequence=[(0,0,0,0),(0,0,0,1),(.2,0,0,1),(0,0,0,0),(-.2,0,0,1),
              (0,.2,0,1),(0,-.2,0,1),(0,0,.3,1),(0,0,-.3,1),(.3,.1,.25,1),(0,0,0,0)]
    failures=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    step=0;max_speed=0.;segments=[]
    with torch.no_grad():
        for command in sequence:
            velocities=[]
            for _ in range(round(5/env.dt)):
                env.set_evaluation_command(torch.tensor(command[:3],device=env.device),command[3])
                env.compute_observations()
                obs,_,_,dones,_=env.step(policy(env.get_observations()))
                failures+=dones.long();step+=1
                max_speed=max(max_speed,float(env.rs01_step_max_speed.max()))
                velocities.append(env.base_lin_vel[:,:2].clone())
                if dones.any() or not torch.isfinite(obs).all():break
            segments.append(dict(command=command,mean_vx_vy=torch.stack(velocities).mean((0,1)).tolist()))
            if failures.any() or not torch.isfinite(obs).all():break
    result=dict(task=args.task,checkpoint=args.checkpoint,seed=args.seed,num_envs=env.num_envs,
                completed_s=round(step*env.dt,3),requested_s=55.,resets=failures.cpu().tolist(),
                finite=bool(torch.isfinite(obs).all()),max_joint_speed_rad_s=max_speed,segments=segments)
    print(json.dumps(result,indent=2))
    output=Path(__file__).resolve().parents[3]/'artifacts/rs01_sensor_sync/physx_transitions.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
