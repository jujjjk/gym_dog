"""Full-strength robustness smoke: original sensor/actuator, 13 fixed commands."""
import json
import sys
import isaacgym
import torch
sys.path.insert(0,'/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')
sys.path.insert(0,'/home/nszb/gym/artifacts/rs01_v22_phase_scratch')
import evaluate_rs01_go2_omni as audit
from select_play import CASES
from legged_gym.utils import get_args, task_registry


def main():
    args=get_args()
    cfg,train=task_registry.get_cfgs(args.task)
    n=args.eval_envs
    audit._set_nominal_eval_cfg(cfg,args.duration_s,len(CASES)*n)
    cfg.domain_rand.randomize_base_mass=True
    cfg.robust_push.ramp_s=0.
    cfg.sensor.clean_fraction=cfg.sensor.robust_fraction=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    # Verify the simulator actually received the randomized masses.
    masses=[sum(p.mass for p in env.gym.get_actor_rigid_body_properties(e,a))
            for e,a in zip(env.envs,env.actor_handles)]
    delta=torch.tensor(masses,device=env.device)-env.payload_kg
    assert float(delta.max()-delta.min())<1e-4
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    cmd=torch.tensor([x[1:4] for x in CASES],device=env.device).repeat_interleave(n,0)
    resets=torch.zeros(env.num_envs,device=env.device)
    records=[]
    with torch.no_grad():
        for _ in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(cmd,1.)
            env.compute_observations()
            obs,_,reward,done,_=env.step(policy(env.get_observations()))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            resets+=done.float()
            records.append(torch.cat((env.base_lin_vel[:,:2],env.rpy[:,:2],
                env.motor_electromagnetic_torques.abs().max(1,keepdim=True).values,
                env.raw_pd_torques.abs().max(1,keepdim=True).values,
                (~env.get_foot_contact_mask().any(1)).float()[:,None]),1).clone())
    data=torch.stack(records)
    cases=[]
    for i,c in enumerate(CASES):
        block=data[:,i*n:(i+1)*n]
        cases.append(dict(name=c[0],command=c[1:4],resets=int(resets[i*n:(i+1)*n].sum()),
            vx_vy=block[:,:,:2].mean((0,1)).tolist(),
            roll_pitch_rms_deg=(block[:,:,2:4].square().mean((0,1)).sqrt()*180/torch.pi).tolist()))
    print(json.dumps(dict(task=args.task,run=args.load_run,checkpoint=args.checkpoint,
        seed=args.seed,duration_s=args.duration_s,envs_per_case=n,full_push_strength=True,
        finite=True,resets=int(resets.sum()),cases=cases,
        payload_kg=env.payload_kg.tolist(),simulator_total_mass_kg=masses,
        pushes_per_env=env.push_count.tolist(),impulse_ns_per_env=env.push_impulse_ns.tolist(),
        nominal_environment=env.nominal_environment.tolist(),
        motor_peak_nm=float(data[:,:,4].max()),raw_peak_nm=float(data[:,:,5].max()),
        flight_ratio=float(data[:,:,6].mean())),indent=2))


if __name__=='__main__':main()
