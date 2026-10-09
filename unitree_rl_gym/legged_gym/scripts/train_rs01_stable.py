"""Single-GPU scratch launcher with an immutable training contract and startup checks."""
import hashlib
import inspect
import json
from pathlib import Path
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict


def main():
    args=get_args()
    if args.task not in ('rs01_stable_phase','rs01_stable_phase_v2','rs01_stable_phase_v3'):raise ValueError('Use an RS01 stable phase task')
    if args.resume:raise ValueError('This launcher starts a genuinely fresh actor; no implicit old checkpoint')
    env,cfg=task_registry.make_env(args.task,args=args)
    runner,train=task_registry.make_alg_runner(env=env,name=args.task,args=args)
    assert type(runner).__name__=='OnPolicyRunner' and not train.runner.amp_enabled
    assert env.num_obs==61 and env.num_actions==12
    hashes={}
    for cls in [*type(env).__mro__[:-1],type(runner),type(runner.alg)]:
        p=Path(inspect.getsourcefile(cls));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    directory=Path(runner.log_dir);directory.mkdir(parents=True,exist_ok=True)
    contract=dict(env=class_to_dict(cfg),train=class_to_dict(train),args=vars(args),source_sha256=hashes)
    if getattr(train.runner,'phase_seed_initialization',False):
        from rs01_phase_initializer import initialize_actor
        p=Path(inspect.getsourcefile(initialize_actor));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        contract['phase_initialization']=initialize_actor(env,runner.alg.actor_critic.actor)
        print('PHASE INITIALIZATION',contract['phase_initialization'],flush=True)
    (directory/'experiment_contract.json').write_text(json.dumps(contract,indent=2,default=str)+'\n')
    with torch.no_grad():
        for _ in range(10):
            obs,_,reward,_,_=env.step(torch.zeros((env.num_envs,12),device=env.device))
            if not bool(torch.isfinite(obs).all() and torch.isfinite(reward).all()):raise RuntimeError('Nonfinite startup')
            if float(env.motor_electromagnetic_torques.abs().max())>cfg.rs01_actuator.peak_torque_limit_nm+.001:
                raise RuntimeError('Motor torque exceeded identified plant limit')
    env.reset()
    print('STARTUP PASS: 61D/12D, non-AMP, fresh actor, finite feedback, real RS01 motor limits',flush=True)
    runner.learn(num_learning_iterations=train.runner.max_iterations,init_at_random_ep_len=True)


if __name__=='__main__':main()
