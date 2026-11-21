"""Finite integration checks for V3 actor/data/phase/deployment contracts."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner,V3StudentPolicy


def main():
    args=get_args();args.task='urdf4_amp_cts_v3';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device)
    runner.load(ROOT/'artifacts/urdf4_amp_v3_training/reference_init/run/model_0.pt',load_optimizer=False)
    model=runner.alg.actor_critic
    command=torch.tensor([[0.,0.,0.],[.25,0.,0.],[0.,.2,0.],[0.,0.,-.6]],device=env.device).repeat(4,1)
    env.set_command_override(command);env.reset();valid_stand=valid_ramp=0
    with torch.no_grad():
        for k in range(190):
            old_phase=env.motion_phase.clone();was_active=env.phase_was_active.clone()
            action=model.act_inference(env.obs_buf)
            assert action.shape==(16,12) and torch.isfinite(action).all()
            result=env.step(action);amp=result[4]['amp']
            assert torch.isfinite(env.obs_buf).all() and env.obs_buf.shape==(16,1121)
            assert torch.isfinite(env.rew_buf).all()
            if k>70:
                valid_stand+=int(amp['valid'][0]);valid_ramp+=int(amp['valid'][1])
                active=was_active&env.phase_was_active&~result[3].bool()
                advance=torch.remainder(env.motion_phase[active]-old_phase[active],1.)
                assert torch.allclose(advance,env.motion_frequency[active]*env.dt,atol=1e-6)
        # Same public measurements, arbitrary privileged inputs and roles.
        original=env.obs_buf.clone();changed=original.clone();changed[:,1020:]=torch.randn_like(changed[:,1020:])
        assert torch.equal(model.act_inference(original),model.act_inference(changed))
        expected=model.act_inference(original)
        jit=torch.jit.script(V3StudentPolicy(model).eval().to(env.device))
        error=float((jit(original[:,:1020])-expected).abs().max());assert error<1e-5
        # PPO must use exactly the bounded executed mean used for deployment.
        student=original.clone();student[:,-1]=0.;model.update_distribution(student)
        assert torch.allclose(model.action_mean,expected,atol=1e-6)
        assert torch.isfinite(model.get_actions_log_prob(expected)).all()
        assert env.external_commands[2,1]>.199 and env.commands[2,1]>.15
        query=torch.tensor([[0.,0.,0.],[.18,0.,0.],[.6,0.,0.]],device=env.device)
        real=runner.style.sample_real(query);assert real.shape==(3,95) and torch.isfinite(real).all()
        assert torch.allclose(real[:,-3:],query/query.new_tensor([.5,.2,.8]))
        assert runner.style.coverage(query)[1]>runner.style.coverage(query)[2]
        assert valid_stand>100 and valid_ramp>100
    result=dict(passed=True,steps=190,observations=1121,public_observations=1020,actions=12,
        valid_standing_pairs=valid_stand,valid_transition_pairs=valid_ramp,
        continuous_phase=True,student_private_input_independent=True,
        JIT_max_error=error,PPO_mean_matches_deployment=True,lateral_point2_not_legacy_clipped=True,
        AMP_query_condition_matches_real_and_fake=True,expert_pairs=runner.style.pair_count,
        urdf_sha256=env.urdf_sha256)
    output=ROOT/'artifacts/urdf4_amp_v3_training/contract_checks.json';output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
