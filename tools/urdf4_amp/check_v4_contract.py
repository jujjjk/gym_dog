"""Finite public-input, executed-action, export and speed-servo integration checks."""
import argparse,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner
from legged_gym.envs.urdf4.amp_v4_env import V4PathController


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_v4';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    saved=torch.load(own.model,map_location='cpu')
    for key in ('balance_archive','balance_variant'):
        setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV4Runner(env,class_to_dict(train),None,device=args.rl_device)
    runner.load(own.model,load_optimizer=False);model=runner.alg.actor_critic
    commands=torch.tensor([[0.,0.,0.],[.25,0.,0.],[.5,0.,0.],[0.,.1,0.]],device=env.device).repeat(4,1)
    env.set_command_override(commands);env.reset()
    with torch.no_grad():
        for _ in range(350):
            env.step(model.act_inference(env.obs_buf))
            assert env.obs_buf.shape==(16,1130) and torch.isfinite(env.obs_buf).all()
            assert torch.isfinite(env.rew_buf).all()
        original=env.obs_buf.clone();changed=original.clone();changed[:,1029:]=torch.randn_like(changed[:,1029:])
        expected=model.act_inference(original)
        assert torch.equal(expected,model.act_inference(changed))
        student=original.clone();student[:,-1]=0.;model.update_distribution(student)
        assert torch.allclose(model.action_mean,expected,atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            file=str(Path(directory)/'student.pt');runner.export_student(file)
            exported=torch.jit.load(file,map_location=env.device)
            error=float((exported(original[:,:1029])-expected).abs().max());assert error<1e-5
            meta=json.loads(Path(file+'.json').read_text())
            assert meta['requires_corrected_external_velocity'] and meta['speed_feedback_enabled']
        low=commands[:,0]<=.35
        low_error=float((expected[low]-model.legacy_policy(original[low,:1020])).abs().max())
        assert low_error<=.050001
        assert float(env.motor_torques.abs().max())<=14.0001
    # The expert anchor must train actions without moving either context encoder.
    model.zero_grad();runner._anchor_loss().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.actor.parameters())
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    assert all(p.grad is None for p in model.student_encoder.parameters())
    # A prolonged speed deficit cannot wind up indefinitely; stop clears memory.
    controller=V4PathController(1,env.dt,env.device,gains=env.path.gains,
        speed_enabled=True,speed_settings=cfg.direction.speed_feedback)
    xy=torch.zeros(1,2,device=env.device);yaw=torch.zeros(1,device=env.device)
    goal=torch.tensor([[.5,0.,0.]],device=env.device)
    with torch.no_grad():
        for _ in range(1000):
            out=controller.update(goal,xy,yaw,xy,yaw)
            assert torch.isfinite(out).all() and out[0,0]<=.900001
        assert controller.speed_integral.abs().max()<=.200001
        assert controller.speed_bias.abs().max()<=.200001
        controller.update(torch.zeros_like(goal),xy,yaw,xy,yaw)
        assert controller.speed_integral.abs().max()==0 and controller.speed_bias.abs().max()==0
        assert controller.speed_applied.abs().max()==0
    result=dict(passed=True,checkpoint=str(own.model),physics_steps=350,public_observations=1029,
        packed_observations=1130,actions=12,JIT_max_error=error,low_speed_action_change_max=low_error,
        private_input_independent=True,PPO_mean_matches_deployment=True,
        anchor_encoder_gradients_absent=True,teacher_context_frozen=True,speed_anti_windup_and_stop_passed=True,
        motor_cap_Nm=14.,contract=runner.contract,source_sha256=runner.experiment['source_sha256'])
    own.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
