"""Export a bounded, public-joint phase-feedback experiment; no PPO or teacher."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner
from legged_gym.algorithms.urdf4_adaptive_phase import AdaptiveFastStudent


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dq-weight',type=float,default=1.);p.add_argument('--phase-gain',type=float,default=1.)
    p.add_argument('--deadzone',type=float,default=.02);p.add_argument('--phase-limit',type=float,default=.08)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;assert not own.output.exists()
    args=get_args();args.task='urdf4_amp_cts_v4';args.num_envs=16;args.headless=True
    saved=torch.load(own.model,map_location='cpu');cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=Urdf4AMPV4Runner(env,class_to_dict(train),None,device=args.rl_device)
    runner.load(own.model,load_optimizer=False);model=runner.alg.actor_critic.eval()
    experiment=AdaptiveFastStudent(model,own.dq_weight,own.phase_gain,own.deadzone,own.phase_limit).eval().to(env.device)
    compiled=torch.jit.script(experiment)
    with torch.no_grad():
        obs=env.obs_buf[:,:1029].clone();obs[:,1020]=.5
        assert torch.allclose(compiled(obs),model.act_inference(obs),atol=1.e-6)
        obs[:,1020]=.4;assert torch.isfinite(compiled(obs)).all()
    own.output.mkdir(parents=True);compiled.save(str(own.output/'student.pt'))
    meta=dict(public_size=1029,urdf_sha256=env.urdf_sha256,pure_public=True,teacher_controller_used=False,policy_accepted=False,
        source_checkpoint=str(own.model.resolve()),source_checkpoint_sha256=hashlib.sha256(own.model.read_bytes()).hexdigest(),
        fast_straight_reference=[.5,0.,0.],fast_straight_threshold_m_s=.35,original_policy_unchanged_at_050_and_low_omni=True,
        phase_observer=dict(inputs='public joint positions/velocities',dq_weight=own.dq_weight,gain=own.phase_gain,
            deadzone_cycles=own.deadzone,limit_cycles=own.phase_limit,trajectory_clock_unchanged=True),
        source_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/legged_gym/algorithms/urdf4_adaptive_phase.py').read_bytes()).hexdigest())
    (own.output/'student.pt.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
