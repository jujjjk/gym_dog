"""Prepare a bounded AMP+CTS seed with public nonlinear support feedback."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_speed_fix_20261117'
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v5 import Urdf4AMPV5Runner
from teacher_v3 import Teacher


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mid-integral',type=float,default=.25)
    own,rest=parser.parse_known_args();sys.argv=sys.argv[:1]+rest;own.output.mkdir(exist_ok=False,parents=True)
    upper=torch.load(BASE/'public_prior1/prior.pt',map_location='cpu')
    upper['yaw_step']=.3
    receipt=json.load(open(BASE/'gym_fast040_receipt60/result.json'))
    case=next(c for c in receipt['cases'] if c['passed'] and c['spec']['command']==[.4,0.,0.])
    spec=copy.deepcopy(case['spec']);spec['design']['yaw_step_integral']=0.;spec['design']['yaw_velocity_gain']=2.
    teacher=Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec);s=teacher.s
    lower=copy.deepcopy(upper)
    lower.update(geometry=torch.tensor(teacher.q),yaw_derivative=torch.tensor(teacher.derivatives[2]),stand=torch.tensor(teacher.q0),
        leads=torch.tensor(np.array([teacher.lead]*4)+np.array(s.get('leg_lead_offsets_s',[0.]*4))),
        duties=torch.tensor(s['leg_stance_ratio']),height=teacher.h-.002,feed=teacher.feed,yaw_gain=2.,yaw_step=.7,yaw_step_integral=0.)
    archive=own.output/'prior.pt';torch.save(dict(lower=lower,upper=upper,lower_goal=.4,upper_goal=.45,
        kind='same-height same-period geometric schedule; online forces recomputed, never interpolate issued feedback actions',
        policy_accepted=False),archive)
    args=get_args();args.task='urdf4_amp_cts_v5';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    cfg.amp_cts.use_public_prior=True;cfg.amp_cts.public_prior_archive=str(archive.resolve())
    cfg.amp_cts.public_feedback_history=True
    cfg.env.public_obs_size=1089;cfg.env.num_observations=1190;cfg.env.num_privileged_obs=1190
    cfg.amp_cts.reward_revision='v14_public_velocity_history_nonlinear_support'
    cfg.direction.mid_cross_integral_gain=own.mid_integral
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    run=own.output/'run';run.mkdir()
    runner=Urdf4AMPV5Runner(env,class_to_dict(train),str(run),device=args.rl_device)
    runner.warm_start(BASE/'selected/model_50.pt');runner.save(str(run/'model_0.pt'))
    model=runner.alg.actor_critic
    env.set_command_override(torch.tensor([.45,0.,0.],device=env.device).repeat(env.num_envs,1));env.reset()
    with torch.no_grad():
        packed=env.obs_buf.clone();expected=model.act_inference(packed)
        packed[:,model.public_size:]=torch.randn_like(packed[:,model.public_size:]);assert torch.equal(expected,model.act_inference(packed))
        exported=torch.jit.load(str(run/'exported/student_0.pt'),map_location=env.device)
        jit_error=float((exported(packed[:,:model.public_size])-expected).abs().max());assert jit_error<1.e-5
        for v in [.25,.35,.5,-.15]:
            probe=packed[:,:model.public_size].clone();probe[:,1020]=v
            assert torch.equal(exported(probe),model.protected_policy(probe[:,:1029]))
    result=dict(PPO_updates=0,private_input_independent=True,protected_commands_exact=True,jit_max_error=jit_error,
        frozen_public_prior=True,inner_yaw_integral=False,outer_mid_heading_integral_gain=own.mid_integral,
        residual_limit_action=.10,policy_accepted=False)
    (own.output/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
