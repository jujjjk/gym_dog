"""Actual PhysX task vs MuJoCo actuator state: same probe q/dq and targets."""
import json,sys
from pathlib import Path
import isaacgym
import torch
import numpy as np
import mujoco
from sim2sim_v14 import V14Sim
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'unitree_rl_gym/legged_gym/scripts'))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_config import Rs01OmniV13DirectionCfg


def main():
    args=get_args();cfg,_=task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg,3.,1);cfg.env.test=False
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    out=Path(__file__).resolve().parents[2]/'artifacts/rs01_v14_actuator_parity'
    sim=V14Sim(out/'scene.xml',out/'model_6850.onnx',[.2,0,0])
    old=Rs01OmniV13DirectionCfg()
    assert class_to_dict(cfg.rewards)==class_to_dict(old.rewards)
    for key in ('response_gain','time_constant_s','observed_closed_loop_delay_s',
                'coulomb_friction_nm','target_rate_limit_rad_s','target_acceleration_limit_rad_s2',
                'joint_to_motor_id','real_to_policy_sign_by_motor_id','peak_torque_limit_nm'):
        assert getattr(cfg.rs01_actuator,key)==getattr(old.rs01_actuator,key),key
    assert np.array_equal(env.rs01_delay_steps.cpu().numpy(),sim.delay_steps)
    assert abs(env.dt-.02)<1e-8 and abs(env.sim_params.dt-.0025)<1e-8
    assert env.rs01_response_substeps==sim.response_substeps==2
    assert torch.allclose(env.dof_vel_limits,torch.full_like(env.dof_vel_limits,32.9867))
    props=env.gym.get_actor_dof_properties(env.envs[0],env.actor_handles[0])
    assert np.all(props['velocity']>32.9867)
    errors={'response_rad':0.,'raw_nm':0.,'motor_nm':0.,'applied_nm':0.}
    actions=torch.zeros(1,12,device=env.device)
    tensor=lambda x:torch.tensor(x,dtype=torch.float,device=env.device)
    env.rs01_feedback_tick=0
    for tick in range(256):
        # Probe both saturated and unsaturated demand, without physics differences.
        q=sim.default+.1*np.sin(np.arange(12)+tick*.07)
        dq=8*np.cos(np.arange(12)*.5+tick*.04)
        target=sim.default+.08*np.sin(np.arange(12)*.4+(tick//8)*.1)
        env.dof_pos[:]=tensor(q);env.dof_vel[:]=tensor(dq)
        env.rs01_limited_position_target_rad[:]=tensor(target)
        sim.data.qpos[sim.qpos_indices]=q;sim.data.qvel[sim.qvel_indices]=dq
        sim.limited_target=target;mujoco.mj_forward(sim.model,sim.data)
        env._compute_torques(actions);sim.physics_step()
        for key,a,b in (
            ('response_rad',env.rs01_response_target_rad,sim.response_target),
            ('raw_nm',env.raw_pd_torques,sim.last_raw_torque),
            ('motor_nm',env.motor_electromagnetic_torques,sim.last_motor_torque),
            ('applied_nm',env.applied_joint_torques,sim.last_applied_torque)):
            errors[key]=max(errors[key],float(np.max(abs(a.cpu().numpy()[0]-b))))
    assert max(errors.values())<1e-4,errors
    # Domain guard must detect overspeed, not edit the joint velocity.
    env.rs01_step_overspeed=torch.zeros(1,dtype=torch.bool,device=env.device)
    env.rs01_step_max_speed=torch.zeros(1,device=env.device)
    env.dof_vel[:]=34.;env._observe_speed_domain()
    assert env.rs01_step_overspeed.item() and torch.all(env.dof_vel==34.)
    sim.data.qvel[sim.qvel_indices]=34.;sim.step_overspeed=False;sim._observe_speed_domain()
    assert sim.step_overspeed and np.all(sim.data.qvel[sim.qvel_indices]==34.)
    result=dict(max_errors=errors,feedback_s=.0025,response_s=.005,policy_s=.02,
                rewards_unchanged=True,identified_parameters_unchanged=True,
                delay_ticks_match=True,speed_guard_no_projection_pass=True)
    (out/'actuator_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
