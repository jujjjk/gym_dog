"""V14 changes numerical feedback resolution, not the identified RS01 hardware."""
from unittest.mock import patch
import isaacgym  # noqa: F401
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_config import Rs01OmniV13DirectionCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_config import Rs01OmniV14ActuatorCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_env import Rs01OmniV13Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_env import Rs01OmniV14Robot
from legged_gym.envs.rs01_go2_straight.rs01_go2_straight_env import Rs01Go2StraightRobot
from legged_gym.utils.helpers import class_to_dict


def test_rewards_observations_commands_and_identification_are_preserved():
    old,new=Rs01OmniV13DirectionCfg(),Rs01OmniV14ActuatorCfg()
    for field in ('commands','rewards','rs01_odometry','normalization','domain_rand'):
        assert class_to_dict(getattr(old,field))==class_to_dict(getattr(new,field)),field
    for field,value in class_to_dict(old.rs01_actuator).items():
        assert getattr(new.rs01_actuator,field)==value,field
    assert old.env.num_observations==new.env.num_observations==61
    assert old.sim.dt==.005 and old.control.decimation==4
    assert new.sim.dt==.0025 and new.sim.substeps==1 and new.control.decimation==8
    assert new.rs01_actuator.response_update_dt_s==.005


def test_response_clock_divider_does_not_speed_up_delay_or_identified_filter():
    for divider,expected in ((1,8),(2,4)):
        r=object.__new__(Rs01Go2StraightRobot)
        r.cfg=Rs01OmniV14ActuatorCfg();r._rs01_actuator_ready=True
        r.rs01_response_substeps=divider;r.rs01_feedback_tick=0
        r.rs01_response_target_rad=torch.zeros(1,12)
        r.dof_pos=torch.zeros(1,12);r.dof_vel=torch.zeros(1,12)
        r.p_gains=torch.full((12,),40.);r.d_gains=torch.ones(12)
        r.peak_torque_limit_nm=torch.full((1,12),17.)
        r.rs01_coulomb_friction_nm=torch.full((12,),.15)
        with patch.object(r,'_advance_rs01_response') as advance:
            for _ in range(8):r._compute_torques(torch.zeros(1,12))
            assert advance.call_count==expected
        assert r.rs01_feedback_tick==8


def test_overspeed_is_latched_until_policy_boundary_and_never_projected():
    r=object.__new__(Rs01OmniV14Robot);r.cfg=Rs01OmniV14ActuatorCfg()
    r.dof_vel=torch.zeros(2,12);r.dof_vel[0,2]=34.
    r.rs01_step_overspeed=torch.zeros(2,dtype=torch.bool)
    r.rs01_step_max_speed=torch.zeros(2);r.reset_buf=torch.zeros(2,dtype=torch.bool)
    r._observe_speed_domain()
    assert r.dof_vel[0,2]==34. and r.rs01_step_overspeed.tolist()==[True,False]
    r.dof_vel.zero_()
    with patch.object(Rs01OmniV13Robot,'check_termination',lambda self:None):
        r.check_termination()
    assert r.reset_buf.tolist()==[True,False]
    assert r.rs01_step_max_speed[0]==34.
