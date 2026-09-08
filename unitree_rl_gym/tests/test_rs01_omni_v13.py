"""Direction reference, coordinate/sign, reward conflict and observation contracts."""

from unittest.mock import patch

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_config import Rs01OmniV12WideCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_env import Rs01OmniV12Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_config import Rs01OmniV13DirectionCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_env import Rs01OmniV13Robot
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.checkpoint_adapter import adapt_observation_input_state


def robot(n=3):
    r = object.__new__(Rs01OmniV13Robot)
    r.cfg = Rs01OmniV13DirectionCfg()
    r.device = 'cpu'; r.num_envs = n
    r._direction_ready = True; r._omni_reference_ready = True
    r.commands = torch.zeros(n, 4); r.command_mode = torch.zeros(n, dtype=torch.long)
    r.gait_enable = torch.ones(n); r.direction_turning = torch.zeros(n, dtype=torch.bool)
    r.base_lin_vel = torch.zeros(n, 3); r.base_ang_vel = torch.zeros(n, 3)
    r.root_states = torch.zeros(n, 13); r.rpy = torch.zeros(n, 3)
    r.omni_desired_position_xy = torch.zeros(n, 2)
    r.omni_estimated_position_xy = torch.zeros(n, 2)
    r.omni_desired_heading_rad = torch.zeros(n)
    r.straight_heading_target_rad = torch.zeros(n)
    r._legal_task_contact_gate = lambda: torch.zeros(n)
    return r


def test_heading_error_correction_sign_for_forward_backward_and_lateral():
    r = robot()
    r.commands[:, :3] = torch.tensor([[.4,0,0],[-.2,0,0],[0,.2,0]])
    r.rpy[:, 2] = .1  # Robot yaw is left of the desired heading.
    t = r._direction_velocity_target()
    assert torch.all(t[:, 2] < 0)
    assert t[0,1] < 0 and t[1,1] > 0 and t[2,0] > 0
    r.rpy.zero_()
    assert torch.equal(r._direction_velocity_target(), r.commands[:, :3])


def test_bounded_target_and_intentional_turn_is_not_fought():
    r = robot()
    r.commands[:, :3] = torch.tensor([[.6,0,0],[-.35,0,0],[.3,.1,.7]])
    r.rpy[:, 2] = 2.0
    r.direction_turning[2] = True
    raw = r.commands.clone()
    t = r._direction_velocity_target()
    assert torch.all(torch.linalg.vector_norm(t[:,:2]-raw[:,:2], dim=1) <= .080001)
    assert torch.all(t[:2,2].abs() <= .35)
    assert torch.equal(t[2], raw[2,:3]) and torch.equal(raw, r.commands)


def test_ordinary_speed_changes_and_march_preserve_heading_reference():
    r = robot()
    r.omni_desired_heading_rad[:] = .2
    r.rpy[:, 2] = .4
    r.set_evaluation_command(torch.tensor([.2,0,0]), 1.)
    r.set_evaluation_command(torch.tensor([.4,0,0]), 1.)
    assert torch.all(r.omni_desired_heading_rad == .2)
    r._set_command_modes(torch.arange(3), torch.full((3,), r.COMMAND_FORWARD))
    assert torch.all(r.omni_desired_heading_rad == .2)
    r.set_evaluation_command(torch.zeros(3), 1.)
    assert torch.all(r.omni_desired_heading_rad == .2)
    r.set_evaluation_command(torch.zeros(3), 1., reset_reference=True)
    assert torch.all(r.omni_desired_heading_rad == .4)


def test_turn_lifecycle_and_hysteresis_reanchor_only_on_events():
    r = robot()
    r.rpy[:, 2] = .3
    r.set_evaluation_command(torch.tensor([0,0,.3]), 1.)
    assert torch.all(r.direction_turning) and torch.all(r.omni_desired_heading_rad == .3)
    r.rpy[:, 2] = .7
    r.set_evaluation_command(torch.tensor([0,0,.03]), 1.)
    assert torch.all(r.direction_turning) and torch.all(r.omni_desired_heading_rad == .3)
    r.set_evaluation_command(torch.tensor([.2,0,0]), 1.)
    assert not torch.any(r.direction_turning)
    assert torch.all(r.omni_desired_heading_rad == .7)
    r.rpy[:, 2] = .8
    r.set_evaluation_command(torch.tensor([.4,0,0]), 1.)
    assert torch.all(r.omni_desired_heading_rad == .7)


def test_reward_prefers_corrective_target_not_raw_command():
    r = robot()
    r.commands[:, 0] = .4; r.rpy[:, 2] = .1
    target = r._direction_velocity_target()
    r.base_lin_vel[:, :2] = target[:, :2]; r.base_ang_vel[:, 2] = target[:, 2]
    assert torch.all(r._reward_tracking_command_velocity() == 1.)
    r.base_lin_vel[:, :2] = r.commands[:, :2]; r.base_ang_vel.zero_()
    assert torch.all(r._reward_tracking_command_velocity() < 1.)
    r.commands.zero_(); r.rpy.zero_(); r.base_lin_vel.zero_()
    assert torch.all(r._reward_tracking_command_velocity() == 1.)
    r.base_lin_vel[:, 1] = .05
    assert torch.all(r._reward_tracking_command_velocity() < 1.)


def test_observations_share_reward_target_without_integral_position_inputs():
    r = robot()
    r.commands[:, 0] = .3; r.rpy[:, 2] = -.1
    r.add_noise = True; r.noise_scale_vec = torch.zeros(61)
    r.commands_scale = torch.tensor([2.,2.,.25])
    r._rs01_observation_estimator_ready = True
    r.estimated_base_lin_vel = torch.zeros(3,3)
    r.estimated_odom_confidence = torch.tensor([0.,.5,1.])
    def parent_obs(env):
        assert not env.add_noise
        env.obs_buf = torch.ones(3,57) * 100.
    with patch.object(Rs01OmniV12Robot, 'compute_observations', parent_obs):
        r.compute_observations()
    r._reward_tracking_command_velocity()
    assert r.obs_buf.shape == (3,61) and r.add_noise
    assert torch.allclose(r.obs_buf[:,9:12], r.direction_reward_target*r.commands_scale)
    assert not torch.any(r.obs_buf[:,[52,55]])
    assert torch.equal(r.obs_buf[:,57:60], r.commands[:,:3]*r.commands_scale)
    assert torch.equal(r.obs_buf[:,60], r.estimated_odom_confidence)


def test_registration_no_duplicate_task_costs_and_hardware_unchanged():
    cfg, ppo = task_registry.get_cfgs('rs01_omni_v13_direction')
    old = Rs01OmniV12WideCfg()
    for key in ('control','rs01_actuator','rs01_odometry','domain_rand'):
        assert class_to_dict(getattr(cfg,key)) == class_to_dict(getattr(old,key))
    a,b = class_to_dict(cfg.rewards.scales),class_to_dict(old.rewards.scales)
    b['pose_error'] = 0.
    assert a == b
    for key in ('pose_error','trajectory_lateral_error','omni_heading_error',
                'tracking_lin_vel','tracking_ang_vel','tracking_planar_velocity','tracking_yaw_velocity'):
        assert a.get(key,0.) == 0., key
    assert a['tracking_command_velocity'] == 14.
    assert cfg.env.num_observations == 61 and cfg.env.num_actions == 12
    assert ppo.algorithm.learning_rate == 1e-4 and ppo.algorithm.schedule == 'fixed'
    assert not ppo.runner.load_optimizer
    for key, value in a.items():
        if value: assert hasattr(Rs01OmniV13Robot, '_reward_'+key)


def test_checkpoint_extension_57_to_61_keeps_existing_weights():
    old = {k:torch.randn(8,57) for k in ('actor.0.weight','critic.0.weight')}
    new = {k:torch.randn(8,61) for k in old}
    result,_ = adapt_observation_input_state(old,new)
    for k in old:
        assert torch.equal(result[k][:,:57],old[k])
        assert not torch.any(result[k][:,57:])


if __name__ == '__main__':
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check(); print('PASS',name)
