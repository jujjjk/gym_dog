"""Behavioral reward contracts and observation migration, without PhysX."""

from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_env import Rs01OmniV10Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_config import Rs01OmniV10RecoveryCfg
from legged_gym.utils.checkpoint_adapter import adapt_observation_input_state


def robot(count):
    r = object.__new__(Rs01OmniV10Robot)
    r.cfg = Rs01OmniV10RecoveryCfg()
    r.num_envs = count
    r.commands = torch.zeros(count, 4)
    r.base_lin_vel = torch.zeros(count, 3)
    r.base_ang_vel = torch.zeros(count, 3)
    r.gait_enable = torch.ones(count)
    r._v2_command_ready = True
    r.all_feet_contact_time_s = torch.zeros(count)
    r.contacts = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool).repeat(count, 1)
    r.desired = r.contacts.clone()
    r.get_foot_contact_mask = lambda: r.contacts
    r._desired_contact_mask = lambda: r.desired
    return r


def test_zero_command_rewards_hold_and_rejects_drift_in_each_axis():
    r = robot(4)
    r.base_lin_vel[1, 0] = 0.04
    r.base_lin_vel[2, 1] = 0.04
    r.base_ang_vel[3, 2] = 0.1
    rewards = r._reward_tracking_command_velocity()
    assert rewards[0] == 1.0 and torch.all(rewards[1:] < rewards[0])
    r.gait_enable.zero_()
    r.contacts[:] = True
    stand_rewards = r._reward_tracking_command_velocity()
    assert torch.allclose(stand_rewards, rewards)


def test_all_contact_masks_only_correct_diagonal_gets_velocity_payoff():
    r = robot(16)
    r.contacts = torch.tensor([[bool(n & (1 << k)) for k in range(4)] for n in range(16)])
    r.commands[:, 0] = 0.1
    r.base_lin_vel[:, 0] = 0.1
    rewards = r._reward_tracking_command_velocity()
    assert rewards[9] == 1.0
    assert torch.count_nonzero(rewards) == 1


def test_handoff_allowed_but_prolonged_four_contact_gets_no_task_payoff():
    r = robot(3)
    r.contacts[:] = True
    r.desired[:] = True
    r.all_feet_contact_time_s[:] = torch.tensor([0.09, 0.13, 0.5])
    assert torch.equal(r._legal_task_contact_gate(), torch.tensor([1., 0., 0.]))
    r.gait_enable.zero_()
    assert torch.equal(r._legal_task_contact_gate(), torch.ones(3))


def test_pose_cost_covers_both_axes_zero_modes_and_does_not_plateau():
    r = robot(5)
    r.root_states = torch.zeros(5, 13)
    r.omni_desired_position_xy = torch.zeros(5, 2)
    r.root_states[:, 0] = torch.tensor([0., 0.2, 0., 0.7, 1.2])
    r.root_states[2, 1] = 0.2
    r._straight_heading_error = lambda: torch.zeros(5)
    cost = r._reward_pose_error()
    assert cost[0] == 0 and cost[1] == cost[2]
    assert torch.allclose(cost[4]-cost[3], cost[3]-cost[1])
    r.gait_enable.zero_()
    assert torch.equal(r._reward_pose_error(), cost)
    r._straight_heading_error = lambda: torch.tensor([0., 0.4, 0.8, 1.2, 1.6])
    assert torch.all(r._reward_pose_error()[1:] > cost[1:])


def test_append_only_migration_preserves_actor_and_critic_function():
    old = {k: torch.randn(8, 55) for k in ('actor.0.weight', 'critic.0.weight')}
    new = {k: torch.randn(8, 57) for k in old}
    adapted, _ = adapt_observation_input_state(old, new)
    observations = torch.randn(12, 57)
    for k in old:
        assert torch.count_nonzero(adapted[k][:, 55:]) == 0
        assert torch.allclose(observations @ adapted[k].T,
                              observations[:, :55] @ old[k].T, atol=1e-5)


def test_external_command_reanchors_once_not_every_step():
    r = robot(2)
    r._omni_reference_ready = True
    r.root_states = torch.ones(2, 13)
    r.rpy = torch.ones(2, 3) * 0.2
    r.omni_desired_position_xy = torch.zeros(2, 2)
    r.omni_estimated_position_xy = torch.zeros(2, 2)
    r.omni_desired_heading_rad = torch.zeros(2)
    r.straight_heading_target_rad = torch.zeros(2)
    cmd = torch.tensor([0.1, 0., 0.])
    r.set_evaluation_command(cmd, 1.)
    assert torch.equal(r.omni_desired_position_xy, torch.ones(2, 2))
    r.root_states[:, :2] = 2.
    r.set_evaluation_command(cmd, 1.)
    assert torch.equal(r.omni_desired_position_xy, torch.ones(2, 2))
    r.set_evaluation_command(torch.zeros(3), 0.)
    assert torch.equal(r.omni_desired_position_xy, torch.ones(2, 2)*2.)


if __name__ == '__main__':
    checks = [(name, value) for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for name, check in checks:
        check()
        print('PASS', name)
