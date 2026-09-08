"""Exhaustive reward merge and single-variable A/B contracts, without PhysX."""

from itertools import product

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_env import Rs01OmniV10Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_config import Rs01OmniV10RecoveryCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v11_env import Rs01OmniV11Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v11_config import (
    Rs01OmniV11HardGateCfg, Rs01OmniV11ContinuousCfg,
    Rs01OmniV11HardGateCfgPPO, Rs01OmniV11ContinuousCfgPPO,
)
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils import task_registry


def robot(count, continuous=False):
    r = object.__new__(Rs01OmniV11Robot)
    r.cfg = Rs01OmniV11ContinuousCfg() if continuous else Rs01OmniV11HardGateCfg()
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


def test_merge_preserves_v10_payoff_for_every_contact_mask():
    cases = list(product(range(16), (9, 6, 15), (0., 1.), (0.09, 0.13)))
    r = robot(len(cases))
    r.contacts = torch.tensor([[bool(mask & (1 << k)) for k in range(4)]
                               for mask, _, _, _ in cases])
    r.desired = torch.tensor([[bool(mask & (1 << k)) for k in range(4)]
                              for _, mask, _, _ in cases])
    r.gait_enable = torch.tensor([gait for _, _, gait, _ in cases])
    r.all_feet_contact_time_s = torch.tensor([duration for _, _, _, duration in cases])
    old_scales = Rs01OmniV10RecoveryCfg.rewards.scales
    old = (Rs01OmniV10Robot._reward_phase_support_tracking(r) * old_scales.phase_support_tracking
           + r._reward_phase_two_contact_quality() * old_scales.phase_two_contact_quality)
    new = r._reward_phase_two_contact_quality() * r.cfg.rewards.scales.phase_two_contact_quality
    assert torch.equal(old, new), (old - new).abs().max()
    assert r.cfg.rewards.scales.phase_support_tracking == 0


def test_hard_branch_is_identical_to_v10_velocity_reward():
    r = robot(16)
    r.contacts = torch.tensor([[bool(n & (1 << k)) for k in range(4)] for n in range(16)])
    r.commands[:, 0] = .1
    r.base_lin_vel[:, 0] = torch.linspace(-.1, .2, 16)
    assert torch.equal(r._reward_tracking_command_velocity(),
                       Rs01OmniV10Robot._reward_tracking_command_velocity(r))


def test_continuous_branch_retains_feedback_on_illegal_contacts():
    r = robot(4, continuous=True)
    r.contacts[:, 1] = True  # Three feet: illegal in this diagonal phase.
    r.base_lin_vel[1, 0] = .04
    r.base_lin_vel[2, 1] = .04
    r.base_ang_vel[3, 2] = .1
    rewards = r._reward_tracking_command_velocity()
    assert rewards[0] == 1 and torch.all((rewards[1:] > 0) & (rewards[1:] < 1))
    assert torch.count_nonzero(r.v11_legal_contact_gate) == 0
    r.cfg = Rs01OmniV11HardGateCfg()
    assert torch.count_nonzero(r._reward_tracking_command_velocity()) == 0


def test_continuous_and_hard_equal_on_legal_support_and_stand():
    r = robot(3, continuous=True)
    r.commands[:, 0] = .1
    r.base_lin_vel[:, 0] = torch.tensor([0., .1, .2])
    for stand in (False, True):
        if stand:
            r.gait_enable.zero_()
            r.contacts[:] = True
        r.cfg = Rs01OmniV11ContinuousCfg()
        continuous = r._reward_tracking_command_velocity()
        r.cfg = Rs01OmniV11HardGateCfg()
        assert torch.equal(continuous, r._reward_tracking_command_velocity())


def test_ablation_changes_only_gate_and_names():
    a, b = class_to_dict(Rs01OmniV11HardGateCfg()), class_to_dict(Rs01OmniV11ContinuousCfg())
    a['asset'].pop('name'); b['asset'].pop('name')
    assert a['rewards'].pop('tracking_contact_gate') is True
    assert b['rewards'].pop('tracking_contact_gate') is False
    assert a == b
    old = class_to_dict(Rs01OmniV10RecoveryCfg())
    old['asset'].pop('name')
    old['rewards']['scales']['phase_support_tracking'] = 0.
    old['rewards']['scales']['phase_two_contact_quality'] = 1.75
    assert a == old
    pa = class_to_dict(Rs01OmniV11HardGateCfgPPO())
    pb = class_to_dict(Rs01OmniV11ContinuousCfgPPO())
    pa['runner'].pop('experiment_name'); pb['runner'].pop('experiment_name')
    assert pa == pb
    assert pa['algorithm']['schedule'] == 'fixed'
    assert pa['algorithm']['learning_rate'] == 1e-4
    assert pa['runner']['load_optimizer'] is False


def test_registration_and_active_rewards_exist():
    for task in ('rs01_omni_v11_hard_gate', 'rs01_omni_v11_continuous'):
        cfg, _ = task_registry.get_cfgs(task)
        assert cfg.env.num_observations == 57 and cfg.env.num_actions == 12
        for name, scale in class_to_dict(cfg.rewards.scales).items():
            if scale:
                assert hasattr(Rs01OmniV11Robot, '_reward_' + name), name


if __name__ == '__main__':
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check()
            print('PASS', name)
