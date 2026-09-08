"""Wide-speed sampling, reward and phase contracts without starting PhysX."""

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_config import Rs01OmniV12WideCfg, Rs01OmniV12WideCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_env import Rs01OmniV12Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v11_config import Rs01OmniV11ContinuousCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils import task_registry


def robot(count):
    r = object.__new__(Rs01OmniV12Robot)
    r.cfg = Rs01OmniV12WideCfg()
    r.device = 'cpu'
    r.num_envs = count
    r.dt = .02
    r.commands = torch.zeros(count, 4)
    r.command_mode = torch.zeros(count, dtype=torch.long)
    r.gait_enable = torch.ones(count)
    r.base_lin_vel = torch.zeros(count, 3)
    r.base_ang_vel = torch.zeros(count, 3)
    r._legal_task_contact_gate = lambda: torch.zeros(count)
    r._omni_reference_ready = False
    r._wide_phase_ready = True
    r.wide_gait_phase = torch.zeros(count)
    return r


def test_speed_bands_cover_entire_range_with_expected_mix():
    torch.manual_seed(37)
    r = robot(30000)
    sample = r._sample_magnitude(30000, [.1, .6])
    bins = torch.bucketize(sample, torch.tensor([.2, .4]))
    fractions = torch.bincount(bins, minlength=3) / 30000.
    assert torch.allclose(fractions, torch.tensor([.3, .4, .3]), atol=.015)
    assert sample.min() >= .1 and sample.max() <= .6 and sample.max() > .59


def test_all_modes_bounds_signs_zero_modes_and_combined_envelope():
    torch.manual_seed(11)
    r = robot(7000)
    modes = torch.arange(7).repeat_interleave(1000)
    r._set_command_modes(torch.arange(7000), modes)
    assert not torch.any(r.commands[:2000, :3])
    assert not torch.any(r.gait_enable[:1000]) and torch.all(r.gait_enable[1000:] == 1)
    specs = ((2, 0, .1, .6), (3, 0, .1, .35), (4, 1, .08, .30), (5, 2, .2, 1.))
    for mode, axis, low, high in specs:
        c = r.commands[modes == mode, :3]
        assert torch.all((c[:, axis].abs() >= low) & (c[:, axis].abs() <= high))
        assert not torch.any(c[:, [k for k in range(3) if k != axis]])
        if mode in (4, 5):
            assert torch.any(c[:, axis] < 0) and torch.any(c[:, axis] > 0)
    assert torch.all(r.commands[modes == 2, 0] > 0)
    assert torch.all(r.commands[modes == 3, 0] < 0)
    combined = r.commands[modes == 6, :3]
    assert torch.all(torch.linalg.vector_norm(combined / torch.tensor([.4, .2, .7]), dim=1) <= 1.000001)
    assert torch.unique(combined > 0, dim=0).shape[0] == 8
    r._set_command_modes(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))


def test_reward_retains_large_error_signal_and_precise_zero_target():
    r = robot(7)
    r.commands[:, 0] = .6
    r.base_lin_vel[:, 0] = torch.tensor([0., .2, .4, .55, .6, .65, .8])
    reward = r._reward_tracking_command_velocity()
    assert torch.all(reward[1:5] > reward[:4]) and reward[4] == 1.
    assert abs(reward[1].item() - 1. / 17.) < 1e-6
    assert torch.allclose(reward[3], reward[5], atol=1e-6)
    r.commands.zero_(); r.base_lin_vel.zero_()
    r.base_lin_vel[1, 0] = .04
    r.base_lin_vel[2, 1] = .04
    r.base_ang_vel[3, 2] = .1
    reward = r._reward_tracking_command_velocity()
    assert reward[0] == 1. and torch.all(reward[1:4] < 1.)
    assert torch.all(reward > 0)  # All contacts illegal in this mock.


def test_frequency_limits_and_phase_continuity_across_commands():
    r = robot(4)
    r.commands[:] = torch.tensor([[0.,0.,0.,0.], [.6,0.,0.,0.], [0.,-.3,0.,0.], [0.,0.,1.,0.]])
    assert torch.allclose(r._command_gait_frequency(), torch.tensor([1/.6,2.5,2.5,2.5]))
    r.wide_gait_phase[:] = .97
    before = r._gait_phase().clone()
    r._advance_wide_gait_phase()
    assert torch.allclose(r._gait_phase(), torch.remainder(before + .02*r._command_gait_frequency(), 1.))
    before = r._gait_phase().clone()
    r._set_command_modes(torch.arange(4), torch.full((4,), r.COMMAND_MARCH))
    assert torch.equal(before, r._gait_phase())
    for _ in range(100):
        r._advance_wide_gait_phase()
    assert torch.all((r._gait_phase() >= 0) & (r._gait_phase() < 1))


def test_diagonal_pairing_unchanged_at_every_phase():
    phase = torch.arange(1000) / 1000.
    a = torch.tensor([1,0,0,1], dtype=torch.bool)
    b = ~a
    mask = Rs01OmniV12Robot._desired_contact_mask_from_phase(phase, a, b, .65)
    assert torch.equal(mask[:,0], mask[:,3]) and torch.equal(mask[:,1], mask[:,2])
    assert torch.all((mask.sum(dim=1) == 2) | (mask.sum(dim=1) == 4))


def test_registration_hardware_observations_and_optimizer_unchanged():
    cfg, ppo = task_registry.get_cfgs('rs01_omni_v12_wide')
    old = Rs01OmniV11ContinuousCfg()
    assert cfg.env.num_observations == 57 and cfg.env.num_actions == 12
    for key in ('control', 'rs01_actuator', 'env', 'domain_rand', 'rs01_odometry'):
        assert class_to_dict(getattr(cfg, key)) == class_to_dict(getattr(old, key))
    assert cfg.asset.file == old.asset.file
    assert class_to_dict(cfg.rewards.scales) == class_to_dict(old.rewards.scales)
    assert ppo.algorithm.learning_rate == 1e-4 and ppo.algorithm.schedule == 'fixed'
    assert not ppo.runner.load_optimizer and ppo.runner.adapt_observation_input
    assert cfg.commands.resampling_time == 4.
    for name, scale in class_to_dict(cfg.rewards.scales).items():
        if scale:
            assert hasattr(Rs01OmniV12Robot, '_reward_' + name)


if __name__ == '__main__':
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check()
            print('PASS', name)
