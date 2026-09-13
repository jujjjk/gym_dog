"""Run directly with the Isaac Gym Python environment."""
import isaacgym  # noqa: F401
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_config import Rs01OmniV16Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15StandCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import guarded_target, update_guard, Rs01OmniV16Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_env import Rs01OmniV15StandRobot


class GuardSupportTests(unittest.TestCase):
    def test_projection(self):
        q = torch.tensor([[0.1, -0.2]])
        dq = torch.tensor([[2.0, -3.0]])
        limit = torch.tensor([[6.0, 10.0]])
        target, raw, safe = guarded_target(q + 1.0, q, dq, 40.0, 1.0, limit, -3.0 * torch.ones_like(q), 3.0 * torch.ones_like(q))
        torch.testing.assert_close(raw, torch.tensor([[38.0, 43.0]]))
        torch.testing.assert_close(safe, limit)
        torch.testing.assert_close(40.0 * (target-q) - dq, safe)

    def test_guard_reference(self):
        import math
        rms, limit = update_guard(torch.tensor([49.0]), torch.tensor([8.0]), .02)
        expected = math.exp(-.01)*49 + (1-math.exp(-.01))*64
        self.assertAlmostEqual(rms.item(), expected, places=5)
        self.assertAlmostEqual(limit.item(), 14-4*(math.sqrt(expected)-6), places=5)

    def test_guard_bounds_and_recovery(self):
        rms = torch.tensor([0.0, 10000.0])
        rms, limits = update_guard(rms, torch.zeros(2), .02)
        torch.testing.assert_close(limits, torch.tensor([14.0, 6.0]))
        self.assertLess(rms[1].item(), 10000.)

    def test_no_plant_or_clock_change(self):
        old, new = Rs01OmniV15StandCfg(), Rs01OmniV16Cfg()
        for key in ('response_gain', 'time_constant_s', 'observed_closed_loop_delay_s',
                    'coulomb_friction_nm', 'target_rate_limit_rad_s',
                    'target_acceleration_limit_rad_s2', 'response_update_dt_s', 'control_dt_s'):
            self.assertEqual(getattr(old.rs01_actuator, key), getattr(new.rs01_actuator, key))
        self.assertEqual(new.control.decimation * new.sim.dt, .02)

    def test_no_random_stand_or_reward_duplicate(self):
        cfg = Rs01OmniV16Cfg()
        self.assertEqual(cfg.commands.mode_probabilities[0], 0.)
        self.assertAlmostEqual(sum(cfg.commands.mode_probabilities), 1.)
        self.assertEqual(cfg.commands.moving_to_stand_probability, 0.)
        self.assertFalse(cfg.rewards.tracking_contact_gate)
        self.assertEqual(cfg.rewards.scales.phase_contact_error, 0.)
        self.assertEqual(cfg.rewards.scales.phase_support_tracking, 0.)

    def test_handoff_grace(self):
        cfg = Rs01OmniV16Cfg()
        self.assertGreaterEqual(cfg.rewards.all_feet_contact_grace_s,
                                (cfg.rewards.gait_stance_ratio-.5)*cfg.rewards.gait_period_s+.04-1e-9)
        self.assertLess(cfg.rewards.all_feet_contact_grace_s, cfg.rewards.gait_period_s/2)

    def test_three_feet_allowed_only_during_handoff(self):
        fake = SimpleNamespace(get_foot_contact_mask=lambda: torch.tensor([
            [1,1,1,0], [1,1,1,0], [1,0,0,0], [1,0,0,1]], dtype=torch.bool),
            _handoff_mask=lambda: torch.tensor([True,False,True,False]),
            _walking_command_gate=lambda: torch.ones(4))
        torch.testing.assert_close(Rs01OmniV16Robot._reward_odd_feet_contact(fake), torch.tensor([0.,1.,1.,0.]))

    def test_projection_and_heat_update_only_once_per_policy_step(self):
        env = Rs01OmniV16Robot.__new__(Rs01OmniV16Robot)
        env.cfg = Rs01OmniV16Cfg()
        env.dt = .02
        env._rs01_actuator_ready = env._v16_project_pending = True
        env.dof_pos = env.dof_vel = torch.zeros(1, 12)
        env.rs01_limited_position_target_rad = torch.ones(1, 12)
        env.motor_electromagnetic_torques = torch.zeros(1, 12)
        env.p_gains, env.d_gains = 40., 1.
        env.guard_joint_lower, env.guard_joint_upper = -torch.ones(12)*3, torch.ones(12)*3
        with patch.object(Rs01OmniV15StandRobot, '_compute_torques', return_value=None):
            env._compute_torques(None)
            first = env.guard_rms_sq.clone()
            sent = env.guard_target.clone()
            for _ in range(7):
                env._compute_torques(None)
            torch.testing.assert_close(env.guard_rms_sq, first)
            torch.testing.assert_close(env.guard_target, sent)
            torch.testing.assert_close(env.rs01_limited_position_target_rad, torch.ones(1, 12))

    def test_response_uses_sent_target_and_restores_history(self):
        env = Rs01OmniV16Robot.__new__(Rs01OmniV16Robot)
        env.guard_target = torch.tensor([.2])
        env.rs01_limited_position_target_rad = torch.tensor([.4])
        def advance():
            torch.testing.assert_close(env.rs01_limited_position_target_rad, torch.tensor([.2]))
        with patch.object(Rs01OmniV15StandRobot, '_advance_rs01_response', side_effect=advance):
            env._advance_rs01_response()
        torch.testing.assert_close(env.rs01_limited_position_target_rad, torch.tensor([.4]))

    def test_group_load_does_not_require_left_right_mirroring(self):
        mask = torch.tensor([[True,False,False,True]])
        fake = SimpleNamespace(_gait_phase=lambda: torch.tensor([.3]), cfg=Rs01OmniV16Cfg(),
            contact_forces=torch.tensor([[[0.,0.,8.],[0.,0.,0.],[0.,0.,0.],[0.,0.,2.]]]),
            feet_indices=torch.arange(4), diagonal_a_contact_mask=mask[0],
            get_foot_contact_mask=lambda: mask, _desired_contact_mask=lambda: mask)
        torch.testing.assert_close(Rs01OmniV16Robot._support_structure_error(fake), torch.zeros(1))


if __name__ == '__main__':
    unittest.main()
