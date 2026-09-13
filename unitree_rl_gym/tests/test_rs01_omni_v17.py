import isaacgym  # noqa: F401
import unittest
from unittest.mock import patch
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_config import Rs01OmniV16Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v17_config import Rs01OmniV17Cfg, Rs01OmniV17CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import Rs01OmniV16Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v17_env import Rs01OmniV17Robot, conditional_roll_cost
from legged_gym.utils.helpers import class_to_dict


class BalanceTests(unittest.TestCase):
    def test_compensation_deadband(self):
        cost = conditional_roll_cost(torch.tensor([0., .02, -.02]), torch.zeros(3,3), Rs01OmniV17Cfg.rewards)
        torch.testing.assert_close(cost, torch.zeros(3))

    def test_both_lean_directions_and_backward(self):
        cmd = torch.tensor([[.2,0.,0.],[-.2,0.,0.]])
        cost = conditional_roll_cost(torch.tensor([.14,-.14]), cmd, Rs01OmniV17Cfg.rewards)
        self.assertGreater(float(cost[0]), 0.)
        torch.testing.assert_close(cost[0], cost[1])

    def test_omni_release(self):
        cmd = torch.tensor([[0.,.08,0.],[0.,0.,.2],[.2,-.08,-.3]])
        torch.testing.assert_close(conditional_roll_cost(torch.ones(3), cmd, Rs01OmniV17Cfg.rewards), torch.zeros(3))

    def test_no_new_reward_or_actuator_observation_change(self):
        old, new = class_to_dict(Rs01OmniV16Cfg()), class_to_dict(Rs01OmniV17Cfg())
        for key in ('rs01_actuator','control','sim','env','commands','normalization'):
            self.assertEqual(old[key], new[key])
        self.assertEqual(old['rewards']['scales'], new['rewards']['scales'])
        self.assertEqual(new['rewards']['scales']['orientation'], 0.)
        self.assertEqual(new['rewards']['scales']['phase_contact_error'], 0.)
        self.assertFalse(new['rewards']['tracking_contact_gate'])
        self.assertEqual(new['rewards']['gait_stance_ratio'], .72)
        self.assertEqual(new['rewards']['all_feet_contact_grace_s'], .172)

    def test_tracking_and_execution_methods_unchanged(self):
        for name in ('_reward_tracking_command_velocity','_compute_torques','step','compute_observations'):
            self.assertIs(getattr(Rs01OmniV17Robot,name), getattr(Rs01OmniV16Robot,name))

    def test_fixed_lr_and_fresh_optimizer(self):
        cfg=Rs01OmniV17CfgPPO()
        self.assertEqual(cfg.algorithm.schedule,'fixed')
        self.assertEqual(cfg.algorithm.learning_rate,.0001)
        self.assertFalse(cfg.runner.load_optimizer)

    def test_lean_does_not_gate_away_support_incentive(self):
        env=Rs01OmniV17Robot.__new__(Rs01OmniV17Robot)
        env.cfg=Rs01OmniV17Cfg()
        env.rpy=torch.tensor([[.14,0.,0.]])
        env.commands=torch.zeros(1,4)
        env._walking_command_gate=lambda:torch.ones(1)
        with patch.object(Rs01OmniV16Robot,'_reward_phase_two_contact_quality',return_value=torch.tensor([.8])):
            good=env._reward_phase_two_contact_quality()
        with patch.object(Rs01OmniV16Robot,'_reward_phase_two_contact_quality',return_value=torch.tensor([.4])):
            poor=env._reward_phase_two_contact_quality()
        torch.testing.assert_close(good-poor,torch.tensor([.4]))

    def test_clearance_changes_height_not_support_window(self):
        a=torch.tensor([True,False,False,True]); b=~a
        height,swing=Rs01OmniV17Robot._swing_height_target_from_phase(
            torch.tensor([.86]),a,b,.72,.016,.018)
        torch.testing.assert_close(height[0,0],torch.tensor(.034))
        self.assertTrue(torch.equal(swing[0],a))


if __name__ == '__main__':
    unittest.main()
