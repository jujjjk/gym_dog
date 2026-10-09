import isaacgym
import unittest
from types import SimpleNamespace
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v23_config import Rs01OmniV23Cfg, Rs01OmniV23WideCfg, Rs01OmniV23CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v23_env import Rs01OmniV23Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_env import envelope_roll_cost


class Tests(unittest.TestCase):
    def test_no_physics_observation_command_or_reward_stack_change(self):
        old = class_to_dict(Rs01OmniV22Cfg())
        for cfg in (Rs01OmniV23Cfg, Rs01OmniV23WideCfg):
            new = class_to_dict(cfg())
            for key in old:
                if key != 'rewards':
                    self.assertEqual(old[key], new[key], key)
            self.assertEqual(old['rewards']['scales'], new['rewards']['scales'])
            changed = {k for k in new['rewards'] if new['rewards'][k] != old['rewards'].get(k)}
            self.assertEqual(changed, {'balance_omni_deadband_rad', 'footprint_omni_min_width_m', 'omni_placement_weight'})

    def test_ab_only_width(self):
        a,b = [class_to_dict(c.rewards) for c in (Rs01OmniV23Cfg, Rs01OmniV23WideCfg)]
        self.assertEqual({k for k in a if a[k] != b[k]}, {'footprint_omni_min_width_m'})

    def test_omni_lean_is_supervised_without_altering_straight(self):
        roll = torch.tensor([.06981317, .06981317])  # 4 degrees
        cmd = torch.tensor([[.2,0.,0.],[0.,.2,0.]])
        old = envelope_roll_cost(roll,cmd,Rs01OmniV22Cfg.rewards)
        new = envelope_roll_cost(roll,cmd,Rs01OmniV23Cfg.rewards)
        self.assertEqual(old[0],new[0]);self.assertEqual(float(old[1]),0.)
        self.assertGreater(float(new[1]),0.)
        env=SimpleNamespace(commands=cmd,cfg=Rs01OmniV23Cfg())
        torch.testing.assert_close(Rs01OmniV23Robot._placement_reward_weight(env),torch.ones(2))

    def test_learning_rate_and_optimizer(self):
        cfg=Rs01OmniV23CfgPPO()
        self.assertEqual(cfg.algorithm.learning_rate,1e-4)
        self.assertEqual(cfg.algorithm.schedule,'fixed')
        self.assertFalse(cfg.runner.load_optimizer)


if __name__ == '__main__':unittest.main()
