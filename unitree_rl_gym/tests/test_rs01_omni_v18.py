import isaacgym  # noqa: F401
import unittest
from unittest.mock import patch
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v17_config import Rs01OmniV17Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import Rs01OmniV16Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_config import Rs01OmniV18Cfg, Rs01OmniV18LowCfg, Rs01OmniV18CfgPPO, Rs01OmniV18SoftCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_env import Rs01OmniV18Robot, envelope_roll_cost
from legged_gym.utils.helpers import class_to_dict


class V18Tests(unittest.TestCase):
    def test_envelope(self):
        cmd = torch.tensor([[.2,0.,0.],[-.2,0.,0.],[0.,.2,0.],[0.,0.,.3]])
        roll = torch.tensor([4.,-4.,4.,8.])*torch.pi/180
        cost = envelope_roll_cost(roll, cmd, Rs01OmniV18Cfg.rewards)
        torch.testing.assert_close(cost[:2], torch.full((2,),2./9.))
        self.assertEqual(float(cost[2]), 0.)
        self.assertGreater(float(cost[3]), 0.)

    def test_compensation_and_symmetry(self):
        r = torch.tensor([0.,.02,.12])
        c = torch.tensor([[0.,0.,0.],[0.,0.,0.],[.2,.1,.3]])
        a = envelope_roll_cost(r,c,Rs01OmniV18Cfg.rewards)
        torch.testing.assert_close(a, envelope_roll_cost(-r,-c,Rs01OmniV18Cfg.rewards))
        torch.testing.assert_close(a[:2],torch.zeros(2))

    def test_contract_and_isolated_ab(self):
        old,a,b = [class_to_dict(x()) for x in (Rs01OmniV17Cfg,Rs01OmniV18Cfg,Rs01OmniV18LowCfg)]
        for key in ('control','rs01_actuator','commands','env','sim','normalization'):
            self.assertEqual(a[key],old[key])
        self.assertEqual(a['rewards']['scales'],old['rewards']['scales'])
        self.assertFalse(a['rewards']['only_positive_rewards'])
        b['asset']['name']=a['asset']['name']
        b['rewards']['swing_clearance_m']=a['rewards']['swing_clearance_m']
        self.assertEqual(a,b)
        for name in ('step','_compute_torques','compute_observations','_reward_tracking_command_velocity'):
            self.assertIs(getattr(Rs01OmniV18Robot,name),getattr(Rs01OmniV16Robot,name))
        cfg=Rs01OmniV18CfgPPO()
        self.assertEqual(cfg.algorithm.schedule,'fixed')
        self.assertEqual(cfg.algorithm.learning_rate,1e-4)
        self.assertFalse(cfg.runner.load_optimizer)

    def test_replaces_cost_without_gating_support(self):
        env=Rs01OmniV18Robot.__new__(Rs01OmniV18Robot)
        env.cfg=Rs01OmniV18Cfg()
        env.rpy=torch.tensor([[4.*torch.pi/180,0.,0.]])
        env.commands=torch.zeros(1,4)
        env._walking_command_gate=lambda:torch.ones(1)
        with patch.object(Rs01OmniV16Robot,'_reward_phase_two_contact_quality',return_value=torch.tensor([.8])):
            result=env._reward_phase_two_contact_quality()
        torch.testing.assert_close(result,torch.tensor([.8-3.*2./9.]))

    def test_soft_only_changes_posture_weight(self):
        a,b=[class_to_dict(x()) for x in (Rs01OmniV18Cfg,Rs01OmniV18SoftCfg)]
        self.assertEqual(b['rewards']['balance_prior_weight'],1.5)
        b['asset']['name']=a['asset']['name']
        b['rewards']['balance_prior_weight']=a['rewards']['balance_prior_weight']
        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()
