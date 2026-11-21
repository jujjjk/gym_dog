import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_lateral import (
    lateral_frequency, Rs01V22PhaseLateralCfg, Rs01V22PhaseLateralPPO)
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_guard import (
    Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard5PPO)


class LateralTest(unittest.TestCase):
    def test_only_cadence_changes(self):
        a = class_to_dict(Rs01V22PhaseLateralCfg()); a.pop('lateral_cadence')
        self.assertEqual(a, class_to_dict(Rs01V22PhaseGuard5Cfg()))
        a,b = class_to_dict(Rs01V22PhaseLateralPPO()),class_to_dict(Rs01V22PhaseGuard5PPO())
        a['runner'].pop('experiment_name'); b['runner'].pop('experiment_name')
        self.assertEqual(a,b)

    def test_march_forward_turn_diagonal_unchanged(self):
        cmd=torch.tensor([[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.],[0.,0.,.45],[.2,.1,0.]])
        old=torch.tensor([1.667,2.,2.3,2.2,2.5])
        torch.testing.assert_close(lateral_frequency(old,cmd,2.1),old)

    def test_lateral_symmetric_bounded_continuous(self):
        vy=torch.linspace(-.3,.3,1001);cmd=torch.zeros(1001,3);cmd[:,1]=vy
        out=lateral_frequency(torch.full_like(vy,2.5),cmd,2.1)
        torch.testing.assert_close(out,out.flip(0))
        self.assertLess(float((out[1:]-out[:-1]).abs().max()),.003)
        self.assertAlmostEqual(float(out[0]),2.1,places=5)
        self.assertTrue(bool(((out>=2.1)&(out<=2.5)).all()))
        low=torch.full_like(vy,1.8)
        torch.testing.assert_close(lateral_frequency(low,cmd,2.1),low)


if __name__=='__main__':
    unittest.main()
