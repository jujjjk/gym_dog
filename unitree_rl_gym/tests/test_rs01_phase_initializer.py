import unittest
import sys
from pathlib import Path
import isaacgym
import torch
import legged_gym.envs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legged_gym/scripts'))
from rs01_phase_initializer import seed_targets
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase_v2 import Rs01StablePhaseV2Cfg
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase_v3 import Rs01StablePhaseV3Cfg
from legged_gym.utils.helpers import class_to_dict


class InitTest(unittest.TestCase):
    def test_only_policy_initialization_changes(self):
        self.assertEqual(class_to_dict(Rs01StablePhaseV2Cfg()),class_to_dict(Rs01StablePhaseV3Cfg()))

    def test_ik_finite_and_static_load_is_target_offset_only(self):
        default=torch.tensor([[0.,-.32987297,1.31853104]*4])
        phase=torch.zeros(8);command=torch.zeros(8,3)
        q=seed_targets(default,command,phase)
        compensated=seed_targets(default,command,phase,mass_kg=11.7317368,kp=torch.full((12,),40.))
        self.assertTrue(torch.isfinite(compensated).all())
        # Compensation counters negative vertical calf Jacobian under upward load.
        self.assertTrue(((compensated-q)[:,2::3]>0).all())
        self.assertLess(float((compensated-q).abs().max()),.22)


if __name__=='__main__':unittest.main()
