import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_bootstrap import (
    nearest_phase_frame, Rs01AMPBootstrapCfg, Rs01AMPBootstrapSlowCfg,
    Rs01AMPBootstrapPPO, Rs01AMPBootstrapSlowPPO)
from legged_gym.envs.rs01_omni_v2.rs01_amp_style_first import Rs01AMPStyleFirstCfg,Rs01AMPStyleFirstSlowCfg


class BootstrapTest(unittest.TestCase):
    def test_phase_wrap(self):
        grid=torch.tensor([[.01,.25,.50,.75]])
        self.assertEqual(nearest_phase_frame(torch.tensor([.99]),grid).item(),0)
        self.assertEqual(nearest_phase_frame(torch.tensor([.49]),grid).item(),2)

    def test_no_plant_change_and_teacher_removal_is_complete(self):
        for base,boot,ppo in ((Rs01AMPStyleFirstCfg,Rs01AMPBootstrapCfg,Rs01AMPBootstrapPPO),
                              (Rs01AMPStyleFirstSlowCfg,Rs01AMPBootstrapSlowCfg,Rs01AMPBootstrapSlowPPO)):
            a,b=class_to_dict(base()),class_to_dict(boot())
            for key in a:
                if key!='rewards':self.assertEqual(a[key],b[key],key)
            self.assertNotIn('reference_motion',a['rewards']['scales'])
            scales=b['rewards']['scales'].copy();self.assertEqual(scales.pop('reference_motion'),8.)
            self.assertEqual(scales,a['rewards']['scales'])
            self.assertFalse(ppo.runner.resume)
            self.assertEqual(ppo.runner.amp_style_weight,0.)


if __name__=='__main__':unittest.main()
