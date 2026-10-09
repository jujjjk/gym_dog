import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase import Rs01StablePhaseCfg
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase_v2 import phase_progress,acquisition_grace,Rs01StablePhaseV2Cfg


class RepairTest(unittest.TestCase):
    def test_standing_earns_zero_phase_income_at_every_phase(self):
        phase=(torch.linspace(0,1,500)[:,None]+torch.tensor([0.,.5,.5,0.])).remainder(1.)
        stance=phase<.7;swing=((phase-.7)/.3).clamp(0,1);lift=.025*torch.sin(torch.pi*swing).square()
        still=phase_progress(torch.ones_like(stance),torch.zeros_like(lift),stance,swing,lift)
        torch.testing.assert_close(still,torch.zeros(500),atol=1e-7,rtol=0)
        ideal=phase_progress(stance,lift,stance,swing,lift)
        # At duty=.70 only30% of each leg's cycle can improve on standing.
        self.assertGreater(float(ideal.mean()),.15)

    def test_learning_grace_is_not_eval_relaxation(self):
        self.assertEqual(acquisition_grace(0,True),12.)
        self.assertEqual(acquisition_grace(12000,True),7.5)
        self.assertEqual(acquisition_grace(24000,True),3.)
        self.assertEqual(acquisition_grace(0,False),3.)

    def test_plant_sensor_and_anti_inward_unchanged(self):
        a,b=class_to_dict(Rs01StablePhaseCfg()),class_to_dict(Rs01StablePhaseV2Cfg())
        for key in a:
            if key not in ('stability','rewards'):self.assertEqual(a[key],b[key],key)
        for key in a['stability']:self.assertEqual(a['stability'][key],b['stability'][key])
        self.assertEqual(a['rewards']['scales'].keys(),b['rewards']['scales'].keys())


if __name__=='__main__':unittest.main()
