import unittest
from types import SimpleNamespace
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v30_config import Rs01OmniV30Cfg
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase import (
    Rs01StablePhaseCfg,Rs01StablePhasePPO,Rs01StablePhaseRobot,advance_lift_clock,corridor_cost)


class StablePhaseTest(unittest.TestCase):
    def test_real_plant_and_sensor_unchanged(self):
        old,new=class_to_dict(Rs01OmniV30Cfg()),class_to_dict(Rs01StablePhaseCfg())
        for k in old:
            if k not in ('commands','rewards'):self.assertEqual(old[k],new[k],k)
        self.assertEqual(new['env']['num_observations'],61)
        self.assertFalse(Rs01StablePhasePPO.runner.amp_enabled)
        self.assertFalse(Rs01StablePhasePPO.runner.resume)
        self.assertEqual(Rs01StablePhasePPO.algorithm.schedule,'fixed')
        self.assertFalse(any('symmetry' in k or 'reference' in k or 'footprint' in k
                             for k in new['rewards']['scales']))

    def test_contact_chatter_cannot_reset_lift_timeout(self):
        age=torch.ones(1,4);air=torch.zeros(1,4);contact=torch.zeros(1,4,dtype=torch.bool)
        age,air=advance_lift_clock(age,air,contact,torch.zeros(1,4),torch.ones(1,dtype=torch.bool),.02)
        self.assertTrue((age>1).all())
        for i in range(2):age,air=advance_lift_clock(age,air,contact,torch.full((1,4),.02),torch.ones(1,dtype=torch.bool),.02)
        self.assertTrue((age==0).all())

    def test_diagonal_phase_and_double_support(self):
        fake=SimpleNamespace(cfg=Rs01StablePhaseCfg(),commands=torch.zeros(4,3),
                             _gait_phase=lambda:torch.tensor([.1,.3,.6,.8]))
        stance,_,height=Rs01StablePhaseRobot._phase_targets(fake)
        self.assertEqual(stance.int().tolist(),[[1,1,1,1],[1,0,0,1],[1,1,1,1],[0,1,1,0]])
        self.assertLessEqual(float(height.max()),.025001)

    def test_one_inward_foot_cannot_hide_behind_pair_average(self):
        good=torch.full((1,4),.14725);bad=good.clone();bad[0,0]=.08
        self.assertEqual(float(corridor_cost(good,torch.ones_like(good))),0.)
        self.assertGreater(float(corridor_cost(bad,torch.ones_like(bad))),0.)


if __name__=='__main__':unittest.main()
