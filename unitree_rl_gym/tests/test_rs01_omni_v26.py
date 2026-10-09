import math
import unittest
import isaacgym
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v26_env import map_omni_hip_targets, landing_bound_weight
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import map_hip_targets
from legged_gym.envs.rs01_omni_v2.rs01_omni_v26_config import Rs01OmniV26Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_config import Rs01OmniV25Cfg
from legged_gym.utils.helpers import class_to_dict


class V26Test(unittest.TestCase):
    def test_bounds_and_outward_freedom(self):
        sides=torch.tensor([1.,-1.,1.,-1.]);ids=[0,3,6,9]
        q=torch.zeros(2,12);q[0,ids]=-.3*sides;q[1,ids]=.3*sides
        out=map_omni_hip_targets(q,torch.tensor([[0.,.2,0.],[0.,0.,.3]]),ids,sides,
                                torch.full((4,),.3),math.radians(8),math.radians(3))
        torch.testing.assert_close(out[0,ids],-sides*math.radians(3))
        torch.testing.assert_close(out[1],q[1])
        self.assertEqual(float(out[:,[1,2,4,5,7,8,10,11]].abs().sum()),0.)

    def test_straight_identical_to_v20(self):
        q=torch.rand(20,12)*.6-.3;cmd=torch.zeros(20,3);cmd[:,0]=.4
        ids=[0,3,6,9];side=torch.tensor([1.,-1.,1.,-1.]);ranges=torch.full((4,),.3)
        old=map_hip_targets(q,cmd,ids,side,ranges,math.radians(8))
        new=map_omni_hip_targets(q,cmd,ids,side,ranges,math.radians(8),math.radians(3))
        torch.testing.assert_close(new,old)

    def test_only_control_parameter_changed(self):
        a=class_to_dict(Rs01OmniV25Cfg());b=class_to_dict(Rs01OmniV26Cfg())
        b['control'].pop('omni_inward_target_rad')
        self.assertEqual(a,b)

    def test_phase_boundary_continuity_and_support_release(self):
        phase=torch.tensor([0.,.22,.5,.72,.999999])
        mask=torch.tensor([True,False,False,True])
        w=landing_bound_weight(phase,mask,.72)
        self.assertAlmostEqual(float(w[0,0]),1.)
        self.assertAlmostEqual(float(w[1,0]),0.)
        self.assertAlmostEqual(float(w[3,0]),0.)
        self.assertAlmostEqual(float(w[2,1]),1.)
        self.assertAlmostEqual(float(w[4,0]),float(w[0,0]),places=5)


if __name__=='__main__':unittest.main()
