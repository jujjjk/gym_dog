import isaacgym
import unittest
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_env import half_cycle_sample,Rs01OmniV21Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_config import Rs01OmniV21Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_config import Rs01OmniV20BCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import Rs01OmniV20BoundedRobot
from legged_gym.utils.helpers import class_to_dict

class Tests(unittest.TestCase):
    def test_contract(self):
        a,b=[class_to_dict(c()) for c in (Rs01OmniV20BCfg,Rs01OmniV21Cfg)]
        for key in ('control','rs01_actuator','sim','env','commands','normalization'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(a['rewards']['scales'],b['rewards']['scales'])
        self.assertEqual(b['rewards']['swing_clearance_m'],.025)
        for name in ('step','compute_observations','_compute_torques','_project_rs01_policy_target'):
            self.assertIs(getattr(Rs01OmniV21Robot,name),getattr(Rs01OmniV20BoundedRobot,name))

    def test_interpolation_and_startup(self):
        stamps=torch.tensor([[1.,.8,.6,.4],[.2,.1,-float('inf'),-float('inf')]])
        history=torch.zeros(2,4,4,2)
        history[0]=stamps[0,:,None,None].expand(4,4,2)
        out,valid=half_cycle_sample(history,stamps,torch.tensor([1.,.2]))
        torch.testing.assert_close(out[0],torch.full((4,2),.5))
        self.assertEqual(valid.tolist(),[True,False]);self.assertTrue(torch.isfinite(out).all())

    def test_antiphase_is_not_simultaneous_mirroring(self):
        stamps=torch.linspace(1.,0.,31)[None]
        phase=stamps[0]
        left=.05*torch.sin(2*torch.pi*phase)
        right=.05*torch.sin(2*torch.pi*(phase+.5))
        h=torch.zeros(1,31,4,2);h[0,:,0,0]=left;h[0,:,1,0]=right
        old,valid=half_cycle_sample(h,stamps,torch.tensor([.75]))
        self.assertTrue(valid.item());self.assertLess(abs(float(old[0,1,0])+.05),.001)

    def test_unwrapped_phase_and_frozen_startup(self):
        stamps=torch.tensor([[3.,2.75,2.5,2.25],[0.,0.,0.,0.]])
        history=torch.ones(2,4,4,2)
        out,valid=half_cycle_sample(history,stamps,torch.tensor([3.,0.]))
        self.assertEqual(valid.tolist(),[True,False])
        torch.testing.assert_close(out,history[:,0])

if __name__=='__main__':unittest.main()
