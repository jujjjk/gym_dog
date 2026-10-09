import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v29_config import Rs01OmniV29Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v30_config import Rs01OmniV30Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import Rs01OmniV28Robot


class V30Test(unittest.TestCase):
    def test_only_two_height_parameters_change(self):
        a=class_to_dict(Rs01OmniV29Cfg()); b=class_to_dict(Rs01OmniV30Cfg())
        for key,value in [('straight_clearance_m',.035),('swing_clearance_m',.028)]:
            self.assertEqual(b['rewards'][key],value)
            b['rewards'][key]=a['rewards'][key]
        self.assertEqual(a,b)  # Includes plant, commands, sensors, reward scales.

    def test_existing_reward_receives_directional_targets(self):
        e=object.__new__(Rs01OmniV28Robot); e.cfg=Rs01OmniV30Cfg()
        e.commands=torch.tensor([[.4,0.,0.],[-.3,0.,0.],[0.,.15,0.],[0.,0.,.6],[.3,.04,0.],[0.,0.,0.]])
        e.diagonal_a_contact_mask=torch.tensor([True,False,False,True])
        e.diagonal_b_contact_mask=~e.diagonal_a_contact_mask
        e._gait_phase=lambda:torch.zeros(6)
        captured=[]
        def target(phase,a,b,duty,radius,height):
            captured.append(height.clone())
            return torch.ones(6,4)*height+radius,torch.ones(6,4,dtype=torch.bool)
        e._swing_height_target_from_phase=target
        e.feet_pos=torch.zeros(6,4,3)
        loss=e._phase_swing_clearance_error()
        torch.testing.assert_close(captured[0].flatten(),torch.tensor([.035,.035,.028,.028,.0315,.028]))
        self.assertTrue(torch.isfinite(loss).all())
        e.feet_pos[:,:,2]=.08
        self.assertTrue((e._phase_swing_clearance_error()==0).all())


if __name__=='__main__': unittest.main()
