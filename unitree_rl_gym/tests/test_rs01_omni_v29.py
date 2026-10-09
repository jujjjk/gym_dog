import unittest
import isaacgym
import torch
import legged_gym.envs  # registers tasks; avoids helpers->envs circular import when run standalone
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_config import Rs01OmniV28Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v29_config import Rs01OmniV29Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import Rs01OmniV28Robot


class V29Test(unittest.TestCase):
    def test_only_cadence_floor_changed(self):
        a=class_to_dict(Rs01OmniV28Cfg());b=class_to_dict(Rs01OmniV29Cfg())
        self.assertEqual(Rs01OmniV28Cfg.rewards.lateral_frequency_floor_hz,2.0)
        self.assertEqual(b['rewards']['lateral_frequency_floor_hz'],3.0)
        b['rewards']=a['rewards'];self.assertEqual(a,b)

    def test_full_blend_lateral_and_yaw_reach_mince_cadence(self):
        e=object.__new__(Rs01OmniV28Robot);e.cfg=Rs01OmniV29Cfg()
        e.commands=torch.tensor([[0.,.15,0.],[0.,.15,0.],[0,0,.6],[0,0,-1.0]])
        f=e._command_gait_frequency()
        torch.testing.assert_close(f[:2],torch.full((2,),3.0),atol=1e-6,rtol=0)
        torch.testing.assert_close(f[2:],torch.full((2,),3.0),atol=1e-6,rtol=0)

    def test_forward_cadence_unchanged(self):
        for cmd in ([.4,0,0],[.6,0,0],[-.3,0,0]):
            v28=object.__new__(Rs01OmniV28Robot);v28.cfg=Rs01OmniV28Cfg()
            v28.commands=torch.tensor([cmd])
            v29=object.__new__(Rs01OmniV28Robot);v29.cfg=Rs01OmniV29Cfg()
            v29.commands=torch.tensor([cmd])
            torch.testing.assert_close(v29._command_gait_frequency(),
                v28._command_gait_frequency(),atol=0,rtol=0)

    def test_low_speed_ramp_below_ceiling(self):
        e=object.__new__(Rs01OmniV28Robot);e.cfg=Rs01OmniV29Cfg()
        e.commands=torch.tensor([[0.,.02,0.]])
        f=float(e._command_gait_frequency())
        self.assertLess(f,1.7+2./3.*(2.5-1.7))
        self.assertGreater(f,1.6)


if __name__=='__main__':unittest.main()
