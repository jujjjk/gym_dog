import unittest
import isaacgym
import torch
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v27_config import Rs01OmniV27StrongCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_config import Rs01OmniV28Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import curriculum_command,Rs01OmniV28Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_env import Rs01OmniV25Robot


class V28Test(unittest.TestCase):
    def test_only_sampling_changed(self):
        a=class_to_dict(Rs01OmniV27StrongCfg());b=class_to_dict(Rs01OmniV28Cfg())
        b['commands']=a['commands'];self.assertEqual(a,b)

    def test_envelope_keeps_straights_forward_combo_and_yaw(self):
        q=torch.tensor([[.6,0,0],[-.35,0,0],[.3,.15,.5],[0,0,1.],[0,.3,0],[-.3,.15,-.5]])
        out=curriculum_command(q,.15,[.18,.08,.25])
        torch.testing.assert_close(out[:4],q[:4])
        self.assertAlmostEqual(float(out[4,1]),.15,places=6)
        self.assertLessEqual(float(torch.linalg.vector_norm(out[5]/torch.tensor([.18,.08,.25]))),1.000001)
        torch.testing.assert_close(curriculum_command(out,.15,[.18,.08,.25]),out)

    def test_sampling_change_does_not_rescale_clock(self):
        e=object.__new__(Rs01OmniV28Robot);e.cfg=Rs01OmniV28Cfg()
        e.commands=torch.tensor([[.4,0,0],[0,.15,0],[-.12,.06,.2],[0,0,.6]])
        actual=e._command_gait_frequency()
        e.cfg=Rs01OmniV27StrongCfg()
        torch.testing.assert_close(actual,Rs01OmniV25Robot._command_gait_frequency(e))


if __name__=='__main__':unittest.main()
