import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_config import Rs01AMPCoreCfg,Rs01AMPCorePPO
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_env import progress_over_stationary
from legged_gym.envs.rs01_omni_v2.rs01_amp_scratch_config import Rs01AMPScratchCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import REFERENCE
from legged_gym.algorithms.rs01_amp import StyleLearner


class CoreTest(unittest.TestCase):
    def test_tracking_cannot_pay_stationary(self):
        target=torch.tensor([[.25,0,0],[0,0,0],[0,0,.45]])
        torch.testing.assert_close(progress_over_stationary(torch.zeros_like(target),target),torch.zeros(3))
        self.assertGreater(float(progress_over_stationary(target,target)[0]),.8)

    def test_physical_plant_preserved(self):
        a=class_to_dict(Rs01AMPScratchCfg());b=class_to_dict(Rs01AMPCoreCfg())
        for key in a:
            if key not in ('control','rewards'):self.assertEqual(a[key],b[key],key)
        for key in a['control']:
            if key!='action_scale_by_joint':self.assertEqual(a['control'][key],b['control'][key])
        self.assertNotIn('alive',b['rewards']['scales'])
        self.assertTrue(Rs01AMPCorePPO.runner.freeze_action_std)

    def test_style_ignores_ideal_root_channels(self):
        cfg=class_to_dict(Rs01AMPCorePPO().runner)
        a=StyleLearner(REFERENCE,allow_kinematic=True,settings=cfg)
        pair=a.expert[0,:2];cmd=a.commands[0].expand(2,-1)
        x=a.inputs(pair,cmd);changed=pair.clone()
        changed[:,36:46]+=10;changed[:,82:92]-=10
        torch.testing.assert_close(x,a.inputs(changed,cmd))
        old=StyleLearner(REFERENCE,allow_kinematic=True)
        with self.assertRaises(ValueError):a.load_state_dict(old.state_dict())


if __name__=='__main__':unittest.main()
