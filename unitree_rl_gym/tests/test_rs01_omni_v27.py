import unittest
import isaacgym
import torch
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_config import Rs01OmniV25Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v27_config import Rs01OmniV27Cfg,Rs01OmniV27StrongCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_env import landing_region_cost


class V27Test(unittest.TestCase):
    def test_only_existing_geometry_parameters_change(self):
        a=class_to_dict(Rs01OmniV25Cfg());b=class_to_dict(Rs01OmniV27Cfg())
        for name in ('stance_inner_m','support_region_scale_m'):
            b['rewards'][name]=a['rewards'][name]
        self.assertEqual(a,b)

    def test_more_signal_for_narrow_not_safe_footholds(self):
        contact=torch.ones((1,4),dtype=torch.bool);no=~contact;p=torch.zeros((1,4))
        for cls in (Rs01OmniV27Cfg,Rs01OmniV27StrongCfg):
            cfg=cls().rewards
            self.assertEqual(float(landing_region_cost(torch.full((1,4),.13),contact,no,p,cfg)),0.)
            self.assertGreater(float(landing_region_cost(torch.full((1,4),.09),contact,no,p,cfg)),0.)


if __name__=='__main__':unittest.main()
