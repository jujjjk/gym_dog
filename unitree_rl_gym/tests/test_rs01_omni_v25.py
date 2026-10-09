import unittest
import isaacgym
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_env import Rs01OmniV25Robot, landing_region_cost
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_config import Rs01OmniV25Cfg, Rs01OmniV25FastCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v24_config import Rs01OmniV24WideCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_env import Rs01OmniV12Robot
from legged_gym.utils.helpers import class_to_dict


class V25Test(unittest.TestCase):
    def robot(self):
        e=object.__new__(Rs01OmniV25Robot);e.cfg=Rs01OmniV25Cfg();e.dt=.02
        e.commands=torch.tensor([[.4,0,0],[0,.08,0],[0,0,.2],[0,.3,0],[0,0,0]])
        e.wide_gait_phase=torch.full((5,),.98)
        return e

    def test_cadence_preserves_straight_and_existing_max(self):
        e=self.robot();f=e._command_gait_frequency()
        old=Rs01OmniV12Robot._command_gait_frequency(e)
        torch.testing.assert_close(f[[0,4]],old[[0,4]])
        torch.testing.assert_close(f[1:4],torch.tensor([2.,2.,2.5]))
        e.cfg=Rs01OmniV25FastCfg()
        torch.testing.assert_close(e._command_gait_frequency()[1:4],torch.tensor([2.2,2.2,2.5]))

    def test_phase_integrates_once_and_slews(self):
        e=self.robot();e.v25_frequency_hz=torch.full((5,),1.7)
        previous=e.wide_gait_phase.clone();e._advance_wide_gait_phase()
        self.assertLessEqual(float((e.v25_frequency_hz-1.7).abs().max()),.040001)
        torch.testing.assert_close(e.wide_gait_phase,(previous+.02*e.v25_frequency_hz)%1)
        p=e.wide_gait_phase.clone();e._command_gait_frequency();e._command_gait_frequency()
        torch.testing.assert_close(e.wide_gait_phase,p)

    def test_landing_not_early_swing(self):
        cfg=Rs01OmniV25Cfg().rewards;y=torch.full((1,4),.06)
        no=torch.zeros((1,4),dtype=torch.bool)
        early=landing_region_cost(y,no,no,torch.full((1,4),.2),cfg)
        late=landing_region_cost(y,no,no,torch.full((1,4),.9),cfg)
        self.assertEqual(float(early),0.);self.assertGreater(float(late),0.)

    def test_stance_allows_motion_but_landing_not_inward(self):
        cfg=Rs01OmniV25Cfg().rewards;y=torch.full((1,4),.09)
        yes=torch.ones((1,4),dtype=torch.bool);no=~yes;p=torch.zeros_like(y)
        self.assertEqual(float(landing_region_cost(y,yes,no,p,cfg)),0.)
        self.assertGreater(float(landing_region_cost(y,yes,yes,p,cfg)),0.)
        self.assertGreater(float(landing_region_cost(y-.1,yes,no,p,cfg)),0.)

    def test_no_reward_or_actuator_changes(self):
        a=class_to_dict(Rs01OmniV24WideCfg());b=class_to_dict(Rs01OmniV25Cfg())
        self.assertEqual(a['rewards']['scales'],b['rewards']['scales'])
        for key in a:
            if key!='rewards':self.assertEqual(a[key],b[key],key)

    def test_clearance_only_increases_for_straight_walking(self):
        e=self.robot();e.commands=torch.tensor([[.2,0,0],[0,.2,0],[0,0,0]])
        e._wide_phase_ready=True;e.wide_gait_phase=torch.full((3,),.86)
        e.diagonal_a_contact_mask=torch.tensor([True,False,False,True])
        e.diagonal_b_contact_mask=~e.diagonal_a_contact_mask
        e.feet_pos=torch.zeros((3,4,3))
        e.feet_pos[:,:,2]=e.cfg.rewards.foot_collision_radius_m+.025
        error=e._phase_swing_clearance_error()
        self.assertGreater(float(error[0]),.02)
        torch.testing.assert_close(error[1:],torch.zeros(2),atol=1e-6,rtol=0.)


if __name__=='__main__':unittest.main()
