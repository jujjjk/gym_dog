import isaacgym
import math
import unittest
from unittest.mock import patch
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_config import Rs01OmniV18Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_env import Rs01OmniV18Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_config import Rs01OmniV19Cfg,Rs01OmniV19SoftCfg,Rs01OmniV19CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_env import Rs01OmniV19Robot,placement_cost
from legged_gym.utils.helpers import class_to_dict


class PlacementTests(unittest.TestCase):
    def test_contract(self):
        old,new,soft=[class_to_dict(x()) for x in (Rs01OmniV18Cfg,Rs01OmniV19Cfg,Rs01OmniV19SoftCfg)]
        for k in ('rs01_actuator','control','sim','env','commands','normalization'):
            self.assertEqual(old[k],new[k])
        a=old['rewards']['scales'];b=new['rewards']['scales']
        self.assertEqual([k for k in b if a[k]!=b[k]],['action_saturation'])
        soft['asset']['name']=new['asset']['name'];soft['rewards']['placement_weight']=new['rewards']['placement_weight']
        self.assertEqual(soft,new)
        for n in ('step','_compute_torques','compute_observations','_reward_tracking_command_velocity'):
            self.assertIs(getattr(Rs01OmniV19Robot,n),getattr(Rs01OmniV18Robot,n))
        p=Rs01OmniV19CfgPPO()
        self.assertEqual(p.algorithm.schedule,'fixed');self.assertEqual(p.algorithm.learning_rate,1e-4)
        self.assertFalse(p.runner.load_optimizer)

    def test_deadband_and_omni_release(self):
        width=torch.full((3,2),.23);dx=torch.tensor([[.01,-.01],[.1,0.],[.1,0.]])
        cmd=torch.tensor([[0.,0.,0.],[.2,0.,0.],[0.,.2,0.]])
        c=placement_cost(width,dx,cmd,torch.ones(3),Rs01OmniV19Cfg.rewards)
        self.assertEqual(float(c[0]),0.);self.assertGreater(float(c[1]),0.);self.assertEqual(float(c[2]),0.)
        torch.testing.assert_close(c,placement_cost(width,-dx,cmd,torch.ones(3),Rs01OmniV19Cfg.rewards))

    def test_normal_alternation_filter(self):
        # 90mm instantaneous opposite-phase separation at .6s period
        # must not be treated as 90mm persistent placement bias.
        avg=0.;history=[];alpha=math.exp(-.02/.6)
        for i in range(600):
            avg=alpha*avg+(1-alpha)*.09*math.sin(2*math.pi*i*.02/.6)
            if i>300:history.append(abs(avg))
        self.assertLess(max(history),.015)

    def test_reset_command_and_no_old_width_multiplier(self):
        e=Rs01OmniV19Robot.__new__(Rs01OmniV19Robot);e.cfg=Rs01OmniV19Cfg();e.dt=.02
        e.feet_pos=torch.tensor([[[.05,.1,-.3],[-.05,-.1,-.3],[0.,.1,-.3],[0.,-.1,-.3]]])
        e.root_states=torch.zeros(1,13);e.base_quat=torch.tensor([[0.,0.,0.,1.]])
        e.commands=torch.zeros(1,4);e.foot_slot_by_leg=dict(zip(('FL','FR','RL','RR'),range(4)))
        for _ in range(100): torch.testing.assert_close(e._footprint_quality(),torch.ones(1))
        self.assertGreater(float(e.placement_dx[0,0]),.09)
        e.commands[0,1]=.2;e._footprint_quality()
        self.assertLess(float(e.placement_dx[0,0]),.004)
        with patch.object(Rs01OmniV18Robot,'_reset_dofs'):
            e._reset_dofs(torch.tensor([0]))
        self.assertEqual(float(e.placement_dx.abs().sum()),0.)
        self.assertEqual(float(e.placement_age.sum()),0.)


if __name__=='__main__':unittest.main()
