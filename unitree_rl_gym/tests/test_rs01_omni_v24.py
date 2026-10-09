import isaacgym
import unittest
from types import SimpleNamespace as NS
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v24_env import Rs01OmniV24Robot, support_region_cost
from legged_gym.envs.rs01_omni_v2.rs01_omni_v24_config import Rs01OmniV24Cfg, Rs01OmniV24WideCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v23_config import Rs01OmniV23Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_env import Rs01OmniV21Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_env import huber_positive
from legged_gym.utils.helpers import class_to_dict


class Tests(unittest.TestCase):
    def cost(self,y,contact=None,touchdown=None):
        y=torch.tensor([y],dtype=torch.float)
        c=torch.ones_like(y,dtype=torch.bool) if contact is None else torch.tensor([contact])
        td=torch.zeros_like(c) if touchdown is None else torch.tensor([touchdown])
        return float(support_region_cost(y,c,td,.10,.22,.03,1.))

    def test_safe_region_no_mirror(self):
        self.assertEqual(self.cost([.11,.21,.17,.13]),0.)

    def test_pair_width_cannot_hide_one_inward_foot(self):
        self.assertGreater(self.cost([.06,.16,.15,.15]),0.)
        self.assertGreater(self.cost([-.01,.23,.15,.15]),self.cost([.06,.16,.15,.15]))

    def test_swing_unconstrained_and_landing_accounted(self):
        y=[.04,.15,.15,.15]
        self.assertEqual(self.cost(y,[False,True,True,True]),0.)
        self.assertGreater(self.cost(y,touchdown=[True,False,False,False]),self.cost(y))
        self.assertEqual(self.cost(y,[False]*4),0.)  # flight penalty remains in parent

    def test_outward_excess_not_incentivized(self):
        self.assertGreater(self.cost([.30]*4),0.)

    def test_legacy_width_math_unchanged(self):
        env=NS(cfg=Rs01OmniV23Cfg())
        widths=torch.tensor([[.16,.19],[.21,.25]])
        straight=torch.tensor([0.,1.])
        expected=huber_positive((torch.tensor([.18,.22])[:,None]-widths)/.04).mean(1)
        torch.testing.assert_close(Rs01OmniV21Robot._width_geometry_cost(env,widths,straight),expected)

    def test_execution_and_reward_stack_unchanged(self):
        old=class_to_dict(Rs01OmniV23Cfg());new=class_to_dict(Rs01OmniV24Cfg())
        for key in old:
            if key!='rewards':self.assertEqual(old[key],new[key],key)
        self.assertEqual(old['rewards']['scales'],new['rewards']['scales'])
        self.assertFalse(new['domain_rand']['push_robots'])
        a,b=[class_to_dict(x()) for x in (Rs01OmniV24Cfg,Rs01OmniV24WideCfg)]
        self.assertEqual({k for k in a['rewards'] if a['rewards'][k]!=b['rewards'][k]}, {'support_inner_m'})

    def test_force_duration_impulse_and_no_state_rewrite(self):
        env=object.__new__(Rs01OmniV24Robot)
        env.device='cpu';env.cfg=NS(sim=NS(dt=.0025));env.sim=None;env.v24_trunk_index=0
        env.v24_push_vector=torch.zeros(2,3);env.v24_push_ticks=torch.zeros(2,dtype=torch.long)
        env.v24_push_impulse=torch.zeros(2,3);env.v24_force_tensor=torch.zeros(2,1,3)
        env.root_states=torch.randn(2,13);state=env.root_states.clone()
        env.gym=NS(apply_rigid_body_force_tensors=lambda *a:True)
        env.queue_push([0],[0,10,0],.2)
        with self.assertRaises(ValueError):env.queue_push([0],[0,10,0],.2)
        for _ in range(80):env._apply_push_substep()
        torch.testing.assert_close(env.v24_push_impulse,torch.tensor([[0.,2.,0.],[0.,0.,0.]]))
        env._apply_push_substep();self.assertEqual(float(env.v24_force_tensor.abs().sum()),0.)
        torch.testing.assert_close(env.root_states,state)


if __name__=='__main__':unittest.main()
