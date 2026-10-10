import unittest
from types import SimpleNamespace
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_v22_guard_robust import pulse_envelope, Rs01V22GuardRobustCfg, Rs01V22GuardRobustRobot
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_guard import Rs01V22PhaseGuard5Cfg


class RobustTest(unittest.TestCase):
    def test_pulse_finite_duration_and_impulse(self):
        dt=.0025; duration=torch.tensor([.1,.2])
        t=(torch.arange(100)*dt+.5*dt)[:,None]
        f=pulse_envelope(t,duration[None,:])
        torch.testing.assert_close(f.sum(0)*dt,.5*duration)
        self.assertTrue(bool((f[t[:,0]>=.2]==0.).all()))
        self.assertTrue(bool((pulse_envelope(torch.tensor([-1.,0.,1.]),torch.ones(3))==0.).all()))

    def test_no_reward_actor_actuator_cadence_changes(self):
        a,b=class_to_dict(Rs01V22GuardRobustCfg()),class_to_dict(Rs01V22PhaseGuard5Cfg())
        a.pop('robust_push')
        for c in (a,b):
            c['domain_rand'].pop('randomize_base_mass');c['domain_rand'].pop('added_mass_range')
        self.assertEqual(a,b)
        self.assertFalse(a['domain_rand']['push_robots'])

    def test_reset_clears_only_selected_external_wrench(self):
        env=SimpleNamespace(cfg=Rs01V22GuardRobustCfg(),
            push_elapsed=torch.zeros(3),push_duration=torch.ones(3),push_wait=torch.zeros(3),
            push_vector=torch.ones(3,3),push_force_tensor=torch.ones(3,2,3),
            push_torque_tensor=torch.ones(3,2,3),
            _uniform_push=lambda n,bounds:torch.full((n,),bounds[0]))
        Rs01V22GuardRobustRobot._clear_push(env,torch.tensor([1]))
        self.assertEqual(float(env.push_force_tensor[1].sum()),0.)
        self.assertEqual(float(env.push_torque_tensor[1].sum()),0.)
        self.assertEqual(float(env.push_force_tensor[0].sum()),6.)
        self.assertEqual(float(env.push_wait[1]),4.)


if __name__=='__main__': unittest.main()
