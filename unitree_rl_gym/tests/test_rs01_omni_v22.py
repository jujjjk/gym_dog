import isaacgym
import unittest
from types import SimpleNamespace
import torch
from legged_gym.envs.rs01_omni_v2.rs01_sensor_snapshot import SensorSnapshot, assemble61
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CleanCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_env import Rs01OmniV22Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_config import Rs01OmniV21Cfg
from legged_gym.utils.helpers import class_to_dict


class Tests(unittest.TestCase):
    def make(self):
        s=SensorSnapshot(2,'cpu',.0025,Rs01OmniV22CleanCfg.sensor)
        q=torch.zeros(2,12);quat=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1);g=torch.zeros(2,3)
        self.truth=(q,q.clone(),quat,g,q.clone())
        s.reset(torch.arange(2),0,*self.truth)
        return s

    def test_clean_fresh(self):
        s=self.make();self.truth[0][:]=.3;self.truth[3][:,2]=2
        s.acquire(1,*self.truth);s.read(1)
        torch.testing.assert_close(s.q,self.truth[0]);torch.testing.assert_close(s.gyro,self.truth[3])
        self.assertEqual(float(s.imu_age_s.max()),0.)

    def test_delays_are_causal_and_skew_derived(self):
        s=self.make();s.clean[:]=False;s.motor_delay[:]=2;s.imu_delay[:]=4
        s.motor_phase[:]=0;s.imu_phase[:]=0
        for t in range(1,9):
            self.truth[0][:]=t;s.acquire(t,*self.truth);s.read(t)
            self.assertTrue(bool((s.motor_stamp<=t-2).all()) if t>=2 else True)
        self.assertEqual(s.motor_stamp.tolist(),[6,6]);self.assertEqual(s.imu_stamp.tolist(),[4,4])
        torch.testing.assert_close(s.skew_s,s.imu_age_s-s.motor_age_s)

    def test_reset_flush(self):
        s=self.make();s.acquire(5,*self.truth);s.read(5)
        s.reset(torch.tensor([0]),6,*self.truth);s.read(6)
        self.assertEqual(s.motor_stamp.tolist(),[6,5])
        self.assertTrue(bool((s.ms[0]<0).all()))

    def test_mount_rotates_gyro_and_gravity_consistently(self):
        from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
        s=self.make();a=torch.full((2,),.03);zero=torch.zeros(2)
        s.mount=quat_from_euler_xyz(a,zero,zero)
        s.acquire(1,*self.truth);s.read(1)
        expected=quat_rotate_inverse(s.mount,torch.tensor([[0.,0.,-1.]]).repeat(2,1))
        torch.testing.assert_close(s.gravity,expected)
        torch.testing.assert_close(s.gravity.norm(dim=1),torch.ones(2))

    def test_assembly(self):
        s=self.make();s.read(0)
        v=torch.ones(2,3);target=v*.2;cmd=v*.1;phase=torch.zeros(2);heading=phase.clone()
        scale=SimpleNamespace(lin_vel=2.,ang_vel=.25,dof_pos=1.,dof_vel=.05)
        obs=assemble61(s,v,torch.ones(2),target,cmd,torch.zeros(1,12),scale,
                       torch.tensor([2.,2.,.25]),torch.zeros(2,12),phase,heading,torch.ones(2))
        self.assertEqual(obs.shape,(2,61));self.assertTrue(bool((obs[:,[52,55]]==0).all()))
        torch.testing.assert_close(obs[:,9:12],target*torch.tensor([2.,2.,.25]))
        self.assertTrue(bool((obs[:,53]==1.6).all()))

    def test_root_refresh_not_stale_policy_gyro(self):
        env=object.__new__(Rs01OmniV22Robot)
        env.root_states=torch.zeros(1,13);env.root_states[:,6]=1
        env.dof_pos=env.dof_vel=env.motor_electromagnetic_torques=torch.zeros(1,12)
        env.base_ang_vel=torch.full((1,3),99.);env.sim=None
        def refresh(sim):env.root_states[:,10:13]=torch.tensor([1.,2.,3.])
        env.gym=SimpleNamespace(refresh_actor_root_state_tensor=refresh,refresh_dof_state_tensor=lambda sim:None)
        gyro=env._fresh_sensor_truth()[3]
        torch.testing.assert_close(gyro,torch.tensor([[1.,2.,3.]]))

    def test_no_reward_or_plant_config_change(self):
        a,b=[class_to_dict(x()) for x in (Rs01OmniV21Cfg,Rs01OmniV22Cfg)]
        for key in ('rewards','control','rs01_actuator','env','commands','sim','normalization'):
            self.assertEqual(a[key],b[key])


if __name__=='__main__':unittest.main()
