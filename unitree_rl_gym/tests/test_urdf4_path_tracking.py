"""Fixed-frame/path semantics; run without constructing a simulator."""
import math,unittest
import isaacgym
import torch
from legged_gym.envs.urdf4.path_tracking import PathGains,PlanarPathController,StudentDeadReckoning,FixedOdometryFrame,rotate_xy,wrap


class PathTrackingTests(unittest.TestCase):
    def setUp(self):
        self.zero=torch.zeros(1,2);self.yaw=torch.zeros(1);self.cmd=torch.tensor([[.28,0.,0.]])

    def controller(self,mode=2):
        c=PlanarPathController(1,.02,'cpu',mode=mode);c.reset(self.zero,self.yaw)
        c.update(self.cmd,self.zero,self.yaw,self.zero,self.yaw)
        return c

    def test_cross_track_correction_sign(self):
        c=self.controller()
        for _ in range(20):out=c.update(self.cmd,torch.tensor([[0.,.1]]),self.yaw,self.zero,self.yaw)
        self.assertLess(float(out[0,1]),0.)

    def test_heading_correction_sign_and_fixed_reference(self):
        c=self.controller(mode=1)
        for _ in range(60):out=c.update(self.cmd,self.zero,self.yaw-.1,self.zero,self.yaw)
        self.assertGreater(float(out[0,2]),0.)
        self.assertEqual(float(c.reference_yaw[0]),0.)

    def test_forward_speed_change_does_not_move_the_line(self):
        c=self.controller();p=torch.tensor([[1.,.1]])
        for _ in range(10):c.update(self.cmd,p,self.yaw,self.zero,self.yaw)
        out=c.update(torch.tensor([[.18,0.,0.]]),p,self.yaw,self.zero,self.yaw)
        self.assertAlmostEqual(float(c.reference_xy[0,1]),0.)
        self.assertLess(float(out[0,1]),0.)

    def test_stop_is_not_in_place_marching_or_corrective_yaw(self):
        c=self.controller()
        for _ in range(20):c.update(self.cmd,torch.tensor([[0.,.2]]),self.yaw-.2,self.zero,self.yaw)
        out=c.update(torch.zeros(1,3),self.zero,self.yaw,self.zero,self.yaw)
        self.assertTrue(torch.equal(out,torch.zeros(1,3)))

    def test_integral_does_not_change_qualified_forward_and_clears_at_stop(self):
        c=PlanarPathController(1,.02,'cpu',gains=PathGains(nonforward_heading_integral_gain=.25))
        c.reset(self.zero,self.yaw)
        for _ in range(50):c.update(self.cmd,self.zero,self.yaw-.05,self.zero,self.yaw)
        self.assertEqual(float(c.heading_integral[0]),0.)
        for _ in range(50):c.update(torch.tensor([[0.,.1,0.]]),self.zero,self.yaw-.05,self.zero,self.yaw)
        self.assertGreater(float(c.heading_integral[0]),0.)
        c.update(torch.zeros(1,3),self.zero,self.yaw,self.zero,self.yaw)
        self.assertEqual(float(c.heading_integral[0]),0.)

    def test_indexed_reset_keeps_other_robots_reference_and_integral(self):
        c=PlanarPathController(2,.02,'cpu');c.reset(torch.zeros(2,2),torch.zeros(2))
        c.reference_xy[1]=torch.tensor([4.,5.]);c.reference_yaw[1]=.7;c.heading_integral[:]=.03
        c.reset_ids(torch.tensor([0]),torch.tensor([[1.,2.]]),torch.tensor([-.4]))
        self.assertTrue(torch.equal(c.reference_xy,torch.tensor([[1.,2.],[4.,5.]])))
        self.assertAlmostEqual(float(c.reference_yaw[1]),.7,places=6)
        self.assertEqual(float(c.heading_integral[0]),0.)
        self.assertAlmostEqual(float(c.heading_integral[1]),.03,places=6)

    def test_amp_pairs_survive_servo_corrections_but_not_user_or_gait_changes(self):
        from unittest.mock import patch
        from legged_gym.envs.urdf4.amp_direction_env import Urdf4AMPDirectionRobot
        from legged_gym.envs.urdf4.amp_expand_env import Urdf4AMPExpandRobot
        env=object.__new__(Urdf4AMPDirectionRobot)
        env.reset_buf=torch.tensor([False,False,True,False,False])
        env.target_commands=torch.tensor([[.25,0.,0.]]).repeat(5,1)
        env.gait_enabled=torch.ones(5,dtype=torch.bool)
        old=env.target_commands.clone()
        def physics_step(instance,action):
            instance.target_commands[1,0]=-.18
            instance.amp_terminal_enabled=torch.tensor([True,True,True,True,False])
            corrected=old+torch.tensor([0.,.01,.02])
            return None,None,None,torch.tensor([False,False,False,True,False]),{'amp':{'command':corrected,'valid':torch.zeros(5,dtype=torch.bool)}}
        with patch.object(Urdf4AMPExpandRobot,'step',physics_step):
            a=env.step(torch.zeros(5,12))[4]['amp']
        self.assertTrue(torch.equal(a['valid'],torch.tensor([True,False,False,False,False])))
        self.assertTrue(torch.equal(a['style_command'],old))
        self.assertTrue(torch.equal(a['command'],old+torch.tensor([0.,.01,.02])))

    def test_off_mode_is_exact_pass_through(self):
        c=self.controller(mode=0);command=torch.tensor([[.2,.08,.3]])
        out=c.update(command,self.zero,self.yaw,self.zero,self.yaw)
        self.assertTrue(torch.equal(command,out))

    def test_rotation_invariance(self):
        a=.8;position=torch.tensor([[.3,.08]]);yaw=torch.tensor([-.07]);v=torch.tensor([[.28,.02]])
        first=self.controller();second=PlanarPathController(1,.02,'cpu');second.reset(self.zero,self.yaw+a)
        second.update(self.cmd,self.zero,self.yaw+a,self.zero,self.yaw)
        for _ in range(30):
            x=first.update(self.cmd,position,yaw,v,self.yaw)
            y=second.update(self.cmd,rotate_xy(self.yaw+a,position),yaw+a,rotate_xy(self.yaw+a,v),self.yaw)
        self.assertTrue(torch.allclose(x,y,atol=1.e-6))

    def test_angle_wrap(self):
        c=PlanarPathController(1,.02,'cpu',mode=1);yaw=torch.tensor([math.pi-.02]);c.reset(self.zero,yaw)
        c.update(self.cmd,self.zero,yaw,self.zero,self.yaw)
        for _ in range(20):out=c.update(self.cmd,self.zero,torch.tensor([-math.pi+.02]),self.zero,self.yaw)
        self.assertLess(float(out[0,2]),0.);self.assertLess(abs(float(c.heading_error[0])),.05)

    def test_student_odometry_public_only(self):
        class Encoder(torch.nn.Module):
            def forward(self,x):return torch.tensor([[.3,-.1,0.]]).expand(x.shape[0],3)
        e=StudentDeadReckoning(Encoder(),1,.02,'cpu');public=torch.zeros(1,1020)
        pos,yaw,vel=e.update(public,self.yaw)
        self.assertTrue(torch.equal(pos,torch.zeros(1,2)))
        pos,yaw,vel=e.update(public,self.yaw)
        self.assertTrue(torch.allclose(pos,torch.tensor([[.006,-.002]])))
        with self.assertRaises(ValueError):e.update(torch.zeros(1,1121),self.yaw)

    def test_external_world_frame_is_anchored_once(self):
        frame=FixedOdometryFrame(torch.tensor([[10.,20.]]),torch.tensor([math.pi/2]))
        p,y,v=frame.transform(torch.tensor([[10.,21.]]),torch.tensor([math.pi/2]),torch.tensor([[0.,.3]]))
        self.assertTrue(torch.allclose(p,torch.tensor([[1.,0.]]),atol=1.e-6))
        self.assertTrue(torch.allclose(v,torch.tensor([[.3,0.]]),atol=1.e-6))
        self.assertAlmostEqual(float(y[0]),0.)

    def test_nonfinite_odometry_is_rejected(self):
        c=self.controller()
        with self.assertRaises(ValueError):c.update(self.cmd,self.zero+self.zero/0,self.yaw,self.zero,self.yaw)

    def test_external_source_cannot_silently_use_simulator_truth(self):
        from legged_gym.envs.urdf4.path_runtime import PathTrackingSession
        with self.assertRaisesRegex(ValueError,'External odometry provider'):
            PathTrackingSession(None,None,pose_source='external')

    def test_off_reference_keeps_original_turn_semantics(self):
        c=self.controller(mode=0);command=torch.tensor([[.22,0.,-.3]])
        for _ in range(10):c.update(command,self.zero,self.yaw,self.zero,self.yaw)
        self.assertAlmostEqual(float(c.reference_yaw[0]),-.3*.02*10,places=6)

    def test_turn_reference_is_slewed_independently_of_correction(self):
        c=self.controller(mode=1);command=torch.tensor([[0.,0.,.4]])
        c.update(command,self.zero,self.yaw,self.zero,self.yaw)
        self.assertAlmostEqual(float(c.reference_yaw[0]),.8*.02*.02,places=6)

    def test_side_command_saturation_keeps_vector_direction(self):
        c=self.controller();command=torch.tensor([[.2,.08,0.]])
        c.update(command,self.zero,self.yaw,self.zero,self.yaw)
        for _ in range(100):out=c.update(command,torch.tensor([[0.,-.2]]),self.yaw,self.zero,self.yaw)
        self.assertAlmostEqual(float(out[0,1]),.1,places=6)
        self.assertLess(float(out[0,0]),.17)  # both components scale, not just y

    def test_student_runtime_has_no_xy_truth_leak_or_extra_history_tick(self):
        from types import SimpleNamespace
        from legged_gym.envs.urdf4.path_runtime import PathTrackingSession
        class Encoder(torch.nn.Module):
            def forward(self,x):return torch.tensor([[.25,.01,0.]]).expand(x.shape[0],3)
        class Env:
            def __init__(self):
                self.num_envs=1;self.dt=.02;self.device='cpu'
                self.root_states=torch.zeros(1,13);self.root_states[:,6]=1.
                self.obs_history=torch.arange(20).float()[None,:,None].expand(1,20,51).clone()
                self.obs_buf=torch.zeros(1,1121);self.gait_enabled=torch.ones(1,dtype=torch.bool);self.target_gait=self.gait_enabled.clone()
                self.commands_scale=torch.tensor([2.,2.,.25]);self.obs_scales=SimpleNamespace(ang_vel=.25)
            def _public_yaw(self):return torch.zeros(1)
            def _post_physics_step_callback(self):pass
            def set_command_override(self,c,gait_enabled=True):self.command=c.clone()
        first,second=Env(),Env();policy=SimpleNamespace(encoder=Encoder())
        a=PathTrackingSession(first,policy,pose_source='student');b=PathTrackingSession(second,policy,pose_source='student')
        second.root_states[:,:2]+=1000.;second.obs_buf[:,1020:]=123.
        history=first.obs_history.clone()
        command=torch.tensor([[.25,0.,0.]])
        for _ in range(20):
            x=a.prepare_command(command,first.obs_buf);y=b.prepare_command(command,second.obs_buf)
            self.assertTrue(torch.equal(x['command'],y['command']))
        self.assertTrue(torch.equal(history[:,1:],first.obs_history[:,1:]))
        with self.assertRaisesRegex(ValueError,'validated'):
            a.prepare_command(torch.tensor([[.28,0.,0.]]),first.obs_buf)
        a.close();b.close()


if __name__=='__main__':unittest.main()
