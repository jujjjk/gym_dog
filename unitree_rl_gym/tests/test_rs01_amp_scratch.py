import unittest
import isaacgym
import torch
import numpy as np
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import Rs01AMPCfg,REFERENCE
from legged_gym.envs.rs01_omni_v2.rs01_amp_scratch_config import *
from legged_gym.envs.rs01_omni_v2.rs01_amp_scratch_env import legal_diagonal_or_handoff
from legged_gym.algorithms.rs01_amp import StyleLearner


class ScratchTest(unittest.TestCase):
    def test_real_plant_unchanged(self):
        a=class_to_dict(Rs01AMPCfg());b=class_to_dict(Rs01AMPScratchCfg())
        for key in a:
            if key!='rewards':self.assertEqual(a[key],b[key],key)
        self.assertNotIn('phase_two_contact_quality',b['rewards']['scales'])
        self.assertNotIn('phase_swing_clearance',b['rewards']['scales'])
        for cls in [Rs01AMPScratchPPO,Rs01AMPScratchSlowPPO]:
            r=cls().runner;self.assertFalse(r.resume);self.assertEqual(r.load_run,-1)
            self.assertFalse(r.amp_allow_reference_warmstart)

    def test_support_not_clock_locked(self):
        c=torch.tensor([[1,0,0,1],[0,1,1,0],[1,1,1,1],[1,1,0,0],[0,0,0,0],[1,1,1,0]],dtype=torch.bool)
        self.assertEqual(legal_diagonal_or_handoff(c,c[0],c[1]).tolist(),[True,True,True,False,False,False])

    def test_timing_and_velocity_scaling(self):
        a=StyleLearner(REFERENCE,allow_kinematic=True);b=StyleLearner(SLOW_REFERENCE,allow_kinematic=True)
        torch.testing.assert_close(b.commands,a.commands*.75)
        self.assertAlmostEqual(b.reference_settings['frequency'],1.5)
        self.assertAlmostEqual(b.reference_settings['duty'],a.reference_settings['duty'])
        # B frame 4k and A frame 3k describe exactly the same reference pose.
        from pathlib import Path
        for name in ['forward','left','combined']:
            with np.load(Path(REFERENCE)/(name+'.npz')) as x,np.load(Path(SLOW_REFERENCE)/(name+'.npz')) as y:
                np.testing.assert_allclose(y['qpos'][::4],x['qpos'][0:375:3],atol=2e-5)
                np.testing.assert_allclose(y['base_lin_vel_body_m_s'][:,:2],y['command'][:,:2],atol=.006)
                np.testing.assert_allclose(y['base_ang_vel_body_rad_s'][:,2],y['command'][:,2],atol=.006)
                self.assertTrue(np.isfinite(y['amp_features']).all())


if __name__=='__main__':unittest.main()
