import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import json
import numpy as np
from reference_gait import trajectory, Kinematics, make_clip, ROOT, MOVES
from read_reference import ReferenceDataset


class ReferenceTest(unittest.TestCase):
    def test_straight_anchor_offset_preserves_phase_and_world_support(self):
        k=Kinematics(ROOT/'artifacts/rs01_v20_sim2sim/scene.xml',ROOT/'artifacts/rs01_v22_sim2sim/B23500.json')
        args=SimpleNamespace(seconds=1,fps=50,frequency=1.03,duty=.62,height=.29,
            half_width=.174,lift=31,directional_lift=30,rear_lift=30)
        original=make_clip(k,(.15,0,0),args)
        args.front_x_offset=.0064;args.rear_x_offset=-.0118
        changed=make_clip(k,(.15,0,0),args)
        np.testing.assert_array_equal(original['desired_contact'],changed['desired_contact'])
        delta=changed['foot_pos_world_m']-original['foot_pos_world_m']
        np.testing.assert_allclose(delta[:,:,0],np.tile([.0064,.0064,-.0118,-.0118],(50,1)),atol=1e-12)
        np.testing.assert_allclose(delta[:,:,1:],0,atol=1e-12)

    def test_diagonal_no_reference_flight(self):
        for t in np.arange(0,1,.001):
            _,_,_,contact,_=trajectory(t,(.2,.1,.3),2,.65,.29,.18,.06)
            self.assertEqual(contact[0],contact[3]);self.assertEqual(contact[1],contact[2])
            self.assertGreaterEqual(contact.sum(),2)

    def test_world_stance_fixed_and_swing_continuous(self):
        cfg=((.2,.1,.3),2,.65,.29,.18,.06)
        for t in np.arange(.01,.3,.01):
            a=trajectory(t,*cfg)[2][0];b=trajectory(t+.001,*cfg)[2][0]
            np.testing.assert_allclose(a,b,atol=1e-12)
        for t in [.325,.5]:
            a=trajectory(t-1e-6,*cfg)[2][0];b=trajectory(t+1e-6,*cfg)[2][0]
            self.assertLess(np.linalg.norm(a-b),1e-7)

    def test_all_ik_and_features(self):
        k=Kinematics(ROOT/'artifacts/rs01_v20_sim2sim/scene.xml',ROOT/'artifacts/rs01_v22_sim2sim/B23500.json')
        args=SimpleNamespace(seconds=1,fps=50,frequency=2,duty=.65,height=.29,half_width=.18,lift=60,directional_lift=50)
        for cmd in MOVES.values():
            a=make_clip(k,cmd,args)
            self.assertEqual(a['amp_features'].shape,(50,46))
            self.assertLess(a['ik_error_m'].max(),1e-5)
            self.assertFalse(a['transition_valid'][-1])
            self.assertTrue((a['joint_pos_rad']>=k.limits[:,0]).all())
            self.assertTrue((a['joint_pos_rad']<=k.limits[:,1]).all())
            np.testing.assert_allclose(a['root_pos_world_m'][:,2],.29)

    def test_loader_never_crosses_clips(self):
        with tempfile.TemporaryDirectory() as p:
            p=Path(p);(p/'manifest.json').write_text(json.dumps(dict(amp_training_approved=False,clips=[dict(clip='a'),dict(clip='b')])))
            for i,name in enumerate(('a','b')):
                np.savez(p/(name+'.npz'),amp_features=np.full((3,46),i),command=np.zeros((3,3)),transition_valid=[True,True,False])
            with self.assertRaises(ValueError):ReferenceDataset(p)
            pairs,_,ids=ReferenceDataset(p,True).sample(100)
            np.testing.assert_array_equal(pairs[:,0],ids);np.testing.assert_array_equal(pairs[:,46],ids)


if __name__=='__main__':unittest.main()
