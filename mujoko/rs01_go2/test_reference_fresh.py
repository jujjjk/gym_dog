import unittest
from types import SimpleNamespace
import numpy as np
from reference_fresh import ContractOnly, rate_audit, MOVES, ROOT, accepts_diagonal_load, phase_control_residual
from reference_gait import trajectory


class FreshReferenceTest(unittest.TestCase):
    def test_control_harmonics_periodic_and_mirrored(self):
        coefficients=np.arange(12)*.001
        phase=np.array([.2,.2,.2,.2])
        q=phase_control_residual(phase,coefficients).reshape(4,3)
        np.testing.assert_allclose(q[[0,2],0],-q[[1,3],0])
        np.testing.assert_allclose(q[[0,2],1:],q[[1,3],1:])
        np.testing.assert_allclose(phase_control_residual(phase+1,coefficients),q.ravel(),atol=1e-12)
        np.testing.assert_allclose(phase_control_residual([0,.5,.5,0],coefficients),0,atol=1e-12)

    def test_handoff_requires_both_feet_and_load(self):
        outgoing=np.array([True,False,False,True])
        self.assertFalse(accepts_diagonal_load([20,100,0,20],outgoing,12))
        self.assertFalse(accepts_diagonal_load([40,10,10,40],outgoing,12))
        self.assertTrue(accepts_diagonal_load([10,45,45,10],outgoing,12))
        self.assertFalse(accepts_diagonal_load([45,10,10,45],outgoing,12))

    def test_no_learned_policy(self):
        c=ContractOnly(ROOT/'artifacts/rs01_v22_sim2sim/B23500.json')
        self.assertTrue(c.get_modelmeta().custom_metadata_map)
        with self.assertRaises(RuntimeError):c.run([], {})

    def test_diagonal_and_no_crossing(self):
        for cmd in MOVES.values():
            for t in np.arange(0,2,.01):
                b,R,f,c,_=trajectory(t,cmd,1.25,.65,.29,.175,.030)
                self.assertEqual(c[0],c[3]);self.assertEqual(c[1],c[2])
                self.assertGreaterEqual(c.sum(),2)
                self.assertGreater(((f-b)@R)[:,1].dot([1,-1,1,-1])/4,.14)
                self.assertGreater(np.min(((f-b)@R)[:,1]*[1,-1,1,-1]),.14)

    def test_rate_uses_forward_difference(self):
        cfg={'control':{'target_rate_limit_rad_s':[2.]*12,
                        'target_acceleration_limit_rad_s2':[100.]*12}}
        q=np.tile(np.array([0,.1,0])[:,None],(1,12))
        v,a=rate_audit({'joint_pos_rad':q},cfg,.02)
        self.assertAlmostEqual(v,2.5);self.assertAlmostEqual(a,5.)

    def test_speed_profiles_keep_command_and_diagonal_support(self):
        for speed,freq,duty in [(.15,1.25,.65),(.20,1.40,.63),(.25,1.55,.61),(.25,1.25,.60)]:
            for t in np.arange(0,2,.02):
                b,R,f,c,_=trajectory(t,(speed,0,0),freq,duty,.29,.175,.030)
                self.assertAlmostEqual(b[0],speed*t)
                self.assertEqual(c[0],c[3]);self.assertEqual(c[1],c[2])
                self.assertGreaterEqual(c.sum(),2)


if __name__=='__main__':unittest.main()
