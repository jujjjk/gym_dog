import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_lift import (
    lift_credit, attitude_cost, Rs01V22PhaseLiftCfg, Rs01V22PhaseLiftPPO)
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_scratch import (
    Rs01V22PhaseScratchCfg, Rs01V22PhaseScratchPPO, update_exchange)


class LiftRepairTest(unittest.TestCase):
    def test_preserves_reward_weights_and_plant(self):
        a, b = class_to_dict(Rs01V22PhaseScratchCfg()), class_to_dict(Rs01V22PhaseLiftCfg())
        b.pop('phase_acquisition')
        self.assertTrue(b['phase_exchange'].pop('require_lift_event'))
        self.assertEqual(a, b)
        a, b = class_to_dict(Rs01V22PhaseScratchPPO()), class_to_dict(Rs01V22PhaseLiftPPO())
        for cfg in (a, b):
            cfg['runner'].pop('experiment_name')
            cfg['runner'].pop('action_std_value')
        self.assertEqual(a, b)

    def test_every_recovered_foot_has_credit(self):
        clocks = torch.tensor([[5.,5.,5.,5.], [0.,5.,5.,5.], [0.,0.,5.,5.], [0.,0.,0.,5.], [0.,0.,0.,0.]])
        gate, stale = lift_credit(clocks, 1.15, .5, .05)
        self.assertTrue(bool((gate[1:] > gate[:-1]).all()))
        self.assertAlmostEqual(float(gate[0]), .05)
        self.assertAlmostEqual(float(gate[-1]), 1.)
        torch.testing.assert_close(stale, torch.tensor([1.,.75,.5,.25,0.]))

    def test_pitch_cost_replaces_not_duplicates_roll(self):
        roll = torch.tensor([0., .5, 1.])
        cfg = Rs01V22PhaseLiftCfg.phase_acquisition
        cost = attitude_cost(roll, torch.tensor([0., 0., 0.]), cfg.pitch_deadband_rad, cfg.pitch_scale_rad)
        torch.testing.assert_close(cost, roll)
        bent = attitude_cost(torch.zeros(1), torch.tensor([.476]), cfg.pitch_deadband_rad, cfg.pitch_scale_rad)
        self.assertGreater(float(bent), 1.5)

    def test_hanging_foot_cannot_refresh_credit_forever(self):
        cfg = Rs01V22PhaseLiftCfg.phase_exchange
        clocks, air = torch.zeros(1, 4), torch.zeros(1, 4)
        for _ in range(100):
            clocks, air, _ = update_exchange(clocks, air, torch.zeros(1, 4, dtype=torch.bool),
                torch.ones(1, 4)*.02, torch.tensor([1./30]), torch.tensor([True]), .02, cfg)
        self.assertTrue(bool((clocks > 3.).all()))

    def test_periodic_lifts_keep_full_credit(self):
        cfg = Rs01V22PhaseLiftCfg.phase_exchange
        clocks, air = torch.zeros(1, 4), torch.zeros(1, 4)
        for i in range(150):
            phase = (torch.tensor([[i/30.]]) + torch.tensor([0.,.5,.5,0.])).remainder(1.)
            contact = phase < .72
            height = .025*torch.sin(torch.pi*((phase-.72)/.28).clamp(0,1))
            clocks, air, _ = update_exchange(clocks, air, contact, height,
                torch.tensor([1./30]), torch.tensor([True]), .02, cfg)
            gate, _ = lift_credit(clocks, 1.15, .5, .05)
            self.assertAlmostEqual(float(gate), 1.)


if __name__ == '__main__':
    unittest.main()
