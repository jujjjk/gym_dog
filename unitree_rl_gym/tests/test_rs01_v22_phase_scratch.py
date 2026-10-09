import unittest
from unittest.mock import patch
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_scratch import (
    Rs01V22PhaseScratchCfg, Rs01V22PhaseScratchPPO,
    Rs01V22PhaseScratchRobot, Rs01OmniV22Robot, update_exchange)


class ExchangeTest(unittest.TestCase):
    def test_exact_original_contract(self):
        a, b = class_to_dict(Rs01OmniV22Cfg()), class_to_dict(Rs01V22PhaseScratchCfg())
        b.pop('phase_exchange')
        self.assertEqual(a, b)  # ALL old settings, including every reward weight.
        a, b = class_to_dict(Rs01OmniV22CfgPPO()), class_to_dict(Rs01V22PhaseScratchPPO())
        for cfg in (a, b):
            for k in ('experiment_name', 'resume', 'load_run', 'checkpoint', 'resume_path'):
                cfg['runner'].pop(k, None)
        self.assertEqual(a, b)

    def simulate(self, kind, steps=180):
        cfg = Rs01V22PhaseScratchCfg.phase_exchange
        clocks, air = torch.zeros(1, 4), torch.zeros(1, 4)
        phase, penalties = torch.zeros(1), []
        for i in range(steps):
            # Original cadence range, with a mid-test frequency transition.
            delta = torch.tensor([.02 * (1. / .6 if i < 80 else 2.5)])
            phase = (phase + delta).remainder(1.)
            local = (phase[:, None] + torch.tensor([0., .5, .5, 0.])).remainder(1.)
            contact = local < .72
            height = .025 * torch.sin(torch.pi * ((local - .72) / .28).clamp(0, 1))
            active = torch.tensor([kind != 'stand'])
            if kind in ('stuck', 'chatter', 'low'):
                contact[:, 0] = kind != 'chatter' or i % 2 == 0
                height[:, 0] = .020 if kind == 'chatter' else .003
                if kind == 'low':
                    contact[:, 0] = False
            clocks, air, penalty = update_exchange(clocks, air, contact, height, delta, active, .02, cfg)
            penalties.append(float(penalty))
        return clocks, air, penalties

    def test_legal_diagonal_and_speed_transition(self):
        self.assertEqual(max(self.simulate('normal')[2]), 0.)

    def test_each_foot_must_really_lift(self):
        for kind in ('stuck', 'chatter', 'low'):
            self.assertEqual(self.simulate(kind)[2][-1], 1., kind)

    def test_stand_clears_clocks(self):
        cfg = Rs01V22PhaseScratchCfg.phase_exchange
        clocks, air, penalty = update_exchange(
            torch.ones(1, 4) * 5, torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4), torch.tensor([.05]), torch.tensor([False]), .02, cfg)
        self.assertEqual(float(clocks.sum() + air.sum() + penalty.sum()), 0.)

    def test_no_double_penalty_and_selective_reset(self):
        env = object.__new__(Rs01V22PhaseScratchRobot)
        env._exchange_ready = True
        env.exchange_penalty = torch.tensor([1., .2])
        env.no_lift_cycles = torch.ones(2, 4)
        env.exchange_lift_time_s = torch.ones(2, 4)
        env._walking_command_gate = lambda: torch.ones(2)
        with patch.object(Rs01OmniV22Robot, '_reward_prolonged_all_feet_contact', return_value=torch.tensor([1., .5])):
            torch.testing.assert_close(env._reward_prolonged_all_feet_contact(), torch.tensor([1., .5]))
        with patch.object(Rs01OmniV22Robot, 'reset_idx'):
            env.reset_idx(torch.tensor([0]))
        self.assertEqual(float(env.no_lift_cycles[0].sum()), 0.)
        self.assertEqual(float(env.no_lift_cycles[1].sum()), 4.)


if __name__ == '__main__':
    unittest.main()
