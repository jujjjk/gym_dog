import unittest
from types import SimpleNamespace
from unittest.mock import patch
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_config import Rs01AMPSeededCfg, Rs01AMPSeededSlowCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_style_first import (
    support_failure, Rs01AMPStyleFirstCfg, Rs01AMPStyleFirstSlowCfg)
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_env import Rs01AMPCoreRobot
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_env import Rs01AMPSeededRobot


class StyleFirstTest(unittest.TestCase):
    def test_evaluation_never_reads_or_injects_reference_state(self):
        env = object.__new__(Rs01AMPSeededRobot)
        env.reference_reset_ready = True
        env.cfg = SimpleNamespace(env=SimpleNamespace(test=True))
        # No reference tensors exist: eval reset must return before touching them.
        with patch.object(Rs01AMPCoreRobot,'reset_idx') as base:
            env.reset_idx(torch.tensor([0]))
            base.assert_called_once()

    def test_three_foot_handoff_is_not_illegal(self):
        contact = torch.tensor([[1,1,1,1], [1,0,1,1], [1,0,0,1],
                                [1,0,1,0], [1,0,0,0], [0,0,0,0]], dtype=torch.bool)
        a = torch.tensor([1,0,0,1], dtype=torch.bool)
        b = ~a
        self.assertEqual(support_failure(contact,a,b).tolist(),[0.,0.,0.,1.,1.,1.])

    def test_only_reward_design_changes(self):
        for old,new in ((Rs01AMPSeededCfg,Rs01AMPStyleFirstCfg),
                        (Rs01AMPSeededSlowCfg,Rs01AMPStyleFirstSlowCfg)):
            a,b = class_to_dict(old()),class_to_dict(new())
            for key in a:
                if key != 'rewards':self.assertEqual(a[key],b[key],key)
            self.assertEqual(a['rewards']['gait_period_s'],b['rewards']['gait_period_s'])
            self.assertNotIn('alive',b['rewards']['scales'])
            self.assertNotIn('prolonged_all_feet_contact',b['rewards']['scales'])


if __name__ == '__main__':unittest.main()
