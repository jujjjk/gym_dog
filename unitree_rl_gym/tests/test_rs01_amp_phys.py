"""The AMP reference must be something the identified RS01 plant can actually execute."""
import json
from pathlib import Path
import unittest
import isaacgym
import numpy as np
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_config import Rs01AMPCoreCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import REFERENCE
from legged_gym.envs.rs01_omni_v2.rs01_amp_phys_config import (
    PHYS_REFERENCE, PHYS_COMMANDS, REFERENCE_FREQUENCY,
    Rs01AMPPhysCfg, Rs01AMPPhysSeededCfg, Rs01AMPPhysPPO)
from legged_gym.algorithms.rs01_amp import StyleLearner

RATE_LIMIT = np.array([2.0, 2.6, 3.2]*4)   # hip/thigh/calf target rate limits, rad/s


def load(root, name):
    manifest = json.loads((Path(root)/'manifest.json').read_text())
    clip = next(c for c in manifest['clips'] if c['clip'] == name)
    return manifest, clip, np.load(Path(root)/(name+'.npz'), allow_pickle=False)


class PhysReferenceTest(unittest.TestCase):
    def test_recorded_reference_is_plant_feasible(self):
        manifest, _, _ = load(PHYS_REFERENCE, 'march')
        self.assertEqual(manifest['kind'], 'recorded_nonideal_plant')
        self.assertEqual([c['command'] for c in manifest['clips']], PHYS_COMMANDS)
        for clip in manifest['clips']:
            self.assertTrue(clip['physically_executed'], clip['clip'])
            self.assertLessEqual(clip['rate_ratio_max'], 1.15, clip['clip'])
            self.assertGreaterEqual(min(clip['foot_height_p95_mm']), 12., clip['clip'])
            self.assertLess(clip['raw_peak_nm'], 17., clip['clip'])
            self.assertGreater(clip['template_tracking_rmse_rad'], 0., clip['clip'])

    def test_rejected_kinematic_design_is_the_control(self):
        # Guards the root cause: the 60 mm / 2 Hz designed style needs ~2.4x the target rate limit.
        _, clip, data = load(REFERENCE, 'forward')
        self.assertGreater(clip['reference_rate_to_target_limit_max'], 1.5)
        rate = np.abs(np.diff(data['joint_pos_rad'], axis=0)/0.02).max(0)/RATE_LIMIT
        self.assertGreater(float(rate.max()), 1.5)

    def test_commands_never_exceed_demonstrated_speed(self):
        _, _, template = load(REFERENCE, 'forward')
        for name in ['forward','backward','left','right','turn_left','turn_right','combined']:
            _, clip, _ = load(PHYS_REFERENCE, name)
            requested = np.array(clip['requested_command']); got = np.array(clip['command'])
            self.assertLessEqual(np.abs(got).max(), 1.05*np.abs(requested).max(), name)
            self.assertGreater(np.abs(got).max(), 0., name)
            # Only the commanded axis may be demanded: the teacher's yaw drift is not a target.
            self.assertTrue(np.all(got[np.abs(requested) == 0] == 0), (name, got, requested))

    def test_style_learner_and_env_agree_on_new_reference(self):
        learner = StyleLearner(PHYS_REFERENCE, allow_kinematic=True,
                               settings=class_to_dict(Rs01AMPPhysPPO().runner))
        self.assertEqual(learner.expert.shape[0], len(PHYS_COMMANDS))
        self.assertEqual(learner.expert.shape[2], 92)
        torch.testing.assert_close(learner.commands, torch.tensor(PHYS_COMMANDS))
        with np.load(PHYS_REFERENCE+'/march.npz') as d:
            self.assertEqual(d['amp_features'].shape[1], 46)
            self.assertTrue(np.isfinite(d['amp_features']).all())

    def test_only_style_inputs_changed(self):
        core, phys = class_to_dict(Rs01AMPCoreCfg()), class_to_dict(Rs01AMPPhysCfg())
        for key in core:
            if key not in ('amp', 'rewards'):
                self.assertEqual(core[key], phys[key], key)
        shared = set(core['rewards']) & set(phys['rewards']) - {'scales'}
        self.assertEqual({k for k in shared if core['rewards'][k] != phys['rewards'][k]},
                         {'gait_period_s'}, 'only the gait clock may follow the new reference')
        self.assertEqual(core['rewards']['scales'], phys['rewards']['scales'])
        self.assertEqual(phys['control'], core['control'])
        self.assertNotEqual(core['amp']['commands'], phys['amp']['commands'])
        self.assertNotIn('reset_reference', phys['amp'])
        seeded = class_to_dict(Rs01AMPPhysSeededCfg())
        self.assertEqual(seeded['amp']['reset_reference'], PHYS_REFERENCE)
        self.assertEqual(seeded['amp']['commands'], PHYS_COMMANDS)

    def test_reward_clock_matches_reference(self):
        # Rs01AMPOnPolicyRunner refuses to start unless frequency*gait_period_s == 1 and duty matches.
        for cfg in (Rs01AMPPhysCfg, Rs01AMPPhysSeededCfg):
            rewards, amp = cfg().rewards, cfg().amp
            self.assertAlmostEqual(REFERENCE_FREQUENCY*rewards.gait_period_s, 1., places=9)
            self.assertAlmostEqual(amp.commands[1][0], .0532)


if __name__ == '__main__':
    unittest.main()
