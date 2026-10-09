"""Reference-state seeding may only reuse the discriminator's own clips and stay inside the policy envelope."""
import json
from pathlib import Path
import unittest
import isaacgym
import numpy as np
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_config import Rs01AMPCoreCfg,Rs01AMPCoreSlowCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_config import (
    Rs01AMPSeededCfg,Rs01AMPSeededSlowCfg,Rs01AMPSeededPPO,Rs01AMPSeededSlowPPO)

DOF_ORDER = [leg+'_'+j+'_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
NEEDED = ['joint_pos_rad','joint_vel_rad_s','command','root_pos_world_m','leg_phase']


def envelope_ratio(cfg):
    """How much of the reset reference the action space cannot express from the default pose."""
    default = torch.tensor([cfg.init_state.default_joint_angles[n] for n in DOF_ORDER])
    scale = torch.tensor([cfg.control.action_scale_by_joint[j] for _ in range(4)
                          for j in ('hip','thigh','calf')])
    root = Path(cfg.amp.reset_reference)
    clips = json.loads((root/'manifest.json').read_text())['clips']
    delta = np.concatenate([np.load(root/(c['clip']+'.npz'))['joint_pos_rad'] for c in clips])
    ratio = torch.from_numpy(np.abs(delta-default.numpy())/scale.numpy())
    return float((ratio > 1).float().mean()), float(ratio.max())


class SeededTest(unittest.TestCase):
    def test_plant_and_style_source_unchanged(self):
        for base,seeded,ppo in ((Rs01AMPCoreCfg,Rs01AMPSeededCfg,Rs01AMPSeededPPO),
                                (Rs01AMPCoreSlowCfg,Rs01AMPSeededSlowCfg,Rs01AMPSeededSlowPPO)):
            a,b = class_to_dict(base()),class_to_dict(seeded())
            for key in a:
                if key != 'amp':
                    self.assertEqual(a[key],b[key],key)
            self.assertEqual(a['amp']['commands'],b['amp']['commands'],
                             'seeding must not retune the commands it is graded on')
            # Seeding reuses the same reference set the discriminator watches; it is not a new teacher.
            runner = class_to_dict(ppo().runner)
            self.assertEqual(b['amp']['reset_reference'],runner['amp_reference'])
            self.assertGreater(b['amp']['reference_reset_probability'],0.)
            self.assertLess(b['amp']['reference_reset_probability'],1.)

    def test_reference_frames_are_expressible_and_finite(self):
        for seeded in (Rs01AMPSeededCfg,Rs01AMPSeededSlowCfg):
            cfg = seeded()
            root = Path(cfg.amp.reset_reference)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['joint_order'],DOF_ORDER)
            frames = set()
            for clip in manifest['clips']:
                with np.load(root/(clip['clip']+'.npz')) as d:
                    for key in NEEDED:
                        self.assertIn(key,d.files)
                        self.assertTrue(np.isfinite(d[key]).all(),(clip['clip'],key))
                    frames.add(d['joint_pos_rad'].shape)
            self.assertEqual(len(frames),1,'stacked reset reference needs equal clip lengths')
            outside,peak = envelope_ratio(cfg)
            self.assertEqual(outside,0.,'reference pose outside expanded action envelope')
            self.assertLess(peak,1.)

    def test_runner_is_fresh_and_seeded_names_are_distinct(self):
        names = []
        for ppo in (Rs01AMPSeededPPO,Rs01AMPSeededSlowPPO):
            runner = class_to_dict(ppo().runner)
            self.assertFalse(runner['resume'])
            self.assertEqual(runner['load_run'],-1)
            self.assertFalse(runner['amp_allow_reference_warmstart'])
            self.assertTrue(runner['freeze_action_std'])
            names.append(runner['experiment_name'])
        self.assertEqual(names,['rs01_amp_seeded_original','rs01_amp_seeded_slow'])


if __name__ == '__main__':
    unittest.main()
