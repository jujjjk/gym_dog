import json, os, tempfile, unittest
from pathlib import Path
import isaacgym
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import Rs01AMPCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_config import Rs01AMPCoreCfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_config import Rs01AMPSeededCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v30_config import Rs01OmniV30Cfg
from legged_gym.envs.rs01_omni_v2.rs01_amp_inplant_config import (
    INPLANT_REFERENCE, MANIFEST, INPLANT_COMMANDS,
    Rs01AMPInPlantCfg, Rs01AMPInPlantPPO, Rs01AMPInPlantSeededCfg)
from legged_gym.algorithms.rs01_amp import StyleLearner


class AMPInPlantTest(unittest.TestCase):
    def test_only_feasible_clips_were_written(self):
        self.assertTrue(MANIFEST['feasibility_gate_passed'])
        self.assertTrue(MANIFEST['amp_training_approved'])
        for clip in MANIFEST['clips']:
            self.assertTrue(all(l >= 3 for l in clip['liftoffs_per_foot']), clip['clip'])
            self.assertGreaterEqual(clip['diagonal_alternations'], 4)
            self.assertGreaterEqual(min(clip['foot_height_p95_mm']), 5., clip['clip'])
            self.assertLess(clip['raw_peak_nm'], 17.)
            self.assertLessEqual(clip['rate_ratio_p95'], 1.15)

    def test_config_is_read_from_the_recording(self):
        cfg = class_to_dict(Rs01AMPInPlantCfg())
        self.assertEqual(cfg['amp']['commands'], INPLANT_COMMANDS)
        self.assertEqual([[float(v) for v in c['command']] for c in MANIFEST['clips']],
                         INPLANT_COMMANDS)
        self.assertTrue(cfg['amp']['command_dependent_clock'])
        self.assertEqual(cfg['rewards']['gait_period_s'], MANIFEST['settings']['period_s'])
        self.assertEqual(cfg['rewards']['gait_stance_ratio'], MANIFEST['settings']['duty'])
        self.assertFalse(Rs01AMPInPlantPPO.runner.amp_allow_kinematic)
        seeded = class_to_dict(Rs01AMPInPlantSeededCfg())['amp']
        self.assertEqual(seeded['reset_reference'], INPLANT_REFERENCE)
        # A nested amp class shadows the parent's, so the seeded branch's own fields have to
        # survive the merge -- the reference reset died on a missing probability once.
        self.assertEqual(seeded['reference_reset_probability'],
                         class_to_dict(Rs01AMPSeededCfg())['amp']['reference_reset_probability'])
        self.assertTrue(seeded['command_dependent_clock'])
        self.assertEqual(seeded['commands'], INPLANT_COMMANDS)
        # The kinematic experiments keep their single tempo clock.
        self.assertFalse(class_to_dict(Rs01AMPCfg())['amp']['command_dependent_clock'])

    def test_plant_limits_are_untouched(self):
        demonstrator, inplant = class_to_dict(Rs01OmniV30Cfg()), class_to_dict(Rs01AMPInPlantCfg())
        # The clips were executed through the demonstrator's own output range, so the in-plant
        # task trains in it too; only the from-scratch experiments widen it for exploration.
        for key in ('sim', 'noise', 'domain_rand', 'asset', 'control'):
            self.assertEqual(demonstrator[key], inplant[key], key)
            self.assertEqual(demonstrator[key], class_to_dict(Rs01AMPInPlantSeededCfg())[key], key)
        self.assertNotEqual(inplant['control']['action_scale_by_joint'],
                            class_to_dict(Rs01AMPCoreCfg())['control']['action_scale_by_joint'])

    def test_learner_carries_the_recorded_tempos(self):
        style = StyleLearner(INPLANT_REFERENCE)
        self.assertEqual([round(v, 4) for v in style.clip_cadence.tolist()],
                         [round(float(c['cadence_hz']), 4) for c in MANIFEST['clips']])
        self.assertEqual(style.commands.shape, (len(MANIFEST['clips']), 3))

    def test_reference_without_a_measured_tempo_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads((Path(INPLANT_REFERENCE)/'manifest.json').read_text())
            for clip in manifest['clips']:
                del clip['cadence_hz']
                os.symlink(os.path.join(INPLANT_REFERENCE, clip['clip']+'.npz'),
                           os.path.join(tmp, clip['clip']+'.npz'))
            Path(tmp, 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'no measured cadence'):
                StyleLearner(tmp)


if __name__ == '__main__':
    unittest.main()
