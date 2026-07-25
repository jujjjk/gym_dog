import unittest

import isaacgym  # noqa: F401 - must precede torch imports

from legged_gym.envs.dog.dog_stage2_actuator_config import (
    DogRs01Stage2ActuatorACfg,
    DogRs01Stage2ActuatorBCfg,
)
from legged_gym.envs.dog.telemetry_schema import build_headers
from legged_gym.utils.helpers import class_to_dict


class Stage2ConfigurationTests(unittest.TestCase):
    def test_a_preserves_pd_and_targets_sagittal_dynamics(self):
        cfg = DogRs01Stage2ActuatorACfg()
        self.assertEqual(
            cfg.control.stiffness,
            {"hip": 60.0, "thigh": 70.0, "calf": 70.0},
        )
        self.assertEqual(cfg.control.hip_action_scale, 0.16)
        self.assertEqual(cfg.control.thigh_action_scale, 0.21)
        self.assertEqual(cfg.control.calf_action_scale, 0.20)
        self.assertEqual(
            cfg.control.final_target_rate_limits_initial,
            {"hip": 2.0, "thigh": 2.5, "calf": 3.05},
        )
        self.assertFalse(cfg.control.preserve_thermal_state_on_reset)
        self.assertTrue(cfg.control.preserve_thermal_state_in_test)
        self.assertEqual(
            cfg.control.thermal_reset_ratio_range, [0.75, 0.95]
        )

    def test_b_is_only_a_modest_pd_ablation(self):
        cfg = DogRs01Stage2ActuatorBCfg()
        self.assertEqual(
            cfg.control.stiffness,
            {"hip": 60.0, "thigh": 68.0, "calf": 68.0},
        )
        self.assertEqual(
            cfg.control.damping,
            {"hip": 1.2, "thigh": 1.5, "calf": 1.5},
        )
        self.assertEqual(cfg.control.calf_action_scale, 0.20)
        self.assertAlmostEqual(
            cfg.control.stiffness["calf"] * 0.10, 6.8
        )

    def test_stage2_reward_set_is_small_and_orthogonal(self):
        scales = class_to_dict(DogRs01Stage2ActuatorACfg.rewards.scales)
        active = {name for name, value in scales.items() if float(value)}
        self.assertEqual(len(active), 21)
        self.assertIn("torque_clip", active)
        self.assertIn("raw_torque_rate", active)
        self.assertIn("motor_thermal_overload", active)
        self.assertNotIn("motor_continuous_usage", active)
        self.assertNotIn("peak_torque", active)
        self.assertIn("normalized_command_tracking", active)

    def test_stage2_csv_duration_and_reward_units(self):
        headers = build_headers(
            ["FL_calf_joint"], reward_names=["torque_clip"]
        )
        self.assertIn(
            "motor_over_12_duration_s_FL_calf_joint", headers
        )
        self.assertIn("foot_contact_duration_s_FL", headers)
        self.assertIn("touchdown_event_FL", headers)
        self.assertIn(
            "reward_scaled_torque_clip_per_step", headers
        )


if __name__ == "__main__":
    unittest.main()
