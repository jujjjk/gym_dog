import unittest

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs.base.actuator_torque import (
    apply_coulomb_friction,
    limit_electromagnetic_torque,
)
from legged_gym.envs.base.contact_state import (
    update_consecutive_true_count,
    update_contact_mask,
)
from legged_gym.envs.base.terminal_snapshot import TerminalSnapshot
from legged_gym.envs.dog.telemetry_schema import build_headers
from legged_gym.utils.checkpoint_adapter import (
    adapt_observation_input_state,
)
from legged_gym.algorithms.conservative_ppo import executed_action_delta


class ContactStateTests(unittest.TestCase):
    def test_two_newton_threshold_and_hysteresis(self):
        previous = torch.tensor([[False, True, True, False]])
        force = torch.tensor([[1.99, 1.60, 1.49, 2.00]])
        contact = update_contact_mask(force, previous, 2.0, 1.5)
        self.assertEqual(
            contact.tolist(), [[False, True, False, True]]
        )

    def test_equal_threshold_has_no_hidden_hysteresis(self):
        previous = torch.tensor([True, False])
        force = torch.tensor([1.99, 2.0])
        contact = update_contact_mask(force, previous, 2.0, 2.0)
        self.assertEqual(contact.tolist(), [False, True])

    def test_illegal_contact_consecutive_frames(self):
        count = torch.zeros(2, dtype=torch.long)
        count = update_consecutive_true_count(
            torch.tensor([True, False]), count
        )
        count = update_consecutive_true_count(
            torch.tensor([True, True]), count
        )
        count = update_consecutive_true_count(
            torch.tensor([False, True]), count
        )
        self.assertEqual(count.tolist(), [0, 2])


class TorqueDomainTests(unittest.TestCase):
    def test_active_limit_and_non_alias(self):
        raw = torch.tensor([[20.0, -18.0, 5.0]])
        active = torch.tensor([[17.0, 12.0, 6.0]])
        motor = limit_electromagnetic_torque(raw, active)
        applied = apply_coulomb_friction(
            motor,
            torch.tensor([[1.0, -1.0, 0.0]]),
            torch.tensor([[0.2, 0.2, 0.2]]),
            0.05,
        )
        self.assertTrue(torch.all(torch.abs(motor) <= active))
        self.assertNotEqual(raw.data_ptr(), motor.data_ptr())
        self.assertNotEqual(motor.data_ptr(), applied.data_ptr())
        motor[0, 0] = 0.0
        self.assertEqual(float(raw[0, 0]), 20.0)
        self.assertNotEqual(float(applied[0, 0]), 0.0)

    def test_friction_opposes_velocity(self):
        motor = torch.tensor([[4.0, -4.0, 2.0]])
        velocity = torch.tensor([[1.0, -2.0, 0.0]])
        applied = apply_coulomb_friction(
            motor, velocity, torch.full_like(motor, 0.3), 0.05
        )
        friction_effect = applied - motor
        self.assertTrue(
            torch.all(friction_effect * velocity <= 1.0e-7)
        )

    def test_reference_delta_uses_executed_tanh_space(self):
        policy = torch.tensor([[4.0, 0.5]])
        reference = torch.tensor([[3.0, -0.5]])
        delta = executed_action_delta(policy, reference)
        self.assertTrue(torch.allclose(
            delta,
            torch.tanh(policy) - torch.tanh(reference),
        ))
        self.assertLess(float(delta[0, 0]), 0.01)

    def test_reference_delta_can_match_hard_clipped_action_space(self):
        policy = torch.tensor([[4.0, 0.5, -2.0]])
        reference = torch.tensor([[3.0, -0.5, 0.25]])
        delta = executed_action_delta(
            policy,
            reference,
            transform="clip",
            clip_value=1.0,
        )
        self.assertTrue(torch.equal(
            delta,
            torch.tensor([[0.0, 1.0, -1.25]]),
        ))


class CheckpointAdapterTests(unittest.TestCase):
    class ActorCritic(torch.nn.Module):
        def __init__(self, input_width):
            super().__init__()
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(input_width, 8, bias=False)
            )
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(input_width, 8, bias=False)
            )

    @staticmethod
    def state(input_width):
        return {
            "actor.0.weight": torch.ones(8, input_width),
            "critic.0.weight": torch.ones(8, input_width),
        }

    def test_52_to_52(self):
        target = self.ActorCritic(52)
        adapted, changes = adapt_observation_input_state(
            self.state(52), target.state_dict()
        )
        target.load_state_dict(adapted, strict=True)
        self.assertEqual(adapted["actor.0.weight"].shape[1], 52)
        self.assertEqual(changes, [])

    def test_52_to_76(self):
        target = self.ActorCritic(76)
        adapted, changes = adapt_observation_input_state(
            self.state(52), target.state_dict()
        )
        target.load_state_dict(adapted, strict=True)
        self.assertEqual(adapted["actor.0.weight"].shape[1], 76)
        self.assertTrue(torch.all(adapted["actor.0.weight"][:, 52:] == 0))
        self.assertEqual(len(changes), 2)

    def test_51_scalar_heading_to_52_sin_cos_migration(self):
        target = self.ActorCritic(52)
        source = self.state(51)
        source["actor.0.weight"][:, 50] = 3.0
        source["critic.0.weight"][:, 50] = 4.0
        migration = {
            "source_width": 51,
            "destination_width": 52,
            "copy_prefix": 50,
            "column_mappings": [
                {"source": 50, "destination": 50, "scale": 2.0},
            ],
        }
        adapted, changes = adapt_observation_input_state(
            source,
            target.state_dict(),
            column_migration=migration,
        )
        target.load_state_dict(adapted, strict=True)
        self.assertTrue(
            torch.all(adapted["actor.0.weight"][:, 50] == 6.0)
        )
        self.assertTrue(
            torch.all(adapted["critic.0.weight"][:, 50] == 8.0)
        )
        self.assertTrue(
            torch.all(adapted["actor.0.weight"][:, 51] == 0.0)
        )
        self.assertEqual(len(changes), 2)

    def test_76_to_76(self):
        target = self.ActorCritic(76)
        adapted, changes = adapt_observation_input_state(
            self.state(76), target.state_dict()
        )
        target.load_state_dict(adapted, strict=True)
        self.assertEqual(adapted["critic.0.weight"].shape[1], 76)
        self.assertEqual(changes, [])

    def test_unsupported_dimension_is_explicit(self):
        with self.assertRaisesRegex(
            ValueError, "only supports widening actor/critic"
        ):
            adapt_observation_input_state(
                self.state(88), self.state(76)
            )


class TerminalAndCsvTests(unittest.TestCase):
    def test_terminal_snapshot_survives_source_reset(self):
        snapshot = TerminalSnapshot(1, 4, 3, "cpu")
        raw = torch.tensor([[18.0, 2.0, -3.0]])
        motor = torch.tensor([[17.0, 2.0, -3.0]])
        applied = torch.tensor([[16.8, 2.1, -3.0]])
        snapshot.capture(
            torch.tensor([True]),
            torch.tensor([16], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
            torch.tensor([[True, False, False, True]]),
            torch.tensor([[False, True, True, False]]),
            torch.tensor([[0.20, 0.0, 0.0, 0.18]]),
            torch.tensor([0.75]),
            torch.tensor([[0.1, -0.2, 0.3]]),
            torch.tensor([1.2]),
            raw,
            motor,
            applied,
            torch.tensor([[17.0, 17.0, 17.0]]),
            torch.tensor([[17.0, 12.0, 10.0]]),
        )
        raw.zero_()
        motor.zero_()
        applied.zero_()
        self.assertTrue(snapshot.valid[0])
        self.assertEqual(float(snapshot.raw_pd_torques[0, 0]), 18.0)
        self.assertEqual(
            snapshot.contact_mask.tolist(),
            [[True, False, False, True]],
        )
        self.assertEqual(int(snapshot.illegal_contact_count[0]), 3)

    def test_csv_fields_and_units(self):
        headers = build_headers(["FL_hip_joint"])
        required = {
            "reset_reason",
            "illegal_contact_count",
            "foot_contact_mask",
            "desired_contact_mask",
            "terminal_contact_mask",
            "terminal_phase",
            "terminal_roll_rad",
            "terminal_pitch_rad",
            "terminal_yaw_rate_rad_s",
            "raw_pd_torque_nm_FL_hip_joint",
            "motor_electromagnetic_torque_nm_FL_hip_joint",
            "applied_joint_torque_nm_FL_hip_joint",
            "peak_torque_limit_nm_FL_hip_joint",
            "active_torque_limit_nm_FL_hip_joint",
            "raw_over_17nm_FL_hip_joint",
            "peak_saturation_flag_FL_hip_joint",
            "active_saturation_flag_FL_hip_joint",
            "motor_over_6nm_FL_hip_joint",
            "motor_over_12nm_FL_hip_joint",
            "motor_over_15nm_FL_hip_joint",
        }
        self.assertTrue(required.issubset(headers))


if __name__ == "__main__":
    unittest.main()
