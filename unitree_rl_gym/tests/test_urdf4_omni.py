"""Simulator-free checks of the urdf4_omni_v1 task.

    python -m unittest tests.test_urdf4_omni
"""

import math
import os
import unittest
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401  (must precede torch)
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.urdf4.disturbances import (
    DelayLine,
    ImuDisturbance,
    pulse_envelope,
    quat_from_rpy,
    quat_rotate_inverse,
)
from legged_gym.envs.urdf4.gait_clock import (
    LEG_PHASE_OFFSETS,
    CycleMean,
    leg_phase,
    scheduled_stance,
    swing_height_target,
)
from legged_gym.envs.urdf4.urdf4_omni_config import Urdf4OmniCfg


def imu_cfg(**overrides):
    """Copy of the task's IMU settings with selected values replaced."""
    values = {key: value for key, value in vars(Urdf4OmniCfg.imu).items() if not key.startswith("_")}
    values.update(overrides)
    return type("ImuCfg", (), values)


QUIET = dict(
    clean_env_fraction=0.0, gyro_noise_std_rad_s=0.0, gyro_bias_rad_s=0.0, gyro_bias_walk_rad_s_sqrt_s=0.0,
    gyro_scale_error=0.0, gravity_noise_std=0.0, attitude_drift_std_deg=0.0, spike_probability=0.0,
    stale_frame_probability=0.0, mount_roll_pitch_deg=0.0, mount_yaw_deg=0.0,
)


class ConfigTest(unittest.TestCase):
    def test_observation_and_timing_contract(self):
        cfg = Urdf4OmniCfg
        self.assertEqual(cfg.env.num_observations, cfg.env.frame_size * cfg.env.history_length)
        self.assertEqual(cfg.env.frame_size, 3 + 3 + 3 + 12 + 12 + 12 + 2 + 1)
        self.assertEqual(cfg.env.num_privileged_obs, cfg.env.frame_size + 3 + 1 + 4 + 4 + 4 + 3 + 1 + 1 + 3 + 12 + 12)
        self.assertAlmostEqual(cfg.sim.dt * cfg.control.decimation, cfg.rs01_actuator.control_dt_s)
        self.assertFalse(cfg.domain_rand.push_robots)
        self.assertEqual(cfg.terrain.mesh_type, "plane")

    def test_default_pose_matches_the_urdf(self):
        cfg = Urdf4OmniCfg
        path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.assertTrue(os.path.isfile(path), "run legged_gym/scripts/prepare_urdf4_asset.py")
        limits = {
            joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
            for joint in ET.parse(path).getroot().findall("joint")
            if joint.get("type") == "revolute"
        }
        angles = cfg.init_state.default_joint_angles
        self.assertEqual(sorted(limits), sorted(angles))
        self.assertEqual(sorted(cfg.rs01_actuator.runtime_dof_order), sorted(angles))
        clip = cfg.normalization.clip_actions
        for name, angle in angles.items():
            lower, upper = limits[name]
            scale = cfg.control.action_scale_by_joint[name.split("_")[1]]
            # Even a saturated action must stay inside the joint's travel.
            self.assertGreater(angle - clip * scale, lower, name)
            self.assertLess(angle + clip * scale, upper, name)
        for side_pair in (("FL", "FR"), ("RL", "RR")):
            for joint in ("thigh", "calf"):
                left, right = (angles[f"{leg}_{joint}_joint"] for leg in side_pair)
                self.assertAlmostEqual(left, -right)  # Mirrored joint axes.


class GaitClockTest(unittest.TestCase):
    def setUp(self):
        self.legs = ["FL", "FR", "RL", "RR"]
        self.offsets = torch.tensor([LEG_PHASE_OFFSETS[leg] for leg in self.legs])
        self.ratio = Urdf4OmniCfg.gait.stance_ratio

    def test_diagonal_pairs_and_no_flight(self):
        clock = torch.linspace(0.0, 0.999, 1000)
        stance = scheduled_stance(leg_phase(clock, self.offsets), self.ratio)
        fl, fr, rl, rr = stance.unbind(1)
        self.assertTrue(torch.equal(fl, rr))
        self.assertTrue(torch.equal(fr, rl))
        self.assertGreaterEqual(int(stance.sum(dim=1).min()), 2)  # A diagonal always supports.
        self.assertTrue(bool((fl | fr).all()))
        self.assertAlmostEqual(stance.float().mean().item(), self.ratio, places=2)
        # The pairs alternate: each swings alone at some point.
        self.assertTrue(bool((fl & ~fr).any()) and bool((fr & ~fl).any()))

    def test_swing_height_profile(self):
        radius, clearance = 0.02, 0.03
        phase = torch.tensor([[0.0, self.ratio, 0.5 * (1.0 + self.ratio), 0.9999]])
        target = swing_height_target(phase, self.ratio, radius, clearance)[0]
        self.assertAlmostEqual(target[0].item(), radius, places=5)
        self.assertAlmostEqual(target[1].item(), radius, places=5)
        self.assertAlmostEqual(target[2].item(), radius + clearance, places=5)
        self.assertAlmostEqual(target[3].item(), radius, places=3)


    def test_cycle_mean_keeps_the_offset_and_drops_the_stride_ripple(self):
        steps = 25  # One 0.5 s gait period at 50 Hz.
        mean = CycleMean(num_envs=2, width=1, steps=steps, device="cpu")
        offset = torch.tensor([[0.05], [-0.02]])
        for step in range(3 * steps):
            ripple = 0.3 * math.sin(2.0 * math.pi * step / steps)
            mean.push(offset + ripple)
        torch.testing.assert_close(mean.mean(), offset, atol=1.0e-6, rtol=0.0)
        mean.reset(torch.tensor([0]))
        self.assertEqual(mean.mean()[0].item(), 0.0)
        self.assertAlmostEqual(mean.mean()[1].item(), -0.02, places=6)
        mean.push(torch.ones(2, 1))
        self.assertAlmostEqual(mean.mean()[0].item(), 1.0 / steps, places=6)


class DisturbanceTest(unittest.TestCase):
    def test_pulse_envelope(self):
        duration = torch.full((2001,), 0.2)
        time = torch.linspace(-0.1, 0.3, 2001)
        envelope = pulse_envelope(time, duration)
        self.assertTrue(bool((envelope[(time < 0.0) | (time >= 0.2)] == 0.0).all()))
        self.assertAlmostEqual(envelope.max().item(), 1.0, places=4)
        impulse = torch.trapz(envelope, time).item()
        self.assertAlmostEqual(impulse, 0.1, places=4)  # peak * duration / 2

    def test_delay_line_reads_each_latency(self):
        line = DelayLine(num_envs=3, width=1, length=6, device="cpu")
        line.reset(torch.arange(3), torch.full((3, 1), -1.0))
        for value in range(4):
            line.push(torch.full((3, 1), float(value)))
        read = line.read(torch.tensor([0, 2, 5]))
        self.assertEqual(read.flatten().tolist(), [3.0, 1.0, -1.0])  # Oldest slot still holds the reset sample.
        with self.assertRaises(ValueError):
            line.read(torch.tensor([0, 0, 6]))
        line.reset(torch.tensor([1]), torch.full((1, 1), 9.0))
        self.assertEqual(line.read(torch.tensor([0, 4, 0])).flatten().tolist(), [3.0, 9.0, 3.0])

    def _truth(self, count):
        torch.manual_seed(0)
        quat = quat_from_rpy(*(0.3 * torch.randn(3, count)))
        rate = torch.randn(count, 3)
        down = torch.tensor([0.0, 0.0, -1.0]).repeat(count, 1)
        return quat, rate, quat_rotate_inverse(quat, down)

    def test_clean_imu_reports_truth(self):
        count = 64
        imu = ImuDisturbance(count, "cpu", 0.02, imu_cfg(clean_env_fraction=1.0))
        imu.reset(torch.arange(count))
        quat, rate, gravity = self._truth(count)
        for _ in range(50):
            imu.step()
        measured_rate, measured_gravity = imu.measure(quat, rate)
        torch.testing.assert_close(measured_rate, rate)
        torch.testing.assert_close(measured_gravity, gravity, atol=1.0e-6, rtol=0.0)

    def test_disabled_imu_model_is_clean(self):
        imu = ImuDisturbance(32, "cpu", 0.02, imu_cfg(enabled=False))
        imu.reset(torch.arange(32))
        self.assertTrue(bool(imu.clean.all()))

    def test_mount_misalignment_is_bounded_and_constant(self):
        count, tilt_deg = 512, 2.0
        cfg = imu_cfg(**{**QUIET, "mount_roll_pitch_deg": tilt_deg, "mount_yaw_deg": 0.0})
        imu = ImuDisturbance(count, "cpu", 0.02, cfg)
        imu.reset(torch.arange(count))
        quat, rate, gravity = self._truth(count)
        _, first = imu.measure(quat, rate)
        imu.step()
        _, second = imu.measure(quat, rate)
        torch.testing.assert_close(first, second)
        angle = torch.rad2deg(torch.acos((first * gravity).sum(dim=1).clamp(-1.0, 1.0)))
        self.assertLessEqual(angle.max().item(), math.sqrt(2.0) * tilt_deg + 1.0e-2)
        self.assertGreater(angle.mean().item(), 0.5)
        torch.testing.assert_close(first.norm(dim=1), torch.ones(count))

    def test_slow_error_states_stay_inside_their_limits(self):
        count = 256
        cfg = imu_cfg(clean_env_fraction=0.0, gyro_bias_walk_rad_s_sqrt_s=0.05)
        imu = ImuDisturbance(count, "cpu", 0.02, cfg)
        imu.reset(torch.arange(count))
        seen_tilt = []
        for _ in range(3000):
            imu.step()
            seen_tilt.append(imu.tilt_error.clone())
        self.assertLessEqual(imu.gyro_bias.abs().max().item(), cfg.gyro_bias_limit_rad_s + 1.0e-6)
        tilt = torch.rad2deg(torch.stack(seen_tilt))
        self.assertLessEqual(tilt.abs().max().item(), cfg.attitude_drift_limit_deg + 1.0e-4)
        # Stationary spread of the drift matches the configured value.
        self.assertAlmostEqual(tilt.std().item(), cfg.attitude_drift_std_deg, delta=0.15)

    def test_stale_frame_repeats_the_previous_sample(self):
        count = 16
        imu = ImuDisturbance(count, "cpu", 0.02, imu_cfg(**{**QUIET, "stale_frame_probability": 1.0}))
        imu.reset(torch.arange(count))
        quat, rate, _ = self._truth(count)
        first_rate, first_gravity = imu.measure(quat, rate)  # Nothing to repeat yet.
        torch.testing.assert_close(first_rate, rate)
        second_rate, second_gravity = imu.measure(quat, rate + 1.0)
        torch.testing.assert_close(second_rate, first_rate)
        torch.testing.assert_close(second_gravity, first_gravity)

    def test_reset_only_touches_the_given_robots(self):
        imu = ImuDisturbance(8, "cpu", 0.02, imu_cfg(clean_env_fraction=0.0))
        imu.reset(torch.arange(8))
        bias, mount = imu.gyro_bias.clone(), imu.mount_quat.clone()
        imu.reset(torch.tensor([2, 5]))
        kept = torch.tensor([0, 1, 3, 4, 6, 7])
        torch.testing.assert_close(imu.gyro_bias[kept], bias[kept])
        torch.testing.assert_close(imu.mount_quat[kept], mount[kept])
        self.assertFalse(torch.equal(imu.gyro_bias[[2, 5]], bias[[2, 5]]))


if __name__ == "__main__":
    unittest.main()
