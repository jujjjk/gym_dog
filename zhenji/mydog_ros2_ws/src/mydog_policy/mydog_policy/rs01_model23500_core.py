"""V22 B23500: measured sensor snapshot only; no synthetic noise/delay on hardware."""
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract
from .rs01_model6850_core import Rs01Model6850Core
from .rs01_model6850_guard import Guarded6850PolicyCore
from .rs01_model18000_core import Rs01Model18000Core

EXPECTED_ONNX_SHA256 = '4367b59d30a75bd9150866bd90d0fa304b05f7eeab874c477e8dde452362423f'


class Model23500Contract(Model930Contract):
    expected_task = 'rs01_omni_v22_sensor'
    expected_observations = 61
    model_label = 'B23500'

    @classmethod
    def from_onnx_session(cls, session, onnx_path, expected_sha256=EXPECTED_ONNX_SHA256):
        if expected_sha256 != EXPECTED_ONNX_SHA256:
            raise RuntimeError('B23500 hash override is not permitted')
        c = super().from_onnx_session(session, onnx_path, expected_sha256)
        expected = dict(mapping='conditional_inward_scale_v1', inward_target_rad=np.deg2rad(8.),
                        vy_gate=.08, wz_gate=.20, stand_phase='freeze_and_resume', odometry='legacy',
                        guard=dict(continuous=6., peak=14., full=8., tau=2.))
        v22 = c.raw.get('v22', {})
        if c.raw.get('v20') != expected or c.policy_dt != .02:
            raise RuntimeError('B23500 mapping/guard/20ms contract mismatch')
        if (v22.get('host_guard') != 'policy_sensor_snapshot'
                or v22.get('observation') != 'sensor_snapshot_61'
                or c.raw['observations']['base_linear_velocity_source'] != 'rs01_leg_odometry'
                or v22.get('odometry') != c.raw['observations']['rs01_leg_odometry']):
            raise RuntimeError('B23500 sensor/odometry contract mismatch')
        arrays = dict(default=np.asarray(c.raw['default_joint_angles_rad'], dtype=np.float64))
        for field, key in [('action_scale', 'action_scale_rad'), ('rate_limit', 'target_rate_limit_rad_s'),
                           ('accel_limit', 'target_acceleration_limit_rad_s2')]:
            arrays[field] = np.asarray(c.raw['control'][key], dtype=np.float64)
        return replace(c, **arrays)


class Rs01Model23500Core(Rs01Model18000Core):
    initialize_odometry_on_first_tick = True
    # Only remove float64 roundoff at mathematically exact target arrival.
    # No macroscopic target/rate/acceleration envelope is changed.
    target_reached_atol = 1e-12
    # V22 trained with the original V13 direction controller. Do not inherit
    # the later B18000-only deadband / half-gain / disabled planar correction.
    _mix_direction_command = Rs01Model6850Core._mix_direction_command

    def _direction_heading_error(self, error):
        # Apply the same confidence to direction mixing AND actor sin/cos
        # heading channels. Keep measured yaw and the original target intact.
        return error * getattr(self, 'heading_correction_weight', 1.)


class Guarded23500PolicyCore(Guarded6850PolicyCore):
    actor_type = Rs01Model23500Core

    def build_observation(self, now, base_linear_velocity, base_angular_velocity,
                          projected_gravity, command, q_policy, dq_policy, yaw, kinematics=None):
        if getattr(self, 'common_time_required', False):
            sample = getattr(self, 'aligned_sample', None)
            if sample is None or not 0 <= now-sample['timestamp'] <= .06:
                raise RuntimeError('Missing/stale common-time policy observation')
            q_policy, dq_policy = self.mapper.real_to_policy_abs(sample['q_real'], sample['dq_real'])
            base_angular_velocity = sample['gyro']-self.aligned_gyro_bias
            projected_gravity, yaw = sample['gravity'], sample['yaw']
            # Recompute actor odometry/kinematics from the same delayed sample.
            # step() still receives the current q/dq for PD protection.
            kinematics = None
        return super().build_observation(now, base_linear_velocity, base_angular_velocity,
            projected_gravity, command, q_policy, dq_policy, yaw, kinematics)
