"""V22 B23500: measured sensor snapshot only; no synthetic noise/delay on hardware."""
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract, wrap_pi
from .rs01_model6850_core import Rs01Model6850Core
from .rs01_model6850_guard import Guarded6850PolicyCore
from .rs01_model18000_core import Rs01Model18000Core
from .lateral_position_hold import LateralPositionHold

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

    def __init__(self, session, contract):
        # Built first: the base constructor already calls the overridden reset().
        self.lateral_hold = LateralPositionHold()
        super().__init__(session, contract)

    def reset(self, yaw, q_policy=None, phase=0.):
        super().reset(yaw, q_policy, phase)
        self.lateral_hold.reset()

    # V22 trained with the original V13 direction controller. Do not inherit
    # the later B18000-only deadband / half-gain / disabled planar correction.
    # The bounded lateral position-hold correction is added on top; raw
    # command observation channels are never modified.
    def _mix_direction_command(self, command, error, turning, conf):
        target = Rs01Model6850Core._mix_direction_command(
            self, command, error, turning, conf)
        correction = float(self.lateral_hold.correction)
        if correction:
            lo, hi = self.contract.raw['commands']['ranges']['lin_vel_y']
            target[1] = float(np.clip(target[1]+correction, lo, hi))
        return target

    # March also holds the session heading: never erase its actor heading
    # channels. Intentional turns keep the existing uncorrected behavior.
    straight_min_vx_mps = 1e-3
    straight_max_other = 1e-6

    def heading_correction_active(self):
        command = getattr(self, 'command', None)
        if command is None:
            return False
        return bool(abs(float(command[2])) <= self.straight_max_other)

    def tick(self,q,dq,gyro,gravity,yaw,command,gait=1.,kinematics=None):
        previous_hold=self.heading_correction_active()
        new=np.asarray(command,dtype=float)
        next_hold=abs(new[2]) <= self.straight_max_other
        if self.steps and next_hold and not previous_hold:
            # Start holding the current heading after an intentional turn, rather than
            # steering back to a reference from an earlier action. Parent tick
            # advances the previous yaw command before constructing this obs.
            self.heading=float(yaw)-self.contract.policy_dt*float(self.command[2])
        result = super().tick(q,dq,gyro,gravity,yaw,command,gait,kinematics)
        # Integrates one tick behind the injection point; at 50 Hz this lag is
        # negligible for a drift-rate integrator and keeps tick() side-effect
        # order identical to the trained controller.
        self.lateral_hold.update(
            self.contract.policy_dt, result['estimated_velocity'],
            result['confidence'], self.command, self.turning, self.gait,
            wrap_pi(self.heading-float(yaw)))
        return result

    def _direction_heading_error(self, error):
        # Apply the same confidence to direction mixing AND actor sin/cos
        # heading channels. Keep measured yaw and the original target intact.
        if not self.heading_correction_active():
            return 0.
        return error * getattr(self, 'heading_correction_weight', 1.)


class Guarded23500PolicyCore(Guarded6850PolicyCore):
    actor_type = Rs01Model23500Core
    common_time_max_age_sec = .14

    def build_observation(self, now, base_linear_velocity, base_angular_velocity,
                          projected_gravity, command, q_policy, dq_policy, yaw, kinematics=None):
        if getattr(self, 'common_time_required', False):
            sample = getattr(self, 'aligned_sample', None)
            # Permit up to 140 ms TOTAL sample age, including 50 ms lookback.
            # This tolerates short resampling gaps; longer outages fail closed.
            if sample is None or not 0 <= now-sample['timestamp'] <= self.common_time_max_age_sec:
                raise RuntimeError('Missing/stale common-time policy observation')
            q_policy, dq_policy = self.mapper.real_to_policy_abs(sample['q_real'], sample['dq_real'])
            base_angular_velocity = sample['gyro']-self.aligned_gyro_bias
            projected_gravity, yaw = sample['gravity'], sample['yaw']
            # Recompute actor odometry/kinematics from the same delayed sample.
            # step() still receives the current q/dq for PD protection.
            kinematics = None
        return super().build_observation(now, base_linear_velocity, base_angular_velocity,
            projected_gravity, command, q_policy, dq_policy, yaw, kinematics)
