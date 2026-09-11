"""Hardware-independent A6850 guarded-test primitives; no device I/O."""

import numpy as np

from .rs01_model6850_core import Rs01Model6850Core
from .rs01_model930_core import Rs01Model930TargetLimiter


class Guarded6850PolicyCore:
    """Keep the A6850 actor/limiter intact, then apply real PD protection.

    The 14 Nm hardware envelope differs from the 17 Nm training envelope.
    The old braking limiter is used ONLY for its PD projection, never step().
    """

    def __init__(self, session, contract):
        self.actor = Rs01Model6850Core(session, contract)
        self.mapper = self.actor.mapper
        self.limiter = Rs01Model930TargetLimiter(contract)
        self.previous_action = np.zeros(12)
        self.heading_target = 0.0
        self.pending = None
        # Allocate the ONNX execution buffers before any device is opened.
        self.actor.tick(contract.default, np.zeros(12), np.zeros(3),
                        [0, 0, -1], 0., np.zeros(3), gait=0.)
        self.reset(0., 0.)

    def reset(self, now, yaw, q_policy=None):
        self.actor.reset(yaw, q_policy)
        self.limiter.reset(self.actor.target)
        self.previous_action.fill(0.0)
        self.heading_target = float(yaw)
        self.pending = None

    def build_observation(self, now, base_linear_velocity,
                          base_angular_velocity, projected_gravity, command,
                          q_policy, dq_policy, yaw):
        # A6850 runs its own training-compatible odometry from q/dq/gyro.
        self.pending = self.actor.tick(
            q_policy, dq_policy, base_angular_velocity, projected_gravity,
            yaw, command, gait=1.)
        self.heading_target = self.actor.heading
        return self.pending['observation']

    def step(self, observation, q_policy, dq_policy, active_limit_nm):
        if self.pending is None:
            raise RuntimeError('No A6850 inference pending')
        result = self.pending
        self.pending = None
        safe, info = self.limiter.pd_equivalent_peak_limit(
            result['target_policy'], q_policy, dq_policy, active_limit_nm)
        # Do not silently substitute a different trained limiter or overwrite
        # actor history with the protective projection; log the projection.
        self.previous_action = result['action'].copy()
        return dict(action=result['action'], safe_target_policy=safe,
                    torque_info=info)


class TrialCommand:
    """Explicit one-shot arming, fresh commands, and a bounded motion lease."""

    caps = np.array([0.20, 0.15, 0.30])

    def __init__(self):
        self.vector = np.zeros(3)
        self.stamp = None
        self.armed_at = None
        self.started_at = None
        self.reason = 'not armed'

    @property
    def armed(self):
        return self.armed_at is not None

    def disarm(self, reason):
        self.vector.fill(0.)
        self.stamp = self.armed_at = self.started_at = None
        self.reason = reason

    def arm(self, now):
        self.disarm('armed; waiting for fresh command')
        self.armed_at = float(now)

    def receive(self, values, now):
        values = np.asarray(values, dtype=float).reshape(3)
        if not np.isfinite(values).all() or np.any(np.abs(values) > self.caps + 1e-8):
            self.disarm('invalid/out-of-range command')
            return False
        if not np.any(np.abs(values) > 0.001):
            self.disarm('zero command')
            return True
        if not self.armed:
            return False
        self.vector = values.copy()
        self.stamp = float(now)
        return True

    def active(self, now):
        if not self.armed:
            return False
        if now < self.armed_at or now - self.armed_at > 10.:
            self.disarm('arm lease expired')
        elif self.started_at is not None and now - self.started_at >= 4.:
            self.disarm('four-second motion budget exhausted')
        elif self.stamp is not None and not 0 <= now - self.stamp <= 0.35:
            self.disarm('command timeout')
        return self.armed and self.stamp is not None

    def start(self, now):
        if not self.active(now):
            raise RuntimeError('Cannot start without fresh armed command')
        if self.started_at is None:
            self.started_at = float(now)


class ReceptionGuard:
    """Bound real transport age; NEVER label reception as acquisition time.

    Board age + HTTP cache age + local cache age bounds the reported motor
    age. Board counters must keep progressing. IMU stamps come from parsed
    serial frames, not repeated vendor getter calls. This is not proof of
    sensor clock synchronization and is only a tethered-test envelope.
    """

    def __init__(self):
        self.seq = self.tick = self.progress_at = None
        self.metrics = {}

    def check(self, motor, imu, wall, mono):
        local_age = wall - float(motor.stamp)
        imu_age = wall - float(imu.stamp)
        cache = float(motor.cache_age_ms) / 1000.
        ages = np.asarray(motor.age_ms, dtype=float) / 1000.
        if not np.isfinite(np.r_[local_age, imu_age, cache, ages]).all():
            raise RuntimeError('Non-finite feedback ages')
        if min(local_age, imu_age, cache, float(ages.min())) < 0:
            raise RuntimeError('Future/negative feedback age')
        total = ages + cache + local_age
        if total.max() > .080 or imu_age > .060:
            raise RuntimeError('Motor/IMU transport feedback stale')
        if not np.isfinite(np.r_[motor.q_real, motor.dq_real, motor.torque,
                                  imu.gyro_rad_s, imu.rpy_deg,
                                  imu.projected_gravity, imu.quat_wxyz]).all():
            raise RuntimeError('Non-finite hardware feedback')
        if abs(np.linalg.norm(imu.quat_wxyz) - 1.) > .02:
            raise RuntimeError('Invalid IMU quaternion')
        seq = np.asarray(motor.snapshot_seq, dtype=np.int64)
        tick = np.asarray(motor.board_tick_ms, dtype=np.int64)
        if np.any(seq < 0) or np.any(tick < 0):
            raise RuntimeError('Missing board counters')
        if self.seq is None:
            self.progress_at = np.full(12, mono)
        else:
            # Board reboot/counter wrap requires a new calibration/start.
            if np.any(seq < self.seq) or np.any(tick < self.tick):
                raise RuntimeError('Board counter regressed/reset')
            advanced = (seq > self.seq) & (tick > self.tick)
            self.progress_at[advanced] = mono
            if np.any(mono - self.progress_at > .080):
                raise RuntimeError('Board feedback stopped progressing')
        self.seq, self.tick = seq.copy(), tick.copy()
        self.metrics = dict(feedback_time_basis='host_reception_with_board_age',
                            acquisition_sync_verified=False,
                            effective_motor_age_ms=float(total.max() * 1000),
                            imu_frame_age_ms=float(imu_age * 1000))
