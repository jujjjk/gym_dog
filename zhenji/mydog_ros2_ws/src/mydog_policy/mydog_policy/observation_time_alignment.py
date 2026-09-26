"""Common-time resampling of receiver-stamped data, never hardware sample sync.

MCU ticks timestamp CAN reception, and the serial protocol has no sample clock.
The affine MCU mapping estimates transport delay; its fit residual is not an
absolute timestamp accuracy guarantee. No extrapolation or invented feedback.
"""
from collections import deque
import math
import numpy as np
from .observation_pipeline import quaternion_matrix, rotation_rpy


class BoardClock:
    def __init__(self):
        self.points = deque(maxlen=256)
        self.last_tick = None
        self.unwrapped = 0
        self.slope = 1.
        self.offset = 0.
        self.residual_ms = float('inf')
        self.updates = 0

    def update(self, tick, host):
        tick = int(tick)
        if not 0 <= tick < 2**32 or not math.isfinite(host):
            raise ValueError('invalid board clock')
        if self.last_tick is not None:
            delta = (tick-self.last_tick) & 0xffffffff
            if delta >= 2**31:
                self.__init__()
                raise ValueError('board clock reset/regression')
            if delta == 0:
                return self.unwrapped/1000.
            self.unwrapped += delta
        else:
            self.unwrapped = tick
        self.last_tick = tick
        x = self.unwrapped/1000.
        if self.points and (host <= self.points[-1][1] or abs((host-self.points[-1][1])-(x-self.points[-1][0])) > .1):
            self.__init__()
            raise ValueError('host/board clock discontinuity')
        self.points.append((x, host))
        self.updates += 1
        # Drift changes slowly; fit at 5 Hz after warmup, not per motor tick.
        if len(self.points) > 10 and self.updates % 10:
            if abs(host-self.host_time(x)) > .01:
                self.residual_ms = float('inf')
            return x
        data = np.asarray(self.points)
        dx = data[:, 0]-data[0, 0]
        dy = data[:, 1]-data[0, 1]
        if dx[-1] >= 2.:
            centered = dx-dx.mean()
            self.slope = float(np.clip(np.dot(centered, dy-dy.mean())/np.dot(centered, centered), .999, 1.001))
        offsets = data[:, 1]-self.slope*data[:, 0]
        self.offset = float(np.min(offsets))
        self.residual_ms = float(np.percentile(offsets-self.offset, 95)*1000.)
        return x

    @property
    def ready(self):
        return len(self.points) >= 10 and self.residual_ms <= 5.

    def host_time(self, board_seconds):
        return self.slope*board_seconds+self.offset


def interpolate(records, target, max_gap=.06, quaternion=False):
    """Require real bracketing records; return value and actual source span."""
    left = right = None
    for stamp, value in records:
        if stamp <= target:
            left = (stamp, value)
        if stamp >= target:
            right = (stamp, value)
            break
    if left is None or right is None:
        raise ValueError('missing bracket')
    gap = right[0]-left[0]
    if gap < 0 or gap > max_gap:
        raise ValueError('source gap exceeds limit')
    a, b = np.asarray(left[1], dtype=float), np.asarray(right[1], dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('nonfinite source')
    u = 0. if gap == 0 else (target-left[0])/gap
    if quaternion:
        quaternion_matrix(a); quaternion_matrix(b)
        a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
        dot = float(np.dot(a, b))
        if dot < 0:
            b = -b; dot = -dot
        if dot > .9995:
            value = a+u*(b-a)
        else:
            angle = math.acos(np.clip(dot, -1., 1.))
            value = (math.sin((1-u)*angle)*a+math.sin(u*angle)*b)/math.sin(angle)
        value /= np.linalg.norm(value)
    else:
        value = a+u*(b-a)
    return value, gap*1000.


class CommonTimeAlignment:
    def __init__(self, rotation, delay=.05):
        self.rotation = rotation
        self.delay = delay
        self.clocks = [BoardClock(), BoardClock()]
        self.motors = [deque(maxlen=64) for _ in range(12)]
        self.sample = None
        self.last_target = None
        self.diagnostics = {}

    def update(self, motor, frames, now):
        self.sample = None
        target = now-self.delay
        self.diagnostics = dict(observation_alignment_method='linear_q_dq_gyro_quaternion_slerp',
            observation_time_basis='mapped_mcu_can_reception_and_serial_reception_monotonic',
            acquisition_sync_verified=False, observation_target_monotonic=target,
            observation_delay_ms=self.delay*1000., observation_alignment_verified=False,
            observation_alignment_reason='', observation_resampled_skew_ms=None,
            observation_bracket_max_ms=None, mcu_clock_fit_ready=False,
            mcu_clock_fit_residual_ms=[None, None], mcu_clock_scale=[None, None],
            mcu_clock_offset_sec=[None, None], aligned_motor_channels=0,
            unaligned_motor_indices=list(range(12)), imu_bracket_max_ms=None,
            imu_common_time_ready=False, motor_bracket_failures={}, motor_latest_mapped_age_ms=[])
        try:
            epoch = getattr(motor, 'state_epoch_monotonic', None)
            if epoch is None or not np.isfinite(epoch) or not 0 <= now-epoch <= .06:
                raise ValueError('missing/stale RT monotonic epoch')
            if self.last_target is not None and target <= self.last_target:
                raise ValueError('control clock regression')
            self.last_target = target
            for board in range(2):
                start = board*6
                ticks = np.asarray(motor.board_tick_ms[start:start+6])
                if not np.all(ticks == ticks[0]):
                    raise ValueError('incoherent board snapshot')
                try:
                    board_now = self.clocks[board].update(ticks[0], epoch)
                except ValueError:
                    for index in range(start, start+6): self.motors[index].clear()
                    raise
                for index in range(start, start+6):
                    age = float(motor.age_ms[index])
                    if not motor.online[index] or not 0 <= age <= 60:
                        self.motors[index].clear()
                        continue
                    stamp = board_now-age/1000.
                    history = self.motors[index]
                    if not history or stamp > history[-1][0]:
                        history.append((stamp, (float(motor.q_real[index]), float(motor.dq_real[index]))))
            self.diagnostics.update(mcu_clock_fit_ready=all(c.ready for c in self.clocks),
                mcu_clock_fit_residual_ms=[c.residual_ms for c in self.clocks],
                mcu_clock_scale=[c.slope for c in self.clocks],
                mcu_clock_offset_sec=[c.offset for c in self.clocks])
            if not all(c.ready for c in self.clocks):
                raise ValueError('MCU clock fit warming/unreliable')
            values, gaps, missing = [], [], []
            for index, history in enumerate(self.motors):
                clock = self.clocks[index//6]
                self.diagnostics['motor_latest_mapped_age_ms'].append(
                    (now-clock.host_time(history[-1][0]))*1000. if history else None)
                try:
                    value, gap = interpolate(history, (target-clock.offset)/clock.slope, max_gap=.06/clock.slope)
                    values.append(value); gaps.append(gap*clock.slope)
                except ValueError as exc:
                    missing.append(index)
                    self.diagnostics['motor_bracket_failures'][index] = str(exc)
            self.diagnostics['aligned_motor_channels'] = 12-len(missing)
            self.diagnostics['unaligned_motor_indices'] = missing
            gyro, gyro_gap = interpolate(frames.get('gyro', ()), target)
            quat, quat_gap = interpolate(frames.get('quat', ()), target, quaternion=True)
            self.diagnostics.update(imu_bracket_max_ms=max(gyro_gap, quat_gap), imu_common_time_ready=True)
            if missing:
                raise ValueError('motor brackets unavailable: '+str(missing))
            matrix = quaternion_matrix(quat) @ self.rotation.T
            values = np.asarray(values)
            self.sample = dict(timestamp=target, q_real=values[:, 0], dq_real=values[:, 1],
                gyro=self.rotation@gyro, gravity=matrix.T@np.array([0., 0., -1.]),
                yaw=math.radians(rotation_rpy(matrix)[2]))
            self.diagnostics.update(observation_alignment_verified=True, observation_resampled_skew_ms=0.,
                observation_bracket_max_ms=max(gaps+[gyro_gap, quat_gap]))
        except (ValueError, TypeError) as exc:
            self.diagnostics['observation_alignment_reason'] = str(exc)
        return self.sample
