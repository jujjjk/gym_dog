"""Observation alignment on explicitly labelled host-time estimates, not device sync."""
from copy import deepcopy
import math
import numpy as np


def mount_rotation(roll_deg, pitch_deg, yaw_deg):
    angles = np.radians([roll_deg, pitch_deg, yaw_deg])
    if not np.isfinite(angles).all():
        raise ValueError('IMU mount angles must be finite')
    r, p, y = angles
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def rotation_rpy(matrix):
    return np.degrees([math.atan2(matrix[2, 1], matrix[2, 2]),
                       math.asin(float(np.clip(-matrix[2, 0], -1., 1.))),
                       math.atan2(matrix[1, 0], matrix[0, 0])])


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or abs(norm-1.) > .02:
        raise ValueError('Invalid IMU quaternion')
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ObservationPipeline:
    def __init__(self, mount=(0., 0., 0.), max_age_ms=60., max_skew_ms=10.,
                 preview_tau_ms=5., timing_mode="reception"):
        if not 0 < max_age_ms <= 60 or not 0 < max_skew_ms <= 10:
            raise ValueError('Observation age/skew limits must be <=60/10ms')
        if not 0 <= preview_tau_ms <= 10:
            raise ValueError('Preview filter tau must be in [0,10]ms')
        if timing_mode not in ('reception', 'strict_host_alignment'):
            raise ValueError('Invalid observation timing mode')
        self.timing_mode = timing_mode
        self.rotation = mount_rotation(*mount)
        self.identity_mount = np.array_equal(self.rotation, np.eye(3))
        self.max_age_ms, self.max_skew_ms = max_age_ms, max_skew_ms
        self.tau = preview_tau_ms/1000.
        self.last_imu_stamp = float('-inf')
        self.last_filter_time = None
        self.filtered_dq = self.filtered_gyro = None
        self.diagnostics = {}
        self.quality_since = dict(timing=None, odometry=None, heading=None)

    def process(self, motor, latest_imu, history, wall, mono):
        # Keep the newest motor snapshot for PD protection and encoder limits.
        # Per-motor age + server cache age are estimates, NOT acquisition clocks.
        motor_times = float(motor.stamp) - (
            float(motor.cache_age_ms) + np.asarray(motor.age_ms, dtype=float))/1000.
        if not np.isfinite(motor_times).all():
            raise ValueError('Invalid motor timestamps')
        candidates = [s for s in list(history)+[latest_imu]
                      if s.valid and s.stamp >= self.last_imu_stamp
                      and 0 <= wall-s.stamp <= self.max_age_ms/1000.]
        def stamps(s):
            return np.asarray(getattr(s, 'frame_receive_wall', [s.stamp]), dtype=float)
        def skew(s):
            return float(np.ptp(np.r_[motor_times, stamps(s)]))*1000.
        imu = (min(candidates, key=lambda s: (skew(s), -s.stamp))
               if candidates and self.timing_mode == 'strict_host_alignment' else latest_imu)
        times = stamps(imu)
        regression = imu.stamp < self.last_imu_stamp
        if not regression:
            self.last_imu_stamp = float(imu.stamp)
        motor_age = float(np.max(wall-motor_times))*1000.
        imu_age = float(np.max(wall-times))*1000.
        time_values = np.r_[motor_times, times, wall, mono]
        reception_ok = bool(np.isfinite(time_values).all() and not regression
                           and np.all(wall-motor_times >= 0) and np.all(wall-times >= 0)
                           and motor_age <= self.max_age_ms and imu_age <= self.max_age_ms
                           and float(np.ptp(times))*1000. <= 60. and imu.valid)
        alignment_ok = bool(reception_ok and skew(imu) <= self.max_skew_ms)
        temporal_ok = alignment_ok if self.timing_mode == 'strict_host_alignment' else reception_ok
        result = deepcopy(imu)
        raw_gyro = np.asarray(imu.gyro_rad_s, dtype=float).copy()
        raw_gravity = np.asarray(imu.projected_gravity, dtype=float).copy()
        raw_dq = np.asarray(motor.dq_real, dtype=float).copy()
        result.gyro_rad_s = self.rotation @ raw_gyro
        result.projected_gravity = self.rotation @ raw_gravity
        # Sensor quaternion remains raw in capture. All control Euler angles
        # are base-frame angles, consistent with the transformed gravity/gyro.
        if not self.identity_mount:
            world_base = quaternion_matrix(imu.quat_wxyz) @ self.rotation.T
            result.rpy_deg = rotation_rpy(world_base)
        if not np.isfinite(np.r_[raw_dq, result.gyro_rad_s, result.projected_gravity, result.rpy_deg]).all():
            raise ValueError('Non-finite observation input')
        # Preview only: filters never change the actor or PD input in Phase 1.
        if temporal_ok:
            dt = None if self.last_filter_time is None else mono-self.last_filter_time
            if dt is None or not 0 < dt <= .04:
                self.filtered_dq, self.filtered_gyro = raw_dq.copy(), result.gyro_rad_s.copy()
            else:
                alpha = 1. if self.tau == 0 else 1.-math.exp(-dt/self.tau)
                self.filtered_dq += alpha*(raw_dq-self.filtered_dq)
                self.filtered_gyro += alpha*(result.gyro_rad_s-self.filtered_gyro)
            self.last_filter_time = mono
        else:
            self.last_filter_time = None
            self.filtered_dq, self.filtered_gyro = raw_dq.copy(), result.gyro_rad_s.copy()
        self.diagnostics = dict(
            observation_timing_mode=self.timing_mode, observation_reception_ok=reception_ok,
            observation_alignment_target_ok=alignment_ok,
            observation_time_basis='host_reception_and_board_age_estimate',
            acquisition_sync_verified=False, motor_acquisition_timestamp=None,
            imu_acquisition_timestamp=None, observation_control_timestamp=mono,
            imu_euler_source=getattr(imu,'euler_source','vendor_euler'),
            imu_euler_report_receive_wall=getattr(imu,'euler_report_receive_wall',None),
            motor_estimated_host_timestamp=float(np.min(motor_times)),
            imu_receive_host_timestamp=float(np.min(times)),
            aligned_motor_age_ms=motor_age, aligned_imu_age_ms=imu_age,
            aligned_sensor_skew_ms=skew(imu), imu_internal_frame_skew_ms=float(np.ptp(times))*1000.,
            observation_temporal_ok=temporal_ok, imu_history_selected=imu is not latest_imu,
            filter_applied_to_policy=False, filter_preview_tau_ms=self.tau*1000.,
            raw_dq=raw_dq, preview_filtered_dq=self.filtered_dq.copy(),
            raw_gyro=raw_gyro, base_gyro=result.gyro_rad_s.copy(),
            preview_filtered_gyro=self.filtered_gyro.copy(), raw_gravity=raw_gravity,
            base_gravity=result.projected_gravity.copy(), yaw_raw_deg=float(imu.rpy_deg[2]))
        return motor, result

    def quality(self, now, confidence, heading_ok, walking):
        checks = dict(timing=bool(self.diagnostics.get('observation_temporal_ok', False)),
                      odometry=bool(np.isfinite(confidence) and confidence >= .5),
                      heading=bool(heading_ok))
        durations = {}
        for name, healthy in checks.items():
            if healthy or not walking:
                self.quality_since[name] = None
            elif self.quality_since[name] is None:
                self.quality_since[name] = now
            since = self.quality_since[name]
            durations[name] = 0. if since is None else max(0., now-since)
        # Independent causes must not keep one shared timer alive when they
        # recover in turn. Each unchanged 200ms gate measures its own fault.
        reasons = [name for name, duration in durations.items() if walking and duration >= .20]
        ok = all(checks.values())
        stop = bool(reasons)
        self.diagnostics.update(obs_quality_ok=ok, obs_quality_bad_sec=max(durations.values()),
                                obs_quality_state='ok' if ok else ('soft_hold' if stop else 'transient'),
                                obs_quality_stop_reason=','.join(reasons),
                                obs_quality_timing_bad_sec=durations['timing'],
                                obs_quality_odometry_bad_sec=durations['odometry'],
                                obs_quality_heading_bad_sec=durations['heading'])
        return stop

    def scalar_diagnostics(self):
        return {k: v for k, v in self.diagnostics.items() if not isinstance(v, np.ndarray)}
