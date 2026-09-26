"""B23500 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
import time
from .rt_schedule import LatestTarget, next_deadline
from .observation_pipeline import ObservationPipeline
from .observation_time_alignment import CommonTimeAlignment
from .heading_recovery import HeadingRecovery
from .heading_reference import GyroHeadingReference
from .rs01_timestamped_imu import QuaternionFrameStampedImu
from .motor_state_rt_interface import MotorStateRtInterface
import numpy as np
from .rs01_model18000_node import Rs01Model18000Node, TIMING_SINGLE_GAP_SEC
from .rs01_model23500_core import Model23500Contract, Guarded23500PolicyCore, EXPECTED_ONNX_SHA256


class _FusedYawMonitor:
    """Keep the yaw/gyro consistency monitor on the fused IMU yaw.

    The control loop now receives the gyro-integrated heading, which would
    agree with the gyro by construction. Comparing the fused yaw instead keeps
    the magnetometer disturbance measurable in status and capture.
    """

    def __init__(self, inner, node):
        self._inner = inner
        self._node = node

    def _fused(self, yaw):
        fused = getattr(self._node, '_fused_yaw', None)
        return yaw if yaw is None or fused is None else fused

    def update(self, now, yaw, corrected_gyro_z_rad_s):
        return self._inner.update(now, self._fused(yaw), corrected_gyro_z_rad_s)

    def reset(self, now=None, yaw=None):
        return self._inner.reset(now, self._fused(yaw))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Rs01Model23500Node(Rs01Model18000Node):
    imu_interface_type = QuaternionFrameStampedImu
    motor_interface_type = MotorStateRtInterface
    telemetry_period_sec = .10
    node_name = 'rs01_model23500_node'
    model_label = 'B23500 V22 guarded omni'
    model_filename = 'B23500.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model23500Contract
    policy_core_type = Guarded23500PolicyCore
    topic_namespace = '/mydog/model23500'
    command_topic = '/mydog/model23500/cmd_vel'

    def _validate_deployment_parameters(self):
        super()._validate_deployment_parameters()
        self.declare_parameter('observation_pipeline_enabled', False)
        self.declare_parameter('observation_timing_mode', 'common_time')
        self.declare_parameter('imu_mount_roll_deg', 0.)
        self.declare_parameter('imu_mount_pitch_deg', 0.)
        self.declare_parameter('imu_mount_yaw_deg', 0.)
        self.declare_parameter('observation_filter_preview_tau_ms', 5.)
        self.observation_pipeline = None
        self.common_time_alignment = None
        self.heading_recovery = HeadingRecovery()
        # Direction-controller heading: calibrated gyro integration while
        # walking; the magnetometer-fused yaw only pulls when the field norm
        # is back at the standing baseline. The consistency monitor keeps
        # watching the fused yaw so the disturbance stays visible in status.
        self.declare_parameter('heading_reference_mode', 'gyro_mag_gated')
        self.declare_parameter('heading_mag_tolerance_ut', 8.)
        self.declare_parameter('heading_walking_pull_tau_sec', 5.)
        mode = str(self.get_parameter('heading_reference_mode').value)
        if mode not in ('gyro_mag_gated', 'fused'):
            raise RuntimeError('heading_reference_mode must be gyro_mag_gated or fused')
        self.heading_reference = None if mode == 'fused' else GyroHeadingReference(
            mag_tolerance_ut=float(self.get_parameter('heading_mag_tolerance_ut').value),
            walking_pull_tau_sec=float(self.get_parameter('heading_walking_pull_tau_sec').value))
        self._fused_yaw = None
        if self.get_parameter('observation_pipeline_enabled').value:
            mount = [float(self.get_parameter('imu_mount_'+axis+'_deg').value)
                     for axis in ('roll', 'pitch', 'yaw')]
            mode = str(self.get_parameter('observation_timing_mode').value)
            self.observation_pipeline = ObservationPipeline(
                mount=mount, max_age_ms=60., timing_mode='reception' if mode == 'common_time' else mode, preview_tau_ms=float(self.get_parameter('observation_filter_preview_tau_ms').value))
            if mode == 'common_time':
                self.common_time_alignment = CommonTimeAlignment(self.observation_pipeline.rotation)

    def _observation_state(self, motor, imu):
        pipeline = getattr(self, 'observation_pipeline', None)
        if pipeline is None:
            return super()._observation_state(motor, imu)
        history = []
        if pipeline.timing_mode == 'strict_host_alignment':
            reader = getattr(self.imu, 'get_history_view', getattr(self.imu, 'get_history', None))
            history = reader() if reader is not None else []
        wall, mono = time.time(), time.monotonic()
        result = pipeline.process(motor, imu, history, wall, mono)
        alignment = getattr(self, 'common_time_alignment', None)
        if alignment is not None:
            reader = getattr(self.imu, 'get_frame_history', None)
            motor_history = getattr(self.motor, 'get_history_view', lambda: ())()
            alignment.update(motor, reader() if reader is not None else {}, mono, motor_history=motor_history)
            pipeline.diagnostics['raw_reception_skew_ms'] = pipeline.diagnostics['aligned_sensor_skew_ms']
            fresh = alignment.sample
            sample = fresh
            if fresh is None:
                previous = getattr(self.core, 'aligned_sample', None)
                limit = float(getattr(self.core, 'common_time_max_age_sec', .14))
                if previous is not None and 0 <= mono-float(previous['timestamp']) <= limit:
                    sample = previous
                    reason = alignment.diagnostics.get('observation_alignment_reason') or 'resample missed'
                    alignment.diagnostics['observation_alignment_reason'] = 'holding last common-time sample: '+reason
            pipeline.diagnostics.update(alignment.diagnostics)
            # In common_time this field describes the resampled policy data.
            # The original source span is retained as raw_reception_skew_ms.
            pipeline.diagnostics['aligned_sensor_skew_ms'] = alignment.diagnostics['observation_resampled_skew_ms']
            pipeline.diagnostics['observation_timing_mode'] = 'common_time'
            pipeline.diagnostics['observation_temporal_ok'] = bool(
                pipeline.diagnostics['observation_reception_ok'] and sample is not None)
            pipeline.diagnostics['observation_alignment_target_ok'] = fresh is not None
            self.core.common_time_required = True
            self.core.aligned_sample = sample
            self._latest_base_gyro = result[1].gyro_rad_s.copy()
        # Keep current feedback for all PD, position, attitude and fault guards.
        return result

    def _fresh_state(self):
        motor, imu, roll, pitch, yaw = super()._fresh_state()
        reference = getattr(self, 'heading_reference', None)
        if reference is None:
            return motor, imu, roll, pitch, yaw
        if not isinstance(self.heading_consistency, _FusedYawMonitor):
            self.heading_consistency = _FusedYawMonitor(self.heading_consistency, self)
        self._fused_yaw = float(yaw)
        gyro = np.asarray(imu.gyro_rad_s, dtype=float) - np.asarray(self.gyro_bias_rad_s, dtype=float)
        mag = getattr(imu, 'mag_uT', None)
        mag_norm = None if mag is None else float(np.linalg.norm(np.asarray(mag, dtype=float)))
        heading = reference.update(time.monotonic(), yaw, gyro[2], mag_norm, self.mode == 'walk')
        pipeline = getattr(self, 'observation_pipeline', None)
        if pipeline is not None:
            pipeline.diagnostics.update(reference.diagnostics)
        sample = getattr(self.core, 'aligned_sample', None)
        if sample is not None:
            # The 50 ms resampled quaternion yaw carries the same field error.
            sample['yaw'] = heading
        return motor, imu, roll, pitch, heading

    def _walk_entry_allowed(self):
        pipeline = getattr(self, 'observation_pipeline', None)
        return super()._walk_entry_allowed() and (pipeline is None or
            pipeline.diagnostics.get('observation_temporal_ok', False))

    def arm_callback(self, request, response):
        pipeline = getattr(self, 'observation_pipeline', None)
        if request.data and pipeline is not None and not pipeline.diagnostics.get('observation_temporal_ok', False):
            response.success = False
            response.message = 'Observation timing gate not ready: ' + str(pipeline.scalar_diagnostics())
            return response
        return super().arm_callback(request, response)

    def _update_walk_inhibitors(self, now, odometry, q_policy):
        pipeline = getattr(self, 'observation_pipeline', None)
        if pipeline is None:
            return super()._update_walk_inhibitors(now, odometry, q_policy)
        alignment = getattr(self, 'common_time_alignment', None)
        if alignment is not None:
            self.core.aligned_gyro_bias = self._latest_base_gyro-self.corrected_gyro_rad_s
            if self.mode == 'walk' and not pipeline.diagnostics.get('observation_temporal_ok', False):
                self._enter_soft_hold('Common-time observation unavailable: '+
                    pipeline.diagnostics.get('observation_alignment_reason', 'stale reception'), now, q_policy)
                return True
        heading = self.heading_consistency_state
        recovery = self.heading_recovery.update(now, heading, self.mode == 'walk')
        self.core.actor.heading_correction_weight = recovery['heading_correction_weight']
        pipeline.diagnostics.update(recovery)
        # Heading offset stays in the direction controller. Yaw/gyro
        # disagreement must not latch soft stand. Timing and odometry still can.
        if pipeline.quality(now, float(odometry['confidence']), True, self.mode == 'walk'):
            self._enter_soft_hold('Observation quality persistently unreliable: ' + pipeline.diagnostics['obs_quality_stop_reason'], now, q_policy)
            return True
        if self.mode == 'walk' and recovery['heading_recovery_state'] in ('degraded', 'severe'):
            pipeline.diagnostics['obs_quality_ok'] = False
            if pipeline.diagnostics['obs_quality_state'] == 'ok':
                pipeline.diagnostics['obs_quality_state'] = recovery['heading_recovery_state']
        return False

    def __init__(self):
        self._periodic_target = LatestTarget(max_age=TIMING_SINGLE_GAP_SEC)
        self._send_target_age_ms = 0.
        super().__init__()
        self.declare_parameter('fast_commands', False)
        self.fast_commands = bool(self.get_parameter('fast_commands').value)
        self.declare_parameter('continuous_commands', False)
        self.continuous_commands = bool(self.get_parameter('continuous_commands').value)
        if self.continuous_commands:
            self.trial.lease_sec = float('inf')
        if self.fast_commands:
            self.trial.caps = np.array([.40, .30, .60])

    def _enqueue_send(self, target_real):
        now = time.perf_counter()
        # A watchdog holding an old target must not refresh a stalled policy's
        # walking target lease. The controller will soft-stop when it resumes.
        started = getattr(self, '_control_start', None)
        if self.mode == 'walk' and (started is None or now-started > TIMING_SINGLE_GAP_SEC):
            return
        with self._send_lock:
            self._periodic_target.put(np.asarray(target_real, dtype=np.float32), now)
        if self._send_pump is None or not self._send_pump.is_alive():
            return super()._enqueue_send(target_real)

    def _periodic_send_tick(self, now):
        if not self.enable_send or self.first_send:
            return
        with self._send_lock:
            target = self._periodic_target.get(now)
            stamp = self._periodic_target.stamp
            self._pending_target = target
        if target is None:
            if stamp is not None and self.trial.armed:
                self.trial.disarm('Control target stale: periodic sender requires target age <=50ms')
            return
        self._send_target_age_ms = (now-stamp)*1000.
        self._dispatch_pending_send()

    def _send_pump_loop(self):
        self._apply_thread_scheduling('send')
        period = float(self.contract.policy_dt)
        deadline = time.perf_counter()
        while not self._watchdog_stop.is_set():
            if self._watchdog_stop.wait(max(0., deadline-time.perf_counter())):
                break
            started = time.perf_counter()
            self._periodic_send_tick(started)
            deadline = next_deadline(deadline, max(time.perf_counter(), started+.008), period)

    def _extra_status(self):
        result = super()._extra_status()
        pipeline = getattr(self, 'observation_pipeline', None)
        result['status_publish_hz'] = 1. / self.telemetry_period_sec
        result['heading_correction_active'] = bool(self.core.actor.heading_correction_active())
        result['capture_policy_hz'] = 50.
        result['send_schedule'] = 'absolute_deadline_latest_target'
        result['send_target_age_ms'] = self._send_target_age_ms
        result['observation_pipeline_enabled'] = pipeline is not None
        if pipeline is not None:
            result.update(pipeline.scalar_diagnostics())
            result['imu_mount_rotation_base_from_imu'] = pipeline.rotation.tolist()

        result.update(continuous_commands=self.continuous_commands, fast_commands=self.fast_commands, command_caps=self.trial.caps.tolist(), model='B23500', direction_controller='v13_with_heading_confidence_degradation' if pipeline is not None else 'trained_v13_sensor_heading',
                      observation_contract='sensor_snapshot_61',
                      host_guard='policy_sensor_snapshot',
                      hardware_motion_validated=False)
        # B18000 disables planar correction. B23500 does not: do not publish
        # a false zero or imply that heading-based correction is a position sensor.
        result.pop('lateral_correction_mps', None)
        return result


def main(args=None):
    import sys
    sys.setswitchinterval(.001)
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model23500Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
