"""B23500 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
import time
from .rt_schedule import LatestTarget, next_deadline
from .observation_pipeline import ObservationPipeline
from .odometry_quality import support_quality
from .support_plane_odometry import SupportPlaneOdometry
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
        stationary = bool(self.mode == 'ready' and not self.trial.armed
                          and self.walk_start_stable and np.linalg.norm(gyro) <= self.walk_start_max_gyro)
        heading = reference.update(time.monotonic(), yaw, gyro[2], mag_norm,
                                   self.mode == 'walk', stationary=stationary)
        pipeline = getattr(self, 'observation_pipeline', None)
        if pipeline is not None:
            pipeline.diagnostics.update(reference.diagnostics)
        sample = getattr(self.core, 'aligned_sample', None)
        if sample is not None:
            # The 50 ms resampled quaternion yaw carries the same field error.
            sample['yaw'] = heading
        return motor, imu, roll, pitch, heading

    def _heading_ready_for_stand(self):
        reference = getattr(self, 'heading_reference', None)
        if reference is None:
            return super()._heading_ready_for_stand()
        # The fused-yaw monitor is diagnostic when gyro heading drives policy.
        # Freshness, quiet gyro, attitude and support gates remain independent.
        return bool(reference.heading is not None and np.isfinite(reference.heading))

    def _request_calibration(self, request, response):
        # Keep the last validated bias until a complete replacement is ready.
        # Pending calibration still blocks arm; cancelling does not erase it.
        calibrated = self.imu_calibrated
        result = super()._request_calibration(request, response)
        if result.success:
            self.imu_calibrated = calibrated
        return result

    def _on_gyro_calibration_complete(self, now, yaw, q):
        reference = getattr(self, 'heading_reference', None)
        if reference is not None and reference.heading is not None:
            # Bias replacement must not jump from gyro heading to magnetic yaw.
            yaw = reference.heading
            reference.last_time = now
        super()._on_gyro_calibration_complete(now, yaw, q)
        self.calibration_generation = getattr(self, 'calibration_generation', 0) + 1

    def _walk_entry_allowed(self):
        pipeline = getattr(self, 'observation_pipeline', None)
        return (not self.calibration_requested) and super()._walk_entry_allowed() and (pipeline is None or
            pipeline.diagnostics.get('observation_temporal_ok', False))

    def arm_callback(self, request, response):
        if request.data and self.calibration_requested:
            response.success = False
            response.message = 'IMU calibration in progress; wait for completion or cancel it'
            return response
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
        support_odometry=odometry
        support_source='live_body_axes'
        sample_age=None
        if alignment is not None:
            support_source='common_time_gravity'
            sample=getattr(self.core,'aligned_sample',None)
            support_odometry={}
            if sample is not None:
                sample_age=now-float(sample['timestamp'])
                if 0 <= sample_age <= self.core.common_time_max_age_sec:
                    if self._ground_support is None:
                        self._ground_support=SupportPlaneOdometry.from_estimator(self.walk_guard_odometry)
                    q,dq=self.mapper.real_to_policy_abs(sample['q_real'],sample['dq_real'])
                    support_odometry=self._ground_support.estimate_aligned(
                        q,dq,np.asarray(sample['gyro'])-self.core.aligned_gyro_bias,sample['gravity'])
        support = support_quality(support_odometry, self.walk_guard_odometry)
        pipeline.diagnostics.update(support)
        pipeline.diagnostics.update(obs_support_source=support_source,
            obs_support_sample_age_ms=None if sample_age is None else sample_age*1000.,
            obs_support_confidence=float(support_odometry.get('confidence',0.)),
            obs_support_selected_pair_index=int(support_odometry.get('selected_pair_index',-1)),
            obs_support_pair_residual_m_s=support_odometry.get('pair_residual_m_s'),
            obs_support_velocity=support_odometry.get('base_linear_velocity',np.zeros(3)).tolist())
        # Confidence measures velocity agreement, not contact probability.
        # This independent guard does not replace the actor's trained odometry
        # or confidence. A missing diagonal gets one slow gait cycle (600ms)
        # to recover; malformed velocity and timing retain their 200ms gate.
        if pipeline.quality(now, float(support_odometry.get('confidence',0.)), True, self.mode == 'walk',
                            odometry_support_usable=support['obs_odometry_support_usable'],
                            odometry_timeout_sec=.60 if support['obs_odometry_support_reason'] == 'no_legal_diagonal_pair' else .20):
            self._enter_soft_hold('Observation quality persistently unreliable: ' + pipeline.diagnostics['obs_quality_stop_reason'], now, q_policy)
            return True
        if self.mode == 'walk' and recovery['heading_recovery_state'] in ('degraded', 'severe'):
            pipeline.diagnostics['obs_quality_ok'] = False
            if pipeline.diagnostics['obs_quality_state'] == 'ok':
                pipeline.diagnostics['obs_quality_state'] = recovery['heading_recovery_state']
        return False

    def __init__(self):
        self._ground_support = None
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

    def _reset_walk_session(self, now, yaw, q_policy):
        if self._ground_support is not None:
            self._ground_support.reset()
        pipeline=getattr(self,'observation_pipeline',None)
        if pipeline is not None:
            pipeline.quality_since=dict(timing=None,odometry=None,heading=None)
            pipeline.low_confidence_since=None
        return super()._reset_walk_session(now,yaw,q_policy)

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
        result['deployment_revision'] = 'b23500-ready-20260927'
        cached = getattr(self, '_feedback_q', None)
        if cached is not None:
            errors = np.abs(np.asarray(cached)-self.contract.default)
            worst = int(np.argmax(errors))
            result['stand_max_error_rad'] = float(errors[worst])
            result['stand_worst_joint'] = self.contract.joint_names[worst]
            result['stand_required_error_rad'] = self.startup_ready_error
            result['stand_not_at_target_joints'] = [self.contract.joint_names[i]
                for i in np.flatnonzero(errors > self.startup_ready_error)]
        result['status_publish_hz'] = 1. / self.telemetry_period_sec
        result['heading_correction_active'] = bool(self.core.actor.heading_correction_active())
        result['calibration_requested'] = self.calibration_requested
        result['calibration_generation'] = getattr(self, 'calibration_generation', 0)
        result['gyro_bias_rad_s'] = np.asarray(self.gyro_bias_rad_s).tolist()
        result['heading_ready_for_stand'] = self._heading_ready_for_stand()
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
