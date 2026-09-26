"""B23500 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
import time
from .observation_pipeline import ObservationPipeline
from .observation_time_alignment import CommonTimeAlignment
from .heading_recovery import HeadingRecovery
from .rs01_timestamped_imu import QuaternionFrameStampedImu
from .motor_state_rt_interface import MotorStateRtInterface
import numpy as np
from .rs01_model18000_node import Rs01Model18000Node
from .rs01_model23500_core import Model23500Contract, Guarded23500PolicyCore, EXPECTED_ONNX_SHA256


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
            alignment.update(motor, reader() if reader is not None else {}, mono)
            pipeline.diagnostics['raw_reception_skew_ms'] = pipeline.diagnostics['aligned_sensor_skew_ms']
            pipeline.diagnostics.update(alignment.diagnostics)
            # In common_time this field describes the resampled policy data.
            # The original source span is retained as raw_reception_skew_ms.
            pipeline.diagnostics['aligned_sensor_skew_ms'] = alignment.diagnostics['observation_resampled_skew_ms']
            pipeline.diagnostics['observation_timing_mode'] = 'common_time'
            pipeline.diagnostics['observation_temporal_ok'] = bool(
                pipeline.diagnostics['observation_reception_ok'] and alignment.sample is not None)
            pipeline.diagnostics['observation_alignment_target_ok'] = alignment.sample is not None
            self.core.common_time_required = True
            self.core.aligned_sample = alignment.sample
            self._latest_base_gyro = result[1].gyro_rad_s.copy()
        # Keep current feedback for all PD, position, attitude and fault guards.
        return result

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
        super().__init__()
        self.declare_parameter('fast_commands', False)
        self.fast_commands = bool(self.get_parameter('fast_commands').value)
        self.declare_parameter('continuous_commands', False)
        self.continuous_commands = bool(self.get_parameter('continuous_commands').value)
        if self.continuous_commands:
            self.trial.lease_sec = float('inf')
        if self.fast_commands:
            self.trial.caps = np.array([.40, .30, .60])

    def _extra_status(self):
        result = super()._extra_status()
        pipeline = getattr(self, 'observation_pipeline', None)
        result['status_publish_hz'] = 1. / self.telemetry_period_sec
        result['capture_policy_hz'] = 50.
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
