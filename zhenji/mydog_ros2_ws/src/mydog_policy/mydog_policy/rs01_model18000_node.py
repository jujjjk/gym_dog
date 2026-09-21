"""B18000 on the upstream 5544692 asynchronous 50Hz hardware loop.

No auto-arm; startup stays dry-run/stand-only. A6850 and B18000 share the same
exclusive hardware lock. Timing failure disarms motion, keeping the old soft-stop chain.
"""
from collections import deque
import time
import threading
import numpy as np
import rclpy
from std_srvs.srv import SetBool
from .motor_rt_client import MotorRtClient
from .motor_rt_protocol import MotorCommand
from .rs01_model930_core import REAL_MOTOR_IDS, estimate_stationary_gyro_bias
from .rs01_model930_node import Rs01Model930Node
from .rs01_model6850_node import Rs01Model6850Node
from .rs01_model18000_core import Model18000Contract, Guarded18000PolicyCore, EXPECTED_ONNX_SHA256


def timing_window_ready(intervals):
    values=np.asarray(list(intervals),dtype=float)
    return bool(len(values)>=40 and np.isfinite(values).all()
                and .018<=np.median(values)<=.022 and np.percentile(values,95)<=.030
                and values.min()>=.008 and values.max()<=.040)


class Rs01Model18000Node(Rs01Model6850Node):
    calibrate_gyro_bias = False  # B: explicit calibration after supported standing.
    node_name='rs01_model18000_node'
    model_label='B18000 guarded omni'
    model_filename='B18000.onnx'
    expected_onnx_sha256=EXPECTED_ONNX_SHA256
    contract_type=Model18000Contract
    policy_core_type=Guarded18000PolicyCore
    topic_namespace='/mydog/model18000'
    command_topic='/mydog/model18000/cmd_vel'

    def __init__(self):
        self._calibration_lock = threading.RLock()
        self.imu_calibrated = False
        self.calibration_requested = False
        self.calibration_state = 'awaiting_supported_stand'
        self.calibration_reason = 'request_calibration_after_supported_stand'
        self.calibration_joint_span_rad = 0.
        self._calibration_samples = []
        self._rt = MotorRtClient()
        self._rt_spi_ms = self._rt_total_ms = 0.
        self._b_loop_intervals=deque(maxlen=50)
        self._b_send_intervals=deque(maxlen=50)
        self._b_last_success=None
        super().__init__()
        # Bound the operator-run 165s sequence plus the initial ready gate.
        # The 350ms command deadman and all motion guards remain unchanged.
        self.trial.lease_sec = 180.
        self.calibration_service = self.create_service(
            SetBool, self.topic_namespace + '/calibrate_imu', self.calibrate_callback)

    def calibrate_callback(self, request, response):
        with self._calibration_lock:
            return self._request_calibration(request, response)

    def _request_calibration(self, request, response):
        if self.mode != 'ready' or self.trial.armed:
            response.success = False
            response.message = 'Require unarmed ready stand; support robot first'
            return response
        self.imu_calibrated = False
        self.calibration_requested = bool(request.data)
        self._calibration_samples = []
        self.calibration_state = 'waiting_still' if request.data else 'cancelled'
        self.calibration_reason = 'collecting_stable_window' if request.data else 'operator_cancelled'
        self.calibration_joint_span_rad = 0.
        response.success = True
        response.message = 'Calibration requested; maintain supported still stand for five seconds' if request.data else 'Calibration cancelled'
        return response

    def _walk_entry_allowed(self):
        return self.imu_calibrated and self._b_timing_ready()

    def _prime_live_enable(self):
        # Verify the local protocol before enabling; this opcode is read-only.
        self._rt.exchange()
        return super()._prime_live_enable()

    def _fresh_state(self):
        with self._calibration_lock:
            try:
                return self._calibration_fresh_state()
            except Exception as exc:
                self._calibration_samples.clear()
                if self.calibration_requested:
                    self.calibration_state = 'waiting_fresh_feedback'
                    self.calibration_reason = str(exc)
                raise

    def _calibration_fresh_state(self):
        state = super()._fresh_state()
        if not self.calibration_requested:
            return state
        motor, imu, roll, pitch, yaw = state
        q, dq = self.mapper.real_to_policy_abs(motor.q_real, motor.dq_real)
        stamp = float(imu.stamp)
        gyro = np.asarray(imu.gyro_rad_s, dtype=float)
        rpy = np.radians(np.asarray(imu.rpy_deg, dtype=float))
        blockers = []
        if self.mode != 'ready' or self.trial.armed:
            blockers.append('require_unarmed_ready')
        if abs(roll) > .10 or abs(pitch) > .10:
            blockers.append('body_not_level')
        if np.max(np.abs(q-self.contract.default)) > self.startup_ready_error:
            blockers.append('joints_not_at_stand')
        if np.linalg.norm(gyro) > .08:
            blockers.append('gyro_motion')
        if not np.isfinite(np.r_[q, gyro, rpy, stamp]).all():
            blockers.append('nonfinite_feedback')
        if blockers:
            self._calibration_samples = []
            self.calibration_state = 'waiting_still'
            self.calibration_reason = ','.join(blockers)
            return state
        samples = self._calibration_samples
        if samples and (stamp < samples[-1][0] or stamp-samples[-1][0] > .08):
            samples.clear()
            self.calibration_reason = 'imu_frame_gap_or_regression'
            self.calibration_state = 'waiting_fresh_feedback'
            return state
        # Use actual encoder displacement over the whole calibration window.
        # Instantaneous velocity telemetry can be noisy while position is steady.
        # Check every control tick, even when the IMU frame is unchanged.
        positions = [s[3] for s in samples] + [q]
        self.calibration_joint_span_rad = float(np.ptp(positions, axis=0).max())
        if self.calibration_joint_span_rad > .02:
            samples.clear()
            self.calibration_state = 'waiting_still'
            self.calibration_reason = 'joint_position_span_exceeds_0.02rad'
            return state
        if not samples or stamp > samples[-1][0]:
            samples.append((stamp, gyro.copy(), rpy.copy(), q.copy()))
        self.calibration_state = 'collecting'
        self.calibration_reason = 'collecting_stable_window'
        if samples and (np.ptp(np.unwrap(np.array([s[2] for s in samples]), axis=0), axis=0).max()
                        > self.gyro_calibration_max_rpy_span):
            samples.clear()
            self.calibration_state = 'waiting_still'
            self.calibration_reason = 'body_orientation_changed'
        if len(samples) >= 100 and samples[-1][0]-samples[0][0] >= self.gyro_bias_calibration_sec:
            try:
                bias = estimate_stationary_gyro_bias(
                    [s[1] for s in samples], [s[2] for s in samples],
                    max_std_rad_s=self.gyro_calibration_max_std,
                    max_rpy_span_rad=self.gyro_calibration_max_rpy_span,
                    max_abs_bias_rad_s=self.gyro_bias_max_abs)
            except (ValueError, RuntimeError) as exc:
                samples.clear()
                self.calibration_state = 'waiting_still'
                self.calibration_reason = str(exc)
                return state
            self.gyro_bias_rad_s = np.asarray(bias, dtype=np.float32)
            Rs01Model930Node._reset_walk_session(self, time.monotonic(), yaw, q)
            self.heading_consistency.reset()
            self.walk_ready_since = None
            self.imu_calibrated = True
            self.calibration_requested = False
            self.calibration_state = 'complete'
            self.calibration_reason = ''
        return state

    def _dispatch_pending_send(self):
        with self._send_lock:
            target = self._pending_target
            self._pending_target = None
        if target is None or not self.enable_send:
            return
        started = time.perf_counter()
        try:
            if self.first_send:
                raise RuntimeError('RT send requires completed HTTP safety/enable handshake')
            commands = [MotorCommand(int(mid), float(target[i]), 0., 0.,
                                     float(self.kp_real[i]), float(self.kd_real[i]))
                        for i, mid in enumerate(REAL_MOTOR_IDS)]
            result = self._rt.exchange(commands)
            self.motor.snapshot_from_payload(result['payload'])
            now = time.perf_counter()
            if self._b_last_success is not None:
                self._b_send_intervals.append(now-self._b_last_success)
            self._b_last_success = now
            self.last_send_time = time.monotonic()
            self._rt_spi_ms, self._rt_total_ms = result['spi_ms'], result['total_ms']
        except Exception as exc:
            self.trial.disarm('RT send failed; no command retry/fallback')
            self._b_last_success = None
            self._b_send_intervals.clear()
            self.get_logger().error('RT send failed (enable held): ' + str(exc))
        self._send_dt_ms = (time.perf_counter()-started)*1000
        if self._last_send_perf is not None:
            self._send_hz = 1/max(started-self._last_send_perf, 1e-9)
        self._last_send_perf = started

    def _validate_deployment_parameters(self):
        super()._validate_deployment_parameters()
        for name,want in [('hardware_torque_limit_nm',14.),('continuous_torque_nm',6.),
                          ('thermal_derate_full_rms_nm',8.),('thermal_rms_time_constant_sec',2.)]:
            if abs(float(self.get_parameter(name).value)-want)>1e-6:
                raise RuntimeError('B18000 requires %s=%s'%(name,want))

    def _ingest_send_feedback(self, response=None):
        # Called only after the inherited HTTP send succeeds, not on failed attempts.
        now=time.perf_counter()
        if self._b_last_success is not None:
            self._b_send_intervals.append(now-self._b_last_success)
        self._b_last_success=now
        super()._ingest_send_feedback(response)

    def _b_timing_ready(self):
        return bool(timing_window_ready(self._b_loop_intervals)
                    and (not self.enable_send or
                         (timing_window_ready(self._b_send_intervals)
                          and self._b_last_success is not None
                          and time.perf_counter()-self._b_last_success<=.040)))

    def arm_callback(self, request, response):
        if request.data and not self.imu_calibrated:
            response.success=False
            response.message='B18000 requires supported-stand IMU calibration via /calibrate_imu'
            return response
        if request.data and not self._b_timing_ready():
            response.success=False
            response.message='B18000 requires measured ~50Hz control and successful sends for40 samples'
            return response
        return super().arm_callback(request,response)

    def control_loop(self):
        now=time.monotonic(); previous=self._previous_control
        if self.mode=='walk' and ((previous is not None and now-previous>.040)
                                  or not self._b_timing_ready()):
            self.trial.disarm('B18000 timing contract lost; operator re-arm required')
        super().control_loop()
        if previous is not None and self._previous_control!=previous:
            self._b_loop_intervals.append(self._previous_control-previous)

    def _extra_status(self):
        result=super()._extra_status()
        result.update(model='B18000',timing_ready=self._b_timing_ready(),
                      imu_calibrated=self.imu_calibrated,
                      calibration_state=self.calibration_state,
                      calibration_samples=len(self._calibration_samples),
                      trial_lease_sec=self.trial.lease_sec,
                      calibration_reason=self.calibration_reason,
                      calibration_joint_span_rad=self.calibration_joint_span_rad,
                      transport='unix_socket', rt_spi_ms=self._rt_spi_ms,
                      rt_server_ms=self._rt_total_ms,
                      hardware_motion_validated=False,
                      lateral_feedback_valid=False,
                      lateral_correction_mps=0.,
                      extra_compensation_enabled=False,
                      policy_target_mapping='conditional_inward_scale_v1')
        return result

    def destroy_node(self):
        try:
            return super().destroy_node()
        finally:
            self._rt.close()


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=Rs01Model18000Node();rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
