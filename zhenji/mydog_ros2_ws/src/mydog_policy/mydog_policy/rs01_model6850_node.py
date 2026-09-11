"""A6850 tethered hardware test entry; default dry-run and stand-only.

Uses the existing RS01 verified-limit/thermal/stand/stop chain, with a direct
HTTP + frame-stamped serial adapter. It does NOT claim acquisition-clock
parity with MuJoCo, and does not subscribe to legacy estimator arrays.
"""

import json
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .rs01_model6850_core import EXPECTED_ONNX_SHA256, Model6850Contract
from .rs01_model6850_guard import Guarded6850PolicyCore, ReceptionGuard, TrialCommand
from .rs01_model930_node import Rs01Model930Node
from .rs01_timestamped_imu import FrameStampedImu


class Rs01Model6850Node(Rs01Model930Node):
    node_name = 'rs01_model6850_node'
    model_label = 'A6850 guarded omni'
    model_filename = 'stand_only_6850.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model6850Contract
    policy_core_type = Guarded6850PolicyCore
    imu_interface_type = FrameStampedImu
    observation_count = 61
    topic_namespace = '/mydog/model6850'
    command_topic = '/mydog/model6850/cmd_vel'
    calibrate_gyro_bias = True
    strict_diagonal_odometry = True
    heading_consistency_enabled = True
    soft_inhibit_enabled = True

    def __init__(self):
        self.trial = TrialCommand()
        self.reception = ReceptionGuard()
        self._control_start = self._previous_control = None
        self._feedback_wall = None
        self._feedback_q = None
        self._status_log = None
        self._owner = None
        self._send_lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog = None
        self._stop_attempts = 0
        self._last_stop_attempt = float('-inf')
        self._last_target_real = None
        self._last_fault_reason = ''
        # Fail before opening any device if another A6850 instance exists.
        import fcntl
        self._owner = open('/tmp/mydog_a6850_controller.lock', 'a+')
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            self._owner = None
            raise RuntimeError('Another A6850 controller owns the robot')
        try:
            super().__init__()
            self.arm_service = self.create_service(
                SetBool, self.topic_namespace + '/arm', self.arm_callback)
            self._start_watchdog()
        except Exception:
            self._cleanup_partial()
            raise

    def _validate_deployment_parameters(self):
        # Hard ceilings for this initial test release; no launch override may
        # silently weaken the protection inherited from the standing node.
        caps = {
            'hardware_torque_limit_nm': 14., 'max_temperature_c': 70.,
            'max_motor_age_ms': 250., 'max_imu_age_sec': .25,
            'max_abs_roll_rad': .45, 'max_abs_pitch_rad': .45,
            'startup_hip_rate_rad_s': .12, 'startup_thigh_rate_rad_s': .15,
            'startup_calf_rate_rad_s': .15,
            'hip_current_limit_amp': 12., 'thigh_current_limit_amp': 12.,
            'calf_current_limit_amp': 16., 'http_timeout_sec': .04,
            'startup_max_initial_error_rad': 1.8,
            'startup_ready_error_rad': .12,
            'walk_start_max_abs_roll_rad': .10,
            'walk_start_max_abs_pitch_rad': .10,
            'walk_start_max_gyro_rad_s': .08,
            'walk_start_max_odom_speed_mps': .05,
            'low_odom_confidence_timeout_sec': .60,
            'heading_consistency_bad_hold_sec': .60,
            'gyro_calibration_max_std_rad_s': .05,
            'gyro_calibration_max_rpy_span_rad': .08,
        }
        for name, cap in caps.items():
            value = float(self.get_parameter(name).value)
            if not np.isfinite(value) or not 0 < value <= cap:
                raise RuntimeError(f'{name} must be in (0, {cap}] for A6850')
        if not self.require_online or self.gyro_bias_calibration_sec < 5.:
            raise RuntimeError('Online checks and five-second calibration are required')
        if self.startup_ready_hold_sec < 2. or self.walk_start_stable_sec < 1.:
            raise RuntimeError('A6850 requires two-second stand and one-second walk gates')
        if self.walk_start_min_odom_confidence < .5:
            raise RuntimeError('A6850 odometry confidence gate must be at least 0.5')

    def command_callback(self, message):
        values = [message.linear.x, message.linear.y, message.angular.z]
        unsupported = [message.linear.z, message.angular.x, message.angular.y]
        if not np.isfinite(unsupported).all() or np.any(np.abs(unsupported) > 1e-8):
            self.trial.disarm('unsupported Twist axes')
        elif self.stand_only:
            self.trial.disarm('stand_only')
        elif not self.trial.receive(values, time.monotonic()):
            self.get_logger().warning('Command rejected: ' + self.trial.reason)
        self.cmd_vx = float(self.trial.vector[0])

    def command_active(self, now):
        active = not self.stand_only and self.trial.active(now)
        self.cmd_vx = float(self.trial.vector[0])
        return active

    def _command_vector(self):
        return self.trial.vector.copy()

    def arm_callback(self, request, response):
        now = time.monotonic()
        if not request.data:
            self.trial.disarm('operator disarm')
            response.success = True
            response.message = 'Disarmed; control loop will return softly to stand'
        elif (self.stand_only or self.faulted or self.mode != 'ready'
              or not self.walk_start_stable or self.walk_inhibit_latched
              or self._control_start is None or now - self._control_start > .08):
            response.success = False
            response.message = 'Requires stand_only=false, fresh ready state and walk_start_stable=true'
        elif self.trial.armed:
            response.success = False
            response.message = 'Already armed; repeated arm cannot extend the trial'
        else:
            self.trial.arm(now)
            response.success = True
            response.message = (
                'Armed for continuous omni: zero velocity is step-in-place; '
                'keep publishing, or disarm to return to stand')
        return response

    def _fresh_state(self):
        state = super()._fresh_state()
        motor, imu, _, _, _ = state
        wall, mono = time.time(), time.monotonic()
        self.reception.check(motor, imu, wall, mono)
        q, dq = self.mapper.real_to_policy_abs(motor.q_real, motor.dq_real)
        if np.any(q < self.contract.lower) or np.any(q > self.contract.upper):
            raise RuntimeError('Measured joints outside A6850 URDF limits')
        if np.max(np.abs(dq)) >= self.contract.raw['v14']['speed_validity_limit_rad_s']:
            raise RuntimeError('Motor overspeed')
        self._feedback_wall = wall
        self._feedback_q = q.copy()
        # The inherited ready fallback retains an old stand target after a
        # walk. Explicitly start the stop ramp at LIVE feedback instead.
        if self.mode == 'walk' and not self.command_active(mono):
            self._enter_soft_hold(self.trial.reason, mono, q)
        return state

    def _reset_walk_session(self, now, yaw, q_policy):
        self.trial.start(now)
        super()._reset_walk_session(now, yaw, q_policy)

    def _enter_soft_hold(self, reason, now, q_policy):
        self.trial.disarm(reason)
        super()._enter_soft_hold(reason, now, q_policy)

    def _prime_live_enable(self):
        """Enable motors at the current pose before the 50 Hz send loop."""
        from .rs01_model930_core import REAL_MOTOR_IDS
        motor = self.motor.get_latest()
        q = np.asarray(motor.q_real, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(q)):
            raise RuntimeError('Cannot prime motors from invalid feedback')
        items = []
        for index, motor_id in enumerate(REAL_MOTOR_IDS):
            items.append({
                'motor_id': int(motor_id),
                'position': float(q[index]),
                'speed': 0.0,
                'torque': 0.0,
                'kp': float(self.kp_real[index]),
                'kd': float(self.kd_real[index]),
            })
        self.get_logger().warn(
            'Priming motors at live positions with a 2.0s HTTP budget; '
            'the 50 Hz loop stays at 40 ms afterwards.'
        )
        with self._send_lock:
            response = self.http.post(
                f'{self.motor_base_url}/api/rs04/motion_batch_fast',
                json={
                    'items': items,
                    'enable_first': True,
                    'stop_first': False,
                    'require_hardware_torque_limits': True,
                    'require_verified_hardware_safety_limits': True,
                },
                timeout=max(self.http_timeout, 2.0),
            )
        if response.status_code != 200:
            raise RuntimeError(
                f'motor prime HTTP {response.status_code}: {response.text}'
            )
        self.first_send = False
        self.last_send_time = time.monotonic()
        self._last_target_real = q.copy()
        self._wait_for_fresh_feedback()

    def _wait_for_fresh_feedback(self):
        # enable_first blocks the motor HTTP server; do not enter the 50 Hz
        # loop on the stale cache left behind by that round-trip.
        deadline = time.monotonic() + 0.5
        last_error = 'no fresh sample'
        while time.monotonic() < deadline:
            motor = self.motor.get_latest()
            imu = self.imu.get_latest()
            probe = type(self.reception)()
            try:
                probe.check(motor, imu, time.time(), time.monotonic())
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(0.01)
                continue
            if (probe.metrics['effective_motor_age_ms'] <= 50
                    and probe.metrics['imu_frame_age_ms'] <= 40):
                self.reception = type(self.reception)()
                return
            time.sleep(0.01)
        raise RuntimeError('No fresh motor/IMU sample after enable: ' + last_error)

    def control_loop(self):
        now = time.monotonic()
        if self.enable_send and self.first_send:
            try:
                self._prime_live_enable()
            except Exception as exc:
                self.get_logger().error(
                    'ENABLE HELD (prime retry, no disable): ' + str(exc))
                self._hold_target()
            self._previous_control = None
            return
        if self._previous_control is not None:
            dt = now - self._previous_control
            if dt < 0.008:
                return
            if dt > 0.080:
                self.get_logger().warning(
                    'Slow control dt=%.1fms; motors stay enabled' % (dt * 1000))
        self._previous_control = self._control_start = now
        super().control_loop()

    def _send_target(self, target_real):
        if not self.enable_send:
            return
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(target_real)):
            self.get_logger().error(
                'ENABLE HELD (non-finite target, no disable)')
            self._hold_target()
            return
        self._last_target_real = target_real.copy()
        if (self._control_start is not None
                and time.monotonic() - self._control_start > .040):
            self.get_logger().warning(
                'Compute >40ms; still sending to keep motors enabled')
        with self._send_lock:
            Rs01Model930Node._send_target(self, target_real)

    def _hold_target(self):
        target = self._last_target_real
        if target is None:
            try:
                target = np.asarray(
                    self.motor.get_latest().q_real, dtype=np.float32)
            except Exception:
                return
        if target is None or not np.all(np.isfinite(target)):
            return
        self._last_target_real = np.asarray(target, dtype=np.float32).reshape(12)
        try:
            with self._send_lock:
                Rs01Model930Node._send_target(self, self._last_target_real)
        except Exception as exc:
            self.get_logger().error(
                'Hold send failed (enable kept): ' + str(exc))

    def _start_watchdog(self):
        if not self.enable_send:
            return
        self._watchdog = threading.Thread(target=self._watchdog_loop,
                                          name='a6850-send-watchdog', daemon=True)
        self._watchdog.start()

    def _watchdog_check(self):
        if (self.enable_send and self.last_send_time is not None
                and time.monotonic() - self.last_send_time > .15):
            self.get_logger().warning(
                'Send gap >150ms; resending hold, motors stay enabled')
            self._hold_target()

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(.02):
            self._watchdog_check()

    def _extra_status(self):
        return dict(self.reception.metrics,
                    command_vy_mps=float(self.trial.vector[1]),
                    command_yaw_rad_s=float(self.trial.vector[2]),
                    trial_armed=self.trial.armed, trial_reason=self.trial.reason,
                    trial_elapsed_s=(0. if self.trial.started_at is None else
                                     time.monotonic() - self.trial.started_at),
                    last_fault_reason=self._last_fault_reason,
                    enable_held=True,
                    hardware_motion_validated=False)

    def _open_csv(self, path):
        super()._open_csv(path)
        if path:
            from pathlib import Path
            self._status_log = Path(path + '.status.jsonl').expanduser().open('w')

    def _write_csv(self, now, status, *args):
        super()._write_csv(now, status, *args)
        if self._status_log:
            self._status_log.write(json.dumps(dict(time_monotonic_s=now, **status)) + '\n')
            self._status_log.flush()

    def _emergency_stop(self, reason):
        # Keep MIT enable. Never POST /api/stop; resend the last PD target.
        reason = str(reason)
        self._last_fault_reason = reason
        self.get_logger().error('ENABLE HELD (no disable): ' + reason)
        self._hold_target()
        if self._status_log:
            self._status_log.write(json.dumps(dict(
                time_monotonic_s=time.monotonic(), mode=getattr(self, 'mode', ''),
                enable_held=True, reason=reason, send=bool(self.enable_send),
                stop_request_accepted=False)) + '\n')
            self._status_log.flush()

    def _cleanup_partial(self):
        # Covers failures after opening serial or configuring limits but before
        # the caller obtains a fully constructed node.
        self._watchdog_stop.set()
        if getattr(self, 'enable_send', False) and hasattr(self, 'http'):
            try:
                self.get_logger().error(
                    'ENABLE HELD after init error (no disable): initialization failed')
            except Exception:
                pass
            self._hold_target()
        for name, method in (('imu', 'stop'), ('motor', 'close'), ('http', 'close')):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:
                    pass
        for obj in (getattr(self, 'csv_handle', None), self._status_log, self._owner):
            if obj is not None:
                obj.close()
        self._owner = None

    def destroy_node(self):
        self._watchdog_stop.set()
        if self._watchdog:
            self._watchdog.join(timeout=.5)
        try:
            return super().destroy_node()
        finally:
            if self._status_log:
                self._status_log.close()
            if self._owner:
                self._owner.close()
                self._owner = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model6850Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
