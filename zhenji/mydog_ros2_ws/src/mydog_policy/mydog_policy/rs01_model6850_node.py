"""A6850 tethered hardware test entry; default dry-run and stand-only.

Uses the existing RS01 verified-limit/thermal/stand/stop chain, with a direct
HTTP + frame-stamped serial adapter. It does NOT claim acquisition-clock
parity with MuJoCo, and does not subscribe to legacy estimator arrays.
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .rs01_model6850_core import EXPECTED_ONNX_SHA256, Model6850Contract
from .rs01_model6850_guard import Guarded6850PolicyCore, ReceptionGuard, TrialCommand
from .rs01_model930_node import Rs01Model930Node
from .rs01_timestamped_imu import FrameStampedImu
from . import realtime


class Rs01Model6850Node(Rs01Model930Node):
    node_name = 'rs01_model6850_node'
    model_label = 'A6850 guarded omni'
    model_filename = 'stand_only_6850.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model6850Contract
    policy_core_type = Guarded6850PolicyCore
    imu_interface_type = FrameStampedImu
    observation_count = 61
    telemetry_period_sec = 0.0
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
        self._send_event = threading.Event()
        self._watchdog = None
        self._send_pump = None
        self._telemetry_pump = None
        self._control_thread = None
        self._pending_target = None
        self._pending_telemetry = None
        self._telemetry_lock = threading.Lock()
        self._telemetry_event = threading.Event()
        self._stop_attempts = 0
        self._last_stop_attempt = float('-inf')
        self._last_target_real = None
        self._last_fault_reason = ''
        self._control_dt_ms = 0.
        self._send_dt_ms = 0.
        self._compute_dt_ms = 0.
        self._publish_dt_ms = 0.
        self._status_log_rows = 0
        self._last_send_perf = None
        self._loop_hz = 0.
        self._send_hz = 0.
        self._q_policy_cached = None
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
            # Policy observations use the actor's own filter. The node only
            # needs the legal-diagonal guard estimator, so do not run a
            # second non-strict copy of the same four-leg FK.
            self.leg_odometry = self.walk_guard_odometry
            self.arm_service = self.create_service(
                SetBool, self.topic_namespace + '/arm', self.arm_callback)
            self._declare_realtime_parameters()
            self._start_watchdog()
        except Exception:
            self._cleanup_partial()
            raise

    def _declare_realtime_parameters(self):
        # Host scheduling only. Empty CPU lists and priority 0 leave the OS
        # defaults; refusals are reported in status, never fatal.
        self.declare_parameter('realtime_control_cpus', '')
        self.declare_parameter('realtime_send_cpus', '')
        self.declare_parameter('realtime_background_cpus', '')
        self.declare_parameter('realtime_fifo_priority', 0)
        self.declare_parameter('realtime_gil_switch_interval_sec', .001)
        self._rt_control_cpus = realtime.parse_cpus(self.get_parameter('realtime_control_cpus').value)
        self._rt_send_cpus = realtime.parse_cpus(self.get_parameter('realtime_send_cpus').value)
        self._rt_background_cpus = realtime.parse_cpus(self.get_parameter('realtime_background_cpus').value)
        self._rt_fifo_priority = int(self.get_parameter('realtime_fifo_priority').value)
        if not 0 <= self._rt_fifo_priority <= 95:
            raise RuntimeError('realtime_fifo_priority must be within [0, 95]')
        realtime.configure_interpreter(
            float(self.get_parameter('realtime_gil_switch_interval_sec').value))

    def _apply_thread_scheduling(self, role):
        cpus = {'control': self._rt_control_cpus, 'send': self._rt_send_cpus}.get(
            role, self._rt_background_cpus)
        fifo = {'control': self._rt_fifo_priority,
                'send': max(0, self._rt_fifo_priority - 10)}.get(role, 0)
        report = realtime.apply_to_current_thread(
            role, cpus=cpus, fifo_priority=fifo, nice=None if fifo else 5)
        self.get_logger().info('Thread scheduling %s: %s' % (role, report))

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
            now = time.monotonic()
            last = getattr(self, '_last_cmd_reject', 0.)
            if now - last >= 1.0:
                self._last_cmd_reject = now
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
        # Encoder noise at the fence is not a true URDF violation. 0.01 rad
        # is 0.6 deg; last hang-sag false trip was FR calf 1.920 vs 1.91986.
        slack = 0.01
        outside = (q < self.contract.lower - slack) | (q > self.contract.upper + slack)
        if np.any(outside):
            names = list(self.contract.joint_names)
            detail = ', '.join(
                '%s=%.3f' % (names[i], float(q[i]))
                for i in np.flatnonzero(outside))
            raise RuntimeError(
                'Measured joints outside A6850 URDF limits: ' + detail)
        if np.max(np.abs(dq)) >= self.contract.raw['v14']['speed_validity_limit_rad_s']:
            raise RuntimeError('Motor overspeed')
        self._feedback_wall = wall
        self._feedback_q = q.copy()
        self._q_policy_cached = (q, dq)
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

    def _update_walk_inhibitors(self, now, odometry, q_policy):
        # Keep walking while the trial is publishing. Heading/odom observers
        # remain in /status, but must not latch soft_hold: the sequence keeps
        # sending, and a latched hold would ignore those commands until it
        # stops. Command-timeout still returns to stand via _fresh_state.
        heading_state = getattr(self, 'heading_consistency_state', {}) or {}
        heading_bad = bool(
            self.heading_consistency_enabled
            and heading_state.get('ready')
            and not heading_state.get('healthy')
        )
        odom_bad = float(odometry['confidence']) < self.walk_start_min_odom_confidence
        if heading_bad or odom_bad:
            last = getattr(self, '_last_inhibit_note', 0.)
            if now - last >= 1.0:
                self._last_inhibit_note = now
                self.get_logger().warning(
                    'Walk observer note (policy continues): heading_healthy=%s '
                    'yaw_gyro_err=%+.3f odom_conf=%.2f' % (
                        heading_state.get('healthy'),
                        float(heading_state.get('mean_error_rad_s', 0.)),
                        float(odometry['confidence'])))
        return False

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
        if hasattr(self.motor, 'pause_async_poll'):
            self.motor.pause_async_poll()

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
            self._control_dt_ms = dt * 1000.
            self._loop_hz = (0. if dt <= 0 else 1. / dt)
            if dt > 0.080:
                self.get_logger().warning(
                    'Slow control dt=%.1fms; motors stay enabled' % (dt * 1000))
        self._previous_control = self._control_start = now
        started = time.perf_counter()
        super().control_loop()
        self._compute_dt_ms = (time.perf_counter() - started) * 1000.

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
                and time.monotonic() - self._control_start > .012):
            # Rate-limit console I/O; keep every-cycle timing in telemetry.
            now = time.monotonic()
            if now - getattr(self, '_last_compute_warning', float('-inf')) >= 1.:
                self._last_compute_warning = now
                self.get_logger().warning(
                    'Compute >12ms; still queueing send to keep 50 Hz')
        self._enqueue_send(target_real)

    def _enqueue_send(self, target_real):
        with self._send_lock:
            self._pending_target = np.asarray(
                target_real, dtype=np.float32).reshape(12).copy()
        if self._send_pump is None or not self._send_pump.is_alive():
            self._dispatch_pending_send()
        else:
            self._send_event.set()

    def _dispatch_pending_send(self):
        with self._send_lock:
            target = self._pending_target
            self._pending_target = None
        if target is None:
            return
        started = time.perf_counter()
        try:
            response = Rs01Model930Node._send_target(self, target)
            self._ingest_send_feedback(response)
        except Exception as exc:
            self.get_logger().error(
                'Send failed (enable kept): ' + str(exc))
        elapsed = time.perf_counter() - started
        self._send_dt_ms = elapsed * 1000.
        if self._last_send_perf is not None:
            gap = started - self._last_send_perf
            self._send_hz = (0. if gap <= 0 else 1. / gap)
        self._last_send_perf = started

    def _ingest_send_feedback(self, response=None):
        motor = getattr(self, 'motor', None)
        if motor is None:
            return
        payload = None
        if response is not None:
            try:
                payload = response.json()
            except Exception:
                payload = None
        from_payload = getattr(motor, 'snapshot_from_payload', None)
        if payload is not None and callable(from_payload):
            try:
                if from_payload(payload) is not None:
                    return
            except Exception:
                pass
        refresh = getattr(motor, 'refresh_latest', None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:
                self.get_logger().warning(
                    'Motor refresh after send failed: ' + str(exc))

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
        self._enqueue_send(self._last_target_real)

    def _start_send_pump(self):
        if not self.enable_send:
            return
        if self._send_pump is not None and self._send_pump.is_alive():
            return
        self._send_pump = threading.Thread(
            target=self._send_pump_loop, name='a6850-send-pump', daemon=True)
        self._send_pump.start()

    def _send_pump_loop(self):
        self._apply_thread_scheduling('send')
        while not self._watchdog_stop.is_set():
            with self._send_lock:
                pending = self._pending_target is not None
            if pending:
                self._dispatch_pending_send()
                continue
            self._send_event.wait(.02)
            self._send_event.clear()

    @staticmethod
    def _copy_mapping(value):
        copied = dict(value)
        for key, item in copied.items():
            if isinstance(item, np.ndarray):
                copied[key] = item.copy()
        return copied

    def _snapshot_telemetry(self, observation, action, target_real, odometry,
                            roll, pitch, yaw, motor, torque_info, now,
                            guard_odometry):
        return dict(
            observation=np.asarray(observation, dtype=np.float32).copy(),
            action=np.asarray(action, dtype=np.float32).copy(),
            target_real=np.asarray(target_real, dtype=np.float32).copy(),
            odometry=self._copy_mapping(odometry),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            motor=SimpleNamespace(
                age_ms=np.asarray(motor.age_ms, dtype=np.float32).copy(),
                temp=np.asarray(motor.temp, dtype=np.float32).copy(),
                q_real=np.asarray(motor.q_real, dtype=np.float32).copy(),
                dq_real=np.asarray(motor.dq_real, dtype=np.float32).copy(),
                torque=np.asarray(motor.torque, dtype=np.float32).copy(),
            ),
            torque_info=self._copy_mapping(torque_info),
            now=float(now),
            guard_odometry=self._copy_mapping(guard_odometry),
        )

    def _publish(self, observation, action, target_real, odometry, roll, pitch,
                 yaw, motor, torque_info, now, guard_odometry):
        # Capture61 has already queued the full-rate cycle before this call.
        # Reduce ROS/JSON/secondary CSV GIL contention without slowing control,
        # capture, fault checks, or successful motor sends.
        signature = (self.mode, self.trial.armed, self.walk_inhibit_reason,
                     self._last_fault_reason)
        due = now - getattr(self, '_last_telemetry_tick', float('-inf')) >= self.telemetry_period_sec
        if not due and signature == getattr(self, '_last_telemetry_signature', None):
            return
        self._last_telemetry_tick = now
        self._last_telemetry_signature = signature
        payload = self._snapshot_telemetry(
            observation, action, target_real, odometry, roll, pitch, yaw,
            motor, torque_info, now, guard_odometry)
        with self._telemetry_lock:
            self._pending_telemetry = payload
        if self._telemetry_pump is None or not self._telemetry_pump.is_alive():
            self._dispatch_pending_telemetry()
        else:
            self._telemetry_event.set()

    def _dispatch_pending_telemetry(self):
        with self._telemetry_lock:
            payload = self._pending_telemetry
            self._pending_telemetry = None
        if payload is None:
            return
        started = time.perf_counter()
        Rs01Model930Node._publish(
            self,
            payload['observation'],
            payload['action'],
            payload['target_real'],
            payload['odometry'],
            payload['roll'],
            payload['pitch'],
            payload['yaw'],
            payload['motor'],
            payload['torque_info'],
            payload['now'],
            payload['guard_odometry'],
        )
        self._publish_dt_ms = (time.perf_counter() - started) * 1000.

    def _start_telemetry_pump(self):
        if self._telemetry_pump is not None and self._telemetry_pump.is_alive():
            return
        self._telemetry_pump = threading.Thread(
            target=self._telemetry_pump_loop, name='a6850-telemetry', daemon=True)
        self._telemetry_pump.start()

    def _telemetry_pump_loop(self):
        self._apply_thread_scheduling('telemetry')
        idle_cycles = 0
        while not self._watchdog_stop.is_set():
            with self._telemetry_lock:
                pending = self._pending_telemetry is not None
            if pending:
                self._dispatch_pending_telemetry()
                continue
            self._telemetry_event.wait(.02)
            self._telemetry_event.clear()
            idle_cycles += 1
            if idle_cycles % 100 == 0:
                # Threads started after construction (IMU serial reader,
                # capture writer) also stay off the control cores.
                realtime.confine_other_threads(self._rt_background_cpus, nice=5)

    def _start_control_thread(self):
        timer = getattr(self, 'timer', None)
        if timer is not None:
            try:
                self.destroy_timer(timer)
            except Exception:
                pass
            self.timer = None
        if self._control_thread is not None and self._control_thread.is_alive():
            return
        self._control_thread = threading.Thread(
            target=self._control_thread_loop, name='a6850-control', daemon=True)
        self._control_thread.start()

    def _control_thread_loop(self):
        self._apply_thread_scheduling('control')
        period = float(self.contract.policy_dt)
        next_t = time.perf_counter()
        while not self._watchdog_stop.is_set():
            self.control_loop()
            finished = time.perf_counter()
            next_t += period
            if next_t < finished:
                next_t = finished
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._watchdog_stop.wait(delay)

    def _start_watchdog(self):
        # Everything allocated so far (ONNX session, ROS entities, contract)
        # lives for the whole run; keep it out of later collections.
        realtime.freeze_startup_objects()
        self._start_telemetry_pump()
        self._start_control_thread()
        if self.enable_send:
            self._watchdog = threading.Thread(target=self._watchdog_loop,
                                              name='a6850-send-watchdog', daemon=True)
            self._watchdog.start()
            self._start_send_pump()
        self.get_logger().info('Thread scheduling background: ' + realtime.confine_other_threads(
            self._rt_background_cpus, nice=5))

    def _watchdog_check(self):
        if (self.enable_send and self.last_send_time is not None
                and time.monotonic() - self.last_send_time > .15):
            self.get_logger().warning(
                'Send gap >150ms; resending hold, motors stay enabled')
            self._hold_target()

    def _watchdog_loop(self):
        self._apply_thread_scheduling('watchdog')
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
                    hardware_motion_validated=False,
                    loop_dt_ms=float(self._control_dt_ms),
                    loop_hz=float(self._loop_hz),
                    compute_dt_ms=float(self._compute_dt_ms),
                    publish_dt_ms=float(self._publish_dt_ms),
                    send_dt_ms=float(self._send_dt_ms),
                    send_hz=float(self._send_hz),
                    send_pump=bool(self._send_pump is not None
                                   and self._send_pump.is_alive()),
                    telemetry_pump=bool(self._telemetry_pump is not None
                                        and self._telemetry_pump.is_alive()),
                    control_thread=bool(self._control_thread is not None
                                        and self._control_thread.is_alive()),
                    realtime_scheduling=realtime.reports())

    def _open_csv(self, path):
        super()._open_csv(path)
        if path:
            from pathlib import Path
            self._status_log = Path(path + '.status.jsonl').expanduser().open('w')

    def _write_csv(self, now, status, *args):
        super()._write_csv(now, status, *args)
        if self._status_log:
            self._status_log.write(json.dumps(dict(time_monotonic_s=now, **status)) + '\n')
            self._status_log_rows += 1
            if self._status_log_rows % 50 == 0:
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
        if hasattr(self, '_send_event'):
            self._send_event.set()
        if hasattr(self, '_telemetry_event'):
            self._telemetry_event.set()
        for thread in (self._send_pump, self._telemetry_pump,
                       self._control_thread, self._watchdog):
            if thread is not None:
                try:
                    thread.join(timeout=.2)
                except Exception:
                    pass
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
        self._send_event.set()
        self._telemetry_event.set()
        if hasattr(self, 'motor') and hasattr(self.motor, 'resume_async_poll'):
            try:
                self.motor.resume_async_poll()
            except Exception:
                pass
        if self._send_pump:
            self._send_pump.join(timeout=.5)
        if self._telemetry_pump:
            self._telemetry_pump.join(timeout=.5)
        if self._control_thread:
            self._control_thread.join(timeout=.5)
        if self._watchdog:
            self._watchdog.join(timeout=.5)
        try:
            return super().destroy_node()
        finally:
            if self._status_log:
                try:
                    self._status_log.flush()
                except Exception:
                    pass
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
