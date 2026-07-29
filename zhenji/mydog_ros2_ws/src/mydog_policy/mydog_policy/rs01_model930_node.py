#!/usr/bin/env python3
"""Guarded ROS 2 deployment node for RS01 model_930."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
import requests
from std_msgs.msg import Float32MultiArray, String

from .imu_serial_interface import ImuSerialInterface
from .motor_state_interface import MotorStateHttpInterface
from .rs01_model930_core import (
    EXPECTED_ONNX_SHA256,
    Model930Contract,
    REAL_MOTOR_IDS,
    Rs01ContinuousTorqueGuard,
    Rs01Model930PolicyCore,
    Rs01NewMachineLegOdometry,
    estimate_stationary_gyro_bias,
)


class Rs01Model930Node(Node):
    """Run model_930 with exact observation and motor-routing semantics."""

    node_name = "rs01_model930_node"
    model_label = "model_930"
    model_filename = "model_930_rs01_heading52.onnx"
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model930Contract
    policy_core_type = Rs01Model930PolicyCore
    observation_count = 52
    topic_namespace = "/mydog/model930"
    include_path_state = False
    calibrate_gyro_bias = False

    def __init__(self):
        super().__init__(self.node_name)
        default_onnx = str(
            Path(get_package_share_directory("mydog_policy"))
            / "models"
            / self.model_filename
        )
        self.declare_parameter("onnx_path", default_onnx)
        self.declare_parameter(
            "expected_onnx_sha256", self.expected_onnx_sha256
        )
        self.declare_parameter(
            "motor_base_url", "http://127.0.0.1:8000"
        )
        self.declare_parameter("imu_port", "/dev/myimu")
        self.declare_parameter("enable_send", False)
        # First hardware launch must remain stand-only. Walking requires an
        # explicit override after the stand and sign checks pass.
        self.declare_parameter("stand_only", True)
        self.declare_parameter("require_online", True)
        self.declare_parameter("max_motor_age_ms", 100.0)
        self.declare_parameter("max_imu_age_sec", 0.10)
        self.declare_parameter("gyro_bias_calibration_sec", 5.0)
        self.declare_parameter("gyro_bias_max_abs_rad_s", 0.35)
        self.declare_parameter("gyro_calibration_max_std_rad_s", 0.05)
        self.declare_parameter(
            "gyro_calibration_max_rpy_span_rad", 0.08
        )
        self.declare_parameter("max_temperature_c", 70.0)
        self.declare_parameter("max_abs_roll_rad", 0.60)
        self.declare_parameter("max_abs_pitch_rad", 0.60)
        self.declare_parameter("command_timeout_sec", 0.50)
        self.declare_parameter("command_min_vx_mps", 0.21)
        self.declare_parameter("command_max_vx_mps", 0.25)
        self.declare_parameter("command_zero_threshold_mps", 0.02)
        self.declare_parameter("hardware_torque_limit_nm", 14.0)
        self.declare_parameter("continuous_torque_nm", 6.0)
        self.declare_parameter("thermal_derate_full_rms_nm", 8.0)
        self.declare_parameter("thermal_rms_time_constant_sec", 2.0)
        self.declare_parameter("hip_current_limit_amp", 12.0)
        self.declare_parameter("thigh_current_limit_amp", 12.0)
        self.declare_parameter("calf_current_limit_amp", 16.0)
        self.declare_parameter("startup_hold_sec", 0.50)
        self.declare_parameter("startup_ready_error_rad", 0.08)
        self.declare_parameter("startup_ready_hold_sec", 1.0)
        self.declare_parameter("startup_max_initial_error_rad", 1.80)
        self.declare_parameter("startup_hip_rate_rad_s", 0.12)
        self.declare_parameter("startup_thigh_rate_rad_s", 0.15)
        self.declare_parameter("startup_calf_rate_rad_s", 0.15)
        self.declare_parameter("low_odom_confidence_timeout_sec", 0.60)
        self.declare_parameter("http_timeout_sec", 0.08)
        self.declare_parameter("debug_csv_path", "")

        self.onnx_path = Path(
            str(self.get_parameter("onnx_path").value)
        ).expanduser().resolve()
        self.expected_sha256 = str(
            self.get_parameter("expected_onnx_sha256").value
        )
        self.motor_base_url = str(
            self.get_parameter("motor_base_url").value
        ).rstrip("/")
        self.enable_send = bool(
            self.get_parameter("enable_send").value
        )
        self.stand_only = bool(
            self.get_parameter("stand_only").value
        )
        self.require_online = bool(
            self.get_parameter("require_online").value
        )
        self.max_motor_age_ms = float(
            self.get_parameter("max_motor_age_ms").value
        )
        self.max_imu_age_sec = float(
            self.get_parameter("max_imu_age_sec").value
        )
        self.gyro_bias_calibration_sec = float(
            self.get_parameter("gyro_bias_calibration_sec").value
        )
        self.gyro_bias_max_abs = float(
            self.get_parameter("gyro_bias_max_abs_rad_s").value
        )
        self.gyro_calibration_max_std = float(
            self.get_parameter(
                "gyro_calibration_max_std_rad_s"
            ).value
        )
        self.gyro_calibration_max_rpy_span = float(
            self.get_parameter(
                "gyro_calibration_max_rpy_span_rad"
            ).value
        )
        self.max_temperature_c = float(
            self.get_parameter("max_temperature_c").value
        )
        self.max_abs_roll_rad = float(
            self.get_parameter("max_abs_roll_rad").value
        )
        self.max_abs_pitch_rad = float(
            self.get_parameter("max_abs_pitch_rad").value
        )
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )
        self.command_min_vx = float(
            self.get_parameter("command_min_vx_mps").value
        )
        self.command_max_vx = float(
            self.get_parameter("command_max_vx_mps").value
        )
        self.command_zero_threshold = float(
            self.get_parameter("command_zero_threshold_mps").value
        )
        self.active_torque_limit = float(
            self.get_parameter("hardware_torque_limit_nm").value
        )
        self.continuous_torque = float(
            self.get_parameter("continuous_torque_nm").value
        )
        self.http_timeout = float(
            self.get_parameter("http_timeout_sec").value
        )
        self.startup_hold_sec = float(
            self.get_parameter("startup_hold_sec").value
        )
        self.startup_ready_error = float(
            self.get_parameter("startup_ready_error_rad").value
        )
        self.startup_ready_hold_sec = float(
            self.get_parameter("startup_ready_hold_sec").value
        )
        self.startup_max_initial_error = float(
            self.get_parameter("startup_max_initial_error_rad").value
        )
        self.low_odom_timeout = float(
            self.get_parameter("low_odom_confidence_timeout_sec").value
        )
        if not 0.0 < self.active_torque_limit <= 17.0:
            raise RuntimeError(
                "hardware_torque_limit_nm must be in (0, 17]"
            )
        if not (
            0.0 < self.command_min_vx <= self.command_max_vx
        ):
            raise RuntimeError("Invalid positive command speed range")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers
        )
        self.contract = self.contract_type.from_onnx_session(
            self.session,
            self.onnx_path,
            expected_sha256=self.expected_sha256,
        )
        self.core = self.policy_core_type(
            self.session, self.contract
        )
        self.mapper = self.core.mapper
        self.leg_odometry = Rs01NewMachineLegOdometry()
        self.http = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=self.max_motor_age_ms,
            enable_stale_recheck=False,
            async_poll=True,
            poll_hz=50.0,
        )
        self.imu = ImuSerialInterface(
            port=str(self.get_parameter("imu_port").value),
            read_hz=100.0,
        )
        self.get_logger().info(f"Starting {self.model_label} IMU...")
        self.imu.start()
        if not self.imu.wait_until_ready(timeout=3.0):
            raise RuntimeError("IMU is not ready")
        self.gyro_bias_rad_s = np.zeros(3, dtype=np.float32)
        if self.calibrate_gyro_bias:
            self.gyro_bias_rad_s = self._calibrate_gyro_bias()
        self.corrected_gyro_rad_s = np.zeros(3, dtype=np.float32)

        self.kp_real = self.mapper.policy_values_to_real(
            self.contract.kp
        )
        self.kd_real = self.mapper.policy_values_to_real(
            self.contract.kd
        )
        self.active_limit_policy = np.minimum(
            np.full(12, self.active_torque_limit, dtype=np.float32),
            np.full(
                12,
                self.contract.peak_torque_limit,
                dtype=np.float32,
            ),
        )
        if abs(
            self.continuous_torque - self.contract.continuous_torque
        ) > 1.0e-6:
            raise RuntimeError(
                "continuous_torque_nm must match the ONNX RS01 contract "
                f"({self.contract.continuous_torque:.1f} Nm)"
            )
        self.torque_guard = Rs01ContinuousTorqueGuard(
            continuous_torque_nm=self.continuous_torque,
            peak_torque_nm=float(np.min(self.active_limit_policy)),
            derate_full_rms_nm=float(
                self.get_parameter(
                    "thermal_derate_full_rms_nm"
                ).value
            ),
            time_constant_s=float(
                self.get_parameter(
                    "thermal_rms_time_constant_sec"
                ).value
            ),
        )
        self.startup_rate = np.tile(
            np.asarray(
                [
                    float(
                        self.get_parameter(
                            "startup_hip_rate_rad_s"
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            "startup_thigh_rate_rad_s"
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            "startup_calf_rate_rad_s"
                        ).value
                    ),
                ],
                dtype=np.float32,
            ),
            4,
        )
        self.current_limit_real = np.tile(
            np.asarray(
                [
                    float(
                        self.get_parameter(
                            "hip_current_limit_amp"
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            "thigh_current_limit_amp"
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            "calf_current_limit_amp"
                        ).value
                    ),
                ],
                dtype=np.float32,
            ),
            4,
        )
        if (
            np.any(self.current_limit_real <= 0.0)
            or np.any(self.current_limit_real > 23.0)
        ):
            raise RuntimeError("RS01 current limits must be in (0, 23]")

        self.cmd_vx = 0.0
        self.last_command_time = 0.0
        self.command_active_last = False
        self.mode = "waiting_feedback"
        self.mode_start = time.monotonic()
        self.ready_since = None
        self.initialized = False
        self.first_send = True
        self.faulted = False
        self.stop_sent = False
        self.low_odom_since = None
        self.stand_target_policy = self.contract.default.copy()
        self.last_log_time = 0.0
        self.last_send_time = None
        self.last_loop_time = None

        self.pub_obs = self.create_publisher(
            Float32MultiArray,
            f"{self.topic_namespace}/observation",
            10,
        )
        self.pub_action = self.create_publisher(
            Float32MultiArray,
            f"{self.topic_namespace}/action",
            10,
        )
        self.pub_target = self.create_publisher(
            Float32MultiArray,
            f"{self.topic_namespace}/target_real",
            10,
        )
        self.pub_status = self.create_publisher(
            String,
            f"{self.topic_namespace}/status",
            10,
        )
        self.sub_cmd = self.create_subscription(
            Twist, "/cmd_vel", self.command_callback, 10
        )
        self.csv_handle = None
        self.csv_writer = None
        self.csv_rows = 0
        self._open_csv(
            str(self.get_parameter("debug_csv_path").value)
        )

        if self.enable_send:
            self._configure_verified_hardware_limits()
            self.get_logger().warn(
                "MOTOR SEND ENABLED: startup will prime live positions, "
                f"then ramp softly to the {self.model_label} standing pose."
            )
        else:
            self.get_logger().warn(
                "DRY RUN: enable_send=False; no motor command will be sent."
            )
        if self.stand_only:
            self.get_logger().warn(
                "stand_only=True; walking commands are intentionally ignored."
            )

        self.timer = self.create_timer(
            self.contract.policy_dt, self.control_loop
        )
        self.get_logger().info(
            f"RS01 {self.model_label} node ready | "
            f"onnx={self.onnx_path} | "
            f"{self.observation_count}->12 | "
            f"policy_order={list(self.contract.joint_names)} | "
            f"reversed_ids={[hex(x) for x in (0x11,0x13,0x21,0x22,0x32,0x43)]} | "
            f"torque_limit={self.active_torque_limit:.1f}Nm"
        )

    def command_callback(self, message: Twist):
        values = np.asarray(
            [
                message.linear.x,
                message.linear.y,
                message.angular.z,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            self.get_logger().error("Rejected non-finite /cmd_vel")
            return
        if abs(float(values[1])) > 0.01 or abs(float(values[2])) > 0.01:
            self.get_logger().error(
                f"{self.model_label} is straight-only: "
                "linear.y and angular.z must be zero"
            )
            return
        requested = float(values[0])
        if requested <= self.command_zero_threshold:
            self.cmd_vx = 0.0
            self.last_command_time = 0.0
            return
        self.cmd_vx = float(
            np.clip(
                requested, self.command_min_vx, self.command_max_vx
            )
        )
        self.last_command_time = time.monotonic()

    def command_active(self, now):
        return bool(
            not self.stand_only
            and self.last_command_time > 0.0
            and now - self.last_command_time <= self.command_timeout_sec
            and self.cmd_vx >= self.command_min_vx
        )

    def _calibrate_gyro_bias(self):
        duration = self.gyro_bias_calibration_sec
        if duration < 1.0:
            raise RuntimeError(
                "gyro_bias_calibration_sec must be at least 1 second"
            )
        self.get_logger().warn(
            f"Hold robot stationary for {duration:.1f}s gyro calibration; "
            "no motor command is sent during calibration."
        )
        deadline = time.monotonic() + duration
        gyro_samples = []
        rpy_samples = []
        last_stamp = None
        while time.monotonic() < deadline:
            snapshot = self.imu.get_latest()
            stamp = float(snapshot.stamp)
            age = time.time() - stamp
            if (
                snapshot.valid
                and snapshot.backend_alive
                and np.isfinite(age)
                and age <= self.max_imu_age_sec
                and stamp != last_stamp
            ):
                gyro = np.asarray(
                    snapshot.gyro_rad_s, dtype=np.float64
                ).reshape(3)
                rpy = np.radians(
                    np.asarray(
                        snapshot.rpy_deg, dtype=np.float64
                    ).reshape(3)
                )
                if np.all(np.isfinite(gyro)) and np.all(
                    np.isfinite(rpy)
                ):
                    gyro_samples.append(gyro)
                    rpy_samples.append(rpy)
                    last_stamp = stamp
            time.sleep(0.005)
        minimum_samples = max(20, int(duration * 20.0))
        if len(gyro_samples) < minimum_samples:
            raise RuntimeError(
                "Insufficient fresh IMU samples for gyro calibration: "
                f"{len(gyro_samples)} < {minimum_samples}"
            )
        bias = estimate_stationary_gyro_bias(
            gyro_samples,
            rpy_samples,
            max_std_rad_s=self.gyro_calibration_max_std,
            max_rpy_span_rad=self.gyro_calibration_max_rpy_span,
            max_abs_bias_rad_s=self.gyro_bias_max_abs,
        )
        self.get_logger().info(
            "Gyro bias calibrated: "
            f"{bias.tolist()} rad/s from {len(gyro_samples)} samples"
        )
        return bias

    def _configure_verified_hardware_limits(self):
        items = []
        for index, motor_id in enumerate(REAL_MOTOR_IDS):
            items.append(
                {
                    "motor_id": int(motor_id),
                    "torque_limit_nm": self.active_torque_limit,
                    "current_limit_amp": float(
                        self.current_limit_real[index]
                    ),
                }
            )
        response = self.http.post(
            f"{self.motor_base_url}"
            "/api/rs04/configure_verified_motion_safety_limits",
            json={"items": items},
            timeout=max(8.0, self.http_timeout),
        )
        if response.status_code != 200:
            raise RuntimeError(
                "RS01 verified safety handshake failed: "
                f"HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        if not payload.get("verified") or int(
            payload.get("count", 0)
        ) != 12:
            raise RuntimeError(
                f"RS01 safety handshake incomplete: {payload}"
            )

    def _fresh_state(self):
        motor = self.motor.get_latest()
        if not motor.valid:
            raise RuntimeError("motor snapshot invalid")
        age = np.asarray(motor.age_ms, dtype=np.float32)
        if not np.all(np.isfinite(age)):
            raise RuntimeError("motor age contains NaN/Inf")
        if float(np.max(age)) > self.max_motor_age_ms:
            raise RuntimeError(
                f"motor feedback stale: {float(np.max(age)):.1f}ms"
            )
        if self.require_online and not np.all(motor.online):
            offline = np.flatnonzero(~np.asarray(motor.online))
            raise RuntimeError(
                f"offline motor indices: {offline.tolist()}"
            )
        error_code = np.asarray(motor.error_code)
        if np.any(error_code != 0):
            bad = np.flatnonzero(error_code != 0)
            raise RuntimeError(
                f"motor fault indices: {bad.tolist()}"
            )
        temperature = np.asarray(motor.temp, dtype=np.float32)
        if not np.all(np.isfinite(temperature)):
            raise RuntimeError("motor temperature contains NaN/Inf")
        if float(np.max(temperature)) > self.max_temperature_c:
            raise RuntimeError(
                f"motor over-temperature: {float(np.max(temperature)):.1f}C"
            )

        imu = self.imu.get_latest()
        imu_age = time.time() - float(imu.stamp)
        if (
            not imu.valid
            or not imu.backend_alive
            or not np.isfinite(imu_age)
            or imu_age > self.max_imu_age_sec
        ):
            raise RuntimeError(
                f"IMU invalid/stale: age={imu_age:.3f}s "
                f"error={imu.backend_error!r}"
            )
        roll = math.radians(float(imu.rpy_deg[0]))
        pitch = math.radians(float(imu.rpy_deg[1]))
        yaw = math.radians(float(imu.rpy_deg[2]))
        if abs(roll) > self.max_abs_roll_rad:
            raise RuntimeError(f"roll safety limit: {roll:.3f}rad")
        if abs(pitch) > self.max_abs_pitch_rad:
            raise RuntimeError(f"pitch safety limit: {pitch:.3f}rad")
        return motor, imu, roll, pitch, yaw

    def _initialize_from_feedback(self, now, q_policy, yaw):
        max_error = float(
            np.max(np.abs(q_policy - self.contract.default))
        )
        if max_error > self.startup_max_initial_error:
            raise RuntimeError(
                "initial pose too far from model stand after semantic mapping: "
                f"{max_error:.3f}rad > {self.startup_max_initial_error:.3f}rad"
            )
        self.stand_target_policy = q_policy.copy()
        self.core.reset(now, yaw, q_policy=q_policy)
        self.torque_guard.reset()
        self.mode = "startup_hold"
        self.mode_start = now
        self.initialized = True

    def _stand_target(self, q_policy, dq_policy):
        max_step = self.startup_rate * self.contract.policy_dt
        delta = np.clip(
            self.contract.default - self.stand_target_policy,
            -max_step,
            max_step,
        )
        self.stand_target_policy = np.clip(
            self.stand_target_policy + delta,
            self.contract.lower,
            self.contract.upper,
        ).astype(np.float32)
        safe, torque_info = (
            self.core.limiter.pd_equivalent_peak_limit(
                self.stand_target_policy,
                q_policy,
                dq_policy,
                self.active_limit_policy,
            )
        )
        self.core.limiter.reset(safe)
        self.core.previous_action.fill(0.0)
        return safe, torque_info

    def _send_target(self, target_real):
        if not self.enable_send:
            return
        target_real = np.asarray(
            target_real, dtype=np.float32
        ).reshape(12)
        if not np.all(np.isfinite(target_real)):
            raise RuntimeError("target contains NaN/Inf")
        items = []
        for index, motor_id in enumerate(REAL_MOTOR_IDS):
            items.append(
                {
                    "motor_id": int(motor_id),
                    "position": float(target_real[index]),
                    "speed": 0.0,
                    "torque": 0.0,
                    "kp": float(self.kp_real[index]),
                    "kd": float(self.kd_real[index]),
                }
            )
        response = self.http.post(
            f"{self.motor_base_url}/api/rs04/motion_batch_fast",
            json={
                "items": items,
                "enable_first": bool(self.first_send),
                "stop_first": False,
                "require_hardware_torque_limits": True,
                "require_verified_hardware_safety_limits": True,
            },
            timeout=self.http_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"motor send HTTP {response.status_code}: {response.text}"
            )
        self.first_send = False
        self.last_send_time = time.monotonic()

    def _emergency_stop(self, reason):
        if self.faulted and self.stop_sent:
            return
        self.faulted = True
        self.mode = "fault"
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        if self.enable_send and not self.stop_sent:
            self.stop_sent = True
            try:
                self.http.post(
                    f"{self.motor_base_url}/api/stop?clear_error=false",
                    json={
                        "motor_ids": [
                            int(value) for value in REAL_MOTOR_IDS
                        ]
                    },
                    timeout=max(self.http_timeout, 0.2),
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Emergency stop request failed: {exc}"
                )

    def control_loop(self):
        if self.faulted:
            return
        now = time.monotonic()
        try:
            motor, imu, roll, pitch, yaw = self._fresh_state()
            self.corrected_gyro_rad_s = (
                np.asarray(imu.gyro_rad_s, dtype=np.float32)
                - self.gyro_bias_rad_s
            )
            if not np.all(np.isfinite(self.corrected_gyro_rad_s)):
                raise RuntimeError(
                    "bias-corrected gyro contains NaN/Inf"
                )
            q_policy, dq_policy = self.mapper.real_to_policy_abs(
                motor.q_real, motor.dq_real
            )
            odometry = self.leg_odometry.estimate(
                q_policy, dq_policy, self.corrected_gyro_rad_s
            )
            if not self.initialized:
                self._initialize_from_feedback(now, q_policy, yaw)

            command_active = self.command_active(now)
            # A command can arrive while the soft stand ramp is still active.
            # Keep the gait/path origin fresh until the node can actually enter
            # walk mode, otherwise a path-aware policy would start from a stale
            # integration timestamp and immediately trip its stale-state guard.
            if command_active and (
                not self.command_active_last or self.mode != "walk"
            ):
                self.core.reset(now, yaw, q_policy=q_policy)
            self.command_active_last = command_active

            action = np.zeros(12, dtype=np.float32)
            observation = np.zeros(
                self.observation_count, dtype=np.float32
            )
            if self.mode == "startup_hold":
                target_policy = q_policy.copy()
                torque_info = {
                    "raw_pd_torque_nm": np.zeros(12, dtype=np.float32),
                    "safe_pd_torque_nm": np.zeros(12, dtype=np.float32),
                    "limited_mask": np.zeros(12, dtype=bool),
                }
                if now - self.mode_start >= self.startup_hold_sec:
                    self.mode = "stand_ramp"
                    self.mode_start = now
            elif self.mode == "stand_ramp":
                target_policy, torque_info = self._stand_target(
                    q_policy, dq_policy
                )
                error = float(
                    np.max(
                        np.abs(q_policy - self.contract.default)
                    )
                )
                if error <= self.startup_ready_error:
                    if self.ready_since is None:
                        self.ready_since = now
                    elif now - self.ready_since >= self.startup_ready_hold_sec:
                        self.mode = "ready"
                        self.mode_start = now
                        self.core.reset(now, yaw, q_policy=q_policy)
                else:
                    self.ready_since = None
            elif command_active:
                self.mode = "walk"
                observation = self.core.build_observation(
                    now=now,
                    base_linear_velocity=odometry[
                        "base_linear_velocity"
                    ],
                    base_angular_velocity=self.corrected_gyro_rad_s,
                    projected_gravity=imu.projected_gravity,
                    command=np.asarray(
                        [self.cmd_vx, 0.0, 0.0], dtype=np.float32
                    ),
                    q_policy=q_policy,
                    dq_policy=dq_policy,
                    yaw=yaw,
                )
                result = self.core.step(
                    observation,
                    q_policy,
                    dq_policy,
                    self.active_limit_policy,
                )
                action = result["action"]
                target_policy = result["safe_target_policy"]
                torque_info = result["torque_info"]
                if odometry["confidence"] < 0.5:
                    if self.low_odom_since is None:
                        self.low_odom_since = now
                    elif now - self.low_odom_since > self.low_odom_timeout:
                        raise RuntimeError(
                            "leg odometry confidence remained below 0.5"
                        )
                else:
                    self.low_odom_since = None
            else:
                self.mode = "ready"
                target_policy, torque_info = self._stand_target(
                    q_policy, dq_policy
                )
                self.core.reset(now, yaw, q_policy=q_policy)
                self.low_odom_since = None

            target_real = self.mapper.policy_target_to_real(
                target_policy
            )
            measured_policy_abs = np.abs(
                np.asarray(
                    motor.torque, dtype=np.float32
                )[self.mapper.policy_to_real]
            )
            thermal_input = np.maximum(
                np.abs(torque_info["safe_pd_torque_nm"]),
                measured_policy_abs,
            )
            self.active_limit_policy = self.torque_guard.update(
                thermal_input, self.contract.policy_dt
            )
            self._send_target(target_real)
            self._publish(
                observation,
                action,
                target_real,
                odometry,
                roll,
                pitch,
                yaw,
                motor,
                torque_info,
                now,
            )
        except Exception as exc:
            self._emergency_stop(str(exc))

    def _publish(
        self,
        observation,
        action,
        target_real,
        odometry,
        roll,
        pitch,
        yaw,
        motor,
        torque_info,
        now,
    ):
        self._publish_array(self.pub_obs, observation)
        self._publish_array(self.pub_action, action)
        self._publish_array(self.pub_target, target_real)
        status = {
            "mode": self.mode,
            "send": bool(self.enable_send),
            "stand_only": bool(self.stand_only),
            "command_vx_mps": float(self.cmd_vx),
            "base_vx_mps": float(
                odometry["base_linear_velocity"][0]
            ),
            "odom_confidence": float(odometry["confidence"]),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "yaw_rad": float(yaw),
            "max_motor_age_ms": float(np.max(motor.age_ms)),
            "max_temperature_c": float(np.max(motor.temp)),
            "pd_limited_count": int(
                np.count_nonzero(torque_info["limited_mask"])
            ),
            "max_thermal_rms_nm": float(
                np.max(self.torque_guard.rms)
            ),
            "min_active_torque_limit_nm": float(
                np.min(self.active_limit_policy)
            ),
        }
        if self.include_path_state:
            status.update({
                "path_lateral_displacement_m": float(
                    self.core.last_path_lateral_displacement_m
                ),
                "path_lateral_velocity_m_s": float(
                    self.core.last_path_lateral_velocity_m_s
                ),
            })
        if self.calibrate_gyro_bias:
            status.update({
                "raw_gyro_z_rad_s": float(
                    self.corrected_gyro_rad_s[2]
                    + self.gyro_bias_rad_s[2]
                ),
                "corrected_gyro_z_rad_s": float(
                    self.corrected_gyro_rad_s[2]
                ),
                "gyro_bias_x_rad_s": float(
                    self.gyro_bias_rad_s[0]
                ),
                "gyro_bias_y_rad_s": float(
                    self.gyro_bias_rad_s[1]
                ),
                "gyro_bias_z_rad_s": float(
                    self.gyro_bias_rad_s[2]
                ),
            })
        message = String()
        message.data = json_dumps_compact(status)
        self.pub_status.publish(message)
        self._write_csv(
            now,
            status,
            observation,
            action,
            target_real,
            motor,
            torque_info,
        )
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            self.get_logger().info(message.data)

    def _open_csv(self, path):
        if not path:
            return
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.csv_handle = output.open("w", newline="", encoding="utf-8")
        headers = [
            "time_monotonic_s",
            "mode",
            "enable_send",
            "command_vx_mps",
            "base_vx_mps",
            "odom_confidence",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "max_motor_age_ms",
            "max_temperature_c",
            "max_thermal_rms_nm",
            "min_active_torque_limit_nm",
        ]
        if self.include_path_state:
            headers.extend([
                "path_lateral_displacement_m",
                "path_lateral_velocity_m_s",
            ])
        if self.calibrate_gyro_bias:
            headers.extend([
                "raw_gyro_z_rad_s",
                "corrected_gyro_z_rad_s",
                "gyro_bias_x_rad_s",
                "gyro_bias_y_rad_s",
                "gyro_bias_z_rad_s",
            ])
        for prefix in (
            "observation",
            "action",
            "target_real_rad",
            "q_real_rad",
            "dq_real_rad_s",
            "measured_torque_nm",
            "raw_pd_torque_nm",
            "safe_pd_torque_nm",
        ):
            count = (
                self.observation_count
                if prefix == "observation"
                else 12
            )
            headers.extend(
                f"{prefix}_{index}" for index in range(count)
            )
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow(headers)

    def _write_csv(
        self,
        now,
        status,
        observation,
        action,
        target_real,
        motor,
        torque_info,
    ):
        if self.csv_writer is None:
            return
        row = [
            f"{now:.6f}",
            self.mode,
            int(self.enable_send),
            status["command_vx_mps"],
            status["base_vx_mps"],
            status["odom_confidence"],
            status["roll_rad"],
            status["pitch_rad"],
            status["yaw_rad"],
            status["max_motor_age_ms"],
            status["max_temperature_c"],
            status["max_thermal_rms_nm"],
            status["min_active_torque_limit_nm"],
        ]
        if self.include_path_state:
            row.extend([
                status["path_lateral_displacement_m"],
                status["path_lateral_velocity_m_s"],
            ])
        if self.calibrate_gyro_bias:
            row.extend([
                status["raw_gyro_z_rad_s"],
                status["corrected_gyro_z_rad_s"],
                status["gyro_bias_x_rad_s"],
                status["gyro_bias_y_rad_s"],
                status["gyro_bias_z_rad_s"],
            ])
        for values in (
            observation,
            action,
            target_real,
            motor.q_real,
            motor.dq_real,
            motor.torque,
            torque_info["raw_pd_torque_nm"],
            torque_info["safe_pd_torque_nm"],
        ):
            row.extend(
                f"{float(value):.7g}"
                for value in np.asarray(values).reshape(-1)
            )
        self.csv_writer.writerow(row)
        self.csv_rows += 1
        if self.csv_rows % 50 == 0:
            self.csv_handle.flush()

    @staticmethod
    def _publish_array(publisher, values):
        message = Float32MultiArray()
        message.data = (
            np.asarray(values, dtype=np.float32).reshape(-1).tolist()
        )
        publisher.publish(message)

    def destroy_node(self):
        if self.enable_send and not self.stop_sent:
            self._emergency_stop("node shutdown")
        try:
            self.imu.stop()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass
        try:
            self.http.close()
        except Exception:
            pass
        if self.csv_handle is not None:
            self.csv_handle.flush()
            self.csv_handle.close()
        super().destroy_node()


def json_dumps_compact(value):
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model930Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
