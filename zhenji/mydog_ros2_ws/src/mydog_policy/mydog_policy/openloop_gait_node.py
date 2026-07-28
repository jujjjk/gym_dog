#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import csv
import os
import threading
import time

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


DEFAULT_STAND_POSE_REAL_ORDER = [
    (0x11, 0.157100),   # FR_hip
    (0x12, 0.349066),   # FR_thigh
    (0x13, -0.785400),  # FR_calf
    (0x21, -0.157100),  # FL_hip
    (0x22, -0.349067),  # FL_thigh
    (0x23, 0.785400),   # FL_calf
    (0x31, 0.157100),   # RL_hip
    (0x32, -0.226900),  # RL_thigh
    (0x33, 0.349065),   # RL_calf
    (0x41, -0.157100),  # RR_hip
    (0x42, 0.226900),   # RR_thigh
    (0x43, -0.349066),  # RR_calf
]


class MydogOpenLoopGaitNode(Node):
    """
    Open-loop motion generator for hardware bring-up and actuator data collection.

    This node deliberately does not use ONNX or estimator data. It sends a
    deterministic target around the training default pose so you can validate:
    motor order, joint signs, swing clearance, tracking quality, and excitation
    coverage before retraining with real actuator data.
    """

    def __init__(self):
        super().__init__("mydog_openloop_gait_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("gait_hz", 30.0)
        self.declare_parameter("step_hz", 1.0)
        self.declare_parameter("motion_mode", "trot")
        self.declare_parameter("stand_sec", 3.0)
        self.declare_parameter("warmup_sec", 2.0)

        self.declare_parameter("stand_kp", 45.0)
        self.declare_parameter("stand_kd", 5.0)
        self.declare_parameter("send_kp", 40.0)
        self.declare_parameter("send_kd", 5.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.0)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        # Gait shape in policy joint space. These are intentionally modest.
        self.declare_parameter("hip_amp", 0.0)
        self.declare_parameter("thigh_amp", 0.18)
        self.declare_parameter("calf_lift_amp", 0.60)
        self.declare_parameter("stance_calf_amp", 0.08)
        # On this hardware, negative stride sign produces forward motion.
        self.declare_parameter("stride_sign", -1.0)
        self.declare_parameter("duty_factor", 0.60)
        self.declare_parameter("sweep_min_hz", 0.35)
        self.declare_parameter("sweep_max_hz", 2.50)
        self.declare_parameter("sweep_period_sec", 20.0)
        self.declare_parameter("excitation_hip_amp", 0.08)
        self.declare_parameter("excitation_thigh_amp", 0.18)
        self.declare_parameter("excitation_calf_amp", 0.18)
        self.declare_parameter("squat_thigh_amp", 0.08)
        self.declare_parameter("squat_calf_amp", 0.18)
        self.declare_parameter("max_delta", 1.50)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.motion_mode = str(self.get_parameter("motion_mode").value).strip().lower()
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = float(self.get_parameter("stand_kd").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.hip_amp = float(self.get_parameter("hip_amp").value)
        self.thigh_amp = float(self.get_parameter("thigh_amp").value)
        self.calf_lift_amp = float(self.get_parameter("calf_lift_amp").value)
        self.stance_calf_amp = float(self.get_parameter("stance_calf_amp").value)
        self.stride_sign = float(self.get_parameter("stride_sign").value)
        self.duty_factor = float(self.get_parameter("duty_factor").value)
        self.sweep_min_hz = float(self.get_parameter("sweep_min_hz").value)
        self.sweep_max_hz = float(self.get_parameter("sweep_max_hz").value)
        self.sweep_period_sec = float(self.get_parameter("sweep_period_sec").value)
        self.excitation_hip_amp = float(self.get_parameter("excitation_hip_amp").value)
        self.excitation_thigh_amp = float(self.get_parameter("excitation_thigh_amp").value)
        self.excitation_calf_amp = float(self.get_parameter("excitation_calf_amp").value)
        self.squat_thigh_amp = float(self.get_parameter("squat_thigh_amp").value)
        self.squat_calf_amp = float(self.get_parameter("squat_calf_amp").value)
        self.max_delta = float(self.get_parameter("max_delta").value)

        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=self.debug_stale_recheck_ms,
        )
        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.default_real = np.array(
            [pos for _, pos in DEFAULT_STAND_POSE_REAL_ORDER],
            dtype=np.float32,
        )
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
        self._current_drive_hz = self.step_hz
        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._last_debug_csv_time = 0.0
        self._last_feedback_warn_time = 0.0
        self._last_send_info_time = 0.0
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self.setup_debug_csv()
        self.start_debug_collector()

        self.pub_target = self.create_publisher(
            Float32MultiArray,
            "/mydog/openloop_target_real",
            10,
        )
        self.pub_phase = self.create_publisher(
            Float32MultiArray,
            "/mydog/openloop_phase",
            10,
        )

        self.get_logger().warn(
            "Open-loop gait has no balance feedback. Start with the robot supported "
            "or lightly held, and be ready to cut power."
        )

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only, no motor commands sent.")

        self.get_logger().info(
            f"Standing for {self.stand_sec:.2f}s before gait. "
            f"mode={self.motion_mode}, step_hz={self.step_hz:.2f}, gait_hz={self.gait_hz:.1f}, "
            f"thigh_amp={self.thigh_amp:.3f}, calf_lift_amp={self.calf_lift_amp:.3f}, "
            f"stride_sign={self.stride_sign:+.1f}, enable_send={self.enable_send}, "
            f"debug_csv_path={self.debug_csv_path!r}"
        )

        self.timer = self.create_timer(1.0 / self.gait_hz, self.update)

    def send_default_stand(self):
        items = []
        for mid, pos in DEFAULT_STAND_POSE_REAL_ORDER:
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(pos),
                    "speed": 0.0,
                    "torque": 0.0,
                    "kp": self.stand_kp,
                    "kd": self.stand_kd,
                }
            )

        payload = {
            "items": items,
            "enable_first": True,
            "stop_first": False,
        }
        url = f"{self.motor_base_url}/api/rs04/motion_mode_run_batch"
        r = self.http_session.post(url, json=payload, timeout=max(self.http_timeout, 0.5))
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")

        self.get_logger().info(
            f"Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}"
        )

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time
        if elapsed < self.stand_sec:
            target_policy = self.default_policy.copy()
            target_real = self.default_real.copy()
            phase_value = 0.0
            warm = 0.0
            self._phase_acc = 0.0
            self._current_drive_hz = self.step_hz
        else:
            gait_time = elapsed - self.stand_sec
            self._current_drive_hz = self.get_drive_frequency(gait_time)
            self._phase_acc = (self._phase_acc + self._current_drive_hz * dt) % 1.0
            phase_value = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 1e-3))
            target_policy = self.build_target_policy(phase_value, warm, gait_time)
            target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True)

        self.publish_array(self.pub_target, target_real)
        self.publish_array(
            self.pub_phase,
            np.array([phase_value, warm, self._current_drive_hz], dtype=np.float32),
        )

        sent = False
        if not self.enable_send:
            self.update_debug_sample(target_real, target_policy, phase_value, warm, sent)
            return

        delta = target_real - self.last_target_real
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self.max_delta:
            self.get_logger().warn(
                f"[SAFE] open-loop target jump too large: "
                f"{max_delta:.3f} rad > {self.max_delta:.3f} rad. Skip send."
            )
            self.update_debug_sample(target_real, target_policy, phase_value, warm, sent)
            return

        sent = self.send_motion_batch(target_real)
        if sent:
            self.last_target_real = target_real.copy()
        self.update_debug_sample(target_real, target_policy, phase_value, warm, sent)

    def get_drive_frequency(self, gait_time: float) -> float:
        if self.motion_mode in ("excitation", "sine_sweep", "sweep", "chirp"):
            period = max(self.sweep_period_sec, 1e-3)
            u = (gait_time % period) / period
            triangle = 1.0 - abs(2.0 * u - 1.0)
            return self.sweep_min_hz + (self.sweep_max_hz - self.sweep_min_hz) * triangle
        return self.step_hz

    def build_target_policy(self, phase: float, warm: float, gait_time: float) -> np.ndarray:
        if self.motion_mode == "trot":
            return self.build_gait_target_policy(phase, warm, self.gait_offsets_trot())
        if self.motion_mode == "pace":
            return self.build_gait_target_policy(phase, warm, self.gait_offsets_pace())
        if self.motion_mode == "bound":
            return self.build_gait_target_policy(phase, warm, self.gait_offsets_bound())
        if self.motion_mode == "pronk":
            return self.build_gait_target_policy(phase, warm, self.gait_offsets_pronk())
        if self.motion_mode == "walk":
            return self.build_gait_target_policy(phase, warm, self.gait_offsets_walk())
        if self.motion_mode == "squat":
            return self.build_squat_target_policy(phase, warm)
        if self.motion_mode in ("excitation", "sine_sweep", "sweep", "chirp"):
            return self.build_excitation_target_policy(phase, warm, gait_time)

        self.get_logger().warn(f"Unknown motion_mode={self.motion_mode!r}; falling back to trot.")
        self.motion_mode = "trot"
        return self.build_gait_target_policy(phase, warm, self.gait_offsets_trot())

    @staticmethod
    def gait_offsets_trot():
        return {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}

    @staticmethod
    def gait_offsets_pace():
        return {"FR": 0.0, "FL": 0.5, "RR": 0.0, "RL": 0.5}

    @staticmethod
    def gait_offsets_bound():
        return {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5}

    @staticmethod
    def gait_offsets_pronk():
        return {"FR": 0.0, "FL": 0.0, "RR": 0.0, "RL": 0.0}

    @staticmethod
    def gait_offsets_walk():
        # Swing starts when (phase + offset) wraps back to zero.
        # This gives a crawl-like FR -> RL -> FL -> RR sequence.
        return {"FR": 0.00, "RL": 0.75, "FL": 0.50, "RR": 0.25}

    def build_gait_target_policy(self, phase: float, warm: float, leg_offsets: dict, duty_factor: float = None) -> np.ndarray:
        q = self.default_policy.copy()

        duty = self.duty_factor if duty_factor is None else duty_factor
        duty = min(max(float(duty), 0.50), 0.90)
        swing_fraction = max(0.05, 1.0 - duty)
        leg_start = {
            "FR": 0,
            "FL": 3,
            "RR": 6,
            "RL": 9,
        }

        for leg, offset in leg_offsets.items():
            p = (phase + offset) % 1.0
            if p < swing_fraction:
                s = p / swing_fraction
                swing = math.sin(math.pi * s)
                stance = 0.0
                stride = -1.0 + 2.0 * s
            else:
                s = (p - swing_fraction) / max(1.0 - swing_fraction, 1e-3)
                swing = 0.0
                stance = math.sin(math.pi * s)
                stride = 1.0 - 2.0 * s

            i = leg_start[leg]
            q[i + 0] += warm * self.hip_amp * stride
            q[i + 1] += warm * self.stride_sign * self.thigh_amp * stride

            # More negative calf bends the knee and helps foot clearance.
            q[i + 2] += warm * (
                -self.calf_lift_amp * swing
                + self.stance_calf_amp * stance
            )

        return q.astype(np.float32)

    def build_squat_target_policy(self, phase: float, warm: float) -> np.ndarray:
        q = self.default_policy.copy()
        bob = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        leg_starts = (0, 3, 6, 9)
        for i in leg_starts:
            q[i + 1] += warm * self.squat_thigh_amp * bob
            q[i + 2] += warm * -self.squat_calf_amp * bob
        return q.astype(np.float32)

    def build_excitation_target_policy(self, phase: float, warm: float, gait_time: float) -> np.ndarray:
        q = self.default_policy.copy()
        joint_amp = np.array(
            [
                self.excitation_hip_amp,
                self.excitation_thigh_amp,
                self.excitation_calf_amp,
            ],
            dtype=np.float32,
        )
        offsets = np.array([0.00, 0.17, 0.31, 0.46, 0.59, 0.73, 0.11, 0.29, 0.43, 0.67, 0.79, 0.91])
        signs = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1], dtype=np.float32)
        for i in range(12):
            amp = float(joint_amp[i % 3])
            p1 = 2.0 * math.pi * (phase + offsets[i])
            p2 = 2.0 * math.pi * (1.73 * phase + 0.37 * offsets[i] + 0.03 * gait_time)
            q[i] += warm * amp * float(signs[i]) * (0.70 * math.sin(p1) + 0.30 * math.sin(p2))
        return q.astype(np.float32)

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(target_real[i]),
                    "speed": self.send_speed,
                    "torque": self.send_torque,
                    "kp": self.send_kp,
                    "kd": self.send_kd,
                }
            )

        payload = {
            "items": items,
            "enable_first": False,
            "stop_first": False,
        }

        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json=payload,
                timeout=self.http_timeout,
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            now = time.time()
            if now - self._last_send_info_time > 1.0:
                self._last_send_info_time = now
                self.get_logger().info(
                    f"[SEND] open-loop ok | "
                    f"target_min={float(np.min(target_real)):.3f} "
                    f"target_max={float(np.max(target_real)):.3f} "
                    f"kp={self.send_kp:.1f} kd={self.send_kd:.1f}"
                )
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def setup_debug_csv(self):
        path = self.debug_csv_path.strip()
        if not path:
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._debug_csv_file = open(path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        self._debug_csv_writer.writerow(
            [
                "time",
                "elapsed",
                "phase",
                "warm",
                "motion_mode",
                "drive_hz",
                "sent",
                "joint_index",
                "motor_id",
                "joint_name",
                "q_target_real",
                "q_last_sent_real",
                "q_target_policy_abs",
                "q_current_real",
                "dq_current_real",
                "q_error_real",
                "torque_measured",
                "temp",
                "online",
                "error_code",
                "age_ms",
                "feedback_ts",
                "feedback_latency_ms",
                "mode_state",
                "snapshot_seq",
                "board_tick_ms",
                "kp",
                "kd",
                "send_speed",
                "send_torque",
                "gait_hz",
                "step_hz",
                "hip_amp",
                "thigh_amp",
                "calf_lift_amp",
                "stance_calf_amp",
                "stride_sign",
            ]
        )
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing actuator data to {path}")

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(
            target=self.debug_collect_loop,
            name="openloop_actuator_logger",
            daemon=True,
        )
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec
        if period <= 0.0:
            period = 1.0 / max(self.gait_hz, 1e-3)

        while not self._debug_stop_event.wait(period):
            with self._debug_sample_lock:
                sample = self._latest_debug_sample
                if sample is not None:
                    sample = {
                        key: (value.copy() if isinstance(value, np.ndarray) else value)
                        for key, value in sample.items()
                    }
            if sample is None:
                continue
            self.write_debug_csv_sample(**sample)

    def update_debug_sample(
        self,
        target_real: np.ndarray,
        target_policy: np.ndarray,
        phase: float,
        warm: float,
        sent: bool,
    ):
        if self._debug_csv_writer is None:
            return

        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "target_real": np.asarray(target_real, dtype=np.float32).reshape(12).copy(),
                "target_policy": np.asarray(target_policy, dtype=np.float32).reshape(12).copy(),
                "last_sent_real": np.asarray(self.last_target_real, dtype=np.float32).reshape(12).copy(),
                "phase": float(phase),
                "warm": float(warm),
                "motion_mode": str(self.motion_mode),
                "drive_hz": float(self._current_drive_hz),
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(
        self,
        target_real: np.ndarray,
        target_policy: np.ndarray,
        last_sent_real: np.ndarray,
        phase: float,
        warm: float,
        motion_mode: str,
        drive_hz: float,
        sent: bool,
        stamp: float,
    ):
        if self._debug_csv_writer is None:
            return

        now = time.time()
        self._last_debug_csv_time = now

        try:
            snapshot = self.motor.get_latest()
        except Exception as exc:
            if now - self._last_feedback_warn_time > 1.0:
                self._last_feedback_warn_time = now
                self.get_logger().warn(f"[DEBUG_CSV] motor feedback read failed: {exc}")
            return

        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        last_sent_real = np.asarray(last_sent_real, dtype=np.float32).reshape(12)
        q_real = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
        dq_real = np.asarray(snapshot.dq_real, dtype=np.float32).reshape(12)
        torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
        temp = np.asarray(snapshot.temp, dtype=np.float32).reshape(12)
        online = np.asarray(snapshot.online, dtype=bool).reshape(12)
        error_code = np.asarray(snapshot.error_code, dtype=np.int32).reshape(12)
        age_ms = np.asarray(snapshot.age_ms, dtype=np.float32).reshape(12)
        feedback_ts = np.asarray(snapshot.last_update_ts, dtype=np.float64).reshape(12)
        mode_state = np.asarray(snapshot.mode_state, dtype=np.int32).reshape(12)
        snapshot_seq = np.asarray(snapshot.snapshot_seq, dtype=np.int32).reshape(12)
        board_tick_ms = np.asarray(snapshot.board_tick_ms, dtype=np.int64).reshape(12)

        # Convert policy-order target into real motor order for row-wise comparison.
        target_policy_real_order = np.zeros(12, dtype=np.float32)
        target_policy_real_order[self.mapper.policy_to_real_index] = target_policy

        elapsed = float(stamp) - self.start_time
        for i, (mid, name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            self._debug_csv_writer.writerow(
                [
                    f"{now:.6f}",
                    f"{elapsed:.6f}",
                    f"{float(phase):.6f}",
                    f"{float(warm):.6f}",
                    str(motion_mode),
                    f"{float(drive_hz):.6f}",
                    int(bool(sent)),
                    int(i),
                    f"0x{int(mid):02X}",
                    name,
                    f"{float(target_real[i]):.6f}",
                    f"{float(last_sent_real[i]):.6f}",
                    f"{float(target_policy_real_order[i]):.6f}",
                    f"{float(q_real[i]):.6f}",
                    f"{float(dq_real[i]):.6f}",
                    f"{float(target_real[i] - q_real[i]):.6f}",
                    f"{float(torque[i]):.6f}",
                    f"{float(temp[i]):.3f}",
                    int(online[i]),
                    int(error_code[i]),
                    f"{float(age_ms[i]):.3f}",
                    f"{float(feedback_ts[i]):.6f}",
                    f"{float((now - feedback_ts[i]) * 1000.0):.3f}",
                    int(mode_state[i]),
                    int(snapshot_seq[i]),
                    int(board_tick_ms[i]),
                    f"{self.send_kp:.6f}",
                    f"{self.send_kd:.6f}",
                    f"{self.send_speed:.6f}",
                    f"{self.send_torque:.6f}",
                    f"{self.gait_hz:.6f}",
                    f"{self.step_hz:.6f}",
                    f"{self.hip_amp:.6f}",
                    f"{self.thigh_amp:.6f}",
                    f"{self.calf_lift_amp:.6f}",
                    f"{self.stance_calf_amp:.6f}",
                    f"{self.stride_sign:.6f}",
                ]
            )
        self._debug_csv_file.flush()

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def destroy_node(self):
        try:
            self._debug_stop_event.set()
            if self._debug_thread is not None:
                self._debug_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._debug_csv_file is not None:
                self._debug_csv_file.flush()
                self._debug_csv_file.close()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass
        try:
            self.http_session.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MydogOpenLoopGaitNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
