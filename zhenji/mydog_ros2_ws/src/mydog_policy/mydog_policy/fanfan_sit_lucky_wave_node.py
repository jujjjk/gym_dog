#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fanfan sit + lucky-paw wave node.

Purpose:
- Hold zero pose first, ramp to default stand, then move into a manually tuned seated pose.
- Keep right front leg as the main front support.
- Lift left front leg and wave Joint-6 forward/back like a lucky cat paw.

Important:
- This node sends REAL motor-order joint targets directly by motor id, matching the
  Multi-Joint Position Mode Tuner order:
    J1  0x11, J2  0x12, J3  0x13,
    J4  0x21, J5  0x22, J6  0x23,
    J7  0x31, J8  0x32, J9  0x33,
    J10 0x41, J11 0x42, J12 0x43.
- No policy/semantic sign remapping is applied here.
- Kd is clamped to [0, 5.0].
- The sequence is zero hold -> default stand ramp -> stand hold -> sit ramp -> wave hold.
- All transitions are ramped and rate-limited to avoid a sudden protection event.
"""

import csv
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

try:
    from .motor_state_interface import MotorStateHttpInterface
except Exception:  # pragma: no cover
    MotorStateHttpInterface = None


MOTOR_IDS = [0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x31, 0x32, 0x33, 0x41, 0x42, 0x43]
JOINT_NAMES = [f"Joint-{i}" for i in range(1, 13)]

DEFAULT_ZERO_TARGET = np.zeros(12, dtype=np.float32)

# Targets read from your screenshot where visible. J4-J6 are the left-front waving leg;
# their defaults here are deliberately parameterized because your tuner screenshot did
# not show rows 4-6. The defaults are a reasonable raised-paw guess.
DEFAULT_SIT_TARGET = np.array([
    -0.104720,   # J1  RF/FR hip support, screenshot -6 deg
     0.436332,   # J2  RF/FR thigh support, screenshot 25 deg
    -0.872665,   # J3  RF/FR calf support, screenshot -50 deg
     0.104720,   # J4  LF/FL hip, guessed mirror-ish
    -0.950000,   # J5  LF/FL thigh lifted/bent, tune if needed
     1.150000,   # J6  LF/FL calf waving base, tune if needed
     0.157078,   # J7  RR/RH hip, screenshot 9 deg
    -0.593413,   # J8  RR/RH thigh, screenshot -34 deg
     1.762779,   # J9  RR/RH calf, screenshot 101 deg
    -0.157081,   # J10 RL/LH hip, screenshot -9 deg
     0.593413,   # J11 RL/LH thigh, screenshot 34 deg
    -1.762779,   # J12 RL/LH calf, screenshot -101 deg
], dtype=np.float32)

# Default stand target from your Multi-Joint Position Mode Tuner screenshot.
# REAL motor-order angles, no semantic sign remapping:
#   J1  0x11 = +0.157080 rad   J2  0x12 = +0.349066 rad   J3  0x13 = -0.785398 rad
#   J4  0x21 = -0.157080 rad   J5  0x22 = -0.349067 rad   J6  0x23 = +0.785398 rad
#   J7  0x31 = +0.157078 rad   J8  0x32 = -0.226897 rad   J9  0x33 = +0.349066 rad
#   J10 0x41 = -0.157080 rad   J11 0x42 = +0.226904 rad   J12 0x43 = -0.349066 rad
DEFAULT_STAND_FALLBACK = np.array([
     0.157080,  0.349066, -0.785398,
    -0.157080, -0.349067,  0.785398,
     0.157078, -0.226897,  0.349066,
    -0.157080,  0.226904, -0.349066,
], dtype=np.float32)


class FanfanSitLuckyWaveNode(Node):
    def __init__(self):
        super().__init__("fanfan_sit_lucky_wave_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("start_from_feedback", True)
        # New startup sequence: zero hold -> default stand -> sit/wave.
        self.declare_parameter("use_zero_start", True)
        self.declare_parameter("zero_hold_sec", 3.0)
        self.declare_parameter("stand_ramp_sec", 2.5)
        self.declare_parameter("stand_hold_sec", 0.8)
        self.declare_parameter("sit_ramp_sec", 2.8)
        self.declare_parameter("settle_sec", 0.5)
        self.declare_parameter("hold_forever", True)
        self.declare_parameter("return_to_start", False)
        self.declare_parameter("return_sec", 2.5)

        # Safety / smoothness.
        self.declare_parameter("max_target_rate_rad_s", 1.25)
        self.declare_parameter("max_delta", 0.35)
        self.declare_parameter("torque_warn_nm", 6.0)

        # Default gains.
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("default_kp", 40.0)
        self.declare_parameter("default_kd", 5.0)

        # Right-front support is stiffer; left-front waving leg is lighter.
        self.declare_parameter("fr_hip_kp", 42.0)
        self.declare_parameter("fr_thigh_kp", 50.0)
        self.declare_parameter("fr_calf_kp", 50.0)
        self.declare_parameter("fr_hip_kd", 5.0)
        self.declare_parameter("fr_thigh_kd", 5.0)
        self.declare_parameter("fr_calf_kd", 5.0)

        self.declare_parameter("fl_hip_kp", 28.0)
        self.declare_parameter("fl_thigh_kp", 52.0)
        self.declare_parameter("fl_calf_kp", 56.0)
        self.declare_parameter("fl_hip_kd", 4.6)
        self.declare_parameter("fl_thigh_kd", 4.2)
        self.declare_parameter("fl_calf_kd", 3.8)

        self.declare_parameter("rear_hip_kp", 36.0)
        self.declare_parameter("rear_thigh_kp", 46.0)
        self.declare_parameter("rear_calf_kp", 44.0)
        self.declare_parameter("rear_hip_kd", 5.0)
        self.declare_parameter("rear_thigh_kd", 5.0)
        self.declare_parameter("rear_calf_kd", 5.0)

        # Manually tuned seated target. These are REAL joint angles in tuner order.
        for idx, value in enumerate(DEFAULT_SIT_TARGET, start=1):
            self.declare_parameter(f"j{idx}_target", float(value))

        # Default standing target in the same REAL motor order.
        # It is used after the initial zero hold and before sitting down.
        for idx, value in enumerate(DEFAULT_STAND_FALLBACK, start=1):
            self.declare_parameter(f"stand_j{idx}_target", float(value))

        # Extra explicit right-front support and left-front raised-paw shaping.
        # They are added on top of j targets, so set to 0 if your manual pose is already final.
        self.declare_parameter("fr_support_extra_thigh_rad", 0.0)
        self.declare_parameter("fr_support_extra_calf_rad", 0.0)
        self.declare_parameter("fl_lift_extra_thigh_rad", 0.0)
        self.declare_parameter("fl_curl_extra_calf_rad", 0.0)

        # Wave Joint-6. Joint index is 1-based to match your tuner.
        self.declare_parameter("wave_enable", True)
        self.declare_parameter("wave_joint", 6)
        self.declare_parameter("wave_amp_rad", 0.18)
        self.declare_parameter("wave_hz", 1.05)
        self.declare_parameter("wave_phase_rad", 0.0)
        self.declare_parameter("wave_ramp_sec", 0.8)
        self.declare_parameter("wave_sign", 1.0)
        self.declare_parameter("wave_bias_rad", 0.0)

        # Optional coordinated small J5 motion. This makes the raised paw look less like a single-joint flap.
        self.declare_parameter("wave_coupled_thigh_enable", True)
        self.declare_parameter("wave_coupled_thigh_joint", 5)
        self.declare_parameter("wave_coupled_thigh_amp_rad", 0.055)
        self.declare_parameter("wave_coupled_thigh_sign", -1.0)

        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.02)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.start_from_feedback = bool(self.get_parameter("start_from_feedback").value)
        self.use_zero_start = bool(self.get_parameter("use_zero_start").value)
        self.zero_hold_sec = float(self.get_parameter("zero_hold_sec").value)
        self.stand_ramp_sec = float(self.get_parameter("stand_ramp_sec").value)
        self.stand_hold_sec = float(self.get_parameter("stand_hold_sec").value)
        self.sit_ramp_sec = float(self.get_parameter("sit_ramp_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.hold_forever = bool(self.get_parameter("hold_forever").value)
        self.return_to_start = bool(self.get_parameter("return_to_start").value)
        self.return_sec = float(self.get_parameter("return_sec").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.default_kp = float(self.get_parameter("default_kp").value)
        self.default_kd = self.clamp_kd(float(self.get_parameter("default_kd").value))

        target = np.array([float(self.get_parameter(f"j{i}_target").value) for i in range(1, 13)], dtype=np.float32)
        target[1] += float(self.get_parameter("fr_support_extra_thigh_rad").value)
        target[2] += float(self.get_parameter("fr_support_extra_calf_rad").value)
        target[4] += float(self.get_parameter("fl_lift_extra_thigh_rad").value)
        target[5] += float(self.get_parameter("fl_curl_extra_calf_rad").value)
        self.sit_target = target.astype(np.float32)
        self.zero_target = DEFAULT_ZERO_TARGET.copy()
        self.stand_target = np.array([float(self.get_parameter(f"stand_j{i}_target").value) for i in range(1, 13)], dtype=np.float32)

        self.wave_enable = bool(self.get_parameter("wave_enable").value)
        self.wave_joint = int(self.get_parameter("wave_joint").value) - 1
        self.wave_joint = int(min(11, max(0, self.wave_joint)))
        self.wave_amp_rad = float(self.get_parameter("wave_amp_rad").value)
        self.wave_hz = float(self.get_parameter("wave_hz").value)
        self.wave_phase_rad = float(self.get_parameter("wave_phase_rad").value)
        self.wave_ramp_sec = float(self.get_parameter("wave_ramp_sec").value)
        self.wave_sign = -1.0 if float(self.get_parameter("wave_sign").value) < 0.0 else 1.0
        self.wave_bias_rad = float(self.get_parameter("wave_bias_rad").value)
        self.wave_coupled_thigh_enable = bool(self.get_parameter("wave_coupled_thigh_enable").value)
        self.wave_coupled_thigh_joint = int(self.get_parameter("wave_coupled_thigh_joint").value) - 1
        self.wave_coupled_thigh_joint = int(min(11, max(0, self.wave_coupled_thigh_joint)))
        self.wave_coupled_thigh_amp_rad = float(self.get_parameter("wave_coupled_thigh_amp_rad").value)
        self.wave_coupled_thigh_sign = -1.0 if float(self.get_parameter("wave_coupled_thigh_sign").value) < 0.0 else 1.0

        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.http_session = requests.Session()
        self.motor = None
        if MotorStateHttpInterface is not None:
            self.motor = MotorStateHttpInterface(
                base_url=self.motor_base_url,
                timeout=self.http_timeout,
                stale_recheck_ms=self.debug_stale_recheck_ms,
            )

        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._last_send_info_time = 0.0
        self._last_torque_warn_time = 0.0
        self.start_target = self.read_start_pose()
        self.last_target = self.start_target.copy()

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_sit_lucky_wave_target", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_sit_lucky_wave_phase", 10)

        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self.setup_debug_csv()
        self.start_debug_collector()

        if self.enable_send:
            self.get_logger().warn("enable_send=True: moving to seated lucky-paw pose with ramp/rate limit. Keep hand near E-stop.")
        else:
            self.get_logger().warn("enable_send=False: dry run only.")
        self.get_logger().info(
            f"sit_lucky_wave: zero_hold={self.zero_hold_sec:.2f}s -> stand_ramp={self.stand_ramp_sec:.2f}s -> "
            f"stand_hold={self.stand_hold_sec:.2f}s -> sit_ramp={self.sit_ramp_sec:.2f}s; "
            f"wave Joint-{self.wave_joint + 1}, amp={self.wave_amp_rad:.3f} rad, hz={self.wave_hz:.2f}, "
            f"target={np.round(self.sit_target, 3).tolist()}"
        )

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-3), self.update)

    @staticmethod
    def clamp_kd(x: float) -> float:
        return float(min(5.0, max(0.0, x)))

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    def read_start_pose(self) -> np.ndarray:
        # For this node the normal requested startup is a true zero pose,
        # then a smooth ramp to the default stand pose.
        # If use_zero_start is false, fall back to feedback/current pose.
        if self.use_zero_start:
            self.get_logger().info("Start pose set to ZERO_TARGET. The node will hold zero before standing.")
            return DEFAULT_ZERO_TARGET.copy()

        if self.start_from_feedback and self.motor is not None:
            try:
                snapshot = self.motor.get_latest()
                q = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
                if np.all(np.isfinite(q)) and float(np.max(np.abs(q))) > 1e-6:
                    self.get_logger().info("Start pose loaded from real motor feedback.")
                    return q.copy()
            except Exception as exc:
                self.get_logger().warn(f"Could not read motor feedback for start pose; using fallback stand pose. reason={exc}")
        return DEFAULT_STAND_FALLBACK.copy()

    def gain_for_joint(self, i: int) -> tuple[float, float]:
        # i is 0-based in tuner order.
        if i == 0:
            return float(self.get_parameter("fr_hip_kp").value), self.clamp_kd(float(self.get_parameter("fr_hip_kd").value))
        if i == 1:
            return float(self.get_parameter("fr_thigh_kp").value), self.clamp_kd(float(self.get_parameter("fr_thigh_kd").value))
        if i == 2:
            return float(self.get_parameter("fr_calf_kp").value), self.clamp_kd(float(self.get_parameter("fr_calf_kd").value))
        if i == 3:
            return float(self.get_parameter("fl_hip_kp").value), self.clamp_kd(float(self.get_parameter("fl_hip_kd").value))
        if i == 4:
            return float(self.get_parameter("fl_thigh_kp").value), self.clamp_kd(float(self.get_parameter("fl_thigh_kd").value))
        if i == 5:
            return float(self.get_parameter("fl_calf_kp").value), self.clamp_kd(float(self.get_parameter("fl_calf_kd").value))
        # Rear joints.
        j_in_leg = i % 3
        if j_in_leg == 0:
            return float(self.get_parameter("rear_hip_kp").value), self.clamp_kd(float(self.get_parameter("rear_hip_kd").value))
        if j_in_leg == 1:
            return float(self.get_parameter("rear_thigh_kp").value), self.clamp_kd(float(self.get_parameter("rear_thigh_kd").value))
        return float(self.get_parameter("rear_calf_kp").value), self.clamp_kd(float(self.get_parameter("rear_calf_kd").value))

    def compute_target(self, elapsed: float) -> tuple[np.ndarray, str, float, float]:
        # Phase 1: hold zero for a few seconds.
        # This matches your requested workflow: zero -> default stand -> sit/wave.
        if self.use_zero_start and elapsed < self.zero_hold_sec:
            return self.zero_target.copy(), "ZERO_HOLD", 0.0, 0.0

        t = elapsed - (self.zero_hold_sec if self.use_zero_start else 0.0)

        # Phase 2: ramp from zero/current start pose to the default standing pose.
        if t < self.stand_ramp_sec:
            a = self.smootherstep(t / max(self.stand_ramp_sec, 1e-6))
            q = self.start_target + a * (self.stand_target - self.start_target)
            return q.astype(np.float32), "STAND_RAMP", a, 0.0

        # Phase 3: hold default stand briefly before sitting.
        t -= self.stand_ramp_sec
        if t < self.stand_hold_sec:
            return self.stand_target.copy(), "STAND_HOLD", 1.0, 0.0

        # Phase 4: ramp from default stand to manually tuned seated pose.
        t -= self.stand_hold_sec
        if t < self.sit_ramp_sec:
            a = self.smootherstep(t / max(self.sit_ramp_sec, 1e-6))
            q = self.stand_target + a * (self.sit_target - self.stand_target)
            return q.astype(np.float32), "SIT_RAMP", a, 0.0

        # Phase 5: hold seated pose and wave Joint-6.
        sit_elapsed = t - self.sit_ramp_sec
        q = self.sit_target.copy()
        wave_gate = self.smootherstep(sit_elapsed / max(self.wave_ramp_sec, 1e-6))
        wave_value = 0.0
        if self.wave_enable:
            wave_value = self.wave_sign * self.wave_amp_rad * wave_gate * math.sin(2.0 * math.pi * self.wave_hz * sit_elapsed + self.wave_phase_rad)
            q[self.wave_joint] += self.wave_bias_rad * wave_gate + wave_value
            if self.wave_coupled_thigh_enable:
                q[self.wave_coupled_thigh_joint] += (
                    self.wave_coupled_thigh_sign
                    * self.wave_coupled_thigh_amp_rad
                    * wave_gate
                    * math.sin(2.0 * math.pi * self.wave_hz * sit_elapsed + self.wave_phase_rad)
                )

        if self.hold_forever or not self.return_to_start:
            return q.astype(np.float32), "WAVE_HOLD", 1.0, wave_value

        if sit_elapsed < self.settle_sec:
            return q.astype(np.float32), "WAVE_HOLD", 1.0, wave_value

        # Optional return: return to default stand, not to zero.
        ret_t = sit_elapsed - self.settle_sec
        if ret_t < self.return_sec:
            r = self.smootherstep(ret_t / max(self.return_sec, 1e-6))
            q_ret = q + r * (self.stand_target - q)
            return q_ret.astype(np.float32), "RETURN_TO_STAND", 1.0 - r, wave_value
        return self.stand_target.copy(), "DONE", 0.0, 0.0

    def apply_rate_limit(self, q: np.ndarray, dt: float) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32).reshape(12)
        if dt <= 0.0 or self.max_target_rate_rad_s <= 0.0:
            self.last_target = q.copy()
            return q
        max_step = self.max_target_rate_rad_s * dt
        dq = np.clip(q - self.last_target, -max_step, max_step)
        out = self.last_target + dq
        self.last_target = out.astype(np.float32).copy()
        return self.last_target.copy()

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time
        q, state, alpha, wave_value = self.compute_target(elapsed)
        q = self.apply_rate_limit(q, dt)

        self.publish_array(self.pub_target, q)
        self.publish_array(self.pub_phase, np.array([elapsed, alpha, wave_value], dtype=np.float32))

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(q - self.last_target))) > self.max_delta * 2.0:
                self.get_logger().warn("[SAFE] target jump too large; skip send")
            else:
                sent = self.send_motion_batch(q)
        self.update_debug_sample(q, state, alpha, wave_value, sent)

    def send_motion_batch(self, q: np.ndarray) -> bool:
        items = []
        for i, (mid, pos) in enumerate(zip(MOTOR_IDS, q)):
            kp, kd = self.gain_for_joint(i)
            items.append({
                "motor_id": int(mid),
                "position": float(pos),
                "speed": self.send_speed,
                "torque": self.send_torque,
                "kp": float(kp),
                "kd": float(kd),
            })
        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json={"items": items, "enable_first": False, "stop_first": False},
                timeout=self.http_timeout,
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            now = time.time()
            if now - self._last_send_info_time > 1.0:
                self._last_send_info_time = now
                self.get_logger().info(
                    f"[SEND] sit wave ok | q_min={float(np.min(q)):.3f} q_max={float(np.max(q)):.3f} "
                    f"wave_joint=J{self.wave_joint + 1}"
                )
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def publish_array(self, pub, arr: np.ndarray):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in np.asarray(arr).reshape(-1)]
        pub.publish(msg)

    def setup_debug_csv(self):
        path = self.debug_csv_path.strip()
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._debug_csv_file = open(path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        self._debug_csv_writer.writerow([
            "time", "elapsed", "state", "alpha", "wave_value", "joint_index", "motor_id", "joint_name",
            "q_target", "q_current", "q_error", "torque_measured", "temp", "online", "error_code", "age_ms",
            "kp", "kd", "sent",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing sit lucky wave data to {path}")

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(target=self.debug_collect_loop, daemon=True)
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec if self.debug_csv_period_sec > 0.0 else 1.0 / max(self.rate_hz, 1e-3)
        while not self._debug_stop_event.wait(period):
            with self._debug_sample_lock:
                sample = self._latest_debug_sample
                if sample is not None:
                    sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
            if sample is not None:
                self.write_debug_csv_sample(**sample)

    def update_debug_sample(self, q, state, alpha, wave_value, sent):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "q": np.asarray(q, dtype=np.float32).reshape(12).copy(),
                "state": str(state),
                "alpha": float(alpha),
                "wave_value": float(wave_value),
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(self, q, state, alpha, wave_value, sent, stamp):
        if self._debug_csv_writer is None:
            return
        now = time.time()
        q = np.asarray(q, dtype=np.float32).reshape(12)
        q_real = np.full(12, np.nan, dtype=np.float32)
        torque = np.full(12, np.nan, dtype=np.float32)
        temp = np.full(12, np.nan, dtype=np.float32)
        online = np.zeros(12, dtype=bool)
        error_code = np.zeros(12, dtype=np.int32)
        age_ms = np.full(12, np.nan, dtype=np.float32)

        if self.motor is not None:
            try:
                snapshot = self.motor.get_latest()
                q_real = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
                torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
                temp = np.asarray(snapshot.temp, dtype=np.float32).reshape(12)
                online = np.asarray(snapshot.online, dtype=bool).reshape(12)
                error_code = np.asarray(snapshot.error_code, dtype=np.int32).reshape(12)
                age_ms = np.asarray(snapshot.age_ms, dtype=np.float32).reshape(12)
            except Exception:
                pass

        if self.torque_warn_nm > 0.0 and np.all(np.isfinite(torque)):
            torque_abs_max = float(np.max(np.abs(torque)))
            if torque_abs_max > self.torque_warn_nm and now - self._last_torque_warn_time > 1.0:
                self._last_torque_warn_time = now
                self.get_logger().warn(
                    f"[TORQUE] measured max |torque|={torque_abs_max:.2f}Nm > {self.torque_warn_nm:.2f}Nm"
                )

        elapsed = float(stamp) - self.start_time
        for i, (mid, name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
            kp, kd = self.gain_for_joint(i)
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", state, f"{alpha:.6f}", f"{wave_value:.6f}",
                i + 1, f"0x{int(mid):02X}", name,
                f"{float(q[i]):.6f}", f"{float(q_real[i]):.6f}", f"{float(q[i] - q_real[i]):.6f}",
                f"{float(torque[i]):.6f}", f"{float(temp[i]):.3f}", int(bool(online[i])), int(error_code[i]),
                f"{float(age_ms[i]):.2f}", f"{float(kp):.3f}", f"{float(kd):.3f}", int(bool(sent)),
            ])
        self._debug_csv_file.flush()

    def destroy_node(self):
        self._debug_stop_event.set()
        if self._debug_thread is not None:
            self._debug_thread.join(timeout=1.0)
        if self._debug_csv_file is not None:
            self._debug_csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FanfanSitLuckyWaveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
