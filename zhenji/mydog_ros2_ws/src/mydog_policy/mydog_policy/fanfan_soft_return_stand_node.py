#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fanfan soft return-to-stand node.

Purpose
-------
After any open-loop pose/gait/play action leaves the robot frozen in a non-stand pose,
run this node to return the robot to a known default standing pose very softly.

Design notes
------------
- Uses REAL motor-order targets directly, matching the Multi-Joint Position Tuner order:
    J1  0x11, J2  0x12, J3  0x13,
    J4  0x21, J5  0x22, J6  0x23,
    J7  0x31, J8  0x32, J9  0x33,
    J10 0x41, J11 0x42, J12 0x43.
- Reads current motor feedback as the starting pose when available.
- Returns in staged layers to reduce current spikes / oscillation:
    0) hold current pose briefly
    1) hips gently align first, with low Kp
    2) calves partially untuck / reduce extreme folding, still soft
    3) thighs and calves move together to stand
    4) final settle at stand with normal Kp
- Kd is clamped to [0, 5.0].
- Intended as an emergency-friendly "soft park" behavior, not a walking gait.
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
HIP_INDEX = np.array([0, 3, 6, 9], dtype=np.int32)
THIGH_INDEX = np.array([1, 4, 7, 10], dtype=np.int32)
CALF_INDEX = np.array([2, 5, 8, 11], dtype=np.int32)

# Default stand from your tuner-style low/normal stand that was previously used.
# You can override every value with -p j1_stand:=... etc.
DEFAULT_STAND = np.array([
     0.157080,   # J1  0x11
     0.349066,   # J2  0x12
    -0.785398,   # J3  0x13
    -0.157080,   # J4  0x21
    -0.349067,   # J5  0x22
     0.785398,   # J6  0x23
     0.157078,   # J7  0x31
    -0.226897,   # J8  0x32
     0.349066,   # J9  0x33
    -0.157080,   # J10 0x41
     0.226904,   # J11 0x42
    -0.349066,   # J12 0x43
], dtype=np.float32)

# Safe fallback if feedback cannot be read.
FALLBACK_CURRENT = np.array([
    0.0, 1.0472, -2.0944,
    0.0, -1.0472, 2.0944,
    0.0, -1.0472, 1.5708,
    0.0, 1.0472, -2.0944,
], dtype=np.float32)


class FanfanSoftReturnStandNode(Node):
    def __init__(self):
        super().__init__("fanfan_soft_return_stand_node")

        # Communication / timing.
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("start_from_feedback", True)
        self.declare_parameter("hold_current_sec", 0.60)
        self.declare_parameter("hip_align_sec", 1.20)
        self.declare_parameter("calf_half_sec", 1.60)
        self.declare_parameter("main_return_sec", 3.20)
        self.declare_parameter("settle_sec", 1.20)
        self.declare_parameter("hold_after_done", True)
        self.declare_parameter("shutdown_after_done", False)

        # Safety / smoothness.
        self.declare_parameter("max_target_rate_rad_s", 0.85)
        self.declare_parameter("max_delta", 0.24)
        self.declare_parameter("torque_warn_nm", 6.0)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        # Sending defaults.
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)

        # Stage gains. Keep early stages soft to prevent oscillation.
        self.declare_parameter("hold_kp", 24.0)
        self.declare_parameter("hold_kd", 5.0)
        self.declare_parameter("hip_stage_kp", 26.0)
        self.declare_parameter("hip_stage_kd", 5.0)
        self.declare_parameter("calf_stage_kp", 28.0)
        self.declare_parameter("calf_stage_kd", 5.0)
        self.declare_parameter("main_kp", 34.0)
        self.declare_parameter("main_kd", 5.0)
        self.declare_parameter("settle_kp", 40.0)
        self.declare_parameter("settle_kd", 4.2)

        # Optional group scaling in final/main stage.
        self.declare_parameter("hip_kp_scale", 0.85)
        self.declare_parameter("thigh_kp_scale", 1.00)
        self.declare_parameter("calf_kp_scale", 0.95)

        # Stand target in REAL motor order.
        for i, value in enumerate(DEFAULT_STAND, start=1):
            self.declare_parameter(f"j{i}_stand", float(value))

        # Intermediate calf release ratio: how far calves move toward final stand before thighs move fully.
        self.declare_parameter("calf_half_ratio", 0.45)

        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.02)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.start_from_feedback = bool(self.get_parameter("start_from_feedback").value)
        self.hold_current_sec = float(self.get_parameter("hold_current_sec").value)
        self.hip_align_sec = float(self.get_parameter("hip_align_sec").value)
        self.calf_half_sec = float(self.get_parameter("calf_half_sec").value)
        self.main_return_sec = float(self.get_parameter("main_return_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.hold_after_done = bool(self.get_parameter("hold_after_done").value)
        self.shutdown_after_done = bool(self.get_parameter("shutdown_after_done").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.calf_half_ratio = min(1.0, max(0.0, float(self.get_parameter("calf_half_ratio").value)))

        self.stand_target = np.array(
            [float(self.get_parameter(f"j{i}_stand").value) for i in range(1, 13)],
            dtype=np.float32,
        )

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
        self.start_pose = self.read_start_pose()
        self.last_target = self.start_pose.copy()
        self.current_phase = "INIT"

        # Precompute intermediate targets.
        self.hip_target = self.start_pose.copy()
        self.hip_target[HIP_INDEX] = self.stand_target[HIP_INDEX]

        self.calf_half_target = self.hip_target.copy()
        self.calf_half_target[CALF_INDEX] = (
            self.start_pose[CALF_INDEX]
            + self.calf_half_ratio * (self.stand_target[CALF_INDEX] - self.start_pose[CALF_INDEX])
        )

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_soft_return_stand_target", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_soft_return_stand_phase", 10)

        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self.setup_debug_csv()
        self.start_debug_collector()

        if self.enable_send:
            self.get_logger().warn("enable_send=True: softly returning robot to default stand. Keep hand near E-stop.")
        else:
            self.get_logger().warn("enable_send=False: dry run only.")
        self.get_logger().info(
            "soft_return_stand target=" + np.array2string(self.stand_target, precision=3, separator=",")
        )

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-3), self.update)

    @staticmethod
    def clamp_kd(x: float) -> float:
        return float(min(5.0, max(0.0, x)))

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    @staticmethod
    def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        alpha = FanfanSoftReturnStandNode.smootherstep(alpha)
        return (a + alpha * (b - a)).astype(np.float32)

    def read_start_pose(self) -> np.ndarray:
        if self.start_from_feedback and self.motor is not None:
            try:
                snapshot = self.motor.get_latest()
                q = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
                if np.all(np.isfinite(q)) and float(np.max(np.abs(q))) > 1e-6:
                    self.get_logger().info("Start pose loaded from real motor feedback.")
                    return q.copy()
            except Exception as exc:
                self.get_logger().warn(f"Could not read feedback, using fallback current pose. reason={exc}")
        return FALLBACK_CURRENT.copy()

    def compute_target(self, elapsed: float) -> tuple[np.ndarray, str, float]:
        t0 = self.hold_current_sec
        t1 = t0 + self.hip_align_sec
        t2 = t1 + self.calf_half_sec
        t3 = t2 + self.main_return_sec
        t4 = t3 + self.settle_sec

        if elapsed < t0:
            return self.start_pose.copy(), "HOLD_CURRENT", 0.0
        if elapsed < t1:
            a = (elapsed - t0) / max(self.hip_align_sec, 1e-6)
            return self.blend(self.start_pose, self.hip_target, a), "HIP_ALIGN", a
        if elapsed < t2:
            a = (elapsed - t1) / max(self.calf_half_sec, 1e-6)
            return self.blend(self.hip_target, self.calf_half_target, a), "CALF_HALF_RELEASE", a
        if elapsed < t3:
            a = (elapsed - t2) / max(self.main_return_sec, 1e-6)
            return self.blend(self.calf_half_target, self.stand_target, a), "MAIN_RETURN", a
        if elapsed < t4:
            return self.stand_target.copy(), "SETTLE_STAND", 1.0
        if self.hold_after_done:
            return self.stand_target.copy(), "DONE_HOLD", 1.0
        return self.stand_target.copy(), "DONE", 1.0

    def gain_base_for_phase(self, phase: str) -> tuple[float, float]:
        if phase == "HOLD_CURRENT":
            return float(self.get_parameter("hold_kp").value), self.clamp_kd(float(self.get_parameter("hold_kd").value))
        if phase == "HIP_ALIGN":
            return float(self.get_parameter("hip_stage_kp").value), self.clamp_kd(float(self.get_parameter("hip_stage_kd").value))
        if phase == "CALF_HALF_RELEASE":
            return float(self.get_parameter("calf_stage_kp").value), self.clamp_kd(float(self.get_parameter("calf_stage_kd").value))
        if phase == "MAIN_RETURN":
            return float(self.get_parameter("main_kp").value), self.clamp_kd(float(self.get_parameter("main_kd").value))
        return float(self.get_parameter("settle_kp").value), self.clamp_kd(float(self.get_parameter("settle_kd").value))

    def gain_for_joint(self, i: int, phase: str) -> tuple[float, float]:
        kp, kd = self.gain_base_for_phase(phase)
        if phase in ("MAIN_RETURN", "SETTLE_STAND", "DONE_HOLD", "DONE"):
            if i in HIP_INDEX:
                kp *= float(self.get_parameter("hip_kp_scale").value)
            elif i in THIGH_INDEX:
                kp *= float(self.get_parameter("thigh_kp_scale").value)
            elif i in CALF_INDEX:
                kp *= float(self.get_parameter("calf_kp_scale").value)
        return float(kp), self.clamp_kd(float(kd))

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

        q_raw, phase, alpha = self.compute_target(elapsed)
        q = self.apply_rate_limit(q_raw, dt)
        self.current_phase = phase

        self.publish_array(self.pub_target, q)
        phase_id = {
            "HOLD_CURRENT": 0.0,
            "HIP_ALIGN": 1.0,
            "CALF_HALF_RELEASE": 2.0,
            "MAIN_RETURN": 3.0,
            "SETTLE_STAND": 4.0,
            "DONE_HOLD": 5.0,
            "DONE": 6.0,
        }.get(phase, -1.0)
        self.publish_array(self.pub_phase, np.array([elapsed, phase_id, alpha], dtype=np.float32))

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(q - self.last_target))) > self.max_delta * 2.0:
                self.get_logger().warn("[SAFE] target jump too large; skip send")
            else:
                sent = self.send_motion_batch(q, phase)

        self.update_debug_sample(q, phase, alpha, sent)

        if phase == "DONE" and self.shutdown_after_done:
            self.get_logger().info("Soft return done; shutting down node.")
            rclpy.shutdown()

    def send_motion_batch(self, q: np.ndarray, phase: str) -> bool:
        items = []
        for i, (mid, pos) in enumerate(zip(MOTOR_IDS, q)):
            kp, kd = self.gain_for_joint(i, phase)
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
                    f"[SEND] soft return ok | phase={phase} q_min={float(np.min(q)):.3f} q_max={float(np.max(q)):.3f}"
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
            "time", "elapsed", "phase", "alpha", "joint_index", "motor_id", "joint_name",
            "q_target", "q_current", "q_error", "torque_measured", "temp", "online", "error_code", "age_ms",
            "kp", "kd", "sent",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing soft return data to {path}")

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

    def update_debug_sample(self, q, phase, alpha, sent):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "q": np.asarray(q, dtype=np.float32).reshape(12).copy(),
                "phase": str(phase),
                "alpha": float(alpha),
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(self, q, phase, alpha, sent, stamp):
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
            kp, kd = self.gain_for_joint(i, phase)
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", phase, f"{alpha:.6f}",
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
    node = FanfanSoftReturnStandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
