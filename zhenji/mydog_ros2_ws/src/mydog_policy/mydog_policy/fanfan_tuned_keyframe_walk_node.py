#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fanfan tuned keyframe walking gait node.

Purpose:
- Convert a full walking gait manually tuned in Multi-Joint Position Tuner into
  a smooth ROS2 node.
- Send REAL motor-order targets directly:
    J1  0x11, J2  0x12, J3  0x13,
    J4  0x21, J5  0x22, J6  0x23,
    J7  0x31, J8  0x32, J9  0x33,
    J10 0x41, J11 0x42, J12 0x43.
- No policy/semantic remapping is applied.
- Smoothly interpolate through your fixed-point gait using smootherstep.
- Add rate limiting, max jump protection, CSV logging, and per-joint Kp/Kd override.

This node is intentionally traditional/keyframe-based:
    current feedback -> first tuned pose -> continuous keyframe loop
"""

import csv
import math
import os
import threading
import time
from typing import List, Tuple

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

# Your manually tuned gait keyframes, real motor order J1..J12.
# Units: rad.
KEYFRAME_NAMES = [
    "right_lean_lift_FL",    # 右倾抬左前
    "FL_forward_reach",      # 左前前伸
    "FL_touchdown",          # 左前下落
    "LH_lift",               # 左后抬腿
    "LH_forward_reach",      # 左后前伸
    "RH_push",               # 右后蹬
    "FR_forward_reach",      # 右前前伸
    "RH_touchdown",          # 右后落
    "cycle_closure_pose",    # 原文件最后一个未命名姿态
]

KEYFRAMES = np.array([
    [0.0000, 1.0472, -2.0944, 0.0000, -1.0472, 2.2689, 0.0000, -1.0472, 1.5708, 0.0000, 1.0472, -2.0944],
    [0.0000, 1.0472, -2.0944, 0.0000, -0.6109, 2.2689, 0.0000, -1.0472, 1.5708, 0.0000, 1.0472, -2.0944],
    [0.0000, 1.0472, -2.0944, 0.0000, -0.6109, 1.6580, 0.0000, -1.0472, 1.5708, 0.0000, 1.0472, -2.0944],
    [0.0000, 1.0472, -2.2689, 0.0000, -1.0472, 1.8326, 0.0000, -1.0472, 2.0944, 0.0000, 1.0472, -1.6581],
    [0.0000, 1.0472, -2.2689, 0.0000, -1.0472, 1.8326, 0.0000, -0.6981, 1.9199, 0.0000, 1.0472, -1.6581],
    [0.0000, 1.0472, -2.2689, 0.0000, -1.0472, 1.8326, 0.0000, -0.6981, 1.9199, 0.0000, 1.0472, -1.4836],
    [0.0000, 0.3491, -2.2689, 0.0000, -1.0472, 1.8326, 0.0000, -0.6981, 1.9199, 0.0000, 1.0472, -1.4836],
    [0.0000, 0.5236, -1.3962, 0.0000, -1.0472, 2.0944, 0.0000, -0.7854, 1.5708, 0.0000, 0.3491, -1.5709],
    [0.0000, 0.8727, -2.0944, 0.0000, -1.0472, 2.0944, 0.0000, -0.7854, 1.4836, 0.0000, 1.0472, -2.1817],
], dtype=np.float32)


class FanfanTunedKeyframeWalkNode(Node):
    def __init__(self):
        super().__init__("fanfan_tuned_keyframe_walk_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("rate_hz", 80.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("start_from_feedback", True)

        # Startup sequence.
        self.declare_parameter("start_hold_sec", 0.8)
        self.declare_parameter("start_to_first_sec", 2.5)
        self.declare_parameter("loop", True)

        # Keyframe timing.
        # One value per transition:
        # K0->K1, K1->K2, ..., K8->K0.
        self.declare_parameter(
            "segment_times_csv",
            "0.70,0.65,0.80,0.70,0.65,0.70,0.85,0.85,0.90",
        )
        self.declare_parameter("hold_times_csv", "0.04,0.04,0.08,0.04,0.04,0.04,0.06,0.08,0.06")
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("smoothing", "smootherstep")  # linear/smoothstep/smootherstep/sine

        # Optional global pose shaping. Keep zero by default so your tuned points are preserved.
        self.declare_parameter("global_calf_bend_extra", 0.0)
        self.declare_parameter("front_calf_bend_extra", 0.0)
        self.declare_parameter("rear_calf_bend_extra", 0.0)

        # Safety / smoothness.
        self.declare_parameter("max_target_rate_rad_s", 1.6)
        self.declare_parameter("max_delta", 0.42)
        self.declare_parameter("torque_warn_nm", 7.0)

        # Default gains and optional per-joint overrides.
        self.declare_parameter("kp", 40.0)
        self.declare_parameter("kd", 5.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        for i in range(1, 13):
            self.declare_parameter(f"j{i}_kp", -1.0)
            self.declare_parameter(f"j{i}_kd", -1.0)

        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.02)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.start_from_feedback = bool(self.get_parameter("start_from_feedback").value)
        self.start_hold_sec = float(self.get_parameter("start_hold_sec").value)
        self.start_to_first_sec = float(self.get_parameter("start_to_first_sec").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.time_scale = max(0.05, float(self.get_parameter("time_scale").value))
        self.smoothing = str(self.get_parameter("smoothing").value).strip().lower()
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)
        self.kp = float(self.get_parameter("kp").value)
        self.kd = self.clamp_kd(float(self.get_parameter("kd").value))
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.keyframes = KEYFRAMES.copy()
        self.apply_pose_shaping()

        self.segment_times = self.parse_csv_floats(
            str(self.get_parameter("segment_times_csv").value),
            len(self.keyframes),
            default=0.75,
        )
        self.hold_times = self.parse_csv_floats(
            str(self.get_parameter("hold_times_csv").value),
            len(self.keyframes),
            default=0.05,
        )
        self.segment_times = [max(0.08, t * self.time_scale) for t in self.segment_times]
        self.hold_times = [max(0.0, t * self.time_scale) for t in self.hold_times]
        self.cycle_time = float(sum(self.segment_times) + sum(self.hold_times))

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

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_tuned_keyframe_target", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_tuned_keyframe_phase", 10)

        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self.setup_debug_csv()
        self.start_debug_collector()

        self.get_logger().warn(
            "Tuned keyframe walk is open-loop. First run supported/hand-held. "
            "It sends real motor-order targets directly."
        )
        self.get_logger().info(
            f"keyframes={len(self.keyframes)}, cycle_time={self.cycle_time:.2f}s, "
            f"time_scale={self.time_scale:.2f}, smoothing={self.smoothing}, send={self.enable_send}"
        )

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-3), self.update)

    @staticmethod
    def clamp_kd(x: float) -> float:
        return float(min(5.0, max(0.0, x)))

    @staticmethod
    def smoothstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * (3.0 - 2.0 * s)

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    @staticmethod
    def sine_ease(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return 0.5 - 0.5 * math.cos(math.pi * s)

    def ease(self, s: float) -> float:
        if self.smoothing == "linear":
            return min(1.0, max(0.0, float(s)))
        if self.smoothing == "smoothstep":
            return self.smoothstep(s)
        if self.smoothing == "sine":
            return self.sine_ease(s)
        return self.smootherstep(s)

    def parse_csv_floats(self, text: str, n: int, default: float) -> List[float]:
        vals = []
        for item in text.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                vals.append(float(item))
            except ValueError:
                pass
        if not vals:
            vals = [default] * n
        if len(vals) < n:
            vals += [vals[-1]] * (n - len(vals))
        return vals[:n]

    def apply_pose_shaping(self):
        global_extra = float(self.get_parameter("global_calf_bend_extra").value)
        front_extra = float(self.get_parameter("front_calf_bend_extra").value)
        rear_extra = float(self.get_parameter("rear_calf_bend_extra").value)
        # Calf joints in real order: J3, J6, J9, J12. Signs differ by leg.
        # Positive extra means "more folded / lower body" using the same sign convention
        # already present in your tuned keyframes.
        if abs(global_extra) > 1e-9 or abs(front_extra) > 1e-9 or abs(rear_extra) > 1e-9:
            self.keyframes[:, 2] -= (global_extra + front_extra)   # J3  0x13
            self.keyframes[:, 5] += (global_extra + front_extra)   # J6  0x23
            self.keyframes[:, 8] += (global_extra + rear_extra)    # J9  0x33
            self.keyframes[:, 11] -= (global_extra + rear_extra)   # J12 0x43

    def read_start_pose(self) -> np.ndarray:
        if self.start_from_feedback and self.motor is not None:
            try:
                snapshot = self.motor.get_latest()
                q = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
                if np.all(np.isfinite(q)) and float(np.max(np.abs(q))) > 1e-6:
                    self.get_logger().info("Start pose loaded from motor feedback.")
                    return q.copy()
            except Exception as exc:
                self.get_logger().warn(f"Could not read feedback; starting from first keyframe. reason={exc}")
        return self.keyframes[0].copy()

    def gain_for_joint(self, i: int) -> Tuple[float, float]:
        kp_override = float(self.get_parameter(f"j{i + 1}_kp").value)
        kd_override = float(self.get_parameter(f"j{i + 1}_kd").value)
        kp = self.kp if kp_override < 0.0 else kp_override
        kd = self.kd if kd_override < 0.0 else kd_override
        return float(kp), self.clamp_kd(float(kd))

    def keyframe_target_at(self, t: float) -> Tuple[np.ndarray, str, int, int, float]:
        # Returns q, state, from_idx, to_idx, alpha.
        n = len(self.keyframes)
        if self.cycle_time <= 1e-6:
            return self.keyframes[0].copy(), "KEYFRAME", 0, 0, 0.0

        if not self.loop and t >= self.cycle_time:
            return self.keyframes[-1].copy(), "DONE", n - 1, n - 1, 1.0

        tc = t % self.cycle_time
        cursor = 0.0
        for i in range(n):
            hold = self.hold_times[i]
            if tc < cursor + hold:
                return self.keyframes[i].copy(), "HOLD", i, i, 0.0
            cursor += hold

            seg = self.segment_times[i]
            j = (i + 1) % n
            if tc < cursor + seg:
                raw = (tc - cursor) / max(seg, 1e-6)
                a = self.ease(raw)
                q = self.keyframes[i] + a * (self.keyframes[j] - self.keyframes[i])
                return q.astype(np.float32), "TRANSITION", i, j, float(a)
            cursor += seg

        return self.keyframes[-1].copy(), "KEYFRAME", n - 1, n - 1, 1.0

    def compute_target(self, elapsed: float) -> Tuple[np.ndarray, str, int, int, float]:
        if elapsed < self.start_hold_sec:
            return self.start_target.copy(), "START_HOLD", -1, -1, 0.0

        t = elapsed - self.start_hold_sec
        if t < self.start_to_first_sec:
            a = self.ease(t / max(self.start_to_first_sec, 1e-6))
            q = self.start_target + a * (self.keyframes[0] - self.start_target)
            return q.astype(np.float32), "START_TO_FIRST", -1, 0, float(a)

        return self.keyframe_target_at(t - self.start_to_first_sec)

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

        q_raw, state, from_idx, to_idx, alpha = self.compute_target(elapsed)
        q = self.apply_rate_limit(q_raw, dt)

        self.publish_array(self.pub_target, q)
        self.publish_array(
            self.pub_phase,
            np.array([elapsed, float(from_idx), float(to_idx), alpha], dtype=np.float32),
        )

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(q - self.last_target))) > self.max_delta * 2.0:
                self.get_logger().warn("[SAFE] target jump too large; skip send")
            else:
                sent = self.send_motion_batch(q)

        self.update_debug_sample(q, q_raw, state, from_idx, to_idx, alpha, sent)

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
                    f"[SEND] keyframe walk ok | q_min={float(np.min(q)):.3f} q_max={float(np.max(q)):.3f}"
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
            "time", "elapsed", "state", "from_idx", "from_name", "to_idx", "to_name", "alpha",
            "joint_index", "motor_id", "joint_name", "q_target", "q_raw",
            "q_current", "q_error", "torque_measured", "temp", "online", "error_code", "age_ms",
            "kp", "kd", "sent",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing tuned keyframe walk data to {path}")

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

    def update_debug_sample(self, q, q_raw, state, from_idx, to_idx, alpha, sent):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "q": np.asarray(q, dtype=np.float32).reshape(12).copy(),
                "q_raw": np.asarray(q_raw, dtype=np.float32).reshape(12).copy(),
                "state": str(state),
                "from_idx": int(from_idx),
                "to_idx": int(to_idx),
                "alpha": float(alpha),
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(self, q, q_raw, state, from_idx, to_idx, alpha, sent, stamp):
        if self._debug_csv_writer is None:
            return

        now = time.time()
        q = np.asarray(q, dtype=np.float32).reshape(12)
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
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
        from_name = KEYFRAME_NAMES[from_idx] if 0 <= from_idx < len(KEYFRAME_NAMES) else ""
        to_name = KEYFRAME_NAMES[to_idx] if 0 <= to_idx < len(KEYFRAME_NAMES) else ""

        for i, (mid, name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
            kp, kd = self.gain_for_joint(i)
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", state, int(from_idx), from_name, int(to_idx), to_name, f"{alpha:.6f}",
                i + 1, f"0x{int(mid):02X}", name,
                f"{float(q[i]):.6f}", f"{float(q_raw[i]):.6f}",
                f"{float(q_real[i]):.6f}", f"{float(q[i] - q_real[i]):.6f}",
                f"{float(torque[i]):.6f}", f"{float(temp[i]):.3f}",
                int(bool(online[i])), int(error_code[i]), f"{float(age_ms[i]):.2f}",
                f"{float(kp):.3f}", f"{float(kd):.3f}", int(bool(sent)),
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
    node = FanfanTunedKeyframeWalkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
