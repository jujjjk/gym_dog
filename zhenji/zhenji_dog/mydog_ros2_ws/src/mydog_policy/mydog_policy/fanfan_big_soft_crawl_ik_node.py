#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fanfan_flexible_crawl_ik_node.py  (v2 corrected soft crawl)

Purpose
-------
A very soft, non-aggressive crawl gait for the real Fanfan/MyDog robot.
It is designed from the user's walking default stand pose, the fanfan URDF
sagittal link lengths, and RS01 motion-control constraints.

Core idea
---------
1. Use the user's walking stand pose as the calibration neutral pose.
2. Lock/strongly hold the 4 hip joints to reduce side sway.
3. Use a 2-link IK trajectory in sagittal x/z for thigh/calf.
4. Move only one leg in swing at a time: FL -> RR -> FR -> RL.
5. During stance, the other legs move their foot target slowly backward, so the
   trunk moves forward instead of only throwing the swing foot forward.
6. Use smoothstep/cycloidal curves and command rate limiting to protect the
   3D-printed body and RS01 motors.

Install
-------
cp fanfan_flexible_crawl_ik_node.py \
  /home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/fanfan_flexible_crawl_ik_node.py
chmod +x /home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/fanfan_flexible_crawl_ik_node.py

Add to setup.py console_scripts:
'fanfan_flexible_crawl_ik_node = mydog_policy.fanfan_flexible_crawl_ik_node:main',

Build
-----
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
rm -rf build/mydog_policy install/mydog_policy
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

First test, no motor send, only CSV/log:
ros2 run mydog_policy fanfan_flexible_crawl_ik_node --ros-args -p enable_send:=false

First real-machine test, half-hold the body:
ros2 run mydog_policy fanfan_flexible_crawl_ik_node --ros-args \
  -p enable_send:=true \
  -p step_hz:=0.28 \
  -p duty_factor:=0.79 \
  -p stride_length:=0.052 \
  -p swing_height_front:=0.044 \
  -p swing_height_rear:=0.034 \
  -p max_target_rate_rad_s:=0.95 \
  -p target_ema_alpha:=0.40
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

import rclpy
from rclpy.node import Node


# ----------------------------- small math helpers -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smoothstep(u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smootherstep(u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def cycloid(u: float) -> float:
    """0 -> 1 with zero velocity at both ends, good for swing x motion."""
    u = clamp(u, 0.0, 1.0)
    return u - math.sin(2.0 * math.pi * u) / (2.0 * math.pi)


def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class MotorCmd:
    joint: str
    motor_id: int
    pos: float
    vel: float
    torque: float
    kp: float
    kd: float


@dataclass
class LegDef:
    name: str
    hip_joint: str
    thigh_joint: str
    calf_joint: str
    hip_id: int
    thigh_id: int
    calf_id: int
    hip_stand: float
    thigh_stand: float
    calf_stand: float
    sagittal_sign: float
    phase_offset: float
    is_front: bool
    diag_support_leg: str


class MotionHttpClient:
    """Robust HTTP client for the user's local motor FastAPI bridge.

    The screenshot tuner calls look like:
        motion_mode_run(0x11, 0.1571, 0.00, 40.00, 5.00)
    so the real bridge may expect FIVE motion-mode parameters:
        id, position, torque, kp, kd
    not the 6-field MIT tuple used by some other code.

    This client tries the common batch endpoints first, then falls back to
    single-motor HTTP calls with both JSON body and query params. Once one
    format works it caches that exact format, so the first second may probe,
    then later sends are stable.
    """

    def __init__(self, base_url: str, timeout_s: float = 0.045):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._cached_batch_style: Optional[Tuple[str, int]] = None
        self._cached_single_style: Optional[Tuple[str, str, int]] = None  # method, endpoint, payload index
        self.last_error: str = ""
        self.last_success: str = ""

    @staticmethod
    def _hex_id(mid: int) -> str:
        return f"0x{int(mid):02x}"

    def _batch_payloads(self, cmds: List[MotorCmd]) -> List[object]:
        # Five-parameter motion-mode form: id, pos, torque, kp, kd.
        items_id = [
            {"id": int(c.motor_id), "pos": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)}
            for c in cmds
        ]
        items_motor = [
            {"motor_id": int(c.motor_id), "position": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)}
            for c in cmds
        ]
        items_hex = [
            {"motor_id": self._hex_id(c.motor_id), "position": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)}
            for c in cmds
        ]
        tuple5 = [[int(c.motor_id), float(c.pos), float(c.torque), float(c.kp), float(c.kd)] for c in cmds]
        tuple5_hex = [[self._hex_id(c.motor_id), float(c.pos), float(c.torque), float(c.kp), float(c.kd)] for c in cmds]
        ids = [int(c.motor_id) for c in cmds]
        ids_hex = [self._hex_id(c.motor_id) for c in cmds]
        positions = [float(c.pos) for c in cmds]
        torques = [float(c.torque) for c in cmds]
        kps = [float(c.kp) for c in cmds]
        kds = [float(c.kd) for c in cmds]
        # Also include 6-field variants as a fallback for bridges that keep velocity.
        tuple6 = [[int(c.motor_id), float(c.pos), float(c.vel), float(c.torque), float(c.kp), float(c.kd)] for c in cmds]
        return [
            {"commands": items_id},
            {"cmds": items_id},
            {"motors": items_id},
            {"items": items_id},
            {"commands": items_motor},
            {"cmds": items_motor},
            {"motors": items_motor},
            {"commands": items_hex},
            {"ids": ids, "positions": positions, "torques": torques, "kps": kps, "kds": kds},
            {"motor_ids": ids, "positions": positions, "torques": torques, "kps": kps, "kds": kds},
            {"motor_ids": ids_hex, "positions": positions, "torques": torques, "kps": kps, "kds": kds},
            {"ids": ids, "pos": positions, "torque": torques, "kp": kps, "kd": kds},
            {"commands": tuple5},
            {"cmds": tuple5},
            {"commands": tuple5_hex},
            tuple5,
            {"commands": tuple6},
        ]

    def _post_json(self, endpoint: str, payload: object) -> Tuple[bool, str]:
        if requests is None:
            return False, "python requests is not installed"
        try:
            r = requests.post(self.base_url + endpoint, json=payload, timeout=self.timeout_s)
            if 200 <= r.status_code < 300:
                return True, r.text[:180]
            return False, f"POST {endpoint} HTTP {r.status_code}: {r.text[:180]}"
        except Exception as e:
            return False, f"POST {endpoint} {repr(e)}"

    def _request_params(self, method: str, endpoint: str, params: Dict[str, object]) -> Tuple[bool, str]:
        if requests is None:
            return False, "python requests is not installed"
        try:
            if method == "GET":
                r = requests.get(self.base_url + endpoint, params=params, timeout=self.timeout_s)
            else:
                r = requests.post(self.base_url + endpoint, params=params, timeout=self.timeout_s)
            if 200 <= r.status_code < 300:
                return True, r.text[:180]
            return False, f"{method} {endpoint} HTTP {r.status_code}: {r.text[:180]}"
        except Exception as e:
            return False, f"{method} {endpoint} {repr(e)}"

    def send_batch(self, cmds: List[MotorCmd]) -> bool:
        if not cmds:
            return True
        endpoints = [
            "/api/rs04/motion_mode_run_batch",
            "/api/rs04/motion_batch_fast",
            "/api/rs04/motion_mode_run_batch_fast",
            "/api/rs04/motion_run_batch",
        ]
        payloads = self._batch_payloads(cmds)

        if self._cached_batch_style is not None:
            ep, idx = self._cached_batch_style
            ok, msg = self._post_json(ep, payloads[idx])
            self.last_error = msg
            if ok:
                self.last_success = f"batch {ep} style={idx}"
                return True
            self._cached_batch_style = None

        # Probe batch formats.
        for ep in endpoints:
            for i, p in enumerate(payloads):
                ok, msg = self._post_json(ep, p)
                self.last_error = msg
                if ok:
                    self._cached_batch_style = (ep, i)
                    self.last_success = f"batch {ep} style={i}"
                    return True

        # Fallback to single send; slower, but useful and compatible with the tuner.
        all_ok = True
        for c in cmds:
            if not self.send_single(c):
                all_ok = False
        return all_ok

    def _single_payloads(self, c: MotorCmd) -> List[Dict[str, object]]:
        mid = int(c.motor_id)
        hid = self._hex_id(mid)
        return [
            {"motor_id": mid, "pos": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
            {"id": mid, "pos": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
            {"can_id": mid, "pos": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
            {"motor_id": hid, "pos": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
            {"motor_id": mid, "position": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
            {"id": hid, "position": float(c.pos), "torque": float(c.torque), "kp": float(c.kp), "kd": float(c.kd)},
        ]

    def send_single(self, c: MotorCmd) -> bool:
        endpoints = [
            "/api/rs04/motion_mode_run",
            "/api/rs04/motion_mode_run_fast",
            "/api/rs04/motion_run",
        ]
        payloads = self._single_payloads(c)

        if self._cached_single_style is not None:
            method, ep, idx = self._cached_single_style
            if method == "POST_JSON":
                ok, msg = self._post_json(ep, payloads[idx])
            else:
                ok, msg = self._request_params(method, ep, payloads[idx])
            self.last_error = msg
            if ok:
                self.last_success = f"single {method} {ep} style={idx}"
                return True
            self._cached_single_style = None

        for ep in endpoints:
            for i, p in enumerate(payloads):
                ok, msg = self._post_json(ep, p)
                self.last_error = msg
                if ok:
                    self._cached_single_style = ("POST_JSON", ep, i)
                    self.last_success = f"single POST_JSON {ep} style={i}"
                    return True
                for method in ("POST", "GET"):
                    ok, msg = self._request_params(method, ep, p)
                    self.last_error = msg
                    if ok:
                        self._cached_single_style = (method, ep, i)
                        self.last_success = f"single {method} {ep} style={i}"
                        return True
        return False

    def enable_motors(self, motor_ids: List[int]) -> bool:
        """Best-effort enable. Failing here is not fatal; motion endpoints may auto-enable."""
        endpoints = [
            "/api/rs04/enable",
            "/api/rs04/enable_batch",
            "/api/rs04/motor_enable",
            "/api/rs04/motion_mode_enable",
        ]
        ids = [int(x) for x in motor_ids]
        ids_hex = [self._hex_id(x) for x in ids]
        payloads = [
            {"ids": ids}, {"motor_ids": ids}, {"can_ids": ids},
            {"ids": ids_hex}, {"motor_ids": ids_hex},
            ids, ids_hex,
        ]
        for ep in endpoints:
            for p in payloads:
                ok, msg = self._post_json(ep, p)
                self.last_error = msg
                if ok:
                    self.last_success = f"enable {ep}"
                    return True
        # Try single param enable as fallback.
        for ep in endpoints:
            for mid in ids:
                for p in ({"motor_id": mid}, {"id": mid}, {"motor_id": self._hex_id(mid)}):
                    ok, msg = self._request_params("POST", ep, p)
                    self.last_error = msg
                    if ok:
                        self.last_success = f"enable single {ep}"
                        return True
        return False


class FlexibleCrawlIKNode(Node):
    def __init__(self) -> None:
        super().__init__("fanfan_flexible_crawl_ik_node")

        # -------------------------- declared parameters --------------------------
        self.declare_parameter("enable_send", False)
        self.declare_parameter("base_url", "http://127.0.0.1:8000")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("step_hz", 0.28)
        self.declare_parameter("duty_factor", 0.79)
        self.declare_parameter("stride_length", 0.052)  # m; flexible first test, not aggressive
        self.declare_parameter("front_stride_gain", 0.92)
        self.declare_parameter("rear_stride_gain", 1.08)
        self.declare_parameter("swing_height_front", 0.044)  # m, soft front clearance
        self.declare_parameter("swing_height_rear", 0.034)   # m, avoid rear calf hard kick
        self.declare_parameter("nominal_z_front", 0.252)     # m, smaller => lower body
        self.declare_parameter("nominal_z_rear", 0.268)      # m
        self.declare_parameter("stance_press_front", 0.0010) # m, tiny downward press
        self.declare_parameter("stance_press_rear", 0.0015)  # m, keep rear support soft
        self.declare_parameter("diag_support_press", 0.0010) # m, active-leg diagonal support
        self.declare_parameter("stand_time_s", 3.0)
        self.declare_parameter("ramp_time_s", 7.0)
        self.declare_parameter("max_runtime_s", 0.0)  # 0 = unlimited
        self.declare_parameter("max_target_rate_rad_s", 0.95)
        self.declare_parameter("target_ema_alpha", 0.40)  # smaller = softer output, 0.35~0.65 recommended
        self.declare_parameter("torque_ff_nm", 0.0)
        self.declare_parameter("http_timeout_s", 0.045)
        self.declare_parameter("log_csv", True)
        self.declare_parameter("csv_path", "/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/docs/fanfan_flexible_crawl_ik_log.csv")
        self.declare_parameter("print_every_s", 0.5)

        # Kp/Kd. Keep Kd <= 5.0. RS01 motion mode supports Kd 0~5.
        self.declare_parameter("hip_kp_front", 36.0)
        self.declare_parameter("hip_kp_rear", 38.0)
        self.declare_parameter("hip_kd", 5.0)

        self.declare_parameter("front_thigh_kp", 44.0)
        self.declare_parameter("front_calf_kp", 42.0)
        self.declare_parameter("rear_thigh_kp", 38.0)
        self.declare_parameter("rear_calf_kp", 32.0)
        self.declare_parameter("stance_kd", 5.0)

        self.declare_parameter("front_swing_thigh_kp", 46.0)
        self.declare_parameter("front_swing_calf_kp", 44.0)
        self.declare_parameter("rear_swing_thigh_kp", 38.0)
        self.declare_parameter("rear_swing_calf_kp", 34.0)
        self.declare_parameter("swing_kd", 4.5)

        # Hip lock targets, from the user's walking stand image. You can tune these
        # later to -3/+5/+5/-5 deg if the feet still show toe-in/toe-out problems.
        self.declare_parameter("j1_hip_target", 0.157080)
        self.declare_parameter("j4_hip_target", -0.157081)
        self.declare_parameter("j7_hip_target", 0.157078)
        self.declare_parameter("j10_hip_target", -0.157080)

        # Optional small hip soft trim, default disabled. Keep zero for debugging.
        self.declare_parameter("hip_soft_trim_amp", 0.0)  # rad, e.g. <=0.025 only after stable

        # ----------------------------- read parameters ----------------------------
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.base_url = str(self.get_parameter("base_url").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / max(5.0, self.control_hz)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.duty_factor = clamp(float(self.get_parameter("duty_factor").value), 0.76, 0.90)
        self.swing_fraction = 1.0 - self.duty_factor
        if self.swing_fraction > 0.24:
            self.get_logger().warn("swing_fraction is close to leg phase spacing; clamping to avoid overlap")
            self.swing_fraction = 0.24
            self.duty_factor = 0.76
        self.stride_length = float(self.get_parameter("stride_length").value)
        self.front_stride_gain = float(self.get_parameter("front_stride_gain").value)
        self.rear_stride_gain = float(self.get_parameter("rear_stride_gain").value)
        self.swing_height_front = float(self.get_parameter("swing_height_front").value)
        self.swing_height_rear = float(self.get_parameter("swing_height_rear").value)
        self.nominal_z_front = float(self.get_parameter("nominal_z_front").value)
        self.nominal_z_rear = float(self.get_parameter("nominal_z_rear").value)
        self.stance_press_front = float(self.get_parameter("stance_press_front").value)
        self.stance_press_rear = float(self.get_parameter("stance_press_rear").value)
        self.diag_support_press = float(self.get_parameter("diag_support_press").value)
        self.stand_time_s = float(self.get_parameter("stand_time_s").value)
        self.ramp_time_s = float(self.get_parameter("ramp_time_s").value)
        self.max_runtime_s = float(self.get_parameter("max_runtime_s").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.target_ema_alpha = clamp(float(self.get_parameter("target_ema_alpha").value), 0.15, 1.0)
        self.torque_ff_nm = float(self.get_parameter("torque_ff_nm").value)
        self.print_every_s = float(self.get_parameter("print_every_s").value)
        self.log_csv = bool(self.get_parameter("log_csv").value)
        self.csv_path = str(self.get_parameter("csv_path").value)

        self.client = MotionHttpClient(
            self.base_url,
            timeout_s=float(self.get_parameter("http_timeout_s").value),
        )
        self._enable_tried = False

        # URDF parsed values from fanfan_mass_scaled_only_trunk_plus_800g(7).urdf:
        self.L1 = 0.1560608
        self.L2 = 0.148941836101256

        # Real joint limits from the uploaded URDF. They are used as final clamps.
        self.joint_limits: Dict[str, Tuple[float, float]] = {
            "J1": (-0.314159265359, 0.698131700798),
            "J2": (-1.570796326795, 0.645771823238),
            "J3": (-2.443460952792, 2.443460952792),
            "J4": (-0.314159265359, 0.698131700798),
            "J5": (-1.570796326795, 0.645771823238),
            "J6": (-2.443460952792, 2.443460952792),
            "J7": (-0.314159265359, 0.698131700798),
            "J8": (-0.645771823238, 1.570796326795),
            "J9": (-2.443460952792, 2.443460952792),
            "J10": (-0.314159265359, 0.698131700798),
            "J11": (-0.645771823238, 1.570796326795),
            "J12": (-2.443460952792, 2.443460952792),
        }

        # User's walking default stand pose shown in the screenshot.
        self.stand: Dict[str, float] = {
            "J1": float(self.get_parameter("j1_hip_target").value),
            "J2": 0.349066,
            "J3": -1.047197,
            "J4": float(self.get_parameter("j4_hip_target").value),
            "J5": -0.349067,
            "J6": 1.047197,
            "J7": float(self.get_parameter("j7_hip_target").value),
            "J8": -0.226897,
            "J9": 0.785397,
            "J10": float(self.get_parameter("j10_hip_target").value),
            "J11": 0.226904,
            "J12": -0.785397,
        }

        self.legs: Dict[str, LegDef] = {
            # Phase offsets create swing order FL -> RR -> FR -> RL.
            # sagittal_sign maps canonical IK angle to real motor angle.
            "FL": LegDef("FL", "J4", "J5", "J6", 0x21, 0x22, 0x23,
                         self.stand["J4"], self.stand["J5"], self.stand["J6"],
                         -1.0, 0.00, True, "RR"),
            "RR": LegDef("RR", "J7", "J8", "J9", 0x31, 0x32, 0x33,
                         self.stand["J7"], self.stand["J8"], self.stand["J9"],
                         -1.0, 0.75, False, "FL"),
            "FR": LegDef("FR", "J1", "J2", "J3", 0x11, 0x12, 0x13,
                         self.stand["J1"], self.stand["J2"], self.stand["J3"],
                         +1.0, 0.50, True, "RL"),
            "RL": LegDef("RL", "J10", "J11", "J12", 0x41, 0x42, 0x43,
                         self.stand["J10"], self.stand["J11"], self.stand["J12"],
                         +1.0, 0.25, False, "FR"),
        }
        self.swing_order = ["FL", "RR", "FR", "RL"]

        self.last_targets = dict(self.stand)
        self.start_time = time.monotonic()
        self.last_print_t = self.start_time
        self.tick_count = 0
        self.stop_requested = False

        self.csv_file = None
        self.csv_writer = None
        if self.log_csv:
            self._open_csv()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.timer = self.create_timer(self.dt, self._on_timer)
        self.get_logger().info(
            "Flexible crawl IK node started. enable_send=%s, step_hz=%.3f, stride=%.3fm, duty=%.2f, max_rate=%.2frad/s" %
            (self.enable_send, self.step_hz, self.stride_length, self.duty_factor, self.max_target_rate_rad_s)
        )
        self.get_logger().warn(
            "First real test: half-hold the body. Watch torque/temp/error. Kd is clamped <= 5.0."
        )

    # ---------------------------- gait and kinematics ---------------------------

    def _canonical_from_real(self, leg: LegDef, thigh_real: float, calf_real: float) -> Tuple[float, float]:
        return leg.sagittal_sign * thigh_real, leg.sagittal_sign * calf_real

    def _real_from_canonical(self, leg: LegDef, thigh: float, calf: float) -> Tuple[float, float]:
        return leg.sagittal_sign * thigh, leg.sagittal_sign * calf

    def _fk_xz(self, thigh: float, calf: float) -> Tuple[float, float]:
        """Canonical sagittal FK. x forward, z down, both in meters."""
        x = -self.L1 * math.sin(thigh) - self.L2 * math.sin(thigh + calf)
        z = self.L1 * math.cos(thigh) + self.L2 * math.cos(thigh + calf)
        return x, z

    def _ik_xz(self, x: float, z: float) -> Tuple[float, float]:
        """Canonical 2-link IK. Returns thigh and calf angles.

        Uses the knee-bent solution with calf angle negative in canonical space,
        matching FR canonical stand approximately (0.349, -1.047).
        """
        r = math.hypot(x, z)
        # Avoid singular straight-leg and too-folded configurations.
        r_min = abs(self.L1 - self.L2) + 0.020
        r_max = self.L1 + self.L2 - 0.018
        if r < r_min or r > r_max:
            scale = clamp(r, r_min, r_max) / max(1e-6, r)
            x *= scale
            z *= scale
            r = math.hypot(x, z)

        d = (r * r - self.L1 * self.L1 - self.L2 * self.L2) / (2.0 * self.L1 * self.L2)
        d = clamp(d, -0.999, 0.999)
        calf = -math.acos(d)
        beta = math.atan2(-x, z)  # angle from vertical down axis
        thigh = beta - math.atan2(self.L2 * math.sin(calf), self.L1 + self.L2 * math.cos(calf))
        return thigh, calf

    def _leg_base_xz(self, leg: LegDef) -> Tuple[float, float]:
        th, ca = self._canonical_from_real(leg, leg.thigh_stand, leg.calf_stand)
        return self._fk_xz(th, ca)

    def _leg_phase(self, leg: LegDef, gait_clock: float) -> float:
        period = 1.0 / max(1e-5, self.step_hz)
        return ((gait_clock / period) + leg.phase_offset) % 1.0

    def _active_swing_leg(self, gait_clock: float) -> str:
        phases = {name: self._leg_phase(leg, gait_clock) for name, leg in self.legs.items()}
        active = [name for name, p in phases.items() if p < self.swing_fraction]
        if not active:
            return "NONE"
        # If numerical overlap ever happens, report the leg closest to lift-off.
        return min(active, key=lambda n: phases[n])

    def _desired_leg_xz(self, leg: LegDef, phase: float, amp: float, active_leg: str) -> Tuple[float, float, str, float, float]:
        base_x, base_z0 = self._leg_base_xz(leg)
        nominal_z = self.nominal_z_front if leg.is_front else self.nominal_z_rear
        # Keep the neutral forward/back position from the walking stand, but use
        # a slightly lower/folded body height for stable three-leg support.
        base_z = nominal_z

        stride_gain = self.front_stride_gain if leg.is_front else self.rear_stride_gain
        stride = self.stride_length * stride_gain
        swing_h = self.swing_height_front if leg.is_front else self.swing_height_rear
        stance_press = self.stance_press_front if leg.is_front else self.stance_press_rear

        if phase < self.swing_fraction:
            u = phase / max(1e-6, self.swing_fraction)
            x_off = -0.5 * stride + stride * cycloid(u)
            # Rounded high-clearance step: vertical velocity is also soft at top.
            z_lift = swing_h * math.sin(math.pi * smootherstep(u))
            state = "SWING"
        else:
            u = (phase - self.swing_fraction) / max(1e-6, 1.0 - self.swing_fraction)
            x_off = 0.5 * stride - stride * smoothstep(u)
            z_lift = 0.0
            state = "STANCE"

        # Tiny support press. It is deliberately small; earlier tests showed rear
        # calf hard-kicking can shake the trunk.
        support_extra = 0.0
        if state == "STANCE":
            support_extra += stance_press * math.sin(math.pi * u)
            # Strengthen the diagonal support of the active swing leg.
            if active_leg != "NONE" and leg.name == self.legs[active_leg].diag_support_leg:
                support_extra += self.diag_support_press * math.sin(math.pi * u)
            # Pitch coordination: when a front leg swings, rear support carries a
            # little more; when a rear leg swings, front support carries a little more.
            if active_leg != "NONE" and self.legs[active_leg].is_front and not leg.is_front:
                support_extra += 0.0008 * math.sin(math.pi * u)
            if active_leg != "NONE" and (not self.legs[active_leg].is_front) and leg.is_front:
                support_extra += 0.0007 * math.sin(math.pi * u)

        x_des = base_x + amp * x_off
        z_des = base_z - amp * z_lift + amp * support_extra
        return x_des, z_des, state, x_off, (-z_lift + support_extra)

    # ------------------------------ command output ------------------------------

    def _pd_for_joint(self, leg: LegDef, joint_kind: str, leg_state: str) -> Tuple[float, float]:
        kd_hip = clamp(float(self.get_parameter("hip_kd").value), 0.0, 5.0)
        stance_kd = clamp(float(self.get_parameter("stance_kd").value), 0.0, 5.0)
        swing_kd = clamp(float(self.get_parameter("swing_kd").value), 0.0, 5.0)

        if joint_kind == "hip":
            kp = float(self.get_parameter("hip_kp_front").value if leg.is_front else self.get_parameter("hip_kp_rear").value)
            return kp, kd_hip

        if leg_state == "SWING":
            if leg.is_front:
                kp = float(self.get_parameter("front_swing_thigh_kp").value if joint_kind == "thigh" else self.get_parameter("front_swing_calf_kp").value)
            else:
                kp = float(self.get_parameter("rear_swing_thigh_kp").value if joint_kind == "thigh" else self.get_parameter("rear_swing_calf_kp").value)
            return kp, swing_kd

        if leg.is_front:
            kp = float(self.get_parameter("front_thigh_kp").value if joint_kind == "thigh" else self.get_parameter("front_calf_kp").value)
        else:
            kp = float(self.get_parameter("rear_thigh_kp").value if joint_kind == "thigh" else self.get_parameter("rear_calf_kp").value)
        return kp, stance_kd

    def _rate_limit_targets(self, raw: Dict[str, float]) -> Dict[str, float]:
        # Two-stage soft output:
        # 1) hard rate limit prevents sudden position jumps;
        # 2) EMA smoothing makes the sent target more flexible and less punchy.
        max_step = max(0.012, self.max_target_rate_rad_s * self.dt)
        alpha = self.target_ema_alpha
        out = {}
        for j, q in raw.items():
            q_prev = self.last_targets.get(j, q)
            q_limited = q_prev + clamp(q - q_prev, -max_step, max_step)
            q_soft = q_prev + alpha * (q_limited - q_prev)
            lo, hi = self.joint_limits[j]
            # Add small soft margin so we do not scrape the URDF limit.
            margin = 0.020
            out[j] = clamp(q_soft, lo + margin, hi - margin)
        self.last_targets = dict(out)
        return out

    def _make_commands(self, targets: Dict[str, float], states: Dict[str, str]) -> List[MotorCmd]:
        cmds: List[MotorCmd] = []
        for leg_name in self.swing_order:
            leg = self.legs[leg_name]
            leg_state = states.get(leg.name, "STANCE")
            for joint_kind, joint_name, motor_id in [
                ("hip", leg.hip_joint, leg.hip_id),
                ("thigh", leg.thigh_joint, leg.thigh_id),
                ("calf", leg.calf_joint, leg.calf_id),
            ]:
                kp, kd = self._pd_for_joint(leg, joint_kind, leg_state)
                cmds.append(MotorCmd(
                    joint=joint_name,
                    motor_id=motor_id,
                    pos=targets[joint_name],
                    vel=0.0,
                    torque=self.torque_ff_nm,
                    kp=clamp(kp, 0.0, 500.0),
                    kd=clamp(kd, 0.0, 5.0),
                ))
        return cmds

    def _open_csv(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            fieldnames = [
                "t", "mode", "amp", "active_leg",
                "leg", "phase", "leg_state", "x_des_m", "z_des_m", "x_offset_m", "z_offset_m",
                "hip_target", "thigh_target", "calf_target",
                "hip_kp", "hip_kd", "thigh_kp", "thigh_kd", "calf_kp", "calf_kd",
                "enable_send", "send_ok", "http_error",
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().error(f"Failed to open CSV {self.csv_path}: {e}")
            self.csv_file = None
            self.csv_writer = None

    def _write_csv_rows(self, t: float, mode: str, amp: float, active_leg: str,
                        leg_debug: Dict[str, Dict[str, float]], targets: Dict[str, float],
                        states: Dict[str, str], send_ok: bool) -> None:
        if not self.csv_writer:
            return
        try:
            for leg_name in self.swing_order:
                leg = self.legs[leg_name]
                hip_kp, hip_kd = self._pd_for_joint(leg, "hip", states[leg_name])
                th_kp, th_kd = self._pd_for_joint(leg, "thigh", states[leg_name])
                ca_kp, ca_kd = self._pd_for_joint(leg, "calf", states[leg_name])
                dbg = leg_debug[leg_name]
                self.csv_writer.writerow({
                    "t": f"{t:.4f}",
                    "mode": mode,
                    "amp": f"{amp:.4f}",
                    "active_leg": active_leg,
                    "leg": leg_name,
                    "phase": f"{dbg['phase']:.4f}",
                    "leg_state": states[leg_name],
                    "x_des_m": f"{dbg['x_des']:.5f}",
                    "z_des_m": f"{dbg['z_des']:.5f}",
                    "x_offset_m": f"{dbg['x_off']:.5f}",
                    "z_offset_m": f"{dbg['z_off']:.5f}",
                    "hip_target": f"{targets[leg.hip_joint]:.6f}",
                    "thigh_target": f"{targets[leg.thigh_joint]:.6f}",
                    "calf_target": f"{targets[leg.calf_joint]:.6f}",
                    "hip_kp": f"{hip_kp:.2f}",
                    "hip_kd": f"{hip_kd:.2f}",
                    "thigh_kp": f"{th_kp:.2f}",
                    "thigh_kd": f"{th_kd:.2f}",
                    "calf_kp": f"{ca_kp:.2f}",
                    "calf_kd": f"{ca_kd:.2f}",
                    "enable_send": int(self.enable_send),
                    "send_ok": int(send_ok),
                    "http_error": self.client.last_error[:120],
                })
            if self.tick_count % max(1, int(self.control_hz)) == 0:
                self.csv_file.flush()
        except Exception as e:
            self.get_logger().warn(f"CSV write failed: {e}")

    # -------------------------------- main timer --------------------------------

    def _on_timer(self) -> None:
        if self.stop_requested:
            self._safe_hold_stand()
            rclpy.shutdown()
            return

        now = time.monotonic()
        t = now - self.start_time
        self.tick_count += 1

        if self.max_runtime_s > 0.0 and t > self.max_runtime_s:
            self.get_logger().warn("max_runtime_s reached; holding stand and stopping.")
            self.stop_requested = True
            return

        if t < self.stand_time_s:
            mode = "STAND_WARMUP"
            amp = 0.0
            gait_clock = 0.0
        else:
            mode = "GAIT"
            amp = smoothstep((t - self.stand_time_s) / max(0.2, self.ramp_time_s))
            gait_clock = t - self.stand_time_s

        active_leg = self._active_swing_leg(gait_clock)

        raw_targets: Dict[str, float] = {}
        raw_targets.update({"J1": self.stand["J1"], "J4": self.stand["J4"], "J7": self.stand["J7"], "J10": self.stand["J10"]})

        states: Dict[str, str] = {}
        leg_debug: Dict[str, Dict[str, float]] = {}

        for leg_name in self.swing_order:
            leg = self.legs[leg_name]
            phase = self._leg_phase(leg, gait_clock)
            if amp <= 1e-5:
                x_des, z_des = self._leg_base_xz(leg)
                state, x_off, z_off = "STANCE", 0.0, 0.0
                # Still use low/folded IK stand softly after warmup starts.
                if mode == "STAND_WARMUP":
                    th0, ca0 = self._canonical_from_real(leg, leg.thigh_stand, leg.calf_stand)
                    thigh, calf = th0, ca0
                else:
                    nominal_z = self.nominal_z_front if leg.is_front else self.nominal_z_rear
                    thigh, calf = self._ik_xz(x_des, nominal_z)
                    z_des = nominal_z
            else:
                x_des, z_des, state, x_off, z_off = self._desired_leg_xz(leg, phase, amp, active_leg)
                thigh, calf = self._ik_xz(x_des, z_des)

            thigh_real, calf_real = self._real_from_canonical(leg, thigh, calf)
            raw_targets[leg.thigh_joint] = thigh_real
            raw_targets[leg.calf_joint] = calf_real
            states[leg_name] = state
            leg_debug[leg_name] = {
                "phase": phase,
                "x_des": x_des,
                "z_des": z_des,
                "x_off": x_off,
                "z_off": z_off,
            }

        # Optional tiny hip trim. Default zero, because the user observed hip dynamics
        # can make the robot toe out and sway. This is included only for later tuning.
        trim_amp = float(self.get_parameter("hip_soft_trim_amp").value)
        if abs(trim_amp) > 1e-6 and amp > 0.05:
            for leg_name, leg in self.legs.items():
                ph = self._leg_phase(leg, gait_clock)
                trim = trim_amp * math.sin(2.0 * math.pi * ph) * amp
                raw_targets[leg.hip_joint] += trim

        targets = self._rate_limit_targets(raw_targets)
        cmds = self._make_commands(targets, states)

        send_ok = True
        if self.enable_send:
            if not self._enable_tried:
                all_ids = [c.motor_id for c in cmds]
                en_ok = self.client.enable_motors(all_ids)
                self._enable_tried = True
                self.get_logger().warn(f"enable probe result={en_ok}; {self.client.last_success or self.client.last_error}")
            send_ok = self.client.send_batch(cmds)
            if not send_ok and self.tick_count % int(max(1.0, self.control_hz)) == 0:
                self.get_logger().warn(f"motor send failed: {self.client.last_error}")
            elif send_ok and self.tick_count % int(max(2.0, self.control_hz * 2.0)) == 0:
                self.get_logger().info(f"motor send ok via {self.client.last_success}")

        self._write_csv_rows(t, mode, amp, active_leg, leg_debug, targets, states, send_ok)

        if now - self.last_print_t >= self.print_every_s:
            self.last_print_t = now
            swing_states = ",".join([f"{ln}:{states[ln][0]}" for ln in self.swing_order])
            self.get_logger().info(
                f"t={t:.1f}s mode={mode} amp={amp:.2f} active={active_leg} states={swing_states} "
                f"J2={targets['J2']:.3f} J3={targets['J3']:.3f} J8={targets['J8']:.3f} J9={targets['J9']:.3f} send={send_ok}"
            )

    def _safe_hold_stand(self) -> None:
        targets = self._rate_limit_targets(self.stand)
        states = {ln: "STANCE" for ln in self.swing_order}
        cmds = self._make_commands(targets, states)
        if self.enable_send:
            for _ in range(5):
                self.client.send_batch(cmds)
                time.sleep(0.03)
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None

    def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
        self.get_logger().warn(f"signal {signum} received; holding stand then shutdown")
        self.stop_requested = True

    def destroy_node(self) -> bool:
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FlexibleCrawlIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt; safe hold stand.")
        node._safe_hold_stand()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
