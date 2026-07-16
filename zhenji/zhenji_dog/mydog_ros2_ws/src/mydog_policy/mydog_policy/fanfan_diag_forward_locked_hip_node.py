#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fanfan diagonal-order forward gait with locked real hip angles.

Design goal:
- New forward node, separate from fanfan_big_stride_walk_node.
- Four-phase diagonal-order swing sequence: FL -> RR -> FR -> RL.
- One leg swings at a time; the diagonal partner is given time to take load.
- Hip joints are locked in REAL motor-angle space after policy->real mapping:
    J1/0x11  front-right hip
    J4/0x21  front-left  hip
    J7/0x31  rear-right  hip
    J10/0x41 rear-left   hip
- Thigh/calf trajectory is geometric in sagittal x-z plane using IK.
- Adds stance calf-bend offsets to lower the body without changing hip lock.
- Body_y shift is intentionally zero by default to reduce left-right sway.
- Kd is always clamped to 0..5.0.

Install in setup.py, for example:
    'fanfan_diag_forward_locked_hip_node = mydog_policy.fanfan_diag_forward_locked_hip_node:main',
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

from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RR", "RL")
DIAG_PARTNER = {"FL": "RR", "RR": "FL", "FR": "RL", "RL": "FR"}
# User requested order: 左前 -> 右后 -> 右前 -> 左后
DEFAULT_SWING_ORDER = ("FL", "RR", "FR", "RL")

REAL_HIP_MOTOR_IDS = {
    "FR": 0x11,  # Joint-1
    "FL": 0x21,  # Joint-4
    "RR": 0x31,  # Joint-7
    "RL": 0x41,  # Joint-10
}

LEG_DEBUG_DEFAULT = {
    "leg_phase": 0.0,
    "leg_state": "STANCE",
    "stance": 1.0,
    "swing": 0.0,
    "swing_shape": 0.0,
    "x_foot": 0.0,
    "z_foot": 0.0,
    "x_des": 0.0,
    "z_des": 0.0,
    "thigh_target": 0.0,
    "calf_target": 0.0,
    "support_gate": 0.0,
    "support_role": "",
    "active_swing_leg": "",
    "support_legs": "",
}


class FanfanDiagForwardLockedHipNode(Node):
    def __init__(self):
        super().__init__("fanfan_diag_forward_locked_hip_node")

        # Communication / timing.
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("step_hz", 0.56)
        self.declare_parameter("stand_sec", 2.5)
        self.declare_parameter("warmup_sec", 4.0)
        self.declare_parameter("http_timeout", 0.08)

        # Geometric gait.
        self.declare_parameter("swing_order", ",".join(DEFAULT_SWING_ORDER))
        self.declare_parameter("duty_factor", 0.82)
        self.declare_parameter("stride_length", 0.028)
        self.declare_parameter("swing_height", 0.064)
        self.declare_parameter("walk_direction", 1.0)
        self.declare_parameter("advance_start", 0.12)
        self.declare_parameter("advance_end", 0.88)
        self.declare_parameter("swing_lift_up_fraction", 0.18)
        self.declare_parameter("swing_lift_down_fraction", 0.24)
        self.declare_parameter("front_stride_gain", 0.92)
        self.declare_parameter("rear_stride_gain", 0.78)
        self.declare_parameter("front_swing_height_gain", 1.35)
        self.declare_parameter("rear_swing_height_gain", 0.88)
        self.declare_parameter("front_clearance_extra_m", 0.006)
        self.declare_parameter("rear_clearance_extra_m", 0.004)
        self.declare_parameter("front_thigh_delta_scale", 0.26)
        self.declare_parameter("rear_thigh_delta_scale", 0.12)
        self.declare_parameter("front_calf_lift_extra", 0.190)
        self.declare_parameter("rear_calf_lift_extra", 0.075)
        self.declare_parameter("front_swing_thigh_lift_bias", 0.020)
        self.declare_parameter("rear_swing_thigh_lift_bias", 0.006)
        self.declare_parameter("front_swing_forward_unfold", 0.006)
        self.declare_parameter("front_x_bias", 0.002)
        self.declare_parameter("front_z_extend", -0.001)

        # Default posture shaping in policy space. Keep these small because hips are locked in real space.
        self.declare_parameter("hip_default_scale", 0.38)
        self.declare_parameter("front_calf_min_rad", -1.24)
        self.declare_parameter("rear_thigh_default_back_offset", 0.038)

        # Real hip lock. Defaults reduce both front in-toe and rear out-toe compared with previous versions.
        self.declare_parameter("lock_real_hips", True)
        self.declare_parameter("j1_hip_real_target_deg", -2.0)   # FR front inner-toe softened
        self.declare_parameter("j4_hip_real_target_deg", 4.0)    # FL front inner-toe softened
        self.declare_parameter("j7_hip_real_target_deg", 5.0)    # RR rear out-toe softened
        self.declare_parameter("j10_hip_real_target_deg", -5.0)  # RL rear out-toe softened

        # Support shaping. Small z/calf changes only; no dynamic hip bracing.
        self.declare_parameter("diag_support_preload_z_m", 0.006)
        self.declare_parameter("diag_support_calf_push_amp", 0.010)
        self.declare_parameter("diag_support_thigh_back_amp", 0.006)
        self.declare_parameter("other_support_preload_scale", 0.25)
        self.declare_parameter("support_stand_tall_m", 0.002)
        self.declare_parameter("landing_calf_relief_amp", 0.010)
        # Lower-body shaping. Positive values bend calf targets more in stance,
        # making J3/J6/J9/J12 fold more and the body sit lower.
        self.declare_parameter("front_stance_calf_bend_extra", 0.0)
        self.declare_parameter("rear_stance_calf_bend_extra", 0.0)
        # Optional small bend also during swing to keep the visual leg shape consistent.
        self.declare_parameter("front_swing_calf_bend_extra", 0.0)
        self.declare_parameter("rear_swing_calf_bend_extra", 0.0)

        # Motor gains. Per-group + swing phase PD is used by default.
        self.declare_parameter("stand_kp", 40.0)
        self.declare_parameter("stand_kd", 4.2)
        self.declare_parameter("send_kp", 48.0)
        self.declare_parameter("send_kd", 5.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("use_group_pd", True)
        self.declare_parameter("use_phase_pd", True)
        self.declare_parameter("front_hip_kp", 32.0)
        self.declare_parameter("front_hip_kd", 5.0)
        self.declare_parameter("front_thigh_kp", 54.0)
        self.declare_parameter("front_thigh_kd", 4.6)
        self.declare_parameter("front_calf_kp", 54.0)
        self.declare_parameter("front_calf_kd", 4.4)
        self.declare_parameter("rear_hip_kp", 34.0)
        self.declare_parameter("rear_hip_kd", 5.0)
        self.declare_parameter("rear_thigh_kp", 40.0)
        self.declare_parameter("rear_thigh_kd", 5.0)
        self.declare_parameter("rear_calf_kp", 28.0)
        self.declare_parameter("rear_calf_kd", 5.0)
        self.declare_parameter("front_swing_thigh_kp", 60.0)
        self.declare_parameter("front_swing_thigh_kd", 4.0)
        self.declare_parameter("front_swing_calf_kp", 60.0)
        self.declare_parameter("front_swing_calf_kd", 3.8)
        self.declare_parameter("rear_swing_thigh_kp", 42.0)
        self.declare_parameter("rear_swing_thigh_kd", 4.5)
        self.declare_parameter("rear_swing_calf_kp", 40.0)
        self.declare_parameter("rear_swing_calf_kd", 4.3)

        # Geometry.
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)

        # Safety / logging.
        self.declare_parameter("max_target_rate_rad_s", 1.55)
        self.declare_parameter("max_delta", 0.36)
        self.declare_parameter("torque_warn_nm", 6.0)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.02)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        # Read params.
        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)

        self.swing_order = self.parse_swing_order(str(self.get_parameter("swing_order").value))
        self.duty_factor = float(self.get_parameter("duty_factor").value)
        self.stride_length = float(self.get_parameter("stride_length").value)
        self.swing_height = float(self.get_parameter("swing_height").value)
        self.walk_direction = -1.0 if float(self.get_parameter("walk_direction").value) < 0 else 1.0
        self.advance_start = float(self.get_parameter("advance_start").value)
        self.advance_end = float(self.get_parameter("advance_end").value)
        self.swing_lift_up_fraction = float(self.get_parameter("swing_lift_up_fraction").value)
        self.swing_lift_down_fraction = float(self.get_parameter("swing_lift_down_fraction").value)
        self.front_stride_gain = float(self.get_parameter("front_stride_gain").value)
        self.rear_stride_gain = float(self.get_parameter("rear_stride_gain").value)
        self.front_swing_height_gain = float(self.get_parameter("front_swing_height_gain").value)
        self.rear_swing_height_gain = float(self.get_parameter("rear_swing_height_gain").value)
        self.front_clearance_extra_m = float(self.get_parameter("front_clearance_extra_m").value)
        self.rear_clearance_extra_m = float(self.get_parameter("rear_clearance_extra_m").value)
        self.front_thigh_delta_scale = float(self.get_parameter("front_thigh_delta_scale").value)
        self.rear_thigh_delta_scale = float(self.get_parameter("rear_thigh_delta_scale").value)
        self.front_calf_lift_extra = float(self.get_parameter("front_calf_lift_extra").value)
        self.rear_calf_lift_extra = float(self.get_parameter("rear_calf_lift_extra").value)
        self.front_swing_thigh_lift_bias = float(self.get_parameter("front_swing_thigh_lift_bias").value)
        self.rear_swing_thigh_lift_bias = float(self.get_parameter("rear_swing_thigh_lift_bias").value)
        self.front_swing_forward_unfold = float(self.get_parameter("front_swing_forward_unfold").value)
        self.front_x_bias = float(self.get_parameter("front_x_bias").value)
        self.front_z_extend = float(self.get_parameter("front_z_extend").value)

        self.hip_default_scale = float(self.get_parameter("hip_default_scale").value)
        self.front_calf_min_rad = float(self.get_parameter("front_calf_min_rad").value)
        self.rear_thigh_default_back_offset = float(self.get_parameter("rear_thigh_default_back_offset").value)

        self.lock_real_hips = bool(self.get_parameter("lock_real_hips").value)
        self.real_hip_targets = {
            "FR": math.radians(float(self.get_parameter("j1_hip_real_target_deg").value)),
            "FL": math.radians(float(self.get_parameter("j4_hip_real_target_deg").value)),
            "RR": math.radians(float(self.get_parameter("j7_hip_real_target_deg").value)),
            "RL": math.radians(float(self.get_parameter("j10_hip_real_target_deg").value)),
        }

        self.diag_support_preload_z_m = float(self.get_parameter("diag_support_preload_z_m").value)
        self.diag_support_calf_push_amp = float(self.get_parameter("diag_support_calf_push_amp").value)
        self.diag_support_thigh_back_amp = float(self.get_parameter("diag_support_thigh_back_amp").value)
        self.other_support_preload_scale = float(self.get_parameter("other_support_preload_scale").value)
        self.support_stand_tall_m = float(self.get_parameter("support_stand_tall_m").value)
        self.landing_calf_relief_amp = float(self.get_parameter("landing_calf_relief_amp").value)
        self.front_stance_calf_bend_extra = float(self.get_parameter("front_stance_calf_bend_extra").value)
        self.rear_stance_calf_bend_extra = float(self.get_parameter("rear_stance_calf_bend_extra").value)
        self.front_swing_calf_bend_extra = float(self.get_parameter("front_swing_calf_bend_extra").value)
        self.rear_swing_calf_bend_extra = float(self.get_parameter("rear_swing_calf_bend_extra").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = self.clamp_kd(float(self.get_parameter("stand_kd").value))
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = self.clamp_kd(float(self.get_parameter("send_kd").value))
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.use_group_pd = bool(self.get_parameter("use_group_pd").value)
        self.use_phase_pd = bool(self.get_parameter("use_phase_pd").value)

        self.front_hip_kp = float(self.get_parameter("front_hip_kp").value)
        self.front_hip_kd = self.clamp_kd(float(self.get_parameter("front_hip_kd").value))
        self.front_thigh_kp = float(self.get_parameter("front_thigh_kp").value)
        self.front_thigh_kd = self.clamp_kd(float(self.get_parameter("front_thigh_kd").value))
        self.front_calf_kp = float(self.get_parameter("front_calf_kp").value)
        self.front_calf_kd = self.clamp_kd(float(self.get_parameter("front_calf_kd").value))
        self.rear_hip_kp = float(self.get_parameter("rear_hip_kp").value)
        self.rear_hip_kd = self.clamp_kd(float(self.get_parameter("rear_hip_kd").value))
        self.rear_thigh_kp = float(self.get_parameter("rear_thigh_kp").value)
        self.rear_thigh_kd = self.clamp_kd(float(self.get_parameter("rear_thigh_kd").value))
        self.rear_calf_kp = float(self.get_parameter("rear_calf_kp").value)
        self.rear_calf_kd = self.clamp_kd(float(self.get_parameter("rear_calf_kd").value))
        self.front_swing_thigh_kp = float(self.get_parameter("front_swing_thigh_kp").value)
        self.front_swing_thigh_kd = self.clamp_kd(float(self.get_parameter("front_swing_thigh_kd").value))
        self.front_swing_calf_kp = float(self.get_parameter("front_swing_calf_kp").value)
        self.front_swing_calf_kd = self.clamp_kd(float(self.get_parameter("front_swing_calf_kd").value))
        self.rear_swing_thigh_kp = float(self.get_parameter("rear_swing_thigh_kp").value)
        self.rear_swing_thigh_kd = self.clamp_kd(float(self.get_parameter("rear_swing_thigh_kd").value))
        self.rear_swing_calf_kp = float(self.get_parameter("rear_swing_calf_kp").value)
        self.rear_swing_calf_kd = self.clamp_kd(float(self.get_parameter("rear_swing_calf_kd").value))

        self.thigh_length = float(self.get_parameter("thigh_length").value)
        self.calf_length = float(self.get_parameter("calf_length").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.policy_joint_names = self.mapper.get_policy_joint_names()
        self.real_motor_id_to_index = {int(mid): i for i, mid in enumerate(self.motor_ids)}

        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.apply_default_pose_offsets()
        self.default_real = self.mapper.policy_target_to_real_target(self.default_policy, clamp=True).astype(np.float32)
        self.default_real = self.apply_real_hip_lock(self.default_real)
        self.default_foot_xz = self.compute_default_foot_xz()

        self.last_target_policy = self.default_policy.copy()
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
        self._current_leg_debug = {}
        self._last_send_info_time = 0.0
        self._last_torque_warn_time = 0.0
        self._last_feedback_warn_time = 0.0

        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=self.debug_stale_recheck_ms,
        )

        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self.setup_debug_csv()
        self.start_debug_collector()

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_diag_forward_locked_hip_target_real", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_diag_forward_locked_hip_phase", 10)

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only.")

        hip_deg = {k: round(math.degrees(v), 2) for k, v in self.real_hip_targets.items()}
        self.get_logger().warn(
            f"DIAG locked-hip open-loop gait. order={self.swing_order}, step_hz={self.step_hz:.2f}, "
            f"stride={self.stride_length:.3f}, swing={self.swing_height:.3f}, hips_deg={hip_deg}. "
            "First run supported/hand-held."
        )
        self.timer = self.create_timer(1.0 / max(self.gait_hz, 1e-3), self.update)

    @staticmethod
    def clamp_kd(kd: float) -> float:
        return float(min(5.0, max(0.0, kd)))

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    @staticmethod
    def smooth_window(s: float, edge: float = 0.20) -> float:
        edge = min(0.49, max(0.001, float(edge)))
        return FanfanDiagForwardLockedHipNode.smootherstep(s / edge) * FanfanDiagForwardLockedHipNode.smootherstep((1.0 - s) / edge)

    def parse_swing_order(self, text: str) -> tuple[str, str, str, str]:
        parts = [p.strip().upper() for p in text.replace(";", ",").split(",") if p.strip()]
        if len(parts) == 4 and set(parts) == set(LEG_ORDER):
            return tuple(parts)  # type: ignore[return-value]
        self.get_logger().warn(f"Invalid swing_order={text!r}; using {DEFAULT_SWING_ORDER}")
        return DEFAULT_SWING_ORDER

    def apply_default_pose_offsets(self):
        # Keep policy hip small. Real hip final values are locked later in real motor space.
        self.hip_default_scale = min(1.0, max(0.0, self.hip_default_scale))
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            self.default_policy[i + 0] *= self.hip_default_scale
        for leg in REAR_LEGS:
            i = LEG_START[leg]
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset
        for leg in FRONT_LEGS:
            idx = LEG_START[leg] + 2
            if float(self.default_policy[idx]) < self.front_calf_min_rad:
                self.default_policy[idx] = self.front_calf_min_rad

    def apply_real_hip_lock(self, target_real: np.ndarray) -> np.ndarray:
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12).copy()
        if not self.lock_real_hips:
            return target_real
        for leg, mid in REAL_HIP_MOTOR_IDS.items():
            real_i = self.real_motor_id_to_index.get(int(mid))
            if real_i is not None:
                target_real[real_i] = float(self.real_hip_targets[leg])
        return target_real

    def forward_sagittal(self, thigh: float, calf: float) -> tuple[float, float]:
        x = -self.thigh_length * math.sin(thigh) - self.calf_length * math.sin(thigh + calf)
        z = -self.thigh_length * math.cos(thigh) - self.calf_length * math.cos(thigh + calf)
        return float(x), float(z)

    def clamp_reachable_xz(self, x: float, z: float) -> tuple[float, float]:
        r = math.hypot(x, z)
        max_r = self.thigh_length + self.calf_length - 1e-5
        min_r = abs(self.thigh_length - self.calf_length) + 1e-5
        if r < 1e-9:
            return 0.0, -min_r
        if r > max_r:
            scale = max_r / r
            return x * scale, z * scale
        if r < min_r:
            scale = min_r / r
            return x * scale, z * scale
        return x, z

    def inverse_sagittal(self, x: float, z: float) -> tuple[float, float]:
        x, z = self.clamp_reachable_xz(float(x), float(z))
        l1 = self.thigh_length
        l2 = self.calf_length
        cos_calf = (x * x + z * z - l1 * l1 - l2 * l2) / max(2.0 * l1 * l2, 1e-9)
        cos_calf = min(1.0, max(-1.0, cos_calf))
        calf = -math.acos(cos_calf)
        thigh = math.atan2(-x, -z) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
        return float(thigh), float(calf)

    def solve_calf_for_z(self, thigh: float, z_des: float, calf_default: float) -> float:
        value = (-float(z_des) - self.thigh_length * math.cos(float(thigh))) / max(self.calf_length, 1e-9)
        value = min(1.0, max(-1.0, value))
        angle = math.acos(value)
        candidates = (angle - thigh, -angle - thigh)
        return float(min(candidates, key=lambda calf: abs(calf - calf_default)))

    def compute_default_foot_xz(self) -> dict[str, tuple[float, float]]:
        out = {}
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            out[leg] = self.forward_sagittal(float(self.default_policy[i + 1]), float(self.default_policy[i + 2]))
        return out

    def leg_start_phase(self, leg: str) -> float:
        return self.swing_order.index(leg) / 4.0

    def leg_phase(self, leg: str, phase: float) -> float:
        return float((phase - self.leg_start_phase(leg)) % 1.0)

    def swing_fraction(self) -> float:
        # One-leg-at-a-time. Keep below 0.25.
        return min(0.235, max(0.11, 1.0 - min(max(self.duty_factor, 0.72), 0.88)))

    def active_swing_leg(self, phase: float) -> Optional[str]:
        sf = self.swing_fraction()
        for leg in self.swing_order:
            if self.leg_phase(leg, phase) < sf:
                return leg
        return None

    def build_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        active = self.active_swing_leg(phase)
        sf = self.swing_fraction()
        support_legs = [leg for leg in LEG_ORDER if leg != active]

        for leg in LEG_ORDER:
            p = self.leg_phase(leg, phase)
            is_swing = p < sf
            is_front = leg in FRONT_LEGS
            i = LEG_START[leg]
            x0, z0 = self.default_foot_xz[leg]
            x_center = x0 + (self.front_x_bias if is_front else 0.0)
            z_center = z0 + (self.front_z_extend if is_front else 0.0)
            stride_gain = self.front_stride_gain if is_front else self.rear_stride_gain
            height_gain = self.front_swing_height_gain if is_front else self.rear_swing_height_gain
            clearance = self.front_clearance_extra_m if is_front else self.rear_clearance_extra_m
            thigh_scale = self.front_thigh_delta_scale if is_front else self.rear_thigh_delta_scale
            calf_lift = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
            stance_calf_bend_extra = self.front_stance_calf_bend_extra if is_front else self.rear_stance_calf_bend_extra
            swing_calf_bend_extra = self.front_swing_calf_bend_extra if is_front else self.rear_swing_calf_bend_extra
            swing_thigh_bias = self.front_swing_thigh_lift_bias if is_front else self.rear_swing_thigh_lift_bias
            stride = self.stride_length * stride_gain
            swing_h = self.swing_height * height_gain + clearance
            support_gate = 0.0
            support_role = ""
            swing_shape = 0.0
            thigh_bias = 0.0
            calf_push = 0.0

            if is_swing:
                s = p / max(sf, 1e-6)
                u = self.smootherstep((s - self.advance_start) / max(self.advance_end - self.advance_start, 1e-6))
                lift_up = self.smootherstep(s / max(self.swing_lift_up_fraction, 0.001))
                lift_down = self.smootherstep((1.0 - s) / max(self.swing_lift_down_fraction, 0.001))
                swing_shape = lift_up * lift_down
                x_des = x_center + self.walk_direction * (-0.5 * stride + stride * u)
                if is_front:
                    x_des += self.walk_direction * self.front_swing_forward_unfold * swing_shape
                z_des = z_center + swing_h * swing_shape
                thigh_bias += swing_thigh_bias * swing_shape
                calf_push += -calf_lift * swing_shape
                # Keep calf visually bent during swing if requested.
                calf_push += -swing_calf_bend_extra * swing_shape
                leg_state = "SWING"
            else:
                s = (p - sf) / max(1.0 - sf, 1e-6)
                u = self.smootherstep(s)
                stance_shape = math.sin(math.pi * min(1.0, max(0.0, s))) ** 2
                stance_gate = self.smooth_window(s)
                x_des = x_center + self.walk_direction * (0.5 * stride - stride * u)
                z_des = z_center - self.support_stand_tall_m * (0.35 + 0.65 * stance_shape)
                # Positive *_stance_calf_bend_extra folds the calf in stance,
                # lowering the body instead of letting the calf stand too straight.
                calf_push += -stance_calf_bend_extra * (0.45 + 0.55 * stance_shape)

                if active is not None:
                    # The active leg's diagonal partner carries a little more load.
                    diag = DIAG_PARTNER.get(active, "")
                    if leg == diag:
                        support_gate = 1.0
                        support_role = "DIAG_SUPPORT"
                        z_des -= self.diag_support_preload_z_m * stance_gate
                        thigh_bias += self.diag_support_thigh_back_amp * stance_gate
                        calf_push += self.diag_support_calf_push_amp * stance_gate
                    else:
                        support_gate = self.other_support_preload_scale
                        support_role = "OTHER_SUPPORT"
                        z_des -= self.diag_support_preload_z_m * self.other_support_preload_scale * stance_gate
                        calf_push += self.diag_support_calf_push_amp * self.other_support_preload_scale * stance_gate

                # Soften calf near early landing; prevents the foot from stabbing the floor.
                if s < 0.20:
                    calf_push -= self.landing_calf_relief_amp * self.smootherstep((0.20 - s) / 0.20)
                leg_state = "STANCE"

            thigh_ik, calf_ik = self.inverse_sagittal(x_des, z_des)
            thigh_target = float(self.default_policy[i + 1]) + thigh_scale * (thigh_ik - float(self.default_policy[i + 1]))
            thigh_target += thigh_bias
            calf_target = self.solve_calf_for_z(thigh_target, z_des, float(self.default_policy[i + 2]))
            calf_target += calf_push
            if is_front and calf_target < self.front_calf_min_rad:
                calf_target = self.front_calf_min_rad

            # Hip target intentionally stays at default policy; real hip lock happens after mapping.
            q[i + 0] = float(self.default_policy[i + 0])
            q[i + 1] = float(self.default_policy[i + 1]) + warm * (thigh_target - float(self.default_policy[i + 1]))
            q[i + 2] = float(self.default_policy[i + 2]) + warm * (calf_target - float(self.default_policy[i + 2]))

            x_act, z_act = self.forward_sagittal(q[i + 1], q[i + 2])
            leg_debug[leg] = {
                "leg_phase": float(p),
                "leg_state": leg_state,
                "stance": 0.0 if is_swing else 1.0,
                "swing": 1.0 if is_swing else 0.0,
                "swing_shape": float(swing_shape),
                "x_foot": float(x_act),
                "z_foot": float(z_act),
                "x_des": float(x_des),
                "z_des": float(z_des),
                "thigh_target": float(thigh_target),
                "calf_target": float(calf_target),
                "support_gate": float(support_gate),
                "support_role": support_role,
                "active_swing_leg": active or "",
                "support_legs": ",".join(support_legs),
            }
        return q.astype(np.float32), leg_debug

    def default_leg_debug(self, phase: float, warm: float):
        data = {}
        for leg in LEG_ORDER:
            x, z = self.default_foot_xz[leg]
            row = dict(LEG_DEBUG_DEFAULT)
            row.update({
                "leg_phase": float(self.leg_phase(leg, phase)),
                "x_foot": float(x),
                "z_foot": float(z),
                "active_swing_leg": "",
                "support_legs": ",".join(LEG_ORDER),
            })
            data[leg] = row
        return data

    def apply_target_rate_limit(self, target_policy: np.ndarray, dt: float) -> np.ndarray:
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        if self.max_target_rate_rad_s <= 0.0 or dt <= 0.0:
            self.last_target_policy = target_policy.copy()
            return target_policy
        max_step = self.max_target_rate_rad_s * dt
        step = np.clip(target_policy - self.last_target_policy, -max_step, max_step)
        out = self.last_target_policy + step
        out = np.clip(out, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.last_target_policy = out.astype(np.float32).copy()
        return self.last_target_policy.copy()

    def get_pd_for_real_index(self, real_i: int, leg_debug_override=None) -> tuple[float, float]:
        if not self.use_group_pd:
            return float(self.send_kp), self.clamp_kd(float(self.send_kd))
        policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
        policy_name = self.policy_joint_names[policy_i]
        leg = policy_name.split("_", 1)[0].upper()
        name = policy_name.lower()
        is_front = leg in FRONT_LEGS
        leg_debug = leg_debug_override if leg_debug_override is not None else self._current_leg_debug
        state = str(leg_debug.get(leg, {}).get("leg_state", "STANCE")).upper()
        is_swing = self.use_phase_pd and state == "SWING"

        if "hip" in name:
            return (self.front_hip_kp, self.front_hip_kd) if is_front else (self.rear_hip_kp, self.rear_hip_kd)
        if "thigh" in name:
            if is_swing and is_front:
                return self.front_swing_thigh_kp, self.front_swing_thigh_kd
            if is_swing and not is_front:
                return self.rear_swing_thigh_kp, self.rear_swing_thigh_kd
            return (self.front_thigh_kp, self.front_thigh_kd) if is_front else (self.rear_thigh_kp, self.rear_thigh_kd)
        if "calf" in name:
            if is_swing and is_front:
                return self.front_swing_calf_kp, self.front_swing_calf_kd
            if is_swing and not is_front:
                return self.rear_swing_calf_kp, self.rear_swing_calf_kd
            return (self.front_calf_kp, self.front_calf_kd) if is_front else (self.rear_calf_kp, self.rear_calf_kd)
        return float(self.send_kp), self.clamp_kd(float(self.send_kd))

    def send_default_stand(self):
        items = []
        for i, (mid, pos) in enumerate(zip(self.motor_ids, self.default_real)):
            items.append({
                "motor_id": int(mid), "position": float(pos),
                "speed": 0.0, "torque": 0.0,
                "kp": self.stand_kp, "kd": self.stand_kd,
            })
        r = self.http_session.post(
            f"{self.motor_base_url}/api/rs04/motion_mode_run_batch",
            json={"items": items, "enable_first": True, "stop_first": False},
            timeout=max(self.http_timeout, 0.5),
        )
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")
        self.get_logger().info("Default locked-hip stand sent.")

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            kp, kd = self.get_pd_for_real_index(i)
            items.append({
                "motor_id": int(mid),
                "position": float(target_real[i]),
                "speed": self.send_speed,
                "torque": self.send_torque,
                "kp": float(kp),
                "kd": self.clamp_kd(float(kd)),
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
                    f"[SEND] diag locked hip ok | q_min={float(np.min(target_real)):.3f} "
                    f"q_max={float(np.max(target_real)):.3f}"
                )
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time

        if elapsed < self.stand_sec:
            phase = 0.0
            warm = 0.0
            target_policy = self.default_policy.copy()
            leg_debug = self.default_leg_debug(phase, warm)
            self._phase_acc = 0.0
        else:
            gait_time = elapsed - self.stand_sec
            self._phase_acc = (self._phase_acc + self.step_hz * dt) % 1.0
            phase = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 1e-3))
            target_policy, leg_debug = self.build_target_policy(phase, warm)

        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True).astype(np.float32)
        target_real = self.apply_real_hip_lock(target_real)
        self._current_leg_debug = leg_debug

        self.publish_array(self.pub_target, target_real)
        self.publish_array(self.pub_phase, np.array([phase, warm, self.step_hz], dtype=np.float32))

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(target_real - self.last_target_real))) > self.max_delta:
                self.get_logger().warn("[SAFE] real target jump too large; skip send")
            else:
                sent = self.send_motion_batch(target_real)
                if sent:
                    self.last_target_real = target_real.copy()
        self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)

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
            "time", "elapsed", "phase", "warm", "active_swing_leg", "support_legs",
            "leg_name", "joint_index", "motor_id", "joint_name", "policy_joint_name",
            "leg_phase", "leg_state", "stance", "swing", "x_des", "z_des", "x_foot", "z_foot",
            "support_gate", "support_role", "thigh_target", "calf_target",
            "q_target_policy", "q_target_real", "q_current_real", "q_error_real", "torque_measured",
            "temp", "online", "error_code", "age_ms", "sent", "step_hz", "stride_length", "swing_height",
            "duty_factor", "kp", "kd", "lock_real_hips",
            "j1_deg", "j4_deg", "j7_deg", "j10_deg",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing diag locked hip data to {path}")

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(target=self.debug_collect_loop, daemon=True)
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec if self.debug_csv_period_sec > 0 else 1.0 / max(self.gait_hz, 1e-3)
        while not self._debug_stop_event.wait(period):
            with self._debug_sample_lock:
                sample = self._latest_debug_sample
                if sample is not None:
                    sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
            if sample is not None:
                self.write_debug_csv_sample(**sample)

    def update_debug_sample(self, target_real, target_policy, phase, warm, leg_debug, sent):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "target_real": np.asarray(target_real, dtype=np.float32).reshape(12).copy(),
                "target_policy": np.asarray(target_policy, dtype=np.float32).reshape(12).copy(),
                "phase": float(phase), "warm": float(warm),
                "leg_debug": {leg: dict(data) for leg, data in leg_debug.items()},
                "sent": bool(sent), "stamp": time.time(),
            }

    def write_debug_csv_sample(self, target_real, target_policy, phase, warm, leg_debug, sent, stamp):
        if self._debug_csv_writer is None:
            return
        now = time.time()
        try:
            snapshot = self.motor.get_latest()
        except Exception as exc:
            if now - self._last_feedback_warn_time > 1.0:
                self._last_feedback_warn_time = now
                self.get_logger().warn(f"[DEBUG_CSV] motor feedback read failed: {exc}")
            return

        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        q_real = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
        torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
        temp = np.asarray(snapshot.temp, dtype=np.float32).reshape(12)
        online = np.asarray(snapshot.online, dtype=bool).reshape(12)
        error_code = np.asarray(snapshot.error_code, dtype=np.int32).reshape(12)
        age_ms = np.asarray(snapshot.age_ms, dtype=np.float32).reshape(12)

        if self.torque_warn_nm > 0 and np.all(np.isfinite(torque)):
            torque_max = float(np.max(np.abs(torque)))
            if torque_max > self.torque_warn_nm and now - self._last_torque_warn_time > 1.0:
                self._last_torque_warn_time = now
                self.get_logger().warn(f"[TORQUE] measured max |torque|={torque_max:.2f}Nm > {self.torque_warn_nm:.2f}Nm")

        policy_real_order = np.zeros(12, dtype=np.float32)
        policy_real_order[self.mapper.policy_to_real_index] = target_policy
        elapsed = float(stamp) - self.start_time
        hip_deg = {leg: math.degrees(v) for leg, v in self.real_hip_targets.items()}

        for real_i, (mid, real_name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
            policy_name = self.policy_joint_names[policy_i]
            leg = policy_name.split("_", 1)[0].upper()
            info = dict(LEG_DEBUG_DEFAULT)
            info.update(leg_debug.get(leg, {}))
            kp, kd = self.get_pd_for_real_index(real_i, leg_debug)
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", f"{phase:.6f}", f"{warm:.6f}",
                info.get("active_swing_leg", ""), info.get("support_legs", ""),
                leg, int(real_i), int(mid), real_name, policy_name,
                f"{float(info.get('leg_phase', 0.0)):.6f}", info.get("leg_state", ""),
                f"{float(info.get('stance', 0.0)):.6f}", f"{float(info.get('swing', 0.0)):.6f}",
                f"{float(info.get('x_des', 0.0)):.6f}", f"{float(info.get('z_des', 0.0)):.6f}",
                f"{float(info.get('x_foot', 0.0)):.6f}", f"{float(info.get('z_foot', 0.0)):.6f}",
                f"{float(info.get('support_gate', 0.0)):.6f}", info.get("support_role", ""),
                f"{float(info.get('thigh_target', 0.0)):.6f}", f"{float(info.get('calf_target', 0.0)):.6f}",
                f"{float(policy_real_order[real_i]):.6f}", f"{float(target_real[real_i]):.6f}",
                f"{float(q_real[real_i]):.6f}", f"{float(target_real[real_i] - q_real[real_i]):.6f}",
                f"{float(torque[real_i]):.6f}", f"{float(temp[real_i]):.3f}", int(bool(online[real_i])),
                int(error_code[real_i]), f"{float(age_ms[real_i]):.2f}", int(bool(sent)),
                f"{self.step_hz:.6f}", f"{self.stride_length:.6f}", f"{self.swing_height:.6f}",
                f"{self.duty_factor:.6f}", f"{float(kp):.3f}", f"{float(kd):.3f}", int(bool(self.lock_real_hips)),
                f"{hip_deg['FR']:.3f}", f"{hip_deg['FL']:.3f}", f"{hip_deg['RR']:.3f}", f"{hip_deg['RL']:.3f}",
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
    node = FanfanDiagForwardLockedHipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
