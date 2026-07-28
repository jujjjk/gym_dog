#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big-stride forward wave gait for Fanfan.

Design goal:
- Larger, more visible forward steps than stable_calf_walk.
- One-leg-at-a-time wave gait, not diagonal trot, so the diagonal support leg has time to settle.
- Symmetric front-leg amplitudes and symmetric rear-leg amplitudes by default.
- Explicit front-leg swing load transfer:
    FL swing -> RR diagonal rear preload + RL same-side rear unload
    FR swing -> RL diagonal rear preload + RR same-side rear unload
- Conservative rate limiting and CSV logging for real-hardware debugging.

Install as a new ROS2 console script, for example in setup.py:
    'fanfan_big_stride_walk_node = mydog_policy.fanfan_big_stride_walk_node:main',
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
RIGHT_LEGS = ("FR", "RR")
LEFT_LEGS = ("FL", "RL")

# Hip sign convention inherited from the old stable_calf_walk path.
# Kept parameter-scaled and small because real-machine lateral direction can be hardware dependent.
OLD_HIP_OUTWARD_SIGNS = {"FR": 1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
LEG_SIDE = {"FR": -1.0, "RR": -1.0, "FL": 1.0, "RL": 1.0}  # right=-1, left=+1
DIAGONAL_PARTNER = {"FR": "RL", "RL": "FR", "FL": "RR", "RR": "FL"}
SAME_SIDE_REAR = {"FR": "RR", "FL": "RL"}

# New step order: rear leg and its diagonal front leg are not adjacent.
# This avoids the old RR -> FL timing problem where RR had only ~20-30 ms to settle.
DEFAULT_BIG_WALK_ORDER = ("RR", "FR", "RL", "FL")

LEG_FALLBACK_DEBUG = {
    "leg_phase": 0.0,
    "leg_state": "STANCE",
    "stance": 1.0,
    "swing": 0.0,
    "stance_shape": 0.0,
    "swing_shape": 0.0,
    "x_foot": 0.0,
    "z_foot": 0.0,
    "hip_delta": 0.0,
    "thigh_target": 0.0,
    "calf_target": 0.0,
    "thigh_ik": 0.0,
    "calf_ik": 0.0,
    "body_x_shift": 0.0,
    "body_y_shift": 0.0,
    "diag_load_gate": 0.0,
    "diag_support_leg": "",
    "same_rear_unload_gate": 0.0,
    "same_rear_unload_leg": "",
    "pre_swing_leg": "",
    "active_swing_leg": "",
    "support_legs": "",
}


class FanfanBigStrideWalkNode(Node):
    def __init__(self):
        super().__init__("fanfan_big_stride_walk_node")

        # Communication / timing.
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("step_hz", 0.62)
        self.declare_parameter("stand_sec", 3.0)
        self.declare_parameter("warmup_sec", 5.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.0)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        # Motor command gains.
        self.declare_parameter("stand_kp", 40.0)
        self.declare_parameter("stand_kd", 4.2)
        self.declare_parameter("send_kp", 45.0)
        self.declare_parameter("send_kd", 4.8)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)

        # Main gait shape. Larger than the old 0.024 m stride, but slower and wave-based.
        self.declare_parameter("stride_length", 0.038)
        self.declare_parameter("swing_height", 0.072)
        self.declare_parameter("duty_factor", 0.78)
        self.declare_parameter("walk_direction", 1.0)
        self.declare_parameter("swing_order", ",".join(DEFAULT_BIG_WALK_ORDER))
        self.declare_parameter("preload_fraction", 0.12)
        # Front-load gate timing fix: when a front leg enters swing, the diagonal rear
        # support should already be loaded instead of ramping from zero.
        self.declare_parameter("front_load_active_hold_until", 0.82)
        self.declare_parameter("front_load_post_touchdown_hold", 0.04)
        self.declare_parameter("settle_fraction", 0.08)
        self.declare_parameter("advance_start", 0.16)
        self.declare_parameter("advance_end", 0.84)

        # Front/rear symmetry. Keep left/right equal within each group.
        self.declare_parameter("front_stride_gain", 0.92)
        self.declare_parameter("rear_stride_gain", 0.82)
        self.declare_parameter("front_swing_height_gain", 1.26)
        self.declare_parameter("rear_swing_height_gain", 1.12)
        self.declare_parameter("front_thigh_delta_scale", 0.18)
        self.declare_parameter("rear_thigh_delta_scale", 0.16)
        self.declare_parameter("front_calf_lift_extra", 0.210)
        self.declare_parameter("rear_calf_lift_extra", 0.165)
        self.declare_parameter("front_x_bias", 0.004)
        self.declare_parameter("front_z_extend", -0.002)
        self.declare_parameter("front_swing_forward_unfold", 0.018)

        # Default stance correction from your latest stable setup.
        self.declare_parameter("hip_default_scale", 0.38)
        self.declare_parameter("hip_default_inward_offset", 0.010)
        self.declare_parameter("front_calf_min_rad", -1.12)
        self.declare_parameter("rear_hip_default_outward_offset", 0.006)
        self.declare_parameter("rear_thigh_default_back_offset", 0.045)

        # Support / preload logic. These are the key differences from old stable_calf_walk.
        self.declare_parameter("support_stand_tall_m", 0.006)
        self.declare_parameter("diag_support_preload_z_m", 0.012)
        self.declare_parameter("diag_support_calf_push_amp", 0.030)
        self.declare_parameter("diag_support_thigh_back_amp", 0.014)
        self.declare_parameter("diag_support_hip_amp", 0.018)
        self.declare_parameter("same_rear_unload_z_m", 0.006)
        self.declare_parameter("same_rear_calf_relief_amp", 0.018)
        self.declare_parameter("same_rear_unload_hip_amp", 0.014)
        self.declare_parameter("other_support_scale", 0.28)

        # Virtual body shift used to shape support-leg targets and debug. Positive x means rearward load request.
        self.declare_parameter("front_swing_body_x_shift_m", 0.018)
        self.declare_parameter("front_swing_body_y_shift_m", 0.022)
        self.declare_parameter("rear_swing_body_x_shift_m", 0.010)
        self.declare_parameter("rear_swing_body_y_shift_m", 0.012)
        self.declare_parameter("lateral_hip_sign", 1.0)

        # URDF geometry.
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)

        # Safety.
        self.declare_parameter("max_target_rate_rad_s", 2.1)
        self.declare_parameter("max_delta", 0.50)
        self.declare_parameter("torque_warn_nm", 6.0)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = float(self.get_parameter("stand_kd").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)

        self.stride_length = float(self.get_parameter("stride_length").value)
        self.swing_height = float(self.get_parameter("swing_height").value)
        self.duty_factor = float(self.get_parameter("duty_factor").value)
        self.walk_direction = -1.0 if float(self.get_parameter("walk_direction").value) < 0.0 else 1.0
        self.swing_order = self.parse_swing_order(str(self.get_parameter("swing_order").value))
        self.preload_fraction = float(self.get_parameter("preload_fraction").value)
        self.front_load_active_hold_until = float(
            self.get_parameter("front_load_active_hold_until").value
        )
        self.front_load_post_touchdown_hold = float(
            self.get_parameter("front_load_post_touchdown_hold").value
        )
        self.settle_fraction = float(self.get_parameter("settle_fraction").value)
        self.advance_start = float(self.get_parameter("advance_start").value)
        self.advance_end = float(self.get_parameter("advance_end").value)

        self.front_stride_gain = float(self.get_parameter("front_stride_gain").value)
        self.rear_stride_gain = float(self.get_parameter("rear_stride_gain").value)
        self.front_swing_height_gain = float(self.get_parameter("front_swing_height_gain").value)
        self.rear_swing_height_gain = float(self.get_parameter("rear_swing_height_gain").value)
        self.front_thigh_delta_scale = float(self.get_parameter("front_thigh_delta_scale").value)
        self.rear_thigh_delta_scale = float(self.get_parameter("rear_thigh_delta_scale").value)
        self.front_calf_lift_extra = float(self.get_parameter("front_calf_lift_extra").value)
        self.rear_calf_lift_extra = float(self.get_parameter("rear_calf_lift_extra").value)
        self.front_x_bias = float(self.get_parameter("front_x_bias").value)
        self.front_z_extend = float(self.get_parameter("front_z_extend").value)
        self.front_swing_forward_unfold = float(self.get_parameter("front_swing_forward_unfold").value)

        self.hip_default_scale = float(self.get_parameter("hip_default_scale").value)
        self.hip_default_inward_offset = float(self.get_parameter("hip_default_inward_offset").value)
        self.front_calf_min_rad = float(self.get_parameter("front_calf_min_rad").value)
        self.rear_hip_default_outward_offset = float(self.get_parameter("rear_hip_default_outward_offset").value)
        self.rear_thigh_default_back_offset = float(self.get_parameter("rear_thigh_default_back_offset").value)

        self.support_stand_tall_m = float(self.get_parameter("support_stand_tall_m").value)
        self.diag_support_preload_z_m = float(self.get_parameter("diag_support_preload_z_m").value)
        self.diag_support_calf_push_amp = float(self.get_parameter("diag_support_calf_push_amp").value)
        self.diag_support_thigh_back_amp = float(self.get_parameter("diag_support_thigh_back_amp").value)
        self.diag_support_hip_amp = float(self.get_parameter("diag_support_hip_amp").value)
        self.same_rear_unload_z_m = float(self.get_parameter("same_rear_unload_z_m").value)
        self.same_rear_calf_relief_amp = float(self.get_parameter("same_rear_calf_relief_amp").value)
        self.same_rear_unload_hip_amp = float(self.get_parameter("same_rear_unload_hip_amp").value)
        self.other_support_scale = float(self.get_parameter("other_support_scale").value)

        self.front_swing_body_x_shift_m = float(self.get_parameter("front_swing_body_x_shift_m").value)
        self.front_swing_body_y_shift_m = float(self.get_parameter("front_swing_body_y_shift_m").value)
        self.rear_swing_body_x_shift_m = float(self.get_parameter("rear_swing_body_x_shift_m").value)
        self.rear_swing_body_y_shift_m = float(self.get_parameter("rear_swing_body_y_shift_m").value)
        self.lateral_hip_sign = -1.0 if float(self.get_parameter("lateral_hip_sign").value) < 0.0 else 1.0

        self.thigh_length = float(self.get_parameter("thigh_length").value)
        self.calf_length = float(self.get_parameter("calf_length").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)

        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.policy_joint_names = self.mapper.get_policy_joint_names()
        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.apply_default_pose_offsets()
        self.default_real = self.mapper.policy_target_to_real_target(self.default_policy, clamp=True).astype(np.float32)
        self.default_foot_xz = self.compute_default_foot_xz()

        self.last_target_policy = self.default_policy.copy()
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
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

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_big_stride_target_real", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_big_stride_phase", 10)

        self.get_logger().warn(
            "Fanfan BIG STRIDE gait is open-loop. First run supported/hand-held, "
            "then short ground tests only. Be ready to cut power."
        )
        self.get_logger().info(
            f"big_stride_walk order={self.swing_order}, step_hz={self.step_hz:.2f}, "
            f"stride={self.stride_length:.3f}m, swing={self.swing_height:.3f}m, "
            f"duty={self.duty_factor:.2f}, kp={self.send_kp:.1f}, kd={self.send_kd:.1f}, send={self.enable_send}"
        )

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only, no motor commands sent.")

        self.timer = self.create_timer(1.0 / max(self.gait_hz, 1e-3), self.update)

    def parse_swing_order(self, text: str) -> tuple[str, str, str, str]:
        parts = [p.strip().upper() for p in text.replace(";", ",").split(",") if p.strip()]
        if len(parts) == 4 and set(parts) == set(LEG_ORDER):
            return tuple(parts)  # type: ignore[return-value]
        self.get_logger().warn(f"Invalid swing_order={text!r}; using {DEFAULT_BIG_WALK_ORDER}")
        return DEFAULT_BIG_WALK_ORDER

    @staticmethod
    def smoothstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * (3.0 - 2.0 * s)

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    @staticmethod
    def smooth_window(s: float, edge: float = 0.18) -> float:
        edge = min(0.49, max(0.001, float(edge)))
        return FanfanBigStrideWalkNode.smootherstep(s / edge) * FanfanBigStrideWalkNode.smootherstep((1.0 - s) / edge)

    def apply_default_pose_offsets(self):
        # Hip default scale and inward offset.
        self.hip_default_scale = min(1.0, max(0.0, self.hip_default_scale))
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            self.default_policy[i + 0] *= self.hip_default_scale
            # Normal hip sign convention for default inward narrowing is the same as older IK file.
            normal_sign = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}[leg]
            self.default_policy[i + 0] -= normal_sign * self.hip_default_inward_offset

        # Rear leg default posture: both rear thighs farther back, symmetrically.
        for leg in REAR_LEGS:
            i = LEG_START[leg]
            normal_sign = {"RR": -1.0, "RL": 1.0}[leg]
            self.default_policy[i + 0] += normal_sign * self.rear_hip_default_outward_offset
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset

        # Front calf limiter from the older node.
        for leg in FRONT_LEGS:
            idx = LEG_START[leg] + 2
            if float(self.default_policy[idx]) < self.front_calf_min_rad:
                self.default_policy[idx] = self.front_calf_min_rad

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
        # For one-leg wave gait, keep swing_fraction below 0.25 so no two swing windows overlap.
        return min(0.235, max(0.12, 1.0 - min(max(self.duty_factor, 0.70), 0.88)))

    def active_swing_leg(self, phase: float) -> Optional[str]:
        sf = self.swing_fraction()
        for leg in self.swing_order:
            if self.leg_phase(leg, phase) < sf:
                return leg
        return None

    def pre_swing_leg(self, phase: float) -> Optional[str]:
        pf = min(0.22, max(0.03, self.preload_fraction))
        for leg in self.swing_order:
            p = self.leg_phase(leg, phase)
            if p >= 1.0 - pf:
                return leg
        return None

    def front_load_leg_and_gate(
        self,
        phase: float,
        active_leg: Optional[str],
        pre_leg: Optional[str],
    ) -> tuple[Optional[str], float]:
        """Return which front leg is being unloaded and how strongly.

        Important timing fix:
        The previous version ramped the load gate from 0 again when FR/FL entered
        swing. On the real robot this meant the front foot was already trying to
        lift while body load was still on that same foot. Here the pre-swing
        window ramps the gate to 1, and once the front leg enters swing the gate
        stays high through most of swing, only tapering near touchdown.
        """
        sf = self.swing_fraction()
        gate = 0.0
        load_leg: Optional[str] = None

        def take(candidate: Optional[str], candidate_gate: float):
            nonlocal gate, load_leg
            candidate_gate = float(min(1.0, max(0.0, candidate_gate)))
            if candidate in FRONT_LEGS and candidate_gate > gate:
                gate = candidate_gate
                load_leg = candidate

        # 1) Before a front leg swings, ramp load transfer up early.
        if pre_leg in FRONT_LEGS:
            pf = min(0.24, max(0.04, self.preload_fraction))
            p = self.leg_phase(pre_leg, phase)
            s = (p - (1.0 - pf)) / max(pf, 1e-6)
            take(pre_leg, self.smootherstep(s))

        # 2) During active front swing, do NOT restart from zero.
        #    The diagonal rear support should already be carrying load at lift-off.
        if active_leg in FRONT_LEGS:
            p = self.leg_phase(active_leg, phase)
            s = p / max(sf, 1e-6)
            hold_until = min(0.96, max(0.35, self.front_load_active_hold_until))
            if s <= hold_until:
                active_gate = 1.0
            else:
                # Taper only near touchdown so the foot can land without a sharp release.
                active_gate = self.smootherstep((1.0 - s) / max(1.0 - hold_until, 1e-6))
            take(active_leg, active_gate)

        # 3) Short hold just after touchdown. Keep this small because the next rear
        #    leg may soon become the swing leg in the RR->FR->RL->FL order.
        post_hold = min(0.12, max(0.0, self.front_load_post_touchdown_hold))
        if post_hold > 1e-6:
            for leg in FRONT_LEGS:
                p = self.leg_phase(leg, phase)
                if sf <= p < sf + post_hold:
                    post_s = (p - sf) / max(post_hold, 1e-6)
                    take(leg, self.smootherstep(1.0 - post_s))

        return load_leg, float(min(1.0, max(0.0, gate)))

    def desired_body_shift(self, active_leg: Optional[str], front_load_leg: Optional[str], front_load_gate: float) -> tuple[float, float]:
        # Debug/virtual shift. Positive y means shift/load toward left, negative toward right.
        if front_load_leg == "FL" and front_load_gate > 0.0:
            return (
                self.front_swing_body_x_shift_m * front_load_gate,
                -self.front_swing_body_y_shift_m * front_load_gate,
            )
        if front_load_leg == "FR" and front_load_gate > 0.0:
            return (
                self.front_swing_body_x_shift_m * front_load_gate,
                self.front_swing_body_y_shift_m * front_load_gate,
            )
        if active_leg in REAR_LEGS:
            side = LEG_SIDE[active_leg]
            return (
                self.rear_swing_body_x_shift_m,
                -side * self.rear_swing_body_y_shift_m,
            )
        return 0.0, 0.0

    def build_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        sf = self.swing_fraction()
        active = self.active_swing_leg(phase)
        pre = self.pre_swing_leg(phase)
        front_load_leg, front_load_gate = self.front_load_leg_and_gate(phase, active, pre)
        body_x_shift, body_y_shift = self.desired_body_shift(active, front_load_leg, front_load_gate)

        support_legs = [leg for leg in LEG_ORDER if leg != active]

        for leg in LEG_ORDER:
            p = self.leg_phase(leg, phase)
            is_front = leg in FRONT_LEGS
            is_rear = leg in REAR_LEGS
            is_swing = p < sf
            i = LEG_START[leg]
            x0, z0 = self.default_foot_xz[leg]
            x_center = x0 + (self.front_x_bias if is_front else 0.0)
            z_center = z0 + (self.front_z_extend if is_front else 0.0)
            stride_gain = self.front_stride_gain if is_front else self.rear_stride_gain
            height_gain = self.front_swing_height_gain if is_front else self.rear_swing_height_gain
            thigh_scale = self.front_thigh_delta_scale if is_front else self.rear_thigh_delta_scale
            calf_lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
            stride = self.stride_length * stride_gain
            swing_h = self.swing_height * height_gain

            hip_delta = 0.0
            calf_push = 0.0
            thigh_bias = 0.0
            diag_load_gate = 0.0
            same_rear_unload_gate = 0.0
            diag_support_leg = ""
            same_rear_unload_leg = ""
            stance_shape = 0.0
            swing_shape = 0.0

            if is_swing:
                s = p / max(sf, 1e-6)
                denom = max(1e-6, self.advance_end - self.advance_start)
                u = self.smootherstep((s - self.advance_start) / denom)
                lift_up = self.smootherstep(s / max(self.advance_start, 0.001))
                lift_down = self.smootherstep((1.0 - s) / max(1.0 - self.advance_end, 0.001))
                swing_shape = lift_up * lift_down
                x_des = x_center + self.walk_direction * (-0.5 * stride + stride * u)
                if is_front:
                    x_des += self.walk_direction * self.front_swing_forward_unfold * swing_shape
                z_des = z_center + swing_h * swing_shape
                calf_push += -calf_lift_extra * swing_shape
                # Swing hip relax: tiny, only to avoid dragging sideways.
                hip_delta += -0.004 * OLD_HIP_OUTWARD_SIGNS[leg] * swing_shape * self.lateral_hip_sign
                leg_state = "SWING"
            else:
                s = (p - sf) / max(1.0 - sf, 1e-6)
                u = self.smootherstep(s)
                stance_shape = math.sin(math.pi * min(1.0, max(0.0, s))) ** 2
                stance_gate = self.smooth_window(s, edge=0.18)
                x_des = x_center + self.walk_direction * (0.5 * stride - stride * u)
                z_des = z_center - self.support_stand_tall_m * (0.35 + 0.65 * stance_shape)

                # General support: support legs are slightly steadier during any swing.
                if active is not None:
                    calf_push += 0.006 * stance_shape * stance_gate

                # Front swing diagonal rear preload and same-side rear unload.
                if front_load_leg in FRONT_LEGS and front_load_gate > 0.0 and is_rear:
                    diag_rear = DIAGONAL_PARTNER[front_load_leg]
                    same_rear = SAME_SIDE_REAR[front_load_leg]
                    if leg == diag_rear:
                        diag_load_gate = front_load_gate
                        diag_support_leg = diag_rear
                        z_des -= self.diag_support_preload_z_m * front_load_gate
                        # Bias target as if body is moving rearward relative to support foot.
                        x_des += self.front_swing_body_x_shift_m * front_load_gate
                        calf_push += self.diag_support_calf_push_amp * front_load_gate * (0.65 + 0.35 * stance_shape)
                        thigh_bias += self.diag_support_thigh_back_amp * front_load_gate
                        hip_delta += (
                            OLD_HIP_OUTWARD_SIGNS[leg]
                            * self.diag_support_hip_amp
                            * front_load_gate
                            * self.lateral_hip_sign
                        )
                    elif leg == same_rear:
                        same_rear_unload_gate = front_load_gate
                        same_rear_unload_leg = same_rear
                        z_des += self.same_rear_unload_z_m * front_load_gate
                        calf_push -= self.same_rear_calf_relief_amp * front_load_gate
                        hip_delta -= (
                            OLD_HIP_OUTWARD_SIGNS[leg]
                            * self.same_rear_unload_hip_amp
                            * front_load_gate
                            * self.lateral_hip_sign
                        )
                    else:
                        z_des -= self.diag_support_preload_z_m * self.other_support_scale * front_load_gate
                        calf_push += self.diag_support_calf_push_amp * self.other_support_scale * front_load_gate

                # Rear swing diagonal front support: smaller than front->rear transfer.
                if active in REAR_LEGS and is_front:
                    diag_front = DIAGONAL_PARTNER[active]
                    rear_gate = 0.65 + 0.35 * stance_shape
                    if leg == diag_front:
                        z_des -= 0.006 * rear_gate
                        calf_push += 0.018 * rear_gate * stance_shape
                    else:
                        z_des -= 0.003 * rear_gate
                        calf_push += 0.006 * rear_gate * stance_shape

                leg_state = "STANCE"

            thigh_ik, calf_ik = self.inverse_sagittal(x_des, z_des)
            thigh_target = float(self.default_policy[i + 1]) + thigh_scale * (thigh_ik - float(self.default_policy[i + 1]))
            thigh_target += thigh_bias
            # Re-solve calf after thigh bias so foot height target remains meaningful.
            calf_target = self.solve_calf_for_z(thigh_target, z_des, float(self.default_policy[i + 2]))
            calf_target += calf_push

            if is_front and calf_target < self.front_calf_min_rad:
                calf_target = self.front_calf_min_rad

            # Apply warmup interpolation.
            q[i + 0] = float(self.default_policy[i + 0]) + warm * hip_delta
            q[i + 1] = float(self.default_policy[i + 1]) + warm * (thigh_target - float(self.default_policy[i + 1]))
            q[i + 2] = float(self.default_policy[i + 2]) + warm * (calf_target - float(self.default_policy[i + 2]))

            x_actual, z_actual = self.forward_sagittal(q[i + 1], q[i + 2])
            leg_debug[leg] = {
                "leg_phase": float(p),
                "leg_state": leg_state,
                "stance": float(0.0 if is_swing else 1.0),
                "swing": float(1.0 if is_swing else 0.0),
                "stance_shape": float(stance_shape),
                "swing_shape": float(swing_shape),
                "x_foot": float(x_actual),
                "z_foot": float(z_actual),
                "hip_delta": float(hip_delta),
                "thigh_target": float(thigh_target),
                "calf_target": float(calf_target),
                "thigh_ik": float(thigh_ik),
                "calf_ik": float(calf_ik),
                "body_x_shift": float(body_x_shift),
                "body_y_shift": float(body_y_shift),
                "diag_load_gate": float(diag_load_gate),
                "diag_support_leg": diag_support_leg,
                "same_rear_unload_gate": float(same_rear_unload_gate),
                "same_rear_unload_leg": same_rear_unload_leg,
                "pre_swing_leg": pre or "",
                "active_swing_leg": active or "",
                "support_legs": ",".join(support_legs),
            }

        return q.astype(np.float32), leg_debug

    def solve_calf_for_z(self, thigh: float, z_des: float, calf_default: float) -> float:
        value = (-float(z_des) - self.thigh_length * math.cos(float(thigh))) / max(self.calf_length, 1e-9)
        value = min(1.0, max(-1.0, value))
        angle = math.acos(value)
        candidates = (angle - thigh, -angle - thigh)
        return float(min(candidates, key=lambda calf: abs(calf - calf_default)))

    def send_default_stand(self):
        items = []
        for mid, pos in zip(self.motor_ids, self.default_real):
            items.append({
                "motor_id": int(mid),
                "position": float(pos),
                "speed": 0.0,
                "torque": 0.0,
                "kp": self.stand_kp,
                "kd": self.stand_kd,
            })
        payload = {"items": items, "enable_first": True, "stop_first": False}
        r = self.http_session.post(
            f"{self.motor_base_url}/api/rs04/motion_mode_run_batch",
            json=payload,
            timeout=max(self.http_timeout, 0.5),
        )
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")
        self.get_logger().info(f"Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}")

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time

        if elapsed < self.stand_sec:
            target_policy = self.default_policy.copy()
            phase = 0.0
            warm = 0.0
            leg_debug = self.default_leg_debug(phase, warm)
            self._phase_acc = 0.0
        else:
            gait_time = elapsed - self.stand_sec
            self._phase_acc = (self._phase_acc + self.step_hz * dt) % 1.0
            phase = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 1e-3))
            target_policy, leg_debug = self.build_target_policy(phase, warm)

        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True)
        self.publish_array(self.pub_target, target_real)
        self.publish_array(self.pub_phase, np.array([phase, warm, self.step_hz], dtype=np.float32))

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(target_real - self.last_target_real))) > self.max_delta:
                self.get_logger().warn("[SAFE] target jump too large; skip send")
            else:
                sent = self.send_motion_batch(target_real)
                if sent:
                    self.last_target_real = target_real.copy()
        self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)

    def default_leg_debug(self, phase: float, warm: float) -> dict[str, dict[str, float]]:
        data = {}
        for leg in LEG_ORDER:
            row = dict(LEG_FALLBACK_DEBUG)
            x, z = self.default_foot_xz[leg]
            row.update({
                "leg_phase": float(self.leg_phase(leg, phase)),
                "x_foot": float(x),
                "z_foot": float(z),
                "warm": float(warm),
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
        limited = self.last_target_policy + step
        limited = np.clip(limited, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.last_target_policy = limited.astype(np.float32).copy()
        return self.last_target_policy.copy()

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            items.append({
                "motor_id": int(mid),
                "position": float(target_real[i]),
                "speed": self.send_speed,
                "torque": self.send_torque,
                "kp": self.send_kp,
                "kd": self.send_kd,
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
                    f"[SEND] big stride ok | target_min={float(np.min(target_real)):.3f} "
                    f"target_max={float(np.max(target_real)):.3f} kp={self.send_kp:.1f} kd={self.send_kd:.1f}"
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
            "time", "elapsed", "phase", "warm", "active_swing_leg", "pre_swing_leg", "support_legs",
            "leg_name", "joint_index", "motor_id", "joint_name", "policy_joint_name",
            "leg_phase", "leg_state", "stance", "swing", "x_foot", "z_foot",
            "body_x_shift", "body_y_shift", "diag_load_gate", "diag_support_leg",
            "same_rear_unload_gate", "same_rear_unload_leg", "hip_delta", "thigh_target", "calf_target",
            "q_target_policy", "q_target_real", "q_current_real", "q_error_real", "torque_measured",
            "temp", "online", "error_code", "age_ms", "sent", "step_hz", "stride_length", "swing_height",
            "duty_factor", "kp", "kd",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing big stride data to {path}")

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(target=self.debug_collect_loop, daemon=True)
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec if self.debug_csv_period_sec > 0.0 else 1.0 / max(self.gait_hz, 1e-3)
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
                "phase": float(phase),
                "warm": float(warm),
                "leg_debug": {leg: dict(data) for leg, data in leg_debug.items()},
                "sent": bool(sent),
                "stamp": time.time(),
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

        if self.torque_warn_nm > 0.0:
            torque_abs_max = float(np.max(np.abs(torque)))
            if torque_abs_max > self.torque_warn_nm and now - self._last_torque_warn_time > 1.0:
                self._last_torque_warn_time = now
                self.get_logger().warn(
                    f"[TORQUE] measured max |torque|={torque_abs_max:.2f}Nm > {self.torque_warn_nm:.2f}Nm. "
                    "Reduce stride_length/swing_height/step_hz before raising Kp."
                )

        target_policy_real_order = np.zeros(12, dtype=np.float32)
        target_policy_real_order[self.mapper.policy_to_real_index] = target_policy
        elapsed = float(stamp) - self.start_time

        for real_i, (mid, real_name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
            policy_name = self.policy_joint_names[policy_i]
            leg = policy_name.split("_", 1)[0]
            info = dict(LEG_FALLBACK_DEBUG)
            info.update(leg_debug.get(leg, {}))
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", f"{phase:.6f}", f"{warm:.6f}",
                info.get("active_swing_leg", ""), info.get("pre_swing_leg", ""), info.get("support_legs", ""),
                leg, int(real_i), int(mid), real_name, policy_name,
                f"{float(info.get('leg_phase', 0.0)):.6f}", info.get("leg_state", ""),
                f"{float(info.get('stance', 0.0)):.6f}", f"{float(info.get('swing', 0.0)):.6f}",
                f"{float(info.get('x_foot', 0.0)):.6f}", f"{float(info.get('z_foot', 0.0)):.6f}",
                f"{float(info.get('body_x_shift', 0.0)):.6f}", f"{float(info.get('body_y_shift', 0.0)):.6f}",
                f"{float(info.get('diag_load_gate', 0.0)):.6f}", info.get("diag_support_leg", ""),
                f"{float(info.get('same_rear_unload_gate', 0.0)):.6f}", info.get("same_rear_unload_leg", ""),
                f"{float(info.get('hip_delta', 0.0)):.6f}", f"{float(info.get('thigh_target', 0.0)):.6f}",
                f"{float(info.get('calf_target', 0.0)):.6f}",
                f"{float(target_policy_real_order[real_i]):.6f}", f"{float(target_real[real_i]):.6f}",
                f"{float(q_real[real_i]):.6f}", f"{float(target_real[real_i] - q_real[real_i]):.6f}",
                f"{float(torque[real_i]):.6f}", f"{float(temp[real_i]):.3f}",
                int(bool(online[real_i])), int(error_code[real_i]), f"{float(age_ms[real_i]):.2f}",
                int(bool(sent)), f"{self.step_hz:.6f}", f"{self.stride_length:.6f}", f"{self.swing_height:.6f}",
                f"{self.duty_factor:.6f}", f"{self.send_kp:.3f}", f"{self.send_kd:.3f}",
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
    node = FanfanBigStrideWalkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
