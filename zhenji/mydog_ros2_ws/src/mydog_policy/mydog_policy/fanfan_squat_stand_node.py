#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fanfan play-bow squat + rear-rump wag ROS2 node.

Purpose:
- Open-loop cute gesture: default stand -> front-heavy half squat / play bow -> rear haunch lift + hip wag -> stand up.
- Front legs can bend more than rear legs for a dog-like "bow" posture.
- Rear legs can extend slightly during the squat hold to lift the rear haunch before wagging.
- Designed for flat-ground / supported testing first.
- Uses IK in the sagittal plane and keeps all Kd <= 5.0 for RS01 MIT/motion mode.
- This revision removes the old 0.070 rad rear-hip wag cap and adds explicit front-calf bend for a stronger play-bow posture.

Install in setup.py console_scripts, for example:
    'fanfan_squat_stand_node = mydog_policy.fanfan_squat_stand_node:main',
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


class FanfanSquatStandNode(Node):
    def __init__(self):
        super().__init__("fanfan_squat_stand_node")

        # Communication / timing.
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("stand_sec", 2.5)
        self.declare_parameter("squat_down_sec", 2.0)
        self.declare_parameter("squat_hold_sec", 1.2)
        self.declare_parameter("stand_up_sec", 2.0)
        self.declare_parameter("finish_hold_sec", 2.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("http_timeout", 0.08)

        # CSV debug.
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.02)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        # Motion shape.
        # Positive squat_depth_m means body goes down while feet are assumed on ground.
        # In hip frame, foot z becomes less negative, i.e. leg shortens.
        self.declare_parameter("squat_depth_m", 0.040)
        # For the cute play-bow gesture: front legs bend more, rear legs stay taller.
        self.declare_parameter("front_depth_scale", 1.20)
        self.declare_parameter("rear_depth_scale", 0.65)
        self.declare_parameter("x_shift_m", 0.000)
        self.declare_parameter("front_x_shift_m", 0.000)
        self.declare_parameter("rear_x_shift_m", 0.000)
        self.declare_parameter("min_leg_length_margin_m", 0.012)

        # Default pose correction, consistent with your big-stride node.
        self.declare_parameter("hip_default_scale", 0.38)
        self.declare_parameter("hip_default_inward_offset", 0.014)
        self.declare_parameter("rear_hip_default_outward_offset", 0.002)
        self.declare_parameter("rear_thigh_default_back_offset", 0.038)
        self.declare_parameter("front_calf_min_rad", -1.28)
        # Extra negative calf angle applied only during the squat profile.
        # Positive value means the front calves fold more, giving a lower front / butt-up play-bow posture.
        self.declare_parameter("front_calf_bend_extra_rad", 0.0)
        self.declare_parameter("rear_calf_bend_extra_rad", 0.0)

        # Optional walking/squat hip posture: only applied during squat profile.
        self.declare_parameter("squat_hip_inward_offset", 0.004)
        self.declare_parameter("squat_hip_knee_stability_amp", 0.000)

        # Cute rear-hip wag. This is intentionally small and gated by squat alpha/state.
        # style=sway: both rear hip targets move in the same policy-sign direction, creating a side-to-side rump sway.
        # style=mirror: RR/RL move symmetrically inward/outward. Use only if your real machine's sign convention looks better.
        self.declare_parameter("rear_hip_wag_enable", True)
        self.declare_parameter("rear_hip_wag_amp", 0.032)
        self.declare_parameter("rear_hip_wag_hz", 1.35)
        self.declare_parameter("rear_hip_wag_phase", 0.0)
        self.declare_parameter("rear_hip_wag_style", "sway")
        self.declare_parameter("rear_hip_wag_hold_only", True)
        self.declare_parameter("rear_hip_wag_alpha_min", 0.70)
        self.declare_parameter("rear_hip_wag_gate", 1.0)

        # Rear haunch lift during squat-hold. Positive value extends rear legs,
        # making the rear end look raised while the front legs remain lower.
        self.declare_parameter("rear_haunch_lift_enable", True)
        self.declare_parameter("rear_haunch_lift_m", 0.020)
        self.declare_parameter("rear_haunch_lift_hold_only", True)
        self.declare_parameter("rear_haunch_lift_ramp_fraction", 0.25)

        # URDF geometry.
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)

        # Safety / target smoothing.
        self.declare_parameter("max_target_rate_rad_s", 1.20)
        self.declare_parameter("max_delta", 0.30)
        self.declare_parameter("torque_warn_nm", 6.0)

        # Generic fallback PD.
        self.declare_parameter("stand_kp", 40.0)
        self.declare_parameter("stand_kd", 4.2)
        self.declare_parameter("send_kp", 42.0)
        self.declare_parameter("send_kd", 5.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)

        # Group PD. Kd is clamped in code to <= 5.0.
        self.declare_parameter("use_group_pd", True)
        self.declare_parameter("front_hip_kp", 30.0)
        self.declare_parameter("front_hip_kd", 5.0)
        self.declare_parameter("front_thigh_kp", 44.0)
        self.declare_parameter("front_thigh_kd", 5.0)
        self.declare_parameter("front_calf_kp", 40.0)
        self.declare_parameter("front_calf_kd", 5.0)
        self.declare_parameter("rear_hip_kp", 32.0)
        self.declare_parameter("rear_hip_kd", 5.0)
        self.declare_parameter("rear_thigh_kp", 42.0)
        self.declare_parameter("rear_thigh_kd", 5.0)
        self.declare_parameter("rear_calf_kp", 36.0)
        self.declare_parameter("rear_calf_kd", 5.0)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.squat_down_sec = float(self.get_parameter("squat_down_sec").value)
        self.squat_hold_sec = float(self.get_parameter("squat_hold_sec").value)
        self.stand_up_sec = float(self.get_parameter("stand_up_sec").value)
        self.finish_hold_sec = float(self.get_parameter("finish_hold_sec").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)

        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.squat_depth_m = float(self.get_parameter("squat_depth_m").value)
        self.front_depth_scale = float(self.get_parameter("front_depth_scale").value)
        self.rear_depth_scale = float(self.get_parameter("rear_depth_scale").value)
        self.x_shift_m = float(self.get_parameter("x_shift_m").value)
        self.front_x_shift_m = float(self.get_parameter("front_x_shift_m").value)
        self.rear_x_shift_m = float(self.get_parameter("rear_x_shift_m").value)
        self.min_leg_length_margin_m = float(self.get_parameter("min_leg_length_margin_m").value)

        self.hip_default_scale = float(self.get_parameter("hip_default_scale").value)
        self.hip_default_inward_offset = float(self.get_parameter("hip_default_inward_offset").value)
        self.rear_hip_default_outward_offset = float(self.get_parameter("rear_hip_default_outward_offset").value)
        self.rear_thigh_default_back_offset = float(self.get_parameter("rear_thigh_default_back_offset").value)
        self.front_calf_min_rad = float(self.get_parameter("front_calf_min_rad").value)
        self.front_calf_bend_extra_rad = float(self.get_parameter("front_calf_bend_extra_rad").value)
        self.rear_calf_bend_extra_rad = float(self.get_parameter("rear_calf_bend_extra_rad").value)
        self.squat_hip_inward_offset = float(self.get_parameter("squat_hip_inward_offset").value)
        self.squat_hip_knee_stability_amp = float(self.get_parameter("squat_hip_knee_stability_amp").value)
        self.rear_hip_wag_enable = bool(self.get_parameter("rear_hip_wag_enable").value)
        self.rear_hip_wag_amp = float(self.get_parameter("rear_hip_wag_amp").value)
        self.rear_hip_wag_hz = float(self.get_parameter("rear_hip_wag_hz").value)
        self.rear_hip_wag_phase = float(self.get_parameter("rear_hip_wag_phase").value)
        self.rear_hip_wag_style = str(self.get_parameter("rear_hip_wag_style").value).strip().lower()
        self.rear_hip_wag_hold_only = bool(self.get_parameter("rear_hip_wag_hold_only").value)
        self.rear_hip_wag_alpha_min = float(self.get_parameter("rear_hip_wag_alpha_min").value)
        self.rear_hip_wag_gate = float(self.get_parameter("rear_hip_wag_gate").value)
        self.rear_haunch_lift_enable = bool(self.get_parameter("rear_haunch_lift_enable").value)
        self.rear_haunch_lift_m = float(self.get_parameter("rear_haunch_lift_m").value)
        self.rear_haunch_lift_hold_only = bool(self.get_parameter("rear_haunch_lift_hold_only").value)
        self.rear_haunch_lift_ramp_fraction = float(self.get_parameter("rear_haunch_lift_ramp_fraction").value)

        self.thigh_length = float(self.get_parameter("thigh_length").value)
        self.calf_length = float(self.get_parameter("calf_length").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = self.clamp_kd(float(self.get_parameter("stand_kd").value))
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = self.clamp_kd(float(self.get_parameter("send_kd").value))
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.use_group_pd = bool(self.get_parameter("use_group_pd").value)

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

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_squat_target_real", 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, "/mydog/fanfan_squat_phase", 10)

        self.get_logger().warn(
            "Fanfan squat-stand is open-loop. First run supported/hand-held. "
            "Use small squat_depth_m before increasing depth."
        )
        self.get_logger().info(
            f"squat_depth={self.squat_depth_m:.3f}m, down={self.squat_down_sec:.2f}s, "
            f"hold={self.squat_hold_sec:.2f}s, up={self.stand_up_sec:.2f}s, loop={self.loop}, "
            f"wag={self.rear_hip_wag_enable} amp={self.rear_hip_wag_amp:.3f} hz={self.rear_hip_wag_hz:.2f}, "
            f"rear_lift={self.rear_haunch_lift_enable} lift={self.rear_haunch_lift_m:.3f}m, send={self.enable_send}"
        )

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only, no motor commands sent.")

        self.timer = self.create_timer(1.0 / max(self.gait_hz, 1e-3), self.update)

    @staticmethod
    def clamp_kd(kd: float) -> float:
        return float(min(5.0, max(0.0, kd)))

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    def apply_default_pose_offsets(self):
        self.hip_default_scale = min(1.0, max(0.0, self.hip_default_scale))
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            self.default_policy[i + 0] *= self.hip_default_scale
            normal_sign = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}[leg]
            self.default_policy[i + 0] -= normal_sign * self.hip_default_inward_offset

        for leg in REAR_LEGS:
            i = LEG_START[leg]
            normal_sign = {"RR": -1.0, "RL": 1.0}[leg]
            self.default_policy[i + 0] += normal_sign * self.rear_hip_default_outward_offset
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset

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
        max_r = self.thigh_length + self.calf_length - max(1e-5, self.min_leg_length_margin_m)
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

    def squat_profile(self, elapsed: float) -> tuple[float, str, float]:
        """Return squat alpha in [0,1], state name, and local progress."""
        cycle = self.squat_down_sec + self.squat_hold_sec + self.stand_up_sec + self.finish_hold_sec
        t = max(0.0, elapsed - self.stand_sec)
        if self.loop and cycle > 1e-6:
            t = t % cycle

        if elapsed < self.stand_sec:
            return 0.0, "INIT_STAND", elapsed / max(self.stand_sec, 1e-6)
        if t < self.squat_down_sec:
            s = t / max(self.squat_down_sec, 1e-6)
            return self.smootherstep(s), "SQUAT_DOWN", s
        t -= self.squat_down_sec
        if t < self.squat_hold_sec:
            return 1.0, "SQUAT_HOLD", t / max(self.squat_hold_sec, 1e-6)
        t -= self.squat_hold_sec
        if t < self.stand_up_sec:
            s = t / max(self.stand_up_sec, 1e-6)
            return 1.0 - self.smootherstep(s), "STAND_UP", s
        return 0.0, "FINISH_STAND", min(1.0, t / max(self.finish_hold_sec, 1e-6))

    def compute_rear_hip_wag(self, elapsed: float, alpha: float, state: str) -> tuple[float, float]:
        if not self.rear_hip_wag_enable:
            return 0.0, 0.0
        alpha = float(min(1.0, max(0.0, alpha)))
        amin = min(0.98, max(0.0, self.rear_hip_wag_alpha_min))
        alpha_gate = self.smootherstep((alpha - amin) / max(1.0 - amin, 1e-6))
        if self.rear_hip_wag_hold_only and state != "SQUAT_HOLD":
            alpha_gate = 0.0
        gate = float(min(1.0, max(0.0, self.rear_hip_wag_gate))) * alpha_gate
        if gate <= 1e-6:
            return 0.0, 0.0
        # No hard 0.070 rad cap here: user-requested rear-hip wag amplitude is used directly.
        # Keep the sign so negative rear_hip_wag_amp can reverse the visual wag direction if needed.
        # Remaining protection comes from joint limits, max_target_rate_rad_s, max_delta, and Kd clamping.
        amp = float(self.rear_hip_wag_amp)
        hz = min(3.0, max(0.0, float(self.rear_hip_wag_hz)))
        phase = float(self.rear_hip_wag_phase)
        wag = amp * gate * math.sin(2.0 * math.pi * hz * elapsed + phase)
        return float(wag), gate

    def compute_rear_haunch_lift_gate(self, alpha: float, state: str, progress: float) -> float:
        if not self.rear_haunch_lift_enable:
            return 0.0
        alpha = float(min(1.0, max(0.0, alpha)))
        if self.rear_haunch_lift_hold_only and state != "SQUAT_HOLD":
            return 0.0
        if state == "SQUAT_HOLD":
            ramp = min(0.80, max(0.05, float(self.rear_haunch_lift_ramp_fraction)))
            hold_gate = self.smootherstep(float(progress) / max(ramp, 1e-6))
        else:
            hold_gate = 1.0
        return float(min(1.0, max(0.0, alpha * hold_gate)))

    def build_target_policy(self, alpha: float, state: str, elapsed: float, progress: float) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        q = self.default_policy.copy()
        debug = {}
        alpha = float(min(1.0, max(0.0, alpha)))
        rear_wag, rear_wag_gate = self.compute_rear_hip_wag(elapsed, alpha, state)
        rear_lift_gate = self.compute_rear_haunch_lift_gate(alpha, state, progress)
        rear_lift_m = min(0.045, max(0.0, abs(float(self.rear_haunch_lift_m)))) * rear_lift_gate

        for leg in LEG_ORDER:
            i = LEG_START[leg]
            is_front = leg in FRONT_LEGS
            x0, z0 = self.default_foot_xz[leg]
            depth_scale = self.front_depth_scale if is_front else self.rear_depth_scale
            x_extra = self.x_shift_m + (self.front_x_shift_m if is_front else self.rear_x_shift_m)
            x_des = x0 + x_extra * alpha
            z_des = z0 + self.squat_depth_m * depth_scale * alpha
            # During the cute hold, extend the rear legs a little. This raises the rear haunch
            # while the front legs stay deeper, giving a play-bow + rump-wag look.
            rear_haunch_lift_applied = 0.0
            if (not is_front) and rear_lift_m > 0.0:
                z_des -= rear_lift_m
                rear_haunch_lift_applied = rear_lift_m
            thigh, calf = self.inverse_sagittal(x_des, z_des)
            calf_bend_extra = 0.0
            if is_front and self.front_calf_bend_extra_rad != 0.0:
                calf_bend_extra = -abs(float(self.front_calf_bend_extra_rad)) * alpha
                calf += calf_bend_extra
            elif (not is_front) and self.rear_calf_bend_extra_rad != 0.0:
                calf_bend_extra = -abs(float(self.rear_calf_bend_extra_rad)) * alpha
                calf += calf_bend_extra

            hip_delta = 0.0
            if self.squat_hip_inward_offset != 0.0:
                inward_sign = {"FR": 1.0, "FL": -1.0, "RR": 1.0, "RL": -1.0}[leg]
                hip_delta += inward_sign * self.squat_hip_inward_offset * alpha
            if self.squat_hip_knee_stability_amp != 0.0:
                stability_sign = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}[leg]
                hip_delta += stability_sign * self.squat_hip_knee_stability_amp * alpha

            rear_hip_wag_delta = 0.0
            if leg in REAR_LEGS and rear_wag_gate > 0.0:
                if self.rear_hip_wag_style == "mirror":
                    wag_sign = {"RR": -1.0, "RL": 1.0}[leg]
                else:
                    # Sway mode: same policy-sign command on both rear hips.
                    # If the visual direction is wrong on your machine, flip rear_hip_wag_amp sign.
                    wag_sign = 1.0
                rear_hip_wag_delta = wag_sign * rear_wag
                hip_delta += rear_hip_wag_delta

            q[i + 0] = float(self.default_policy[i + 0]) + hip_delta
            q[i + 1] = thigh
            q[i + 2] = calf

            q[i:i+3] = np.clip(q[i:i+3], self.mapper.policy_lower_limit[i:i+3], self.mapper.policy_upper_limit[i:i+3])
            x_actual, z_actual = self.forward_sagittal(float(q[i + 1]), float(q[i + 2]))
            debug[leg] = {
                "alpha": alpha,
                "x_des": float(x_des),
                "z_des": float(z_des),
                "x_foot": float(x_actual),
                "z_foot": float(z_actual),
                "hip_delta": float(hip_delta),
                "rear_hip_wag_delta": float(rear_hip_wag_delta),
                "rear_hip_wag_gate": float(rear_wag_gate),
                "rear_haunch_lift_gate": float(rear_lift_gate),
                "rear_haunch_lift_m": float(rear_haunch_lift_applied),
                "calf_bend_extra": float(calf_bend_extra),
                "thigh_target": float(q[i + 1]),
                "calf_target": float(q[i + 2]),
                "depth_m": float(self.squat_depth_m * depth_scale * alpha),
            }
        return q.astype(np.float32), debug

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

    def get_joint_pd(self, real_i: int, state: str) -> tuple[float, float]:
        if state == "INIT_STAND" or not self.use_group_pd:
            return self.send_kp, self.send_kd
        policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
        policy_name = self.policy_joint_names[policy_i].lower()
        is_front = policy_name.startswith("fr_") or policy_name.startswith("fl_")
        if "hip" in policy_name:
            return (self.front_hip_kp, self.front_hip_kd) if is_front else (self.rear_hip_kp, self.rear_hip_kd)
        if "thigh" in policy_name:
            return (self.front_thigh_kp, self.front_thigh_kd) if is_front else (self.rear_thigh_kp, self.rear_thigh_kd)
        if "calf" in policy_name:
            return (self.front_calf_kp, self.front_calf_kd) if is_front else (self.rear_calf_kp, self.rear_calf_kd)
        return self.send_kp, self.send_kd

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
        r = self.http_session.post(
            f"{self.motor_base_url}/api/rs04/motion_mode_run_batch",
            json={"items": items, "enable_first": True, "stop_first": False},
            timeout=max(self.http_timeout, 0.5),
        )
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")
        self.get_logger().info(f"Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}")

    def send_motion_batch(self, target_real: np.ndarray, state: str) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            kp, kd = self.get_joint_pd(i, state)
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
                    f"[SEND] squat ok | state={state} target_min={float(np.min(target_real)):.3f} "
                    f"target_max={float(np.max(target_real)):.3f}"
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
        alpha, state, progress = self.squat_profile(elapsed)
        target_policy, leg_debug = self.build_target_policy(alpha, state, elapsed, progress)
        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True).astype(np.float32)

        self.publish_array(self.pub_target, target_real)
        self.publish_array(self.pub_phase, np.array([alpha, progress, float(state != "FINISH_STAND")], dtype=np.float32))

        sent = False
        if self.enable_send:
            if float(np.max(np.abs(target_real - self.last_target_real))) > self.max_delta:
                self.get_logger().warn("[SAFE] target jump too large; skip send")
            else:
                sent = self.send_motion_batch(target_real, state)
                if sent:
                    self.last_target_real = target_real.copy()

        self.update_debug_sample(target_real, target_policy, alpha, state, progress, leg_debug, sent)

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
            "time", "elapsed", "state", "alpha", "progress", "leg_name", "joint_index", "motor_id",
            "joint_name", "policy_joint_name", "x_des", "z_des", "x_foot", "z_foot", "depth_m",
            "hip_delta", "rear_hip_wag_delta", "rear_hip_wag_gate", "rear_haunch_lift_gate", "rear_haunch_lift_m", "calf_bend_extra", "thigh_target", "calf_target", "q_target_policy", "q_target_real",
            "q_current_real", "q_error_real", "torque_measured", "temp", "online", "error_code",
            "age_ms", "sent", "kp", "kd", "squat_depth_m",
        ])
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing squat data to {path}")

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

    def update_debug_sample(self, target_real, target_policy, alpha, state, progress, leg_debug, sent):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "target_real": np.asarray(target_real, dtype=np.float32).reshape(12).copy(),
                "target_policy": np.asarray(target_policy, dtype=np.float32).reshape(12).copy(),
                "alpha": float(alpha),
                "state": str(state),
                "progress": float(progress),
                "leg_debug": {leg: dict(data) for leg, data in leg_debug.items()},
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(self, target_real, target_policy, alpha, state, progress, leg_debug, sent, stamp):
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
                    "Reduce squat_depth_m or increase down/up duration."
                )

        target_policy_real_order = np.zeros(12, dtype=np.float32)
        target_policy_real_order[self.mapper.policy_to_real_index] = target_policy
        elapsed = float(stamp) - self.start_time

        for real_i, (mid, real_name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
            policy_name = self.policy_joint_names[policy_i]
            leg = policy_name.split("_", 1)[0]
            info = leg_debug.get(leg, {})
            kp, kd = self.get_joint_pd(real_i, state)
            self._debug_csv_writer.writerow([
                f"{now:.6f}", f"{elapsed:.6f}", state, f"{alpha:.6f}", f"{progress:.6f}",
                leg, int(real_i), int(mid), real_name, policy_name,
                f"{float(info.get('x_des', 0.0)):.6f}", f"{float(info.get('z_des', 0.0)):.6f}",
                f"{float(info.get('x_foot', 0.0)):.6f}", f"{float(info.get('z_foot', 0.0)):.6f}",
                f"{float(info.get('depth_m', 0.0)):.6f}", f"{float(info.get('hip_delta', 0.0)):.6f}",
                f"{float(info.get('rear_hip_wag_delta', 0.0)):.6f}",
                f"{float(info.get('rear_hip_wag_gate', 0.0)):.6f}",
                f"{float(info.get('rear_haunch_lift_gate', 0.0)):.6f}",
                f"{float(info.get('rear_haunch_lift_m', 0.0)):.6f}",
                f"{float(info.get('calf_bend_extra', 0.0)):.6f}",
                f"{float(info.get('thigh_target', 0.0)):.6f}", f"{float(info.get('calf_target', 0.0)):.6f}",
                f"{float(target_policy_real_order[real_i]):.6f}", f"{float(target_real[real_i]):.6f}",
                f"{float(q_real[real_i]):.6f}", f"{float(target_real[real_i] - q_real[real_i]):.6f}",
                f"{float(torque[real_i]):.6f}", f"{float(temp[real_i]):.3f}",
                int(bool(online[real_i])), int(error_code[real_i]), f"{float(age_ms[real_i]):.2f}",
                int(bool(sent)), f"{float(kp):.3f}", f"{self.clamp_kd(float(kd)):.3f}",
                f"{self.squat_depth_m:.6f}",
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
    node = FanfanSquatStandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
