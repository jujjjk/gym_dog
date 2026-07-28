#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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
STEP_ORDER = ("RR", "FL", "RL", "FR")
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RR", "RL")
RIGHT_LEGS = ("FR", "RR")
HIP_OUTWARD_SIGNS = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}


class FanfanStepInPlaceNode(Node):
    """
    URDF-based open-loop stepping-in-place gait for Fanfan.

    The gait keeps one foot in swing at a time. Before lifting a foot it shifts
    the estimated COM toward the other three feet, then lifts/holds/lands the
    swing foot. This is still open loop, so first tests must be hand-held or
    supported.
    """

    def __init__(self):
        super().__init__("fanfan_step_in_place_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("step_hz", 0.55)
        self.declare_parameter("stand_sec", 3.0)
        self.declare_parameter("warmup_sec", 4.0)

        self.declare_parameter("stand_kp", 38.0)
        self.declare_parameter("stand_kd", 4.0)
        self.declare_parameter("send_kp", 38.0)
        self.declare_parameter("send_kd", 4.2)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("http_timeout", 0.08)

        self.declare_parameter("lift_height", 0.026)
        self.declare_parameter("front_lift_height_gain", 1.18)
        self.declare_parameter("rear_lift_height_gain", 1.08)
        self.declare_parameter("front_calf_lift_extra", 0.072)
        self.declare_parameter("rear_calf_lift_extra", 0.070)
        self.declare_parameter("rear_thigh_lift_extra", 0.018)
        self.declare_parameter("front_thigh_lift_scale", 0.42)
        self.declare_parameter("stomp_touchdown_push_m", 0.004)
        self.declare_parameter("front_stomp_touchdown_push_m", 0.005)
        self.declare_parameter("front_companion_press_m", 0.005)
        self.declare_parameter("front_companion_calf_push", 0.018)
        self.declare_parameter("support_stand_tall_m", 0.004)
        self.declare_parameter("diagonal_support_extra_m", 0.004)

        self.declare_parameter("front_swing_body_x_shift", -0.018)
        self.declare_parameter("rear_swing_body_x_shift", 0.040)
        self.declare_parameter("body_y_shift", 0.026)
        self.declare_parameter("body_x_foot_scale", 0.12)
        self.declare_parameter("active_foot_body_x_scale", 0.04)
        self.declare_parameter("hip_body_y_scale", 0.55)
        self.declare_parameter("support_hip_inward_amp", 0.014)
        self.declare_parameter("swing_hip_relief_amp", 0.004)

        self.declare_parameter("hip_default_scale", 0.38)
        self.declare_parameter("hip_default_inward_offset", 0.010)
        self.declare_parameter("rear_hip_default_outward_offset", 0.006)
        self.declare_parameter("rear_thigh_default_back_offset", 0.035)
        self.declare_parameter("front_calf_min_rad", -1.12)

        # From fanfan_mass_scaled_only_trunk_plus_800g.urdf.
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)
        self.declare_parameter("front_foot_x_default", 0.1765)
        self.declare_parameter("rear_foot_x_default", -0.2219)
        self.declare_parameter("foot_y_half_width", 0.138)
        self.declare_parameter("stability_margin_m", 0.020)
        self.declare_parameter("max_target_rate_rad_s", 2.4)
        self.declare_parameter("max_delta", 0.65)
        self.declare_parameter("torque_warn_nm", 6.0)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = float(self.get_parameter("stand_kd").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)

        self.lift_height = float(self.get_parameter("lift_height").value)
        self.front_lift_height_gain = float(self.get_parameter("front_lift_height_gain").value)
        self.rear_lift_height_gain = float(self.get_parameter("rear_lift_height_gain").value)
        self.front_calf_lift_extra = float(self.get_parameter("front_calf_lift_extra").value)
        self.rear_calf_lift_extra = float(self.get_parameter("rear_calf_lift_extra").value)
        self.rear_thigh_lift_extra = float(self.get_parameter("rear_thigh_lift_extra").value)
        self.front_thigh_lift_scale = float(self.get_parameter("front_thigh_lift_scale").value)
        self.stomp_touchdown_push_m = float(self.get_parameter("stomp_touchdown_push_m").value)
        self.front_stomp_touchdown_push_m = float(
            self.get_parameter("front_stomp_touchdown_push_m").value
        )
        self.front_companion_press_m = float(self.get_parameter("front_companion_press_m").value)
        self.front_companion_calf_push = float(
            self.get_parameter("front_companion_calf_push").value
        )
        self.support_stand_tall_m = float(self.get_parameter("support_stand_tall_m").value)
        self.diagonal_support_extra_m = float(self.get_parameter("diagonal_support_extra_m").value)

        self.front_swing_body_x_shift = float(self.get_parameter("front_swing_body_x_shift").value)
        self.rear_swing_body_x_shift = float(self.get_parameter("rear_swing_body_x_shift").value)
        self.body_y_shift = float(self.get_parameter("body_y_shift").value)
        self.body_x_foot_scale = float(self.get_parameter("body_x_foot_scale").value)
        self.active_foot_body_x_scale = float(self.get_parameter("active_foot_body_x_scale").value)
        self.hip_body_y_scale = float(self.get_parameter("hip_body_y_scale").value)
        self.support_hip_inward_amp = float(self.get_parameter("support_hip_inward_amp").value)
        self.swing_hip_relief_amp = float(self.get_parameter("swing_hip_relief_amp").value)

        self.hip_default_scale = float(self.get_parameter("hip_default_scale").value)
        self.hip_default_inward_offset = float(self.get_parameter("hip_default_inward_offset").value)
        self.rear_hip_default_outward_offset = float(
            self.get_parameter("rear_hip_default_outward_offset").value
        )
        self.rear_thigh_default_back_offset = float(
            self.get_parameter("rear_thigh_default_back_offset").value
        )
        self.front_calf_min_rad = float(self.get_parameter("front_calf_min_rad").value)

        self.thigh_length = float(self.get_parameter("thigh_length").value)
        self.calf_length = float(self.get_parameter("calf_length").value)
        self.front_foot_x_default = float(self.get_parameter("front_foot_x_default").value)
        self.rear_foot_x_default = float(self.get_parameter("rear_foot_x_default").value)
        self.foot_y_half_width = float(self.get_parameter("foot_y_half_width").value)
        self.stability_margin_m = float(self.get_parameter("stability_margin_m").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)

        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.apply_default_stance_offsets()
        self.default_real = self.mapper.policy_target_to_real_target(
            self.default_policy,
            clamp=True,
        ).astype(np.float32)
        self.default_foot_xz = self.compute_default_foot_xz()
        self.last_target_policy = self.default_policy.copy()
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
        self._last_send_info_time = 0.0
        self._last_warn_time = 0.0
        self._last_torque_warn_time = 0.0

        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=100.0,
        )

        self.pub_target = self.create_publisher(
            Float32MultiArray,
            "/mydog/fanfan_step_in_place_target_real",
            10,
        )
        self.pub_phase = self.create_publisher(
            Float32MultiArray,
            "/mydog/fanfan_step_in_place_phase",
            10,
        )

        self.get_logger().warn(
            "Fanfan step-in-place is open-loop. First run supported or hand-held, "
            "with emergency power cut ready."
        )
        self.get_logger().info(
            f"step_hz={self.step_hz:.2f}, lift={self.lift_height:.3f}m, "
            f"kp={self.send_kp:.1f}, kd={self.send_kd:.1f}, "
            f"order={'->'.join(STEP_ORDER)}, send={self.enable_send}"
        )

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only, no motor commands sent.")

        self.timer = self.create_timer(1.0 / max(self.gait_hz, 1e-3), self.update)

    def apply_default_stance_offsets(self):
        scale = min(1.0, max(0.0, self.hip_default_scale))
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            self.default_policy[i + 0] *= scale
            self.default_policy[i + 0] -= HIP_OUTWARD_SIGNS[leg] * self.hip_default_inward_offset
        for leg in REAR_LEGS:
            i = LEG_START[leg]
            self.default_policy[i + 0] += HIP_OUTWARD_SIGNS[leg] * self.rear_hip_default_outward_offset
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset
        for leg in FRONT_LEGS:
            idx = LEG_START[leg] + 2
            if self.default_policy[idx] < self.front_calf_min_rad:
                self.default_policy[idx] = self.front_calf_min_rad

    def compute_default_foot_xz(self) -> dict[str, tuple[float, float]]:
        xz = {}
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            xz[leg] = self.forward_sagittal(
                float(self.default_policy[i + 1]),
                float(self.default_policy[i + 2]),
            )
        return xz

    def forward_sagittal(self, thigh: float, calf: float) -> tuple[float, float]:
        x = -self.thigh_length * math.sin(thigh) - self.calf_length * math.sin(thigh + calf)
        z = -self.thigh_length * math.cos(thigh) - self.calf_length * math.cos(thigh + calf)
        return float(x), float(z)

    def inverse_sagittal(self, x: float, z: float) -> tuple[float, float]:
        x, z = self.clamp_reachable_xz(x, z)
        l1 = self.thigh_length
        l2 = self.calf_length
        cos_calf = (x * x + z * z - l1 * l1 - l2 * l2) / max(2.0 * l1 * l2, 1e-9)
        cos_calf = min(1.0, max(-1.0, cos_calf))
        calf = -math.acos(cos_calf)
        thigh = math.atan2(-x, -z) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
        return float(thigh), float(calf)

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
        return float(x), float(z)

    @staticmethod
    def smoothstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return 0.5 - 0.5 * math.cos(math.pi * s)

    def active_step_leg(self, phase: float) -> tuple[str, float]:
        step = (float(phase) * len(STEP_ORDER)) % len(STEP_ORDER)
        idx = int(math.floor(step)) % len(STEP_ORDER)
        return STEP_ORDER[idx], float(step - math.floor(step))

    def support_foot_xy(self, leg: str) -> tuple[float, float]:
        x = self.front_foot_x_default if leg in FRONT_LEGS else self.rear_foot_x_default
        y = -self.foot_y_half_width if leg in RIGHT_LEGS else self.foot_y_half_width
        return float(x), float(y)

    def point_in_support_triangle(
        self,
        swing_leg: str,
        body_x_shift: float,
        body_y_shift: float,
    ) -> tuple[bool, float]:
        points = [self.support_foot_xy(leg) for leg in LEG_ORDER if leg != swing_leg]
        px, py = float(body_x_shift), float(body_y_shift)
        distances = []
        signs = []
        for a, b in zip(points, points[1:] + points[:1]):
            edge_x = b[0] - a[0]
            edge_y = b[1] - a[1]
            cross = edge_x * (py - a[1]) - edge_y * (px - a[0])
            signs.append(cross)
            distances.append(abs(cross) / max(math.hypot(edge_x, edge_y), 1e-9))
        inside = all(v >= -1e-9 for v in signs) or all(v <= 1e-9 for v in signs)
        margin = min(distances) if distances else 0.0
        return bool(inside and margin >= self.stability_margin_m), float(margin)

    def desired_body_shift(self, swing_leg: str, event_s: float) -> tuple[float, float, float]:
        if event_s < 0.16:
            gate = self.smoothstep(event_s / 0.16)
        elif event_s > 0.82:
            gate = self.smoothstep((1.0 - event_s) / 0.18)
        else:
            gate = 1.0

        x_target = self.rear_swing_body_x_shift if swing_leg in REAR_LEGS else self.front_swing_body_x_shift
        y_target = self.body_y_shift if swing_leg in RIGHT_LEGS else -self.body_y_shift
        return float(gate * x_target), float(gate * y_target), float(gate)

    def swing_lift_gate(self, event_s: float) -> tuple[float, str]:
        if event_s < 0.18:
            return 0.0, "SHIFT"
        if event_s < 0.38:
            return self.smoothstep((event_s - 0.18) / 0.20), "LIFT"
        if event_s < 0.54:
            return 1.0, "HOLD"
        if event_s < 0.76:
            return self.smoothstep((0.76 - event_s) / 0.22), "DOWN"
        if event_s < 0.86:
            return 0.0, "TAP"
        return 0.0, "SETTLE"

    def stomp_touchdown_gate(self, event_s: float) -> float:
        if event_s < 0.76 or event_s > 0.90:
            return 0.0
        rise = self.smoothstep((event_s - 0.76) / 0.04)
        fall = self.smoothstep((0.90 - event_s) / 0.08)
        return float(rise * fall)

    def hip_balance_delta(
        self,
        leg: str,
        swing_leg: str,
        body_y_shift: float,
        shift_gate: float,
        is_active: bool,
    ) -> float:
        outward = HIP_OUTWARD_SIGNS[leg]
        if is_active:
            return float(-outward * self.swing_hip_relief_amp * shift_gate)
        lateral = 0.0
        if abs(self.body_y_shift) > 1e-9:
            lateral = (body_y_shift / self.body_y_shift) * self.hip_body_y_scale
        inward = -outward * self.support_hip_inward_amp * shift_gate
        return float(inward + lateral * 0.010 * shift_gate)

    def build_step_target(self, phase: float, warm: float):
        q = self.default_policy.copy()
        swing_leg, event_s = self.active_step_leg(phase)
        body_x_shift, body_y_shift, shift_gate = self.desired_body_shift(swing_leg, event_s)
        stable, margin = self.point_in_support_triangle(swing_leg, body_x_shift, body_y_shift)

        now = time.time()
        if not stable and now - self._last_warn_time > 1.0:
            self._last_warn_time = now
            self.get_logger().warn(
                f"[STABILITY] swing={swing_leg}, body_shift=({body_x_shift:.3f},{body_y_shift:.3f}), "
                f"support_margin={margin:.3f}m"
            )

        lift_gate, state = self.swing_lift_gate(event_s)
        stomp_gate = self.stomp_touchdown_gate(event_s)
        for leg in LEG_ORDER:
            is_front = leg in FRONT_LEGS
            is_rear = leg in REAR_LEGS
            is_active = leg == swing_leg
            i = LEG_START[leg]
            x0, z0 = self.default_foot_xz[leg]
            x = x0
            z = z0
            if is_active:
                x -= warm * body_x_shift * self.active_foot_body_x_scale * shift_gate
                height_gain = self.front_lift_height_gain if is_front else self.rear_lift_height_gain
                z += warm * self.lift_height * height_gain * lift_gate
                stomp_push = (
                    self.front_stomp_touchdown_push_m
                    if is_front
                    else self.stomp_touchdown_push_m
                )
                z -= warm * stomp_push * stomp_gate
            else:
                x -= warm * body_x_shift * self.body_x_foot_scale * shift_gate
                support = self.support_stand_tall_m * shift_gate
                if self.is_diagonal_pair(leg, swing_leg):
                    support += self.diagonal_support_extra_m * shift_gate
                companion_gate = 0.0
                if swing_leg in REAR_LEGS and is_front and self.is_diagonal_pair(leg, swing_leg):
                    companion_gate = max(lift_gate, stomp_gate) * shift_gate
                    support += self.front_companion_press_m * companion_gate
                z -= warm * support

            thigh, calf = self.inverse_sagittal(x, z)
            calf_lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
            calf -= warm * calf_lift_extra * lift_gate if is_active else 0.0
            if (
                not is_active
                and swing_leg in REAR_LEGS
                and is_front
                and self.is_diagonal_pair(leg, swing_leg)
            ):
                calf += warm * self.front_companion_calf_push * companion_gate
            if is_front:
                thigh = float(self.default_policy[i + 1]) + (thigh - float(self.default_policy[i + 1])) * self.front_thigh_lift_scale
                if calf < self.front_calf_min_rad:
                    calf = self.front_calf_min_rad
            elif is_rear and is_active:
                thigh += warm * self.rear_thigh_lift_extra * lift_gate

            hip_delta = self.hip_balance_delta(leg, swing_leg, body_y_shift, shift_gate, is_active)
            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta
            q[i + 1] = self.default_policy[i + 1] + warm * (thigh - self.default_policy[i + 1])
            q[i + 2] = self.default_policy[i + 2] + warm * (calf - self.default_policy[i + 2])

        return q.astype(np.float32), swing_leg, event_s, state, stable, margin

    @staticmethod
    def is_diagonal_pair(leg: str, swing_leg: str) -> bool:
        return (leg, swing_leg) in {
            ("FR", "RL"),
            ("RL", "FR"),
            ("FL", "RR"),
            ("RR", "FL"),
        }

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time
        if elapsed < self.stand_sec:
            target_policy = self.default_policy.copy()
            phase = 0.0
            warm = 0.0
            swing_leg = ""
            event_s = 0.0
            state = "STAND"
            stable = True
            margin = self.stability_margin_m
            self._phase_acc = 0.0
        else:
            gait_time = elapsed - self.stand_sec
            self._phase_acc = (self._phase_acc + self.step_hz * dt) % 1.0
            phase = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 1e-3))
            target_policy, swing_leg, event_s, state, stable, margin = self.build_step_target(phase, warm)

        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True)
        self.publish_array(self.pub_target, target_real)
        self.publish_array(
            self.pub_phase,
            np.array([phase, warm, event_s, 1.0 if stable else 0.0, margin], dtype=np.float32),
        )

        if not self.enable_send:
            return
        max_delta = float(np.max(np.abs(target_real - self.last_target_real)))
        if max_delta > self.max_delta:
            self.get_logger().warn(
                f"[SAFE] target jump {max_delta:.3f} rad > {self.max_delta:.3f}; skip send."
            )
            return
        if self.send_motion_batch(target_real):
            self.last_target_real = target_real.copy()
            if now - self._last_send_info_time > 1.0:
                self._last_send_info_time = now
                self.get_logger().info(
                    f"[STEP] swing={swing_leg} state={state} phase={phase:.3f} "
                    f"event={event_s:.2f} margin={margin:.3f}m"
                )
        self.warn_if_torque_high()

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

    def send_default_stand(self):
        items = []
        for mid, pos in zip(self.motor_ids, self.default_real):
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
        payload = {"items": items, "enable_first": True, "stop_first": False}
        url = f"{self.motor_base_url}/api/rs04/motion_mode_run_batch"
        r = self.http_session.post(url, json=payload, timeout=max(self.http_timeout, 0.5))
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")
        self.get_logger().info(f"Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}")

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
        payload = {"items": items, "enable_first": False, "stop_first": False}
        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json=payload,
                timeout=self.http_timeout,
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def warn_if_torque_high(self):
        if self.torque_warn_nm <= 0.0:
            return
        now = time.time()
        if now - self._last_torque_warn_time < 1.0:
            return
        try:
            snapshot = self.motor.get_latest()
            torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
        except Exception:
            return
        torque_abs_max = float(np.max(np.abs(torque)))
        if torque_abs_max > self.torque_warn_nm:
            self._last_torque_warn_time = now
            self.get_logger().warn(
                f"[TORQUE] measured max |torque|={torque_abs_max:.2f}Nm > "
                f"{self.torque_warn_nm:.2f}Nm. Reduce step_hz/lift_height or support the robot."
            )

    @staticmethod
    def publish_array(pub, values: np.ndarray):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FanfanStepInPlaceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
