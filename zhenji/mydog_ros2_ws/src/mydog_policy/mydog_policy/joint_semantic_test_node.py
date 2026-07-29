#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import numpy as np
import requests

import rclpy
from rclpy.node import Node

from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


class JointSemanticTestNode(Node):
    def __init__(self):
        super().__init__("mydog_joint_semantic_test_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("joint_index", 0)
        self.declare_parameter("amplitude", 0.15)
        self.declare_parameter("freq_hz", 0.5)
        self.declare_parameter("test_hz", 50.0)
        self.declare_parameter("center_mode", "current")
        self.declare_parameter("clamp_targets", False)
        self.declare_parameter("send_kp", 12.0)
        self.declare_parameter("send_kd", 1.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("enable_send", False)
        self.declare_parameter("send_enable_first", False)
        self.declare_parameter("send_stop_first", False)
        self.declare_parameter("http_timeout", 0.05)
        self.declare_parameter("max_target_delta", 0.35)
        self.declare_parameter("print_period_sec", 0.2)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.joint_index = int(self.get_parameter("joint_index").value)
        self.amplitude = float(self.get_parameter("amplitude").value)
        self.freq_hz = float(self.get_parameter("freq_hz").value)
        self.test_hz = float(self.get_parameter("test_hz").value)
        self.center_mode = str(self.get_parameter("center_mode").value).lower()
        self.clamp_targets = bool(self.get_parameter("clamp_targets").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.send_enable_first = bool(self.get_parameter("send_enable_first").value)
        self.send_stop_first = bool(self.get_parameter("send_stop_first").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.max_target_delta = float(self.get_parameter("max_target_delta").value)
        self.print_period_sec = float(self.get_parameter("print_period_sec").value)

        if self.joint_index < 0 or self.joint_index >= 12:
            raise RuntimeError("joint_index must be in [0, 11]")

        self.mapper = JointSemanticMapper()
        try:
            self.motor = MotorStateHttpInterface(
                base_url=self.motor_base_url,
                timeout=self.http_timeout,
                enable_stale_recheck=False,
            )
        except TypeError:
            # Compatible with older motor_state_interface.py on the robot.
            self.motor = MotorStateHttpInterface(
                base_url=self.motor_base_url,
                timeout=self.http_timeout,
            )
        self.session = requests.Session()

        self.q_default_policy = self.mapper.default_joint_angle.copy()
        self.q_default_real = self.mapper.real_default_pose_for_motor_order()
        self.q_center_policy = None
        self.q_center_real = None

        self.real_index = int(self.mapper.policy_to_real_index[self.joint_index])
        self.motor_id = int(self.mapper.real_motor_ids[self.real_index])
        self.policy_name = self.mapper.policy_joint_names[self.joint_index]
        self.real_name = self.mapper.real_joint_names[self.real_index]
        self.sign = float(self.mapper.joint_sign[self.joint_index])

        self.start_time = time.time()
        self._last_print_time = 0.0
        self._enable_sent = False

        self.get_logger().warn(
            "JOINT SEMANTIC TEST: keep robot suspended or supported. "
            "Only one policy joint will move around the training default pose."
        )
        self.get_logger().info(
            f"policy[{self.joint_index:02d}] {self.policy_name} -> "
            f"real[{self.real_index:02d}] motor_id=0x{self.motor_id:02X} {self.real_name} | "
            f"sign={self.sign:+.0f} amp={self.amplitude:.3f}rad freq={self.freq_hz:.2f}Hz "
            f"send={self.enable_send}"
        )

        self.timer = self.create_timer(1.0 / max(self.test_hz, 1e-3), self.update)

    def update(self):
        try:
            snapshot = self.motor.get_latest()
            if not snapshot.valid:
                self.get_logger().warn("motor state invalid")
                return

            q_policy_abs, _ = self.mapper.real_to_policy_abs_q_dq(
                snapshot.q_real,
                snapshot.dq_real,
            )

            if self.q_center_policy is None:
                if self.center_mode == "default":
                    self.q_center_policy = self.q_default_policy.copy()
                    self.q_center_real = self.q_default_real.copy()
                elif self.center_mode == "current":
                    self.q_center_policy = q_policy_abs.copy()
                    self.q_center_real = snapshot.q_real.copy()
                else:
                    raise RuntimeError("center_mode must be 'current' or 'default'")

                default_delta = self.q_default_real - snapshot.q_real
                self.get_logger().warn(
                    f"center_mode={self.center_mode}. "
                    f"default_pose_delta_max={float(np.max(np.abs(default_delta))):.3f}rad. "
                    "Use center_mode=current for mapping/sign test; use default only for zero/default alignment test."
                )

            t = time.time() - self.start_time
            offset = self.amplitude * math.sin(2.0 * math.pi * self.freq_hz * t)

            q_cmd_policy = self.q_center_policy.copy()
            q_cmd_policy[self.joint_index] += offset
            q_cmd_real = self.mapper.policy_target_to_real_target(
                q_cmd_policy,
                clamp=self.clamp_targets,
            )
            measured_offset = float(
                q_policy_abs[self.joint_index]
                - self.q_center_policy[self.joint_index]
            )

            max_delta = float(np.max(np.abs(q_cmd_real - snapshot.q_real)))
            sent = False
            if self.enable_send:
                if max_delta <= self.max_target_delta:
                    sent = self.send_motion_batch(q_cmd_real)
                else:
                    self.get_logger().warn(
                        f"skip send: max_delta={max_delta:.3f}rad "
                        f"> {self.max_target_delta:.3f}rad"
                    )

            now = time.time()
            if now - self._last_print_time >= self.print_period_sec:
                self._last_print_time = now
                q_real_cmd_i = float(q_cmd_real[self.real_index])
                q_real_cur_i = float(snapshot.q_real[self.real_index])
                q_policy_cur_i = float(q_policy_abs[self.joint_index])
                direction_ok = (
                    "near_zero"
                    if abs(offset) < 0.02
                    else ("OK" if offset * measured_offset > 0.0 else "REVERSED_OR_NOT_MOVING")
                )
                self.get_logger().info(
                    f"policy[{self.joint_index:02d}] {self.policy_name} "
                    f"-> motor=0x{self.motor_id:02X} {self.real_name} "
                    f"cmd_offset_policy={offset:+.4f} "
                    f"measured_offset_policy={measured_offset:+.4f} "
                    f"dir={direction_ok} "
                    f"q_cmd_real={q_real_cmd_i:+.4f} "
                    f"q_current_real={q_real_cur_i:+.4f} "
                    f"q_current_policy={q_policy_cur_i:+.4f} "
                    f"max_age={float(np.max(snapshot.age_ms)):.1f}ms "
                    f"max_delta={max_delta:.3f} "
                    f"sent={sent}"
                )

        except Exception as exc:
            self.get_logger().error(f"joint semantic test error: {exc}")

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        motor_ids = self.mapper.get_real_motor_ids()
        items = []
        for i, mid in enumerate(motor_ids):
            items.append({
                "motor_id": int(mid),
                "position": float(target_real[i]),
                "speed": float(self.send_speed),
                "torque": float(self.send_torque),
                "kp": float(self.send_kp),
                "kd": float(self.send_kd),
            })

        payload = {
            "items": items,
            "enable_first": bool(self.send_enable_first and not self._enable_sent),
            "stop_first": bool(self.send_stop_first and not self._enable_sent),
        }

        try:
            r = self.session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json=payload,
                timeout=max(self.http_timeout, 0.2),
            )
            if r.status_code != 200:
                self.get_logger().warn(f"send failed HTTP {r.status_code}: {r.text}")
                return False
            self._enable_sent = True
            return True
        except Exception as exc:
            self.get_logger().warn(f"send request failed: {exc}")
            return False

    def destroy_node(self):
        try:
            self.motor.close()
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JointSemanticTestNode()

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
