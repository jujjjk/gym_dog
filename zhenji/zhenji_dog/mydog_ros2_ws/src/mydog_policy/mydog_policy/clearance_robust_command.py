#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish 5730 parity commands continuously, without zero gaps between actions."""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

# Exact command order from gym_dog/mujoko/sim2sim.py --demo-matrix.
# At 5 seconds per segment this reproduces the 45-second matrix.
PARITY_SEQUENCE = [
    ("forward", 0.35, 0.00, 0.00),
    ("backward", -0.10, 0.00, 0.00),
    ("left_lateral", 0.00, 0.07, 0.00),
    ("right_lateral", 0.00, -0.07, 0.00),
    ("left_yaw", 0.00, 0.00, 0.70),
    ("right_yaw", 0.00, 0.00, -0.70),
    ("left_arc", 0.25, 0.00, 0.50),
    ("backward_left_arc", -0.10, 0.00, 0.35),
    ("left_diagonal", 0.20, 0.07, 0.00),
]

# Symmetric expansion for separate right-side and diagonal checks.
FULL_SEQUENCE = [
    ("forward", 0.35, 0.00, 0.00),
    ("backward", -0.10, 0.00, 0.00),
    ("left_lateral", 0.00, 0.07, 0.00),
    ("right_lateral", 0.00, -0.07, 0.00),
    ("left_yaw", 0.00, 0.00, 0.70),
    ("right_yaw", 0.00, 0.00, -0.70),
    ("left_arc", 0.25, 0.00, 0.50),
    ("right_arc", 0.25, 0.00, -0.50),
    ("backward_left_arc", -0.10, 0.00, 0.35),
    ("backward_right_arc", -0.10, 0.00, -0.35),
    ("left_diagonal", 0.20, 0.07, 0.00),
    ("right_diagonal", 0.20, -0.07, 0.00),
]

LOW_SEQUENCE = [
    ("forward", 0.12, 0.00, 0.00),
    ("backward", -0.05, 0.00, 0.00),
    ("left_lateral", 0.00, 0.03, 0.00),
    ("right_lateral", 0.00, -0.03, 0.00),
    ("left_yaw", 0.00, 0.00, 0.20),
    ("right_yaw", 0.00, 0.00, -0.20),
    ("left_arc", 0.12, 0.00, 0.20),
    ("right_arc", 0.12, 0.00, -0.20),
    ("left_diagonal", 0.12, 0.03, 0.00),
    ("right_diagonal", 0.12, -0.03, 0.00),
]


class ClearanceCommandPublisher(Node):
    def __init__(self):
        super().__init__("clearance_robust_5730_command")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def publish_command(self, vx: float, vy: float, yaw: float) -> None:
        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.angular.z = float(yaw)
        self.publisher.publish(message)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("parity", "full", "low"), default="parity"
    )
    parser.add_argument("--action", default="all")
    parser.add_argument("--segment-sec", type=float, default=None)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--repeat", type=int, default=1)
    parsed, ros_args = parser.parse_known_args(args=args)

    if parsed.segment_sec is not None and parsed.segment_sec <= 0.0:
        raise SystemExit("segment-sec must be positive when provided")
    if parsed.rate <= 0.0:
        raise SystemExit("rate must be positive")
    if parsed.repeat <= 0:
        raise SystemExit("repeat must be positive")

    if parsed.profile == "parity":
        sequence = PARITY_SEQUENCE
    elif parsed.profile == "full":
        sequence = FULL_SEQUENCE
    else:
        sequence = LOW_SEQUENCE
    valid_actions = {name for name, *_ in sequence}
    if parsed.action != "all":
        if parsed.action not in valid_actions:
            raise SystemExit(
                f"unknown action {parsed.action!r}; choose from: "
                + ", ".join(sorted(valid_actions))
            )
        sequence = [item for item in sequence if item[0] == parsed.action]

    segment_sec = parsed.segment_sec
    if segment_sec is None and parsed.action == "all":
        segment_sec = 5.0

    rclpy.init(args=ros_args)
    node = ClearanceCommandPublisher()
    period = 1.0 / parsed.rate
    try:
        for _ in range(parsed.repeat):
            for name, vx, vy, yaw in sequence:
                node.get_logger().warn(
                    f"transition -> {name}: "
                    f"vx={vx:+.3f}, vy={vy:+.3f}, yaw={yaw:+.3f}"
                )
                end = (
                    None
                    if segment_sec is None
                    else time.monotonic() + segment_sec
                )
                while rclpy.ok() and (end is None or time.monotonic() < end):
                    node.publish_command(vx, vy, yaw)
                    rclpy.spin_once(node, timeout_sec=0.0)
                    time.sleep(period)
    except KeyboardInterrupt:
        node.get_logger().warn("stopping command publisher; sending zero command")
    finally:
        for _ in range(max(5, int(parsed.rate * 0.5))):
            node.publish_command(0.0, 0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
