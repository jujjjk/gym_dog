#!/usr/bin/env python3
import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .omni_fast_contract import DEPLOYMENT_LIMITS, MODEL_TASK, PRESETS


def publish_command(publisher, vx, vy, yaw):
    msg = Twist()
    msg.linear.x = float(vx)
    msg.linear.y = float(vy)
    msg.angular.z = float(yaw)
    publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Publish a safe Fanfan Omni preset")
    parser.add_argument("action", choices=sorted(PRESETS))
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.rate <= 0.0:
        parser.error("duration and rate must be positive")

    rclpy.init()
    node = Node("fanfan_omni_command")
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    command = PRESETS[args.action]
    period = 1.0 / args.rate
    deadline = time.monotonic() + args.duration
    try:
        node.get_logger().info(
            f"task={MODEL_TASK} action={args.action} command={command} "
            f"limits={DEPLOYMENT_LIMITS}"
        )
        while rclpy.ok() and time.monotonic() < deadline:
            publish_command(publisher, *command)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        # Respect the launch-file slew limits while returning to the trained stand
        # command, then allow the watchdog to expire.
        return_seconds = max(
            0.6,
            abs(command[0]) / 0.20,
            abs(command[1]) / 0.10,
            abs(command[2]) / 0.60,
        ) + 0.3
        for _ in range(max(3, int(args.rate * return_seconds))):
            publish_command(publisher, 0.0, 0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
