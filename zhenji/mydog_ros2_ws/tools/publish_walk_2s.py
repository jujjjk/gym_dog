#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist


rclpy.init()
node = rclpy.create_node("fixed_walk_command_test")
publisher = node.create_publisher(Twist, "/cmd_vel", 10)

deadline = time.monotonic() + 5.0
while publisher.get_subscription_count() < 1:
    if time.monotonic() >= deadline:
        raise RuntimeError("/cmd_vel没有订阅节点")
    rclpy.spin_once(node, timeout_sec=0.05)

walk = Twist()
walk.linear.x = 0.05

zero = Twist()

print("开始发布100次行走命令，50Hz，约2秒")

start = time.monotonic()
next_tick = start

for index in range(100):
    publisher.publish(walk)
    rclpy.spin_once(node, timeout_sec=0.0)

    next_tick += 0.02
    sleep_time = next_tick - time.monotonic()
    if sleep_time > 0.0:
        time.sleep(sleep_time)

walk_duration = time.monotonic() - start
print(f"行走命令完成：100次，实际耗时{walk_duration:.3f}秒")

print("开始发布20次零命令，20Hz，约1秒")

start = time.monotonic()
next_tick = start

for index in range(20):
    publisher.publish(zero)
    rclpy.spin_once(node, timeout_sec=0.0)

    next_tick += 0.05
    sleep_time = next_tick - time.monotonic()
    if sleep_time > 0.0:
        time.sleep(sleep_time)

zero_duration = time.monotonic() - start
print(f"零命令完成：20次，实际耗时{zero_duration:.3f}秒")

node.destroy_node()
rclpy.shutdown()
