#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist


WALK_DURATION_SEC = 5.0
WALK_HZ = 50.0
ZERO_DURATION_SEC = 1.0
ZERO_HZ = 20.0

walk_count = int(WALK_DURATION_SEC * WALK_HZ)
zero_count = int(ZERO_DURATION_SEC * ZERO_HZ)

rclpy.init()
node = rclpy.create_node("fixed_walk_command_test")
publisher = node.create_publisher(Twist, "/cmd_vel", 10)

deadline = time.monotonic() + 5.0
while publisher.get_subscription_count() < 1:
    if time.monotonic() >= deadline:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("/cmd_vel没有订阅节点，请确认策略节点已经启动")
    rclpy.spin_once(node, timeout_sec=0.05)

print(
    f"检测到 /cmd_vel 订阅者数量："
    f"{publisher.get_subscription_count()}"
)

walk = Twist()
walk.linear.x = 0.05
walk.linear.y = 0.0
walk.linear.z = 0.0
walk.angular.x = 0.0
walk.angular.y = 0.0
walk.angular.z = 0.0

zero = Twist()

print(
    f"开始发布{walk_count}次行走命令，"
    f"{WALK_HZ:.0f}Hz，约{WALK_DURATION_SEC:.1f}秒"
)

start = time.monotonic()
next_tick = start

for index in range(walk_count):
    publisher.publish(walk)
    rclpy.spin_once(node, timeout_sec=0.0)

    next_tick += 1.0 / WALK_HZ
    sleep_time = next_tick - time.monotonic()
    if sleep_time > 0.0:
        time.sleep(sleep_time)

walk_duration = time.monotonic() - start
print(
    f"行走命令完成：{walk_count}次，"
    f"实际耗时{walk_duration:.3f}秒"
)

print(
    f"开始发布{zero_count}次零命令，"
    f"{ZERO_HZ:.0f}Hz，约{ZERO_DURATION_SEC:.1f}秒"
)

start = time.monotonic()
next_tick = start

for index in range(zero_count):
    publisher.publish(zero)
    rclpy.spin_once(node, timeout_sec=0.0)

    next_tick += 1.0 / ZERO_HZ
    sleep_time = next_tick - time.monotonic()
    if sleep_time > 0.0:
        time.sleep(sleep_time)

zero_duration = time.monotonic() - start
print(
    f"零命令完成：{zero_count}次，"
    f"实际耗时{zero_duration:.3f}秒"
)

node.destroy_node()
rclpy.shutdown()

print("5秒行走命令＋1秒零命令发布完成")
