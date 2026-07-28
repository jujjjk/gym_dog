#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


WALK_X = 0.05

WALK_HZ = 50.0
WALK_DURATION_SEC = 5.0

ZERO_HZ = 20.0
ZERO_DURATION_SEC = 1.0

READY_TIMEOUT_SEC = 35.0
READY_REQUIRED_FRAMES = 3


def main():
    rclpy.init()

    node = rclpy.create_node("fixed_walk_after_policy_ready")
    publisher = node.create_publisher(
        Twist,
        "/cmd_vel",
        10,
    )

    state = {
        "policy_frames": 0,
    }

    def policy_action_callback(msg: Float32MultiArray):
        state["policy_frames"] += 1

    subscription = node.create_subscription(
        Float32MultiArray,
        "/mydog/policy_action_raw",
        policy_action_callback,
        10,
    )

    print("等待 /cmd_vel 订阅者连接……")

    deadline = time.monotonic() + READY_TIMEOUT_SEC

    while publisher.get_subscription_count() < 1:
        if time.monotonic() >= deadline:
            node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError(
                "等待 /cmd_vel 订阅者超时，请确认 parity 节点已经启动"
            )

        rclpy.spin_once(node, timeout_sec=0.05)

    print(
        "已检测到 /cmd_vel 订阅者："
        f"{publisher.get_subscription_count()} 个"
    )

    print(
        "等待策略真正进入控制阶段……"
        "必须收到 /mydog/policy_action_raw，"
        "不是只等待订阅器出现。"
    )

    while state["policy_frames"] < READY_REQUIRED_FRAMES:
        if time.monotonic() >= deadline:
            node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError(
                "等待策略READY超时。"
                "请查看终端1是否出现 "
                "[STARTUP_STAND][READY]，"
                "以及状态估计器是否报错。"
            )

        rclpy.spin_once(node, timeout_sec=0.05)

    print(
        f"策略已READY，已收到{state['policy_frames']}帧"
        " /mydog/policy_action_raw"
    )

    # READY以后稍等几帧，确保状态估计和策略循环稳定。
    settle_end = time.monotonic() + 0.20
    while time.monotonic() < settle_end:
        rclpy.spin_once(node, timeout_sec=0.02)

    walk = Twist()
    walk.linear.x = WALK_X
    walk.linear.y = 0.0
    walk.linear.z = 0.0
    walk.angular.x = 0.0
    walk.angular.y = 0.0
    walk.angular.z = 0.0

    zero = Twist()

    walk_count = int(WALK_HZ * WALK_DURATION_SEC)
    zero_count = int(ZERO_HZ * ZERO_DURATION_SEC)

    print(
        f"开始行走：vx={WALK_X:.3f} m/s，"
        f"{WALK_HZ:.0f} Hz，"
        f"{walk_count}次，"
        f"约{WALK_DURATION_SEC:.1f}秒"
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

    actual_walk_duration = time.monotonic() - start

    print(
        f"行走命令完成：{walk_count}次，"
        f"实际耗时{actual_walk_duration:.3f}秒"
    )

    print(
        f"开始发送零命令："
        f"{ZERO_HZ:.0f} Hz，"
        f"{zero_count}次"
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

    actual_zero_duration = time.monotonic() - start

    print(
        f"零命令完成：{zero_count}次，"
        f"实际耗时{actual_zero_duration:.3f}秒"
    )

    # 留一点时间让最后几条零命令通过DDS。
    flush_end = time.monotonic() + 0.20
    while time.monotonic() < flush_end:
        rclpy.spin_once(node, timeout_sec=0.02)

    # 保留引用，防止订阅器被垃圾回收。
    _ = subscription

    node.destroy_node()
    rclpy.shutdown()

    print("5秒行走＋1秒零命令发布完成")


if __name__ == "__main__":
    main()
