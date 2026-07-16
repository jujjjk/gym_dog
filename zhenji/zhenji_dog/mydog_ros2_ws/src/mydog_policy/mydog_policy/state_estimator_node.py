#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .imu_serial_interface import ImuSerialInterface
from .leg_odometry import LegOdometryEstimator, POLICY_LEG_ORDER
from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


class MydogStateEstimatorNode(Node):
    def __init__(self):
        super().__init__("mydog_state_estimator_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("estimator_hz", 50.0)
        self.declare_parameter("max_motor_age_ms", 100.0)
        self.declare_parameter("require_online", False)
        self.declare_parameter("imu_port", "/dev/myimu")
        self.declare_parameter("imu_read_hz", 100.0)
        self.declare_parameter("max_imu_sample_age_sec", 0.10)

        # From fanfan.urdf + current default stance: Trunk/IMU height is about 0.29 m.
        self.declare_parameter("nominal_base_height", 0.293)
        self.declare_parameter("foot_radius", 0.018)
        self.declare_parameter("stance_height_margin", 0.045)
        self.declare_parameter("stance_speed_threshold", 0.65)
        self.declare_parameter("filter_alpha", 0.25)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value)
        self.estimator_hz = float(self.get_parameter("estimator_hz").value)
        self.max_motor_age_ms = float(self.get_parameter("max_motor_age_ms").value)
        self.require_online = bool(self.get_parameter("require_online").value)
        self.imu_port = str(self.get_parameter("imu_port").value)
        self.imu_read_hz = float(self.get_parameter("imu_read_hz").value)
        self.max_imu_sample_age_sec = float(
            self.get_parameter("max_imu_sample_age_sec").value
        )
        if self.max_imu_sample_age_sec <= 0.0:
            raise RuntimeError("max_imu_sample_age_sec must be positive")

        self.mapper = JointSemanticMapper()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=0.08,
            stale_recheck_ms=self.max_motor_age_ms,
            enable_stale_recheck=False,
            async_poll=True,
            poll_hz=self.estimator_hz,
        )
        self.imu = ImuSerialInterface(
            port=self.imu_port,
            read_hz=self.imu_read_hz,
        )
        self.estimator = LegOdometryEstimator(
            nominal_base_height=float(self.get_parameter("nominal_base_height").value),
            foot_radius=float(self.get_parameter("foot_radius").value),
            stance_height_margin=float(self.get_parameter("stance_height_margin").value),
            stance_speed_threshold=float(self.get_parameter("stance_speed_threshold").value),
            filter_alpha=float(self.get_parameter("filter_alpha").value),
        )
        self._last_info_log_time = 0.0
        self._last_error_log_time = 0.0

        self.pub_base_lin_vel = self.create_publisher(
            Float32MultiArray,
            "/mydog/base_lin_vel",
            10,
        )
        self.pub_state = self.create_publisher(
            Float32MultiArray,
            "/mydog/state_estimator",
            10,
        )
        self.pub_debug = self.create_publisher(
            Float32MultiArray,
            "/mydog/leg_odom_debug",
            10,
        )

        self.get_logger().info("Starting IMU for state estimator...")
        self.imu.start()
        if not self.imu.wait_until_ready(timeout=3.0):
            raise RuntimeError("IMU not ready. Please check /dev/myimu.")

        self.timer = self.create_timer(1.0 / self.estimator_hz, self.update)

        self.get_logger().info(
            "mydog_state_estimator_node started | "
            f"hz={self.estimator_hz:.1f}, "
            f"nominal_base_height={self.estimator.nominal_base_height:.3f}, "
            f"foot_radius={self.estimator.foot_radius:.3f}, "
            f"legs={list(POLICY_LEG_ORDER)}"
        )

    def update(self):
        try:
            motor = self.motor.get_latest()
            if not motor.valid:
                self.get_logger().warn("Motor state invalid. Skip estimator frame.")
                return

            max_age = float(np.max(motor.age_ms))
            if max_age > self.max_motor_age_ms:
                self.get_logger().warn(
                    f"Motor feedback too old: max_age={max_age:.1f} ms "
                    f"> {self.max_motor_age_ms:.1f} ms. Skip estimator frame."
                )
                return

            if self.require_online and not np.all(motor.online):
                self.get_logger().warn("Some motors offline. Skip estimator frame.")
                return

            imu = self.get_fresh_imu()

            q_abs_policy, dq_policy = self.mapper.real_to_policy_abs_q_dq(
                motor.q_real,
                motor.dq_real,
            )

            result = self.estimator.estimate(
                q_abs_policy=q_abs_policy,
                dq_policy=dq_policy,
                omega_body=imu.gyro_rad_s,
            )

            self.publish_array(self.pub_base_lin_vel, result.base_lin_vel)

            state = np.concatenate(
                [
                    result.base_lin_vel,
                    imu.gyro_rad_s.astype(np.float32),
                    imu.projected_gravity.astype(np.float32),
                    np.array([np.deg2rad(imu.rpy_deg[2])], dtype=np.float32),
                ]
            )
            self.publish_array(self.pub_state, state)

            debug = np.concatenate(
                [
                    result.raw_base_lin_vel,
                    result.stance_mask.astype(np.float32),
                    result.foot_height.astype(np.float32),
                    result.foot_speed.astype(np.float32),
                    np.array([result.confidence, max_age], dtype=np.float32),
                ]
            )
            self.publish_array(self.pub_debug, debug)

            now = time.monotonic()
            if now - self._last_info_log_time >= 1.0:
                self._last_info_log_time = now
                self.get_logger().info(
                    "base_lin_vel="
                    f"{np.array2string(result.base_lin_vel, precision=3)} "
                    f"raw={np.array2string(result.raw_base_lin_vel, precision=3)} "
                    f"stance={result.stance_mask.astype(int).tolist()} "
                    f"height={np.array2string(result.foot_height, precision=3)} "
                    f"conf={result.confidence:.2f} "
                    f"max_age={max_age:.1f}ms"
                )

        except Exception as exc:
            self.log_state_error(f"state estimator error: {exc}")

    def get_fresh_imu(self):
        imu = self.imu.get_latest()
        age_sec = time.time() - float(imu.stamp)
        if not imu.valid or not imu.backend_alive:
            detail = imu.backend_error or "serial receive thread is not alive"
            raise RuntimeError(f"IMU backend invalid: {detail}")
        if not np.isfinite(age_sec) or age_sec > self.max_imu_sample_age_sec:
            raise RuntimeError(
                f"IMU sample stale: age={age_sec:.3f}s > "
                f"{self.max_imu_sample_age_sec:.3f}s"
            )
        return imu

    def log_state_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log_time >= 1.0:
            self._last_error_log_time = now
            self.get_logger().error(message)

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def destroy_node(self):
        try:
            self.imu.stop()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MydogStateEstimatorNode()

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
