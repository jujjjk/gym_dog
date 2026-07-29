#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed state estimator with robust support-foot selection and snapshot metadata."""

from __future__ import annotations

import time

import numpy as np
import rclpy

from .leg_odometry_robust import RobustLegOdometryEstimator
from .state_estimator_node import MydogStateEstimatorNode
from .state_estimator_contract import append_shared_motor_snapshot


class MydogStateEstimatorFixedNode(MydogStateEstimatorNode):
    """Use robust leg odometry while retaining the existing ROS topic contract."""

    def __init__(self):
        super().__init__()

        self.declare_parameter("no_stance_velocity_decay", 0.85)
        self.declare_parameter("robust_stance_height_margin", 0.035)
        self.declare_parameter("robust_vertical_speed_threshold", 0.22)
        self.declare_parameter("robust_velocity_residual_threshold", 0.30)
        self.declare_parameter("robust_max_stance_feet", 2)
        self.declare_parameter("robust_filter_alpha", 0.35)
        self.declare_parameter("robust_absolute_height_guard", 0.10)
        self.declare_parameter("robust_planar_velocity_clip", 1.0)

        self.no_stance_velocity_decay = float(
            np.clip(
                float(self.get_parameter("no_stance_velocity_decay").value),
                0.0,
                1.0,
            )
        )

        # Replace the original all-foot speed-gated estimator before rclpy.spin()
        # starts invoking the timer created by the base node.
        self.estimator = RobustLegOdometryEstimator(
            nominal_base_height=float(
                self.get_parameter("nominal_base_height").value
            ),
            foot_radius=float(self.get_parameter("foot_radius").value),
            stance_height_margin=float(
                self.get_parameter("robust_stance_height_margin").value
            ),
            stance_vertical_speed_threshold=float(
                self.get_parameter("robust_vertical_speed_threshold").value
            ),
            stance_velocity_residual_threshold=float(
                self.get_parameter(
                    "robust_velocity_residual_threshold"
                ).value
            ),
            max_stance_feet=int(
                self.get_parameter("robust_max_stance_feet").value
            ),
            filter_alpha=float(
                self.get_parameter("robust_filter_alpha").value
            ),
            absolute_height_guard=float(
                self.get_parameter("robust_absolute_height_guard").value
            ),
            planar_velocity_clip=float(
                self.get_parameter("robust_planar_velocity_clip").value
            ),
        )

        self.get_logger().warn(
            "[STATE_FIXED_ROBUST] enabled | "
            f"height_margin={self.estimator.stance_height_margin:.3f}m, "
            "vertical_speed_threshold="
            f"{self.estimator.stance_vertical_speed_threshold:.3f}m/s, "
            "velocity_residual_threshold="
            f"{self.estimator.stance_velocity_residual_threshold:.3f}m/s, "
            f"max_stance_feet={self.estimator.max_stance_feet}, "
            f"filter_alpha={self.estimator.filter_alpha:.2f}, "
            f"no_stance_decay={self.no_stance_velocity_decay:.2f}"
        )

    def update(self):
        try:
            motor = self.motor.get_latest()
            if not motor.valid:
                self.get_logger().warn(
                    "Motor state invalid. Skip estimator frame."
                )
                return

            max_age = float(np.max(motor.age_ms))
            if max_age > self.max_motor_age_ms:
                self.get_logger().warn(
                    f"Motor feedback too old: max_age={max_age:.1f} ms "
                    f"> {self.max_motor_age_ms:.1f} ms. "
                    "Skip estimator frame."
                )
                return

            if self.require_online and not np.all(motor.online):
                self.get_logger().warn(
                    "Some motors offline. Skip estimator frame."
                )
                return

            imu = self.get_fresh_imu()

            q_abs_policy, dq_policy = (
                self.mapper.real_to_policy_abs_q_dq(
                    motor.q_real,
                    motor.dq_real,
                )
            )

            result = self.estimator.estimate(
                q_abs_policy=q_abs_policy,
                dq_policy=dq_policy,
                omega_body=imu.gyro_rad_s,
            )

            if result.confidence <= 0.0:
                self.estimator.filtered *= self.no_stance_velocity_decay
                self.estimator.last_raw = self.estimator.filtered.copy()
                self.estimator.filtered[2] = 0.0
                self.estimator.last_raw[2] = 0.0
                result.base_lin_vel = self.estimator.filtered.copy()
                result.raw_base_lin_vel = self.estimator.last_raw.copy()

            self.publish_array(
                self.pub_base_lin_vel,
                result.base_lin_vel,
            )

            seq = np.asarray(
                motor.snapshot_seq,
                dtype=np.int64,
            ).reshape(12)
            tick = np.asarray(
                motor.board_tick_ms,
                dtype=np.int64,
            ).reshape(12)

            seq_a = int(seq[0]) & 0xFFFF
            seq_b = int(seq[6]) & 0xFFFF
            tick_a = int(tick[0])
            tick_b = int(tick[6])

            # 0:10 remains compatible with the original state estimator.
            # 10:17 is consumed by sim2real_parity_fixed_node.
            state_base = np.concatenate(
                [
                    result.base_lin_vel,
                    imu.gyro_rad_s.astype(np.float32),
                    imu.projected_gravity.astype(np.float32),
                    np.array(
                        [np.deg2rad(imu.rpy_deg[2])],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            result.confidence,
                            float(motor.cache_age_ms),
                            float(seq_a),
                            float(seq_b),
                            float(tick_a),
                            float(tick_b),
                            max_age,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            state = append_shared_motor_snapshot(state_base, motor)
            self.publish_array(self.pub_state, state)

            debug = np.concatenate(
                [
                    result.raw_base_lin_vel,
                    result.stance_mask.astype(np.float32),
                    result.foot_height.astype(np.float32),
                    result.foot_speed.astype(np.float32),
                    np.array(
                        [
                            result.confidence,
                            max_age,
                            float(seq_a),
                            float(seq_b),
                        ],
                        dtype=np.float32,
                    ),
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
                    f"seqA/B={seq_a}/{seq_b} "
                    f"max_age={max_age:.1f}ms"
                )

        except Exception as exc:
            self.log_state_error(
                f"robust fixed state estimator error: {exc}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MydogStateEstimatorFixedNode()
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
