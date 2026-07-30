#!/usr/bin/env python3
"""Guarded ROS 2 deployment node for estimator-parity model_1950."""

from __future__ import annotations

import rclpy

from .rs01_model1950_core import (
    EXPECTED_ONNX_SHA256,
    Model1950Contract,
    Rs01Model1950PolicyCore,
)
from .rs01_model930_node import Rs01Model930Node


class Rs01Model1950Node(Rs01Model930Node):
    """Run model_1950 through the existing verified RS01 safety chain."""

    node_name = "rs01_model1950_node"
    model_label = "model_1950"
    model_filename = "model_1950_rs01_estimator_parity.onnx"
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model1950Contract
    policy_core_type = Rs01Model1950PolicyCore
    observation_count = 54
    topic_namespace = "/mydog/model1950"
    include_path_state = True
    calibrate_gyro_bias = True


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model1950Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
