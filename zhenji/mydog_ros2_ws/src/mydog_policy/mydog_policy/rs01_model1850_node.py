#!/usr/bin/env python3
"""Guarded ROS 2 deployment node for the 54-D RS01 model_1850."""

from __future__ import annotations

import rclpy

from .rs01_model1850_core import (
    EXPECTED_ONNX_SHA256,
    Model1850Contract,
    Rs01Model1850PolicyCore,
)
from .rs01_model930_node import Rs01Model930Node


class Rs01Model1850Node(Rs01Model930Node):
    """Run model_1850 without weakening the existing RS01 safety chain."""

    node_name = "rs01_model1850_node"
    model_label = "model_1850"
    model_filename = "model_1850_rs01_path54.onnx"
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model1850Contract
    policy_core_type = Rs01Model1850PolicyCore
    observation_count = 54
    topic_namespace = "/mydog/model1850"
    include_path_state = True
    calibrate_gyro_bias = True


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model1850Node()
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
