#!/usr/bin/env python3
"""NX deployment node for the real-data curriculum policy."""

from __future__ import annotations

import rclpy

from .realdata_contract import MODEL_TASK, validate_metadata
from .sim2real_hardware_balance_node import MydogHardwareBalanceNode


class MydogRealDataNode(MydogHardwareBalanceNode):
    """Use dynamic gait and complete-target limiting from ONNX metadata."""

    validate_contract = staticmethod(validate_metadata)
    model_task = MODEL_TASK
    expected_fixed_gait_period = None


def main(args=None):
    rclpy.init(args=args)
    node = MydogRealDataNode()
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
