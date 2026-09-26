"""Reuse the coherent capture writer with the isolated V22 controller."""
import rclpy
from .capture61_node import Capture61Node
from .rs01_model23500_node import Rs01Model23500Node


class Capture23500Node(Capture61Node, Rs01Model23500Node):
    # Cooperative super(): capture -> V22 -> latest calibrated transport.
    capture_schema = 'b23500_capture61_v1'


def main(args=None):
    import sys
    sys.setswitchinterval(.001)
    rclpy.init(args=args)
    node = None
    try:
        node = Capture23500Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
