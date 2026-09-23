"""B23500 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
from .rs01_model18000_node import Rs01Model18000Node
from .rs01_model23500_core import Model23500Contract, Guarded23500PolicyCore, EXPECTED_ONNX_SHA256


class Rs01Model23500Node(Rs01Model18000Node):
    node_name = 'rs01_model23500_node'
    model_label = 'B23500 V22 guarded omni'
    model_filename = 'B23500.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model23500Contract
    policy_core_type = Guarded23500PolicyCore
    topic_namespace = '/mydog/model23500'
    command_topic = '/mydog/model23500/cmd_vel'

    def _extra_status(self):
        result = super()._extra_status()
        result.update(model='B23500', direction_controller='trained_v13_sensor_heading',
                      observation_contract='sensor_snapshot_61',
                      host_guard='policy_sensor_snapshot',
                      hardware_motion_validated=False)
        # B18000 disables planar correction. B23500 does not: do not publish
        # a false zero or imply that heading-based correction is a position sensor.
        result.pop('lateral_correction_mps', None)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model23500Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
