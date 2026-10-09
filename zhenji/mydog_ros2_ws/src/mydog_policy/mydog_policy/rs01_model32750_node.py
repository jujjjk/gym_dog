"""B32750 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
from .rs01_model18000_node import Rs01Model18000Node
from .rs01_model32750_core import Model32750Contract, Guarded32750PolicyCore, EXPECTED_ONNX_SHA256


class Rs01Model32750Node(Rs01Model18000Node):
    node_name = 'rs01_model32750_node'
    model_label = 'B32750 guard5 guarded omni'
    model_filename = 'B32750.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model32750Contract
    policy_core_type = Guarded32750PolicyCore
    topic_namespace = '/mydog/model32750'
    command_topic = '/mydog/model32750/cmd_vel'

    def _extra_status(self):
        result = super()._extra_status()
        result.update(model='B32750', direction_controller='trained_v13_sensor_heading',
                      observation_contract='sensor_snapshot_61',
                      host_guard='policy_sensor_snapshot',
                      inward_guard='omnidirectional_inward_scale_v1_5deg',
                      hardware_motion_validated=False)
        result.pop('lateral_correction_mps', None)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model32750Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
