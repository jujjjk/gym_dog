"""B36100 on the latest calibrated, timing-guarded real RS01 transport."""
import rclpy
from .rs01_model18000_node import Rs01Model18000Node
from .rs01_model36100_core import Model36100Contract, Guarded36100PolicyCore, EXPECTED_ONNX_SHA256


class Rs01Model36100Node(Rs01Model18000Node):
    node_name = 'rs01_model36100_node'
    model_label = 'B36100 guard5 robust guarded omni'
    model_filename = 'B36100.onnx'
    expected_onnx_sha256 = EXPECTED_ONNX_SHA256
    contract_type = Model36100Contract
    policy_core_type = Guarded36100PolicyCore
    topic_namespace = '/mydog/model36100'
    command_topic = '/mydog/model36100/cmd_vel'

    def _extra_status(self):
        result = super()._extra_status()
        result.update(model='B36100', direction_controller='trained_v13_sensor_heading',
                      observation_contract='sensor_snapshot_61',
                      host_guard='policy_sensor_snapshot',
                      inward_guard='omnidirectional_inward_scale_v1_5deg',
                      training_disturbances='push_6_18N_payload_0_2kg',
                      hardware_motion_validated=False)
        result.pop('lateral_correction_mps', None)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Rs01Model36100Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
