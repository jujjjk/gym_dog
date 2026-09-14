"""B18000 on the upstream 5544692 asynchronous 50Hz hardware loop.

No auto-arm; startup stays dry-run/stand-only. A6850 and B18000 share the same
exclusive hardware lock. Timing failure disarms motion, keeping the old soft-stop chain.
"""
from collections import deque
import time
import numpy as np
import rclpy
from .rs01_model6850_node import Rs01Model6850Node
from .rs01_model18000_core import Model18000Contract, Guarded18000PolicyCore, EXPECTED_ONNX_SHA256


def timing_window_ready(intervals):
    values=np.asarray(list(intervals),dtype=float)
    return bool(len(values)>=40 and np.isfinite(values).all()
                and .018<=np.median(values)<=.022 and np.percentile(values,95)<=.030
                and values.min()>=.008 and values.max()<=.040)


class Rs01Model18000Node(Rs01Model6850Node):
    node_name='rs01_model18000_node'
    model_label='B18000 guarded omni'
    model_filename='B18000.onnx'
    expected_onnx_sha256=EXPECTED_ONNX_SHA256
    contract_type=Model18000Contract
    policy_core_type=Guarded18000PolicyCore
    topic_namespace='/mydog/model18000'
    command_topic='/mydog/model18000/cmd_vel'

    def __init__(self):
        self._b_loop_intervals=deque(maxlen=50)
        self._b_send_intervals=deque(maxlen=50)
        self._b_last_success=None
        super().__init__()

    def _validate_deployment_parameters(self):
        super()._validate_deployment_parameters()
        for name,want in [('hardware_torque_limit_nm',14.),('continuous_torque_nm',6.),
                          ('thermal_derate_full_rms_nm',8.),('thermal_rms_time_constant_sec',2.)]:
            if abs(float(self.get_parameter(name).value)-want)>1e-6:
                raise RuntimeError('B18000 requires %s=%s'%(name,want))

    def _ingest_send_feedback(self, response=None):
        # Called only after the inherited HTTP send succeeds, not on failed attempts.
        now=time.perf_counter()
        if self._b_last_success is not None:
            self._b_send_intervals.append(now-self._b_last_success)
        self._b_last_success=now
        super()._ingest_send_feedback(response)

    def _b_timing_ready(self):
        return bool(timing_window_ready(self._b_loop_intervals)
                    and (not self.enable_send or
                         (timing_window_ready(self._b_send_intervals)
                          and self._b_last_success is not None
                          and time.perf_counter()-self._b_last_success<=.040)))

    def arm_callback(self, request, response):
        if request.data and not self._b_timing_ready():
            response.success=False
            response.message='B18000 requires measured ~50Hz control and successful sends for40 samples'
            return response
        return super().arm_callback(request,response)

    def control_loop(self):
        now=time.monotonic(); previous=self._previous_control
        if self.mode=='walk' and ((previous is not None and now-previous>.040)
                                  or not self._b_timing_ready()):
            self.trial.disarm('B18000 timing contract lost; operator re-arm required')
        super().control_loop()
        if previous is not None and self._previous_control!=previous:
            self._b_loop_intervals.append(self._previous_control-previous)

    def _extra_status(self):
        result=super()._extra_status()
        result.update(model='B18000',timing_ready=self._b_timing_ready(),
                      hardware_motion_validated=False,
                      policy_target_mapping='conditional_inward_scale_v1')
        return result


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=Rs01Model18000Node();rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
