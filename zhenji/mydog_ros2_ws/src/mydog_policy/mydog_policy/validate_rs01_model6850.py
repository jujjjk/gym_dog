"""Offline validation only; does not import ROS or open hardware interfaces."""
import argparse
import json
import numpy as np
import onnxruntime as ort
from .rs01_model6850_core import Model6850Contract,Rs01Model6850Core,EXPECTED_ONNX_SHA256


def main():
    p=argparse.ArgumentParser();p.add_argument('onnx_path');args=p.parse_args()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(args.onnx_path,sess_options=options,providers=['CPUExecutionProvider'])
    contract=Model6850Contract.from_onnx_session(session,args.onnx_path)
    core=Rs01Model6850Core(session,contract)
    result=core.tick(contract.default,np.zeros(12),np.zeros(3),[0,0,-1],0,[0,0,0],gait=1)
    print(json.dumps(dict(task=contract.raw['task'],sha256=EXPECTED_ONNX_SHA256,
        observations=len(result['observation']),actions=len(result['action']),
        policy_hz=1/contract.policy_dt,finite=bool(np.isfinite(result['target_real']).all()),
        motor_send_supported=False,hardware_validated=False),indent=2))


if __name__=='__main__':main()
