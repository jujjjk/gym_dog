#!/usr/bin/env python3
"""Validate the packaged RS01 model_930 without opening IMU or motor I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import onnxruntime as ort

from .rs01_model930_core import (
    EXPECTED_ONNX_SHA256,
    Model930Contract,
    Rs01Model930Mapper,
    sha256_file,
)


def main(args=None):
    packaged = (
        Path(get_package_share_directory("mydog_policy"))
        / "models"
        / "model_930_rs01_heading52.onnx"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", nargs="?", type=Path, default=packaged)
    parsed = parser.parse_args(args)
    onnx_path = parsed.onnx.expanduser().resolve()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    contract = Model930Contract.from_onnx_session(
        session, onnx_path, EXPECTED_ONNX_SHA256
    )
    mapper = Rs01Model930Mapper(contract)
    real_stand = mapper.policy_target_to_real(contract.default)
    print(f"PASS sha256={sha256_file(onnx_path)}")
    print("graph=52 observations -> 12 actions, policy_hz=50")
    print("policy_order=" + ",".join(contract.joint_names))
    for real_index, motor_id in enumerate(mapper.real_motor_ids):
        policy_index = int(mapper.real_to_policy[real_index])
        sign = float(mapper.sign_policy[policy_index])
        joint = contract.joint_names[policy_index]
        print(
            f"real[{real_index}] motor=0x{int(motor_id):02X} "
            f"joint={joint} sign={sign:+.0f} "
            f"stand={float(real_stand[real_index]):+.8f}rad"
        )


if __name__ == "__main__":
    main()
