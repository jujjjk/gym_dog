#!/usr/bin/env python3
"""Validate packaged RS01 model_1850 without opening motor or IMU I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np
import onnxruntime as ort

from .rs01_model1850_core import (
    EXPECTED_ONNX_SHA256,
    Model1850Contract,
    Rs01Model1850PolicyCore,
)
from .rs01_model930_core import Rs01Model930Mapper, sha256_file


def main(args=None):
    packaged = (
        Path(get_package_share_directory("mydog_policy"))
        / "models"
        / "model_1850_rs01_path54.onnx"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", nargs="?", type=Path, default=packaged)
    parsed = parser.parse_args(args)
    onnx_path = parsed.onnx.expanduser().resolve()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    contract = Model1850Contract.from_onnx_session(
        session, onnx_path, EXPECTED_ONNX_SHA256
    )
    mapper = Rs01Model930Mapper(contract)
    core = Rs01Model1850PolicyCore(session, contract)
    core.reset(now=0.0, yaw=0.0, q_policy=contract.default)
    observation = core.build_observation(
        now=contract.policy_dt,
        base_linear_velocity=[0.23, 0.0, 0.0],
        base_angular_velocity=[0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.23, 0.0, 0.0],
        q_policy=contract.default,
        dq_policy=np.zeros(12, dtype=np.float32),
        yaw=0.0,
    )
    action = session.run(
        ["actions"], {"observations": observation[None, :]}
    )[0][0]
    real_stand = mapper.policy_target_to_real(contract.default)
    if observation.shape != (54,) or action.shape != (12,):
        raise RuntimeError(
            f"Unexpected inference shapes {observation.shape}->{action.shape}"
        )
    if not np.all(np.isfinite(observation)) or not np.all(
        np.isfinite(action)
    ):
        raise RuntimeError("model_1850 dry inference contains NaN/Inf")

    print(f"PASS sha256={sha256_file(onnx_path)}")
    print("graph=54 observations -> 12 actions, policy_hz=50")
    print("path_state=body-velocity integration in initial-heading frame")
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
