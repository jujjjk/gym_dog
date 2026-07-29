#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the ONNX graph identity and embedded force_coord_5280 metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .force_coord_contract import validate_metadata
from .mydog_policy_node import OnnxPolicyRunner


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        default=(
            "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
            "resource/fanfan_force_coord_5280.onnx"
        ),
    )
    args = parser.parse_args(argv)
    path = Path(args.onnx).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    runner = OnnxPolicyRunner(str(path))
    contract = runner.deployment_config
    if contract is None:
        raise RuntimeError("ONNX has no fanfan_deployment_config metadata")
    validate_metadata(contract)

    print("VALID: fanfan_force_coord_5280 deployment contract")
    print("path:", path)
    print("sha256:", file_sha256(path))
    print("task:", contract["task"])
    print("dimensions:", json.dumps(contract["dimensions"]))
    print("joint_names:", json.dumps(contract["joint_names"]))
    print("torque_limits_policy:", json.dumps(contract["control"]["torque_limits"]))
    print(
        "control_period_s:",
        float(contract["control"]["sim_dt"])
        * int(contract["control"]["decimation"]),
    )
    print("gait_period_s:", contract["gait"]["period"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
