#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only start-pose and semantic-mapping checker for force_coord model 5280."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .force_coord_contract import validate_metadata
from .motor_state_interface import MotorStateHttpInterface
from .mydog_policy_node import OnnxPolicyRunner
from .semantic_mapper import JointSemanticMapper


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        default=(
            "/home/jetson/mydog_ros2_ws/src/mydog_policy/"
            "resource/fanfan_force_coord_5280.onnx"
        ),
    )
    parser.add_argument("--motor-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warn-rad", type=float, default=0.08)
    parser.add_argument("--fail-rad", type=float, default=0.30)
    parser.add_argument(
        "--assume-physically-at-model-default",
        action="store_true",
        help="Print candidate zero offsets; no file or motor command is modified.",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not (0.0 <= args.warn_rad < args.fail_rad):
        raise ValueError("require 0 <= warn-rad < fail-rad")

    runner = OnnxPolicyRunner(args.onnx)
    config = runner.deployment_config
    if not config:
        raise RuntimeError("ONNX does not contain fanfan_deployment_config metadata")
    validate_metadata(config)

    mapper = JointSemanticMapper()
    mapper.configure_policy_contract(
        config["joint_names"],
        config["default_joint_angles"],
    )

    motor = MotorStateHttpInterface(
        base_url=args.motor_url,
        timeout=0.2,
        stale_recheck_ms=100.0,
        enable_stale_recheck=False,
        async_poll=False,
    )
    try:
        snap = motor.get_latest()
    finally:
        motor.close()

    if not snap.valid:
        raise RuntimeError("motor snapshot is invalid or server cache is stale")
    if not np.all(snap.online):
        offline = np.where(~snap.online)[0].tolist()
        raise RuntimeError(f"offline motor indices: {offline}")

    q_abs, _ = mapper.real_to_policy_abs_q_dq(
        snap.q_real,
        np.zeros(12, dtype=np.float32),
    )
    default = mapper.default_joint_angle.astype(np.float32)
    error = q_abs - default

    print("joint,start_policy_rad,model_default_rad,error_rad,status")
    statuses = []
    for name, current, target, err in zip(
        mapper.policy_joint_names,
        q_abs,
        default,
        error,
    ):
        magnitude = abs(float(err))
        if magnitude >= args.fail_rad:
            status = "FAIL"
        elif magnitude >= args.warn_rad:
            status = "WARN"
        else:
            status = "OK"
        statuses.append(status)
        print(
            f"{name},{float(current):+.6f},{float(target):+.6f},"
            f"{float(err):+.6f},{status}"
        )

    max_i = int(np.argmax(np.abs(error)))
    print(
        "summary: "
        f"max_abs_error={float(np.max(np.abs(error))):.6f} rad "
        f"joint={mapper.policy_joint_names[max_i]} "
        f"state_cache_age_ms={float(snap.cache_age_ms):.1f}"
    )

    if args.assume_physically_at_model_default:
        q_real_ordered = np.asarray(snap.q_real, dtype=np.float32)[
            mapper.policy_to_real_index
        ]
        candidate = (
            q_real_ordered - mapper.joint_sign * mapper.default_joint_angle
        ).astype(np.float32)
        print("\nCandidate real_zero_offset_policy_order (NOT applied):")
        print("[" + ", ".join(f"{float(x):+.6f}" for x in candidate) + "]")

    if "FAIL" in statuses:
        return 2
    if "WARN" in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
