#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the exact ONNX contract for symmetric-transition checkpoint 5530."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnxruntime as ort

from .symmetric_transition_contract import validate_metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--expected-sha256", default="")
    parsed = parser.parse_args(args=args)

    model = parsed.model.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")

    digest = file_sha256(model)
    expected = parsed.expected_sha256.strip().lower()
    if expected and digest.lower() != expected:
        raise SystemExit(
            f"SHA256 mismatch: expected={expected}, actual={digest}"
        )

    session = ort.InferenceSession(
        str(model),
        providers=["CPUExecutionProvider"],
    )
    metadata = session.get_modelmeta().custom_metadata_map
    raw = metadata.get("fanfan_deployment_config")
    if not raw:
        raise SystemExit("ONNX lacks fanfan_deployment_config metadata")

    contract = json.loads(raw)
    validate_metadata(contract)

    print("symmetric-transition 5530 ONNX validation PASSED")
    print(f"model={model}")
    print(f"sha256={digest}")
    print(f"input={session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"output={session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print(f"task={contract['task']}")
    print(f"command_ranges={contract['commands']['ranges']}")
    print(f"gait={contract['gait']}")
    print(f"control_feedback={{"
          f"vx:{contract['control']['command_feedback_longitudinal_gain']}, "
          f"vy:{contract['control']['command_feedback_lateral_gain']}, "
          f"yaw:{contract['control']['command_feedback_yaw_gain']}, "
          f"heading:{contract['control']['command_feedback_heading_gain']}, "
          f"damping:{contract['control']['command_feedback_heading_damping']}"
          f"}}")
    print(
        "strict_symmetry="
        f"{contract['control']['enforce_policy_symmetry']}"
    )


if __name__ == "__main__":
    main()
