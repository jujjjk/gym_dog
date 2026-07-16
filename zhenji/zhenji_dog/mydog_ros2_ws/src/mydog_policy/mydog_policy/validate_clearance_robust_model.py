#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the exact ONNX contract for clearance-robust checkpoint 5730."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnxruntime as ort

from .clearance_robust_contract import MODEL_SHA256, validate_metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--expected-sha256", default=MODEL_SHA256)
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

    print("clearance-robust 5730 ONNX validation PASSED")
    print(f"model={model}")
    print(f"sha256={digest}")
    print(f"input={session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"output={session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print(f"task={contract['task']}")
    print(f"command_ranges={contract['commands']['ranges']}")
    print(f"torque_limits_policy={contract['control']['torque_limits']}")
    print(f"gait={contract['gait']}")


if __name__ == "__main__":
    main()
