#!/usr/bin/env python3
"""Fail-closed validation for selected tilt-recovery checkpoint 5650."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort

from .tilt_recovery_contract import MODEL_SHA256, validate_metadata
from .validate_symmetric_transition_model import file_sha256


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--expected-sha256", default=MODEL_SHA256)
    parsed = parser.parse_args(args=args)

    model = parsed.model.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    digest = file_sha256(model)
    if digest.lower() != parsed.expected_sha256.strip().lower():
        raise SystemExit(
            "SHA256 mismatch: "
            f"expected={parsed.expected_sha256}, actual={digest}"
        )

    session = ort.InferenceSession(
        str(model),
        providers=["CPUExecutionProvider"],
    )
    raw = session.get_modelmeta().custom_metadata_map.get(
        "fanfan_deployment_config"
    )
    if not raw:
        raise SystemExit("ONNX lacks fanfan_deployment_config metadata")
    contract = json.loads(raw)
    validate_metadata(contract)

    print("tilt-recovery 5530/5650 ONNX validation PASSED")
    print(f"model={model}")
    print(f"sha256={digest}")
    print(f"task={contract['task']}")
    print(f"command_ranges={contract['commands']['ranges']}")
    print(f"torque_limits={contract['control']['torque_limits']}")
    print(
        "lateral_feedback="
        f"{contract['control']['command_feedback_lateral_gain']}"
    )


if __name__ == "__main__":
    main()
