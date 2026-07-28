#!/usr/bin/env python3
import argparse
import hashlib
import json

from .omni_fast_contract import (
    ACTIONS,
    MODEL_SHA256,
    MODEL_TASK,
    OBSERVATIONS,
    validate_metadata,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Validate Fanfan Omni yaw-clean ONNX")
    parser.add_argument("model")
    args = parser.parse_args()

    actual_hash = sha256(args.model)
    if actual_hash != MODEL_SHA256:
        raise SystemExit(f"FAIL SHA256 expected={MODEL_SHA256} actual={actual_hash}")

    import onnxruntime as ort

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if int(model_input.shape[-1]) != OBSERVATIONS:
        raise SystemExit(f"FAIL input shape={model_input.shape}")
    if int(model_output.shape[-1]) != ACTIONS:
        raise SystemExit(f"FAIL output shape={model_output.shape}")

    metadata = session.get_modelmeta().custom_metadata_map
    raw_contract = metadata.get("fanfan_deployment_config")
    if not raw_contract:
        raise SystemExit("FAIL missing fanfan_deployment_config")
    try:
        validate_metadata(json.loads(raw_contract))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL metadata: {exc}") from exc

    print(
        f"OK task={MODEL_TASK} sha256={actual_hash} "
        f"input={model_input.shape} output={model_output.shape}"
    )


if __name__ == "__main__":
    main()
