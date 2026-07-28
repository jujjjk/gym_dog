import hashlib
import json
from pathlib import Path

import onnx

from mydog_policy.saturation_recovery_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    validate_metadata,
)


def test_packaged_model_hash_and_metadata():
    model_path = Path(__file__).parents[1] / "resource" / MODEL_FILENAME
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == MODEL_SHA256
    model = onnx.load(model_path)
    metadata = {item.key: item.value for item in model.metadata_props}
    assert validate_metadata(json.loads(metadata["fanfan_deployment_config"]))


def test_launch_is_hash_locked_and_fail_safe():
    source = (
        Path(__file__).parents[1]
        / "launch"
        / "sim2real_saturation_recovery.launch.py"
    ).read_text(encoding="utf-8")
    assert '"expected_policy_sha256": MODEL_SHA256' in source
    assert '"sim2real_symmetric_transition_5530.launch.py"' in source

