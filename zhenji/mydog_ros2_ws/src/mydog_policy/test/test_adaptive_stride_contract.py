import hashlib
import json
from pathlib import Path

import onnx

from mydog_policy.adaptive_stride_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    validate_metadata,
)


def test_packaged_adaptive_stride_model_hash_and_metadata():
    resource = Path(__file__).parents[1] / "resource" / MODEL_FILENAME
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == MODEL_SHA256
    model = onnx.load(resource)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert validate_metadata(json.loads(metadata["fanfan_deployment_config"]))


def test_adaptive_stride_launch_reuses_fail_safe_launch():
    launch = (
        Path(__file__).parents[1]
        / "launch"
        / "sim2real_adaptive_stride.launch.py"
    ).read_text(encoding="utf-8")
    assert '"sim2real_symmetric_transition_5530.launch.py"' in launch
    assert '"mydog_adaptive_stride_node"' in launch
    assert '"expected_policy_sha256": MODEL_SHA256' in launch

