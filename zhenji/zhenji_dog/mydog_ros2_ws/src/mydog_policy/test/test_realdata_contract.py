import hashlib
import json
from pathlib import Path

import onnx

from mydog_policy.realdata_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    validate_metadata,
)


def test_packaged_realdata_model_hash_and_metadata():
    resource = Path(__file__).parents[1] / "resource" / MODEL_FILENAME
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == MODEL_SHA256
    model = onnx.load(resource)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    contract = json.loads(metadata["fanfan_deployment_config"])
    assert validate_metadata(contract)


def test_realdata_launch_is_safe_by_default():
    launch = (
        Path(__file__).parents[1]
        / "launch"
        / "sim2real_realdata.launch.py"
    ).read_text(encoding="utf-8")
    assert '"policy_executable": "mydog_realdata_node"' in launch
    assert '"expected_policy_sha256": MODEL_SHA256' in launch
