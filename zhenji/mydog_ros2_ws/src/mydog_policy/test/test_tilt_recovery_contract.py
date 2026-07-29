import hashlib
import json
from pathlib import Path

import onnx
import pytest

from mydog_policy.tilt_recovery_contract import (
    MODEL_FILENAME,
    MODEL_SHA256,
    validate_metadata,
)


def packaged_contract():
    resource = Path(__file__).parents[1] / "resource" / MODEL_FILENAME
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == MODEL_SHA256
    model = onnx.load(resource)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    return json.loads(metadata["fanfan_deployment_config"])


def test_packaged_tilt_recovery_model_hash_and_metadata():
    assert validate_metadata(packaged_contract())


def test_tilt_recovery_contract_rejects_old_lateral_feedback():
    contract = packaged_contract()
    contract["control"]["command_feedback_lateral_gain"] = 0.80
    with pytest.raises(ValueError, match="command_feedback_lateral_gain"):
        validate_metadata(contract)


def test_tilt_recovery_node_resets_previous_action_without_phase_reset():
    node = (
        Path(__file__).parents[1]
        / "mydog_policy"
        / "sim2real_tilt_recovery_node.py"
    ).read_text(encoding="utf-8")
    assert "guarded_obs[36:48] = 0.0" in node
    assert "_reset_contract_state(reset_phase=False)" in node
