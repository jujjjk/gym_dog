import json
from pathlib import Path

from mydog_policy.force_coord_contract import (
    MODEL_TASK,
    TORQUE_LIMITS_POLICY,
    validate_metadata,
)


def test_packaged_force_coord_json():
    path = Path(__file__).resolve().parents[1] / "resource" / "fanfan_force_coord_5280.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert validate_metadata(contract)
    assert contract["task"] == MODEL_TASK
    assert contract["control"]["torque_limits"] == TORQUE_LIMITS_POLICY
