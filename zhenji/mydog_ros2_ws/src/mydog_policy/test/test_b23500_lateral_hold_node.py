"""Lateral-hold telemetry contract. Skipped where rclpy is unavailable."""
import csv
import pytest

pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node, advance  # noqa: E402

HOLD_KEYS = ('lateral_hold_active', 'lateral_hold_offset_m',
             'lateral_hold_correction_mps', 'lateral_hold_reason')


@pytest.mark.parametrize('make_node', ['B23500', 'capture23500'], indirect=True)
def test_status_reports_the_hold_while_standing(make_node):
    node, clock, calls = make_node(send=False, stand=False)
    advance(node, clock, 180)
    assert node.mode == 'ready' and not calls
    status = node._extra_status()
    for key in HOLD_KEYS:
        assert key in status
    assert status['lateral_hold_active'] is False
    assert status['lateral_hold_offset_m'] == 0.
    assert status['lateral_hold_correction_mps'] == 0.
    assert not node.trial.armed


@pytest.mark.parametrize('make_node', ['capture23500'], indirect=True)
def test_capture_columns_are_stable_from_the_first_row(make_node):
    node, clock, calls = make_node(send=False, stand=False)
    advance(node, clock, 180)
    assert not calls and not node.capture.error and not node.capture.dropped
    node.capture.close()
    with (node.capture.path/'cycles.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(r['policy_evaluated'] == 'False' for r in rows)
    for row in rows:
        for key in HOLD_KEYS:
            assert key in row, 'capture column set must not change mid-session'
        assert row['lateral_hold_active'] in ('True', 'False')
        assert float(row['lateral_hold_offset_m']) == 0.
        assert float(row['lateral_hold_correction_mps']) == 0.
