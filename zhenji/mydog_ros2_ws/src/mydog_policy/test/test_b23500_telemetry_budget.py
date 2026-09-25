import csv
from types import SimpleNamespace as NS
import pytest
from test_rs01_model6850_node_offline import make_node, advance


@pytest.mark.parametrize('make_node',['capture23500'],indirect=True)
def test_full_rate_capture_with_decimated_status(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    advance(n,clock,100)
    n.capture.close()
    with (n.capture.path/'cycles.csv').open() as f:
        rows=list(csv.DictReader(f))
    status=[msg for topic,msg in n.messages if topic.endswith('/status')]
    assert len(rows)==100
    assert 10 <= len(status) <= 25
    assert n.capture.error=='' and n.capture.dropped==0
    assert n._extra_status()['status_publish_hz']==10.
    # An operator authorization transition is published without waiting 100ms.
    n.trial.arm(clock.t)
    n.trial.receive([0.,0.,0.],clock.t)
    advance(n,clock,1)
    assert len([msg for topic,msg in n.messages if topic.endswith('/status')])>len(status)
    assert not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_timing_fault_reports_window_and_keeps_threshold(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    n._b_loop_intervals.clear()
    n._b_loop_intervals.extend([.040]*50)
    assert not n._b_timing_ready()
    detail=n._timing_fault_detail(clock.t,clock.t-.05)
    assert 'control_gap_ms=50.00' in detail
    assert 'median=40.00' in detail and 'p95=40.00' in detail
    assert not calls
