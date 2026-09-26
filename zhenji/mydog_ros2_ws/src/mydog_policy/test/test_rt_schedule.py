import numpy as np
import pytest
from types import SimpleNamespace as NS
from mydog_policy.rt_schedule import LatestTarget, next_deadline


def test_deadlines_keep_phase_and_skip_missed_slots():
    assert next_deadline(1.,1.003,.02)==pytest.approx(1.02)
    assert next_deadline(1.,1.035,.02)==pytest.approx(1.04)
    assert next_deadline(1.,1.081,.02)==pytest.approx(1.1)


def test_latest_slot_expires_and_does_not_queue_old_targets():
    slot=LatestTarget();q=np.zeros(12)
    slot.put(q,1.);q[:]=9
    np.testing.assert_array_equal(slot.get(1.02),np.zeros(12))
    slot.put(np.ones(12),1.025)
    np.testing.assert_array_equal(slot.get(1.04),np.ones(12))
    assert slot.get(1.076) is None
    assert slot.get(.9) is None


pytest.importorskip('rclpy')
from test_rs01_model6850_node_offline import make_node


@pytest.mark.parametrize('make_node',['B23500','capture23500'],indirect=True)
def test_periodic_sender_repeats_only_fresh_targets_and_never_primes(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    n.enable_send=True;n.first_send=False;n._send_pump=NS(is_alive=lambda:True)
    sent=[]
    def dispatch():
        sent.append(n._pending_target.copy());n._pending_target=None
    n._dispatch_pending_send=dispatch
    n._enqueue_send(np.zeros(12))
    n._periodic_send_tick(clock.t+.005)
    n._periodic_send_tick(clock.t+.025)
    assert len(sent)==2
    n.trial.arm(clock.t)
    n._periodic_send_tick(clock.t+.055)
    assert len(sent)==2 and not n.trial.armed
    n.first_send=True
    n._enqueue_send(np.ones(12));n._periodic_send_tick(clock.t+.01)
    assert len(sent)==2 and not calls


@pytest.mark.parametrize('make_node',['B23500'],indirect=True)
def test_watchdog_cannot_renew_stalled_walking_target(make_node):
    n,clock,calls=make_node(send=False,stand=False)
    n.mode='walk';n._control_start=clock.t-.1
    n._enqueue_send(np.zeros(12))
    assert n._periodic_target.stamp is None and not calls
