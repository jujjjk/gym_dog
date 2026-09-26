from types import SimpleNamespace as NS
from mydog_policy import motor_state_rt_interface as module


def test_reader_only_gets_state_and_command_reply_cannot_replace_cache(monkeypatch):
    calls=[]
    class Client:
        def __init__(self, **kw):pass
        def exchange(self, *args):
            calls.append(args)
            return {'payload':{'state':1}}
        def close(self):calls.append('close')
    monkeypatch.setattr(module,'MotorRtClient',Client)
    m=module.MotorStateRtInterface(async_poll=False)
    monkeypatch.setattr(m,'_snapshot_from_all_states',lambda p:p)
    assert m._fetch_latest_sync()=={'state':1}
    assert m.get_history_view() == ({'state':1},)
    m._fetch_latest_sync()
    assert len(m.get_history_view()) == 2
    sentinel=object();m._latest_snapshot=sentinel
    assert m.snapshot_from_payload({'stale':True}) is sentinel
    m.pause_async_poll();assert not m._poll_paused.is_set()
    m.close()
    assert calls==[(),(),'close'] and m.poll_hz==100.
