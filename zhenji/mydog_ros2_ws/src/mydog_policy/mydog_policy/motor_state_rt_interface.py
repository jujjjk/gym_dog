"""B23500 state-only RT reader, separate socket from motor command ownership."""
import threading
from .motor_rt_client import MotorRtClient
from .motor_state_interface import MotorStateHttpInterface


class MotorStateRtInterface(MotorStateHttpInterface):
    def __init__(self, **kwargs):
        self._state_rt = MotorRtClient(timeout=float(kwargs.get('timeout', .04)))
        self._state_rt_lock = threading.Lock()
        kwargs['poll_hz'] = 100.
        super().__init__(**kwargs)

    def _fetch_latest_sync(self):
        with self._state_rt_lock:
            # Only GET_STATE; never OP_COMMAND, STOP, enable or HTTP fallback.
            payload = self._state_rt.exchange()['payload']
            return self._snapshot_from_all_states(payload)

    def snapshot_from_payload(self, data):
        # The independent reader owns cache chronology. Do not let a delayed
        # command reply replace newer sensor state from the read socket.
        with self._cache_lock:
            return self._latest_snapshot

    def pause_async_poll(self):
        # The old HTTP reader had to yield to HTTP motor commands. Cached RT
        # GET_STATE is independent and must keep refreshing during stand/walk.
        pass

    def close(self):
        super().close()
        with self._state_rt_lock:
            self._state_rt.close()
