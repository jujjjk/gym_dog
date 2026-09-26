"""B23500 state-only RT reader, separate socket from motor command ownership."""
import threading
import time
from collections import deque
from .rt_schedule import next_deadline
from .motor_rt_client import MotorRtClient
from .motor_state_interface import MotorStateHttpInterface


class MotorStateRtInterface(MotorStateHttpInterface):
    def __init__(self, **kwargs):
        self._state_rt = MotorRtClient(timeout=float(kwargs.get('timeout', .04)))
        self._state_rt_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._history = deque(maxlen=128)
        kwargs['poll_hz'] = 100.
        super().__init__(**kwargs)

    def _fetch_latest_sync(self):
        with self._state_rt_lock:
            # Only GET_STATE; never OP_COMMAND, STOP, enable or HTTP fallback.
            payload = self._state_rt.exchange()['payload']
            snapshot = self._snapshot_from_all_states(payload)
            with self._history_lock:
                self._history.append(snapshot)
            return snapshot

    def get_history_view(self):
        with self._history_lock:
            return tuple(self._history)

    def _poll_loop(self):
        period = 1./max(self.poll_hz, 1.)
        deadline = time.perf_counter()
        while not self._stop_event.is_set():
            if self._stop_event.wait(max(0., deadline-time.perf_counter())):
                break
            try:
                snapshot = self._fetch_latest_sync()
                with self._cache_lock:
                    self._latest_snapshot, self._latest_error = snapshot, None
            except Exception as exc:
                with self._cache_lock:
                    self._latest_error = exc
            deadline = next_deadline(deadline, time.perf_counter(), period)

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
