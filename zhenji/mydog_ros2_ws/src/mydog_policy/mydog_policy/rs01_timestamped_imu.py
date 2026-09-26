"""Yahboom IMU with frame-reception freshness and coherent getter locking."""

from collections import deque
from copy import deepcopy
import threading
import time

from YbImuLib import YbImuSerial

from .imu_serial_interface import ImuSerialInterface


class FrameStampedVendor(YbImuSerial):
    def __init__(self, *args, **kwargs):
        self.frame_lock = threading.RLock()
        self.frame_stamps = {}
        self.frame_history = {}
        self.closed = threading.Event()
        super().__init__(*args, **kwargs)

    def _parse_data(self, ext_type, ext_data):
        # Called by the vendor parser only AFTER checksum validation.
        with self.frame_lock:
            super()._parse_data(ext_type, ext_data)
            if ext_type in (self.FUNC_REPORT_IMU_RAW,
                            self.FUNC_REPORT_IMU_QUAT,
                            self.FUNC_REPORT_IMU_EULER):
                self.frame_stamps[ext_type] = time.time()
                mono = time.monotonic()
                if ext_type == self.FUNC_REPORT_IMU_RAW:
                    value = tuple(self.get_gyroscope_data())
                elif ext_type == self.FUNC_REPORT_IMU_QUAT:
                    value = tuple(self.get_imu_quaternion_data())
                else:
                    value = None
                if value is not None:
                    self.frame_history.setdefault(ext_type, deque(maxlen=128)).append((mono, value))

    def _data_handle(self):
        self._dev.flushInput()
        while not self.closed.is_set():
            if self._dev.inWaiting() <= 0:
                self.closed.wait(.001)
                continue
            for value in self._dev.read_all():
                self._receive_data(value)

    def close(self):
        self.closed.set()
        # Let the receive loop leave before closing its descriptor.
        time.sleep(.01)
        self._dev.close()


class FrameStampedImu(ImuSerialInterface):
    vendor_type = FrameStampedVendor

    def __init__(self, *args, **kwargs):
        self._history_lock = threading.Lock()
        self._history = deque(maxlen=32)
        super().__init__(*args, **kwargs)

    def get_history(self):
        with self._history_lock:
            return [deepcopy(s) for s in self._history]

    def get_history_view(self):
        """Internal read-only view; published history entries are never mutated."""
        with self._history_lock:
            return tuple(self._history)

    def get_frame_history(self):
        """Independent immutable RAW/QUAT records, stamped at parser receipt."""
        with self.imu.frame_lock:
            return {name: tuple(self.imu.frame_history.get(kind, ())) for name, kind in
                    [('gyro', self.imu.FUNC_REPORT_IMU_RAW), ('quat', self.imu.FUNC_REPORT_IMU_QUAT)]}

    def _required_frame_kinds(self):
        return (self.imu.FUNC_REPORT_IMU_RAW, self.imu.FUNC_REPORT_IMU_QUAT,
                self.imu.FUNC_REPORT_IMU_EULER)

    def _finish_snapshot(self, snapshot):
        return snapshot

    def _read_snapshot(self):
        with self.imu.frame_lock:
            kinds = self._required_frame_kinds()
            stamps = [self.imu.frame_stamps.get(k, 0.) for k in kinds]
            snapshot = super()._read_snapshot()
            snapshot.stamp = min(stamps)
            # The oldest required frame controls freshness. Re-reading the
            # same cached values cannot refresh this timestamp.
            snapshot.valid = (min(stamps) > 0 and
                              0 <= time.time() - min(stamps) <= .080 and
                              max(stamps) - min(stamps) <= .060)
            snapshot.frame_receive_wall = tuple(stamps)
            snapshot = self._finish_snapshot(snapshot)
            snapshot.acquisition_timestamp = None
            snapshot.acquisition_sync_verified = False
            with self._history_lock:
                if snapshot.valid and (not self._history or self._history[-1].frame_receive_wall != tuple(stamps)):
                    owned = deepcopy(snapshot)
                    for value in vars(owned).values():
                        if hasattr(value, 'setflags'):
                            value.setflags(write=False)
                    self._history.append(owned)
            return snapshot


class QuaternionFrameStampedImu(FrameStampedImu):
    """B23500: Euler and gravity share the same validated quaternion frame."""
    def start(self):
        if self.running:
            return
        super().start()
        # Vendor supported range is 10..100 Hz. Configure reporting, then
        # measure actual frame cadence; never relabel a 25 Hz stream as 100 Hz.
        self.imu.set_report_rate(100)

    def _required_frame_kinds(self):
        return (self.imu.FUNC_REPORT_IMU_RAW, self.imu.FUNC_REPORT_IMU_QUAT)

    def _finish_snapshot(self, snapshot):
        from .observation_pipeline import quaternion_matrix, rotation_rpy
        snapshot.rpy_deg = rotation_rpy(quaternion_matrix(snapshot.quat_wxyz))
        snapshot.euler_source = 'quaternion'
        snapshot.euler_report_receive_wall = self.imu.frame_stamps.get(self.imu.FUNC_REPORT_IMU_EULER, 0.)
        return snapshot
