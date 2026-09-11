"""Yahboom IMU with frame-reception freshness and coherent getter locking."""

import threading
import time

from YbImuLib import YbImuSerial

from .imu_serial_interface import ImuSerialInterface


class FrameStampedVendor(YbImuSerial):
    def __init__(self, *args, **kwargs):
        self.frame_lock = threading.RLock()
        self.frame_stamps = {}
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

    def _read_snapshot(self):
        with self.imu.frame_lock:
            kinds = (self.imu.FUNC_REPORT_IMU_RAW,
                     self.imu.FUNC_REPORT_IMU_QUAT,
                     self.imu.FUNC_REPORT_IMU_EULER)
            stamps = [self.imu.frame_stamps.get(k, 0.) for k in kinds]
            snapshot = super()._read_snapshot()
            snapshot.stamp = min(stamps)
            # The oldest required frame controls freshness. Re-reading the
            # same cached values cannot refresh this timestamp.
            snapshot.valid = (min(stamps) > 0 and
                              0 <= time.time() - min(stamps) <= .060 and
                              max(stamps) - min(stamps) <= .030)
            return snapshot
