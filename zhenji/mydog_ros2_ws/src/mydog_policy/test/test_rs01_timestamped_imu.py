"""Serial-frame timestamp regression tests, without opening a serial port."""
import threading
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('YbImuLib')
from mydog_policy import rs01_timestamped_imu as module


def test_cache_reads_cannot_refresh_missing_serial_frames(monkeypatch):
    backend = NS(frame_lock=threading.RLock(), FUNC_REPORT_IMU_RAW=4,
                 FUNC_REPORT_IMU_QUAT=22, FUNC_REPORT_IMU_EULER=38,
                 frame_stamps={4: 10., 22: 10.005, 38: 10.01})
    interface = module.FrameStampedImu.__new__(module.FrameStampedImu)
    interface.imu = backend
    monkeypatch.setattr(module.ImuSerialInterface, '_read_snapshot',
                        lambda s: NS(stamp=999., valid=True))
    clock = NS(t=10.02)
    monkeypatch.setattr(module, 'time', NS(time=lambda: clock.t))
    snapshot = interface._read_snapshot()
    assert snapshot.valid and snapshot.stamp == 10.
    clock.t = 10.061
    snapshot = interface._read_snapshot()
    assert not snapshot.valid and snapshot.stamp == 10.
    backend.frame_stamps[4] = 10.06
    backend.frame_stamps[22] = 10.06
    # Euler must progress too; refreshing gyro alone is insufficient.
    assert not interface._read_snapshot().valid
    backend.frame_stamps[38] = 10.06
    assert interface._read_snapshot().valid


def test_parser_updates_only_required_frame_stamp(monkeypatch):
    vendor = module.FrameStampedVendor.__new__(module.FrameStampedVendor)
    vendor.frame_lock = threading.RLock()
    vendor.frame_stamps = {}
    vendor.FUNC_REPORT_IMU_RAW = 4
    vendor.FUNC_REPORT_IMU_QUAT = 22
    vendor.FUNC_REPORT_IMU_EULER = 38
    monkeypatch.setattr(module.YbImuSerial, '_parse_data', lambda *a: None)
    monkeypatch.setattr(module, 'time', NS(time=lambda: 123.))
    vendor._parse_data(4, [])
    vendor._parse_data(1, [])
    assert vendor.frame_stamps == {4: 123.}
