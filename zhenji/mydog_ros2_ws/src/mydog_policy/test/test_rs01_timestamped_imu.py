"""Serial-frame timestamp regression tests, without opening a serial port."""
import threading
from types import SimpleNamespace as NS

import pytest
import numpy as np

pytest.importorskip('YbImuLib')
from mydog_policy import rs01_timestamped_imu as module


def test_cache_reads_cannot_refresh_missing_serial_frames(monkeypatch):
    backend = NS(frame_lock=threading.RLock(), FUNC_REPORT_IMU_RAW=4,
                 FUNC_REPORT_IMU_QUAT=22, FUNC_REPORT_IMU_EULER=38,
                 frame_stamps={4: 10., 22: 10.005, 38: 10.01})
    interface = module.FrameStampedImu()
    interface.imu = backend
    monkeypatch.setattr(module.ImuSerialInterface, '_read_snapshot',
                        lambda s: NS(stamp=999., valid=True))
    clock = NS(t=10.02)
    monkeypatch.setattr(module, 'time', NS(time=lambda: clock.t))
    snapshot = interface._read_snapshot()
    assert snapshot.valid and snapshot.stamp == 10.
    clock.t = 10.091
    snapshot = interface._read_snapshot()
    assert not snapshot.valid and snapshot.stamp == 10.
    backend.frame_stamps[4] = 10.09
    backend.frame_stamps[22] = 10.09
    # Euler must progress too; refreshing gyro alone is insufficient.
    assert not interface._read_snapshot().valid
    backend.frame_stamps[38] = 10.09
    assert interface._read_snapshot().valid


def test_parser_updates_only_required_frame_stamp(monkeypatch):
    vendor = module.FrameStampedVendor.__new__(module.FrameStampedVendor)
    vendor.frame_lock = threading.RLock()
    vendor.frame_stamps = {}
    vendor.frame_history = {}
    vendor.get_gyroscope_data = lambda: (1., 2., 3.)
    vendor.FUNC_REPORT_IMU_RAW = 4
    vendor.FUNC_REPORT_IMU_QUAT = 22
    vendor.FUNC_REPORT_IMU_EULER = 38
    monkeypatch.setattr(module.YbImuSerial, '_parse_data', lambda *a: None)
    monkeypatch.setattr(module, 'time', NS(time=lambda: 123., monotonic=lambda: 23.))
    vendor._parse_data(4, [])
    vendor._parse_data(1, [])
    assert vendor.frame_stamps == {4: 123.}
    assert list(vendor.frame_history[4]) == [(23., (1., 2., 3.))]


def test_quaternion_controls_euler_and_only_used_frames_set_age(monkeypatch):
    backend=NS(frame_lock=threading.RLock(),FUNC_REPORT_IMU_RAW=4,
               FUNC_REPORT_IMU_QUAT=22,FUNC_REPORT_IMU_EULER=38,
               frame_stamps={4:10.,22:10.005,38:1.})
    interface=module.QuaternionFrameStampedImu();interface.imu=backend
    quaternion=np.array([np.cos(np.pi/8),0.,0.,np.sin(np.pi/8)])
    monkeypatch.setattr(module.ImuSerialInterface,'_read_snapshot',
                        lambda s:NS(stamp=999.,valid=True,quat_wxyz=quaternion.copy(),rpy_deg=np.array([0.,0.,-90.])))
    clock=NS(t=10.02)
    monkeypatch.setattr(module,'time',NS(time=lambda:clock.t))
    snapshot=interface._read_snapshot()
    assert snapshot.valid and snapshot.stamp==10.
    np.testing.assert_allclose(snapshot.rpy_deg,[0,0,45],atol=1e-6)
    assert snapshot.frame_receive_wall==(10.,10.005)
    assert snapshot.euler_report_receive_wall==1.
    view=interface.get_history_view()
    assert view[0] is interface.get_history_view()[0]
    with pytest.raises(ValueError):view[0].quat_wxyz[0]=0.
    clock.t=10.1
    assert not interface._read_snapshot().valid
