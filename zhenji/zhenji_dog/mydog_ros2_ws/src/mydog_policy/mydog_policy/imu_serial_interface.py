#!/usr/bin/env python3
import time
import threading
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from YbImuLib import YbImuSerial


@dataclass
class ImuSnapshot:
    """
    IMU 最新数据快照。

    gyro_rad_s:
        陀螺仪角速度，单位 rad/s
    acc_g:
        加速度计数据，单位 g
    quat_wxyz:
        四元数，顺序为 [w, x, y, z]
    rpy_deg:
        欧拉角，单位 degree
    projected_gravity:
        重力方向在机器人 base 坐标系下的投影
        IsaacLab 平地策略水平静止时通常应接近 [0, 0, -1]
    """
    stamp: float
    gyro_rad_s: np.ndarray
    acc_g: np.ndarray
    mag_uT: np.ndarray
    quat_wxyz: np.ndarray
    rpy_deg: np.ndarray
    projected_gravity: np.ndarray
    valid: bool
    backend_alive: bool = False
    backend_error: str = ""


class ImuSerialInterface:
    """
    纯 Python IMU 串口接口。

    注意：
    1. 这个类不依赖 ROS2。
    2. 同一时间只能有一个程序打开 /dev/myimu。
    3. 后面 policy_node 或 imu_ros2_node 都可以调用这个接口。
    """

    def __init__(
        self,
        port: str = "/dev/myimu",
        read_hz: float = 100.0,
        debug: bool = False,
    ):
        self.port = port
        self.read_hz = read_hz
        self.debug = debug

        self.imu: Optional[YbImuSerial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self._vendor_threads: list[threading.Thread] = []
        self._backend_failed = False
        self._backend_error = ""
        self._previous_excepthook = None
        self._process_lock_file = None
        self._last_backend_error_print = 0.0

        self.latest = ImuSnapshot(
            stamp=0.0,
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            acc_g=np.zeros(3, dtype=np.float32),
            mag_uT=np.zeros(3, dtype=np.float32),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rpy_deg=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            valid=False,
            backend_alive=False,
        )

        # 如果你的 IMU 安装方向已经和 IsaacLab base_link 一致，这里保持单位阵。
        # IsaacLab / base_link 通常理解为：
        # +X 前方，+Y 左方，+Z 上方。
        self.R_BASE_IMU = np.eye(3, dtype=np.float32)
    

    def wait_until_ready(self, timeout: float = 3.0) -> bool:

        start_time = time.time()

        while time.time() - start_time < timeout:
            s = self.get_latest()
            if s.valid:
                return True
            time.sleep(0.01)

        return False

    def start(self):
        """打开串口并启动后台读取线程。"""
        if self.running:
            return

        self._acquire_process_lock()
        self._install_thread_exception_watchdog()
        before = {thread.ident for thread in threading.enumerate()}
        try:
            self.imu = YbImuSerial(self.port, debug=self.debug)

            # 厂家串口模式需要后台线程解析数据帧。
            self.imu.create_receive_threading()
            time.sleep(0.05)
            created = [
                thread
                for thread in threading.enumerate()
                if thread.ident not in before
            ]
            receive_threads = [
                thread
                for thread in created
                if "receive" in thread.name.lower()
                or "serial" in thread.name.lower()
            ]
            self._vendor_threads = receive_threads or created
            if not self._vendor_threads:
                raise RuntimeError(
                    "YbImuLib did not create a detectable serial receive thread"
                )

            self.running = True
            self.thread = threading.Thread(
                target=self._loop,
                name="mydog_imu_snapshot",
                daemon=True,
            )
            self.thread.start()
        except Exception:
            self._release_process_lock()
            self._restore_thread_exception_watchdog()
            raise

        print(f"[IMU] started on {self.port}")

    def stop(self):
        """停止后台线程。"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.imu is not None:
            for name in ("close", "stop", "disconnect"):
                callback = getattr(self.imu, name, None)
                if callable(callback):
                    try:
                        callback()
                    except Exception:
                        pass
                    break
        self._restore_thread_exception_watchdog()
        self._release_process_lock()
        print("[IMU] stopped")

    def get_latest(self) -> ImuSnapshot:
        """获取最新 IMU 快照。神经网络节点后面就调用这个函数。"""
        backend_alive = self._backend_is_alive()
        with self.lock:
            return ImuSnapshot(
                stamp=self.latest.stamp,
                gyro_rad_s=self.latest.gyro_rad_s.copy(),
                acc_g=self.latest.acc_g.copy(),
                mag_uT=self.latest.mag_uT.copy(),
                quat_wxyz=self.latest.quat_wxyz.copy(),
                rpy_deg=self.latest.rpy_deg.copy(),
                projected_gravity=self.latest.projected_gravity.copy(),
                valid=bool(self.latest.valid and backend_alive),
                backend_alive=backend_alive,
                backend_error=self._backend_error,
            )

    def _acquire_process_lock(self) -> None:
        """Ensure only one project process owns the IMU serial port."""
        try:
            import fcntl
        except ImportError:
            return

        safe_port = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.port)
        path = os.path.join("/tmp", f"mydog_imu{safe_port}.lock")
        handle = open(path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"IMU port {self.port} is already owned by another mydog process"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} port={self.port}\n")
        handle.flush()
        self._process_lock_file = handle

    def _release_process_lock(self) -> None:
        handle = self._process_lock_file
        self._process_lock_file = None
        if handle is None:
            return
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    def _install_thread_exception_watchdog(self) -> None:
        self._previous_excepthook = threading.excepthook

        def hook(args):
            name = str(getattr(args.thread, "name", ""))
            if (
                args.thread in self._vendor_threads
                or "serial_receive" in name.lower()
                or "task_serial" in name.lower()
            ):
                self._backend_failed = True
                self._backend_error = (
                    f"{args.exc_type.__name__}: {args.exc_value}"
                )
                with self.lock:
                    self.latest.valid = False
                    self.latest.backend_alive = False
                    self.latest.backend_error = self._backend_error
            previous = self._previous_excepthook
            if previous is not None:
                previous(args)

        threading.excepthook = hook

    def _restore_thread_exception_watchdog(self) -> None:
        if self._previous_excepthook is not None:
            threading.excepthook = self._previous_excepthook
            self._previous_excepthook = None

    def _backend_is_alive(self) -> bool:
        if self._backend_failed or not self.running:
            return False
        return bool(
            self._vendor_threads
            and all(thread.is_alive() for thread in self._vendor_threads)
        )

    def get_policy_imu_obs(self):
        """
        直接返回 IsaacLab 36维策略需要的 IMU 部分。

        对应：
        obs[3:6] = base_ang_vel
        obs[6:9] = projected_gravity
        """
        s = self.get_latest()
        return s.gyro_rad_s, s.projected_gravity, s.valid

    def _loop(self):
        period = 1.0 / self.read_hz

        # 给厂家后台解析线程一点启动时间
        time.sleep(0.1)

        while self.running:
            t0 = time.time()

            try:
                if not self._backend_is_alive():
                    raise RuntimeError(
                        self._backend_error
                        or "YbImuLib serial receive thread stopped"
                    )
                snapshot = self._read_snapshot()
                with self.lock:
                    self.latest = snapshot
            except Exception as e:
                with self.lock:
                    self.latest.valid = False
                    self.latest.backend_alive = False
                    self.latest.backend_error = str(e)
                now = time.monotonic()
                if now - self._last_backend_error_print >= 1.0:
                    self._last_backend_error_print = now
                    print(f"[IMU] read error: {e}")

            dt = time.time() - t0
            sleep_time = max(0.0, period - dt)
            time.sleep(sleep_time)

    def _read_snapshot(self) -> ImuSnapshot:
        if self.imu is None:
            raise RuntimeError("IMU is not started.")

        ax, ay, az = self.imu.get_accelerometer_data()
        gx, gy, gz = self.imu.get_gyroscope_data()
        mx, my, mz = self.imu.get_magnetometer_data()

        # 厂家接口返回顺序是 qw, qx, qy, qz
        qw, qx, qy, qz = self.imu.get_imu_quaternion_data()

        # ToAngle=True 返回角度 degree
        roll, pitch, yaw = self.imu.get_imu_attitude_data(ToAngle=True)

        gyro_imu = np.array([gx, gy, gz], dtype=np.float32)
        acc_imu = np.array([ax, ay, az], dtype=np.float32)
        mag_imu = np.array([mx, my, mz], dtype=np.float32)
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
        rpy_deg = np.array([roll, pitch, yaw], dtype=np.float32)

        # 先由四元数计算 IMU 坐标系下的 projected_gravity
        projected_g_imu = self._calc_projected_gravity_from_quat(qw, qx, qy, qz)

        # 再转到机器人 base 坐标系
        gyro_base = self.R_BASE_IMU @ gyro_imu
        projected_g_base = self.R_BASE_IMU @ projected_g_imu

        return ImuSnapshot(
            stamp=time.time(),
            gyro_rad_s=gyro_base.astype(np.float32),
            acc_g=acc_imu,
            mag_uT=mag_imu,
            quat_wxyz=quat_wxyz,
            rpy_deg=rpy_deg,
            projected_gravity=projected_g_base.astype(np.float32),
            valid=True,
            backend_alive=True,
        )

    @staticmethod
    def _quat_to_rotmat(qw, qx, qy, qz):
        """
        四元数转旋转矩阵。

        这里默认四元数表示 body/IMU -> world 的姿态。
        projected_gravity 使用 R.T @ [0, 0, -1]。
        """
        w, x, y, z = qw, qx, qy, qz

        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-8:
            return np.eye(3, dtype=np.float32)

        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
            [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

        return R

    def _calc_projected_gravity_from_quat(self, qw, qx, qy, qz):
        """
        计算 projected_gravity。

        IsaacLab 中常用含义：
        世界坐标系重力方向 [0, 0, -1] 投影到机身坐标系下。

        水平静止时应该接近：
        [0, 0, -1]
        """
        R = self._quat_to_rotmat(qw, qx, qy, qz)

        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # 如果四元数是 body -> world，则 world 向量转 body 用 R.T
        projected_g_imu = R.T @ g_world

        return projected_g_imu.astype(np.float32)


if __name__ == "__main__":
    imu = ImuSerialInterface(port="/dev/myimu", read_hz=100.0)
    imu.start()

    if not imu.wait_until_ready(timeout=3.0):
        print("[IMU] not ready, please check serial connection.")
        imu.stop()
        exit(1)

    try:
        while True:
            s = imu.get_latest()
            print(
                f"valid={s.valid} | "
                f"gyro={s.gyro_rad_s} | "
                f"rpy_deg={s.rpy_deg} | "
                f"proj_g={s.projected_gravity}"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        imu.stop()
