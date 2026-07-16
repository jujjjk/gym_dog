#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
import threading

import numpy as np
import requests


@dataclass
class MotorSnapshot:
    stamp: float
    q_real: np.ndarray       # 12 个真机角度，单位 rad
    dq_real: np.ndarray      # 12 个真机速度，单位 rad/s
    torque: np.ndarray
    temp: np.ndarray
    online: np.ndarray
    error_code: np.ndarray
    age_ms: np.ndarray
    last_update_ts: np.ndarray
    mode_state: np.ndarray
    snapshot_seq: np.ndarray
    board_tick_ms: np.ndarray
    cache_age_ms: float
    poll_dt_ms: float
    valid: bool
    raw: dict


class MotorStateHttpInterface:
    """
    从你现有 FastAPI 读取 12 个电机状态。

    当前使用接口：
        /api/state?motor_id=17
        /api/state?motor_id=18
        ...

    注意：
        17 是十进制，等于 0x11。
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:8000",
        timeout=0.08,
        stale_recheck_ms=300.0,
        enable_stale_recheck=True,
        async_poll=False,
        poll_hz=50.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stale_recheck_ms = float(stale_recheck_ms)
        self.enable_stale_recheck = bool(enable_stale_recheck)
        self.async_poll = bool(async_poll)
        self.poll_hz = float(poll_hz)
        self.session = requests.Session()
        self._cache_lock = threading.Lock()
        self._latest_snapshot: MotorSnapshot | None = None
        self._latest_error: Exception | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_poll_stamp_perf: float | None = None

        # 真机电机顺序：FR, FL, RL, RR
        # 这里用十六进制写，Python 发送请求时会自动变成对应十进制整数。
        self.real_motor_ids = [
            0x11, 0x12, 0x13,   # 右前 FR
            0x21, 0x22, 0x23,   # 左前 FL
            0x31, 0x32, 0x33,   # 左后 RL
            0x41, 0x42, 0x43,   # 右后 RR
        ]

        self.real_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",

            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",

            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",

            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]

        if self.async_poll:
            self.start()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="motor-state-http", daemon=True)
        self._thread.start()

    def _poll_loop(self):
        period = 1.0 / max(self.poll_hz, 1.0)
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                snap = self._fetch_latest_sync()
                with self._cache_lock:
                    self._latest_snapshot = snap
                    self._latest_error = None
            except Exception as exc:
                with self._cache_lock:
                    self._latest_error = exc
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, period - elapsed))

    def get_one_motor_state(self, motor_id: int) -> dict:
        """
        读取单个电机状态。

        例如：
            motor_id = 0x11
        实际请求：
            /api/state?motor_id=17
        """
        url = f"{self.base_url}/api/state"
        r = self.session.get(
            url,
            params={"motor_id": int(motor_id)},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_all_motor_states(self) -> dict:
        """
        Read all motor states with one HTTP request.

        The FastAPI /api/state endpoint refreshes both motor boards on each
        request. Calling it once per motor makes one policy frame do 12 full
        board refreshes, which is too slow for closed-loop walking.
        """
        url = f"{self.base_url}/api/state"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected /api/state response type: {type(data)!r}")
        return data

    @staticmethod
    def _lookup_motor_state(all_states: dict, motor_id: int) -> dict:
        keys = (
            hex(motor_id),
            f"0x{motor_id:02X}",
            str(int(motor_id)),
        )

        for key in keys:
            item = all_states.get(key)
            if isinstance(item, dict):
                return item

        for item in all_states.values():
            if isinstance(item, dict) and int(item.get("can_id", -1)) == int(motor_id):
                return item

        raise KeyError(f"motor_id 0x{motor_id:02X} not found in /api/state response")

    def _fetch_latest_sync(self) -> MotorSnapshot:
        poll_stamp_perf = time.perf_counter()
        if self._last_poll_stamp_perf is None:
            poll_dt_ms = 0.0
        else:
            poll_dt_ms = max(
                0.0,
                (poll_stamp_perf - self._last_poll_stamp_perf) * 1000.0,
            )
        self._last_poll_stamp_perf = poll_stamp_perf

        q = []
        dq = []
        torque = []
        temp = []
        online = []
        error_code = []
        age_ms = []
        last_update_ts = []
        mode_state = []
        snapshot_seq = []
        board_tick_ms = []
        raw = {}
        all_states = self.get_all_motor_states()
        meta = all_states.get("__meta__", {}) if isinstance(all_states, dict) else {}

        for mid in self.real_motor_ids:
            item = self._lookup_motor_state(all_states, mid)
            age = float(item.get("age_ms", 999999.0))

            # The web monitor reads /api/state?motor_id=<id>.  Keep the policy
            # path consistent when a bulk snapshot returns a stale channel.
            if self.enable_stale_recheck and age > self.stale_recheck_ms:
                try:
                    refreshed = self.get_one_motor_state(mid)
                    refreshed_age = float(refreshed.get("age_ms", 999999.0))
                    if refreshed_age <= age:
                        item = refreshed
                        age = refreshed_age
                except Exception:
                    pass

            raw[hex(mid)] = item

            q.append(float(item.get("angle", 0.0)))
            dq.append(float(item.get("speed", 0.0)))
            torque.append(float(item.get("torque", 0.0)))
            temp.append(float(item.get("temp", 0.0)))
            online.append(bool(item.get("online", False)))
            error_code.append(int(item.get("error_code", 0)))
            age_ms.append(age)
            last_update_ts.append(float(item.get("last_update_ts", 0.0)))
            mode_state.append(int(item.get("mode_state", 0)))
            snapshot_seq.append(int(item.get("snapshot_seq", 0)))
            board_tick_ms.append(int(item.get("board_tick_ms", 0)))
            if not meta:
                meta = {k: item.get(k) for k in (
                    "state_cache_age_ms",
                    "board_a_seq",
                    "board_b_seq",
                    "state_refresh_dt_ms",
                    "communication_ok",
                    "warning",
                ) if k in item}

        q = np.asarray(q, dtype=np.float32)
        dq = np.asarray(dq, dtype=np.float32)
        torque = np.asarray(torque, dtype=np.float32)
        temp = np.asarray(temp, dtype=np.float32)
        online = np.asarray(online, dtype=bool)
        error_code = np.asarray(error_code, dtype=np.int32)
        age_ms = np.asarray(age_ms, dtype=np.float32)
        last_update_ts = np.asarray(last_update_ts, dtype=np.float64)
        mode_state = np.asarray(mode_state, dtype=np.int32)
        snapshot_seq = np.asarray(snapshot_seq, dtype=np.int32)
        board_tick_ms = np.asarray(board_tick_ms, dtype=np.int64)

        # 当前先只判断数值是否有效。
        # 注意：真正闭环控制时，还应该要求 online=True 且 age_ms 足够小。
        cache_age_ms = float(meta.get("state_cache_age_ms", 0.0) or 0.0)
        server_comm_ok = bool(meta.get("communication_ok", True))
        valid = (
            np.all(np.isfinite(q))
            and np.all(np.isfinite(dq))
            and np.all(np.isfinite(age_ms))
            and server_comm_ok
        )

        return MotorSnapshot(
            stamp=time.time(),
            q_real=q,
            dq_real=dq,
            torque=torque,
            temp=temp,
            online=online,
            error_code=error_code,
            age_ms=age_ms,
            last_update_ts=last_update_ts,
            mode_state=mode_state,
            snapshot_seq=snapshot_seq,
            board_tick_ms=board_tick_ms,
            cache_age_ms=cache_age_ms,
            poll_dt_ms=poll_dt_ms,
            valid=valid,
            raw=raw,
        )

    def get_latest(self) -> MotorSnapshot:
        if not self.async_poll:
            return self._fetch_latest_sync()

        with self._cache_lock:
            snap = self._latest_snapshot
            err = self._latest_error

        if snap is None:
            snap = self._fetch_latest_sync()
            with self._cache_lock:
                self._latest_snapshot = snap
                self._latest_error = None
            return snap

        age_ms = (time.time() - snap.stamp) * 1000.0
        if snap.valid and age_ms <= self.stale_recheck_ms:
            return snap

        if err is not None or age_ms > self.stale_recheck_ms:
            return MotorSnapshot(
                stamp=snap.stamp,
                q_real=snap.q_real.copy(),
                dq_real=snap.dq_real.copy(),
                torque=snap.torque.copy(),
                temp=snap.temp.copy(),
                online=snap.online.copy(),
                error_code=snap.error_code.copy(),
                age_ms=snap.age_ms.copy(),
                last_update_ts=snap.last_update_ts.copy(),
                mode_state=snap.mode_state.copy(),
                snapshot_seq=snap.snapshot_seq.copy(),
                board_tick_ms=snap.board_tick_ms.copy(),
                cache_age_ms=age_ms,
                poll_dt_ms=snap.poll_dt_ms,
                valid=False,
                raw=dict(snap.raw),
            )

        return snap

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.session.close()
        except Exception:
            pass

    def print_debug(self, snapshot: MotorSnapshot):
        print("=" * 90)
        print("valid:", snapshot.valid)
        print("real motor order = FR, FL, RL, RR")

        for i, mid in enumerate(self.real_motor_ids):
            print(
                f"real[{i:02d}] "
                f"motor_id=0x{mid:02X}({int(mid):02d}) "
                f"{self.real_joint_names[i]:16s} | "
                f"angle={snapshot.q_real[i]:+.5f} rad | "
                f"speed={snapshot.dq_real[i]:+.5f} rad/s | "
                f"torque={snapshot.torque[i]:+.3f} | "
                f"temp={snapshot.temp[i]:.1f} | "
                f"online={snapshot.online[i]} | "
                f"err={snapshot.error_code[i]} | "
                f"age_ms={snapshot.age_ms[i]:.0f} | "
                f"seq={snapshot.snapshot_seq[i]} | "
                f"tick={snapshot.board_tick_ms[i]}"
            )


if __name__ == "__main__":
    # 如果这个脚本和 FastAPI 都在 Jetson NX 上运行，用 127.0.0.1
    # 如果你从另一台电脑访问 NX，才用 http://172.19.19.145:8000
    m = MotorStateHttpInterface(base_url="http://127.0.0.1:8000")

    while True:
        s = m.get_latest()
        m.print_debug(s)
        time.sleep(0.5)
