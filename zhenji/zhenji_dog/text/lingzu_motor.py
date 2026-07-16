# -*- coding: utf-8 -*-
import atexit
import os
import struct
import threading
import time
from dataclasses import dataclass, asdict

import spidev


MASTER_ID = 0xFD

CMD_MOTION_CTRL = 0x01
CMD_REPLY       = 0x02
CMD_ENABLE      = 0x03
CMD_STOP        = 0x04
CMD_SET_PARAM   = 0x12
CMD_GET_PARAM   = 0x11
CMD_SET_ZERO    = 0x06

IDX_RUN_MODE      = 0x7005
IDX_IQ_REF        = 0x7006
IDX_SPEED_REF     = 0x700A
IDX_TORQUE_LIMIT  = 0x700B
IDX_POS_REF       = 0x7016
IDX_CSP_V_LIMIT   = 0x7017
IDX_CUR_LIMIT     = 0x7018
IDX_SPEED_ACCEL   = 0x7022
IDX_PP_SPEED      = 0x7024
IDX_PP_ACCEL      = 0x7025

MODE_MOTION   = 0
MODE_PP_POS   = 1
MODE_SPEED    = 2
MODE_CURRENT  = 3
MODE_CSP_POS  = 5

P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -44.0, 44.0
T_MIN, T_MAX = -17.0, 17.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

SPI_FRAME_LEN     = 96
SPI_CMD_MAGIC     = 0xFA
SPI_OP_NOP        = 0x00
SPI_OP_SEND_CAN   = 0x10
SPI_OP_SEND_CAN_BATCH = 0x12
SPI_OP_READ_MOTOR_PARAM = 0x13

SPI_BATCH_HEADER_LEN = 4
SPI_BATCH_ENTRY_LEN = 13
SPI_BATCH_MAX_ENTRIES = (SPI_FRAME_LEN - SPI_BATCH_HEADER_LEN) // SPI_BATCH_ENTRY_LEN

SNAPSHOT_MAGIC    = 0x5A
PARAM_RESPONSE_MAGIC = 0x5B
BOARD_A_TAG       = 0xA1
BOARD_B_TAG       = 0xB1
SNAPSHOT_CAP_BATCH_CAN = 1 << 0
SNAPSHOT_CAP_PARAM_READBACK = 1 << 1

DEFAULT_SPI_MAX_SPEED_HZ = int(os.environ.get("LINGZU_SPI_MAX_SPEED_HZ", "4000000"))
DEFAULT_SPI_PERSISTENT_OPEN = os.environ.get("LINGZU_SPI_PERSISTENT_OPEN", "1").lower() not in ("0", "false", "no")
DEBUG_TX = os.environ.get("LINGZU_DEBUG_TX", "0").lower() in ("1", "true", "yes")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def float_to_uint(x, x_min, x_max, bits):
    x = clamp(x, x_min, x_max)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def uint_to_float(x, x_min, x_max, bits):
    span = x_max - x_min
    return x * span / ((1 << bits) - 1) + x_min


def build_ext_id(cmd_type: int, data_info_16: int, target_id: int) -> int:
    return ((cmd_type & 0x1F) << 24) | ((data_info_16 & 0xFFFF) << 8) | (target_id & 0xFF)


def motor_to_target_and_port(motor_id: int):
    group = (motor_id >> 4) & 0x0F

    if group == 0x1:
        return (0, 0), 1   # A板 FDCAN1
    if group == 0x2:
        return (0, 0), 2   # A板 FDCAN2
    if group == 0x3:
        return (1, 0), 1   # B板 FDCAN1
    if group == 0x4:
        return (1, 0), 2   # B板 FDCAN2

    raise ValueError(f"Unknown motor id group: 0x{motor_id:02X}")


def build_spi_nop_frame() -> bytes:
    frame = bytearray(SPI_FRAME_LEN)
    frame[0] = SPI_CMD_MAGIC
    frame[1] = SPI_OP_NOP
    return bytes(frame)


def build_spi_can_frame(can_port: int, ext_id: int, data8: bytes) -> bytes:
    if len(data8) != 8:
        raise ValueError("CAN data must be 8 bytes")

    frame = bytearray(SPI_FRAME_LEN)
    frame[0] = SPI_CMD_MAGIC
    frame[1] = SPI_OP_SEND_CAN
    frame[2] = can_port & 0xFF
    frame[3] = 0x00
    frame[4:8] = int(ext_id).to_bytes(4, "little")
    frame[8:16] = bytes(data8)
    return bytes(frame)


def build_spi_param_read_frame(motor_id: int, index: int) -> bytes:
    frame = bytearray(SPI_FRAME_LEN)
    frame[0] = SPI_CMD_MAGIC
    frame[1] = SPI_OP_READ_MOTOR_PARAM
    frame[2] = int(motor_id) & 0xFF
    frame[4] = int(index) & 0xFF
    frame[5] = (int(index) >> 8) & 0xFF
    return bytes(frame)


def build_spi_can_batch_frame(entries) -> bytes:
    """Pack up to six CAN frames for one STM32 into one SPI transaction."""
    entries = list(entries)
    if not 1 <= len(entries) <= SPI_BATCH_MAX_ENTRIES:
        raise ValueError(f"batch must contain 1..{SPI_BATCH_MAX_ENTRIES} CAN frames")

    frame = bytearray(SPI_FRAME_LEN)
    frame[0] = SPI_CMD_MAGIC
    frame[1] = SPI_OP_SEND_CAN_BATCH
    frame[2] = len(entries)
    offset = SPI_BATCH_HEADER_LEN
    for can_port, ext_id, data8 in entries:
        if len(data8) != 8:
            raise ValueError("CAN data must be 8 bytes")
        frame[offset] = int(can_port) & 0xFF
        frame[offset + 1:offset + 5] = int(ext_id).to_bytes(4, "little")
        frame[offset + 5:offset + 13] = bytes(data8)
        offset += SPI_BATCH_ENTRY_LEN
    return bytes(frame)


@dataclass
class MotorState:
    can_id: int
    angle: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    temp: float = 0.0
    error_code: int = 0
    online: bool = False
    last_update_ts: float = 0.0
    mode_state: int = 0
    age_ms: int = 0
    board_tag: int = 0
    snapshot_seq: int = 0
    board_tick_ms: int = 0
    board_capabilities: int = 0


class SPITransport:
    def __init__(
        self,
        targets=((0, 0), (1, 0)),
        max_speed_hz=None,
        frame_len=SPI_FRAME_LEN,
        persistent_open=None,
    ):
        self.targets = targets
        self.max_speed_hz = int(DEFAULT_SPI_MAX_SPEED_HZ if max_speed_hz is None else max_speed_hz)
        self.frame_len = frame_len
        self.persistent_open = DEFAULT_SPI_PERSISTENT_OPEN if persistent_open is None else bool(persistent_open)
        self.lock = threading.Lock()
        self.spi_devs = {}

        if self.persistent_open:
            for bus, dev in self.targets:
                try:
                    self._open_dev(bus, dev)
                except Exception as exc:
                    print(f"[SPI] persistent open deferred bus={bus} dev={dev}: {exc}")
        print(
            f"[SPI] max_speed_hz={self.max_speed_hz} "
            f"persistent_open={self.persistent_open} targets={tuple(self.targets)}"
        )
        atexit.register(self.close)

    def _configure_dev(self, spi):
        spi.mode = 0
        spi.bits_per_word = 8
        spi.max_speed_hz = self.max_speed_hz

    def _open_dev(self, bus: int, dev: int):
        key = (int(bus), int(dev))
        old = self.spi_devs.get(key)
        if old is not None:
            return old

        spi = spidev.SpiDev()
        spi.open(key[0], key[1])
        self._configure_dev(spi)
        self.spi_devs[key] = spi
        return spi

    def _close_dev(self, bus: int, dev: int):
        key = (int(bus), int(dev))
        spi = self.spi_devs.pop(key, None)
        if spi is not None:
            try:
                spi.close()
            except Exception:
                pass

    def close(self):
        with self.lock:
            for bus, dev in list(self.spi_devs):
                self._close_dev(bus, dev)

    def _xfer_to_one(self, bus: int, dev: int, payload: bytes) -> bytes:
        if len(payload) != self.frame_len:
            raise ValueError(f"payload must be {self.frame_len} bytes")

        if not self.persistent_open:
            spi = spidev.SpiDev()
            try:
                spi.open(bus, dev)
                self._configure_dev(spi)
                rx = spi.xfer2(list(payload))
                return bytes(rx)
            finally:
                spi.close()

        try:
            spi = self._open_dev(bus, dev)
            rx = spi.xfer2(list(payload))
            return bytes(rx)
        except Exception:
            self._close_dev(bus, dev)
            spi = self._open_dev(bus, dev)
            rx = spi.xfer2(list(payload))
            return bytes(rx)

    def exchange_frame(self, target, payload: bytes) -> bytes:
        bus, dev = target
        with self.lock:
            return self._xfer_to_one(bus, dev, payload)


class LingZuMotorController:
    registry: dict[int, "LingZuMotorController"] = {}

    def __init__(self, motor_id=0x11, transport=None, debug_tx=None):
        self.motor_id = motor_id
        self.state = MotorState(can_id=motor_id)
        self.lock = threading.Lock()
        self.transport = transport if transport is not None else SPITransport()
        self.debug_tx = DEBUG_TX if debug_tx is None else bool(debug_tx)
        LingZuMotorController.registry[motor_id] = self

    def open(self):
        pass

    def close(self):
        pass

    @staticmethod
    def _parse_snapshot_frame(frame: bytes):
        if len(frame) != SPI_FRAME_LEN:
            return None
        if frame[0] != SNAPSHOT_MAGIC:
            return None
        if frame[1] not in (BOARD_A_TAG, BOARD_B_TAG):
            return None

        seq = struct.unpack_from("<H", frame, 2)[0]
        board_tick_ms = struct.unpack_from("<I", frame, 4)[0]
        capabilities = struct.unpack_from("<I", frame, 92)[0]

        motors = []
        offset = 8
        for _ in range(6):
            can_id, online, fault_bits, mode_state, pos_mrad, vel_crad, tq_cNm, temp_dC, age_ms = \
                struct.unpack_from("<BBBBhhhhH", frame, offset)
            offset += 14

            motors.append({
                "can_id": can_id,
                "online": bool(online),
                "fault_bits": int(fault_bits),
                "mode_state": int(mode_state),
                "angle": pos_mrad / 1000.0,
                "speed": vel_crad / 100.0,
                "torque": tq_cNm / 100.0,
                "temp": temp_dC / 10.0,
                "age_ms": int(age_ms),
            })

        return {
            "board_tag": frame[1],
            "seq": seq,
            "board_tick_ms": board_tick_ms,
            "capabilities": capabilities,
            "motors": motors,
        }

    @staticmethod
    def _parse_param_response_frame(frame: bytes):
        if len(frame) != SPI_FRAME_LEN or frame[0] != PARAM_RESPONSE_MAGIC:
            return None
        if frame[1] not in (BOARD_A_TAG, BOARD_B_TAG):
            return None
        return {
            "board_tag": int(frame[1]),
            "motor_id": int(frame[2]),
            "status": int(frame[3]),
            "index": int.from_bytes(frame[4:6], "little"),
            "value": float(struct.unpack_from("<f", frame, 8)[0]),
            "response_tick_ms": int.from_bytes(frame[12:16], "little"),
        }

    @classmethod
    def _apply_snapshot_to_registry(cls, snap: dict | None):
        if not snap:
            return

        now_ts = time.time()
        for item in snap["motors"]:
            ctrl = cls.registry.get(item["can_id"])
            if ctrl is None:
                continue

            with ctrl.lock:
                ctrl.state.angle = item["angle"]
                ctrl.state.speed = item["speed"]
                ctrl.state.torque = item["torque"]
                ctrl.state.temp = item["temp"]
                ctrl.state.error_code = item["fault_bits"]
                ctrl.state.online = item["online"]
                ctrl.state.last_update_ts = now_ts
                ctrl.state.mode_state = item["mode_state"]
                ctrl.state.age_ms = item["age_ms"]
                ctrl.state.board_tag = snap["board_tag"]
                ctrl.state.snapshot_seq = snap["seq"]
                ctrl.state.board_tick_ms = snap["board_tick_ms"]
                ctrl.state.board_capabilities = snap["capabilities"]
    def refresh_board_snapshot(self, read_count=1, inter_read_delay_s=0.0):
        target, _ = motor_to_target_and_port(self.motor_id)

        rx = None
        for index in range(max(1, int(read_count))):
            rx = self.transport.exchange_frame(target, build_spi_nop_frame())
            if index + 1 < read_count and inter_read_delay_s > 0.0:
                time.sleep(inter_read_delay_s)

        snap = self._parse_snapshot_frame(rx)

        self._apply_snapshot_to_registry(snap)
        return snap

    def _send(self, ext_id: int, data: bytes):
        if len(data) != 8:
            raise ValueError("CAN data must be 8 bytes")

        target, can_port = motor_to_target_and_port(self.motor_id)
        payload = build_spi_can_frame(can_port, ext_id, data)

        if self.debug_tx:
            print(
                "[TX] TARGET:", target,
                "| CAN_PORT:", can_port,
                "| EXT_ID: 0x%08X" % ext_id,
                "| DATA:", " ".join(f"{b:02X}" for b in data)
            )

        rx = self.transport.exchange_frame(target, payload)
        snap = self._parse_snapshot_frame(rx)
        self._apply_snapshot_to_registry(snap)
    def enable(self):
        ext_id = build_ext_id(CMD_ENABLE, MASTER_ID, self.motor_id)
        self._send(ext_id, bytes([0] * 8))

    def stop(self, clear_error=False):
        data = bytearray(8)
        data[0] = 1 if clear_error else 0
        ext_id = build_ext_id(CMD_STOP, MASTER_ID, self.motor_id)
        self._send(ext_id, bytes(data))
    def set_zero(self):
        ext_id = build_ext_id(CMD_SET_ZERO, MASTER_ID, self.motor_id)
        data = bytes([1, 0, 0, 0, 0, 0, 0, 0])
        self._send(ext_id, data)

    def get_param(self, index: int):
        data = bytearray(8)
        data[0] = index & 0xFF
        data[1] = (index >> 8) & 0xFF
        ext_id = build_ext_id(CMD_GET_PARAM, MASTER_ID, self.motor_id)
        self._send(ext_id, bytes(data))

    def read_param_f32(self, index: int, timeout: float = 0.35) -> float:
        """Read a float parameter through STM32's verified response channel."""
        if not (
            self.state.board_capabilities & SNAPSHOT_CAP_PARAM_READBACK
        ):
            self.refresh_board_snapshot()
        if not (
            self.state.board_capabilities & SNAPSHOT_CAP_PARAM_READBACK
        ):
            raise RuntimeError(
                "STM32 firmware lacks parameter-readback capability; flash "
                "the updated DOG_0.1A firmware on both boards"
            )
        target, _ = motor_to_target_and_port(self.motor_id)
        request = build_spi_param_read_frame(self.motor_id, index)
        deadline = time.monotonic() + max(0.05, float(timeout))

        # Hold the shared SPI lock for request + polls so the state-refresh
        # thread cannot consume the one-shot 0x5B response frame.
        with self.transport.lock:
            rx = self.transport._xfer_to_one(target[0], target[1], request)
            snap = self._parse_snapshot_frame(rx)
            self._apply_snapshot_to_registry(snap)
            while time.monotonic() < deadline:
                time.sleep(0.005)
                rx = self.transport._xfer_to_one(
                    target[0],
                    target[1],
                    build_spi_nop_frame(),
                )
                response = self._parse_param_response_frame(rx)
                if response is not None:
                    if (
                        response["motor_id"] != int(self.motor_id)
                        or response["index"] != int(index)
                    ):
                        continue
                    if response["status"] != 0:
                        raise RuntimeError(
                            "motor parameter read failed: "
                            f"motor=0x{self.motor_id:02X}, "
                            f"index=0x{index:04X}, status={response['status']}"
                        )
                    if not float("-inf") < response["value"] < float("inf"):
                        raise RuntimeError("motor parameter read returned non-finite value")
                    return float(response["value"])
                snap = self._parse_snapshot_frame(rx)
                self._apply_snapshot_to_registry(snap)
        raise TimeoutError(
            f"parameter readback timeout: motor=0x{self.motor_id:02X}, "
            f"index=0x{index:04X}"
        )

    def write_and_verify_param_f32(
        self,
        index: int,
        value: float,
        *,
        atol: float = 0.05,
    ) -> float:
        self.set_param_f32(index, value)
        time.sleep(0.01)
        actual = self.read_param_f32(index)
        if abs(float(actual) - float(value)) > float(atol):
            raise RuntimeError(
                f"parameter verify mismatch motor=0x{self.motor_id:02X} "
                f"index=0x{index:04X}: requested={float(value):.4f}, "
                f"readback={float(actual):.4f}"
            )
        return float(actual)

    def set_param_u8(self, index: int, value: int):
        data = bytearray(8)
        data[0] = index & 0xFF
        data[1] = (index >> 8) & 0xFF
        data[4] = value & 0xFF
        ext_id = build_ext_id(CMD_SET_PARAM, MASTER_ID, self.motor_id)
        self._send(ext_id, bytes(data))

    def set_param_f32(self, index: int, value: float):
        raw = struct.pack("<f", float(value))
        data = bytearray(8)
        data[0] = index & 0xFF
        data[1] = (index >> 8) & 0xFF
        data[4:8] = raw
        ext_id = build_ext_id(CMD_SET_PARAM, MASTER_ID, self.motor_id)
        self._send(ext_id, bytes(data))

    def set_mode(self, mode: int):
        self.set_param_u8(IDX_RUN_MODE, mode)

    def set_current_limit(self, amp: float):
        self.set_param_f32(IDX_CUR_LIMIT, amp)

    def set_torque_limit(self, torque_nm: float):
        """Set RS01's volatile internal total-torque limit (index 0x700B)."""
        torque_nm = float(torque_nm)
        if not 0.0 < torque_nm <= T_MAX:
            raise ValueError(f"torque limit must be in (0, {T_MAX}] Nm")
        self.set_param_f32(IDX_TORQUE_LIMIT, torque_nm)

    def set_motion_mode(self):
        self.set_mode(MODE_MOTION)

    def set_speed_mode_params(self, accel: float = 2.0):
        self.set_mode(MODE_SPEED)
        time.sleep(0.02)
        self.set_param_f32(IDX_SPEED_ACCEL, accel)

    def set_speed_target(self, speed_rad_s: float):
        self.set_param_f32(IDX_SPEED_REF, speed_rad_s)

    def set_pp_mode_params(self, speed: float = 2.0, accel: float = 1.0):
        self.set_mode(MODE_PP_POS)
        time.sleep(0.02)
        self.set_param_f32(IDX_PP_SPEED, speed)
        time.sleep(0.02)
        self.set_param_f32(IDX_PP_ACCEL, accel)

    def set_position_target(self, pos_rad: float):
        self.set_param_f32(IDX_POS_REF, pos_rad)

    def set_csp_mode_params(self, v_limit: float = 2.0):
        self.set_mode(MODE_CSP_POS)
        time.sleep(0.02)
        self.set_param_f32(IDX_CSP_V_LIMIT, v_limit)

    def set_current_mode_params(self):
        self.set_mode(MODE_CURRENT)

    def set_iq_target(self, iq_amp: float):
        self.set_param_f32(IDX_IQ_REF, iq_amp)

    def motion_control(self, torque: float, position: float, speed: float, kp: float, kd: float):
        ext_id, data = self.build_motion_control_command(
            torque=torque, position=position, speed=speed, kp=kp, kd=kd
        )
        self._send(ext_id, data)

    def build_motion_control_command(self, torque: float, position: float, speed: float, kp: float, kd: float):
        p_u = float_to_uint(position, P_MIN, P_MAX, 16)
        v_u = float_to_uint(speed,    V_MIN, V_MAX, 16)
        kp_u = float_to_uint(kp,      KP_MIN, KP_MAX, 16)
        kd_u = float_to_uint(kd,      KD_MIN, KD_MAX, 16)
        t_u = float_to_uint(torque,   T_MIN, T_MAX, 16)

        data = bytearray(8)
        data[0] = (p_u >> 8) & 0xFF
        data[1] = p_u & 0xFF
        data[2] = (v_u >> 8) & 0xFF
        data[3] = v_u & 0xFF
        data[4] = (kp_u >> 8) & 0xFF
        data[5] = kp_u & 0xFF
        data[6] = (kd_u >> 8) & 0xFF
        data[7] = kd_u & 0xFF

        ext_id = build_ext_id(CMD_MOTION_CTRL, t_u, self.motor_id)
        return ext_id, bytes(data)

    def get_state_dict(self):
        with self.lock:
            return asdict(self.state)


def send_motion_control_batch(commands) -> int:
    """Send motion commands using one SPI frame per STM32 board.

    ``commands`` contains ``(controller, torque, position, speed, kp, kd)``.
    Returns the number of SPI frames sent.
    """
    grouped = {}
    transport = None
    for ctrl, torque, position, speed, kp, kd in commands:
        if transport is None:
            transport = ctrl.transport
        elif ctrl.transport is not transport:
            raise ValueError("all batched controllers must share one SPITransport")
        target, can_port = motor_to_target_and_port(ctrl.motor_id)
        ext_id, data = ctrl.build_motion_control_command(torque, position, speed, kp, kd)
        group = grouped.setdefault(target, {"entries": [], "controllers": []})
        group["entries"].append((can_port, ext_id, data))
        group["controllers"].append(ctrl)

    frames_sent = 0
    for target, group in grouped.items():
        entries = group["entries"]
        batch_supported = any(
            ctrl.state.board_capabilities & SNAPSHOT_CAP_BATCH_CAN
            for ctrl in group["controllers"]
        )
        chunks = (
            [entries[start:start + SPI_BATCH_MAX_ENTRIES]
             for start in range(0, len(entries), SPI_BATCH_MAX_ENTRIES)]
            if batch_supported
            else [[entry] for entry in entries]
        )
        for chunk in chunks:
            payload = (
                build_spi_can_batch_frame(chunk)
                if batch_supported
                else build_spi_can_frame(*chunk[0])
            )
            rx = transport.exchange_frame(target, payload)
            snap = LingZuMotorController._parse_snapshot_frame(rx)
            LingZuMotorController._apply_snapshot_to_registry(snap)
            frames_sent += 1
    return frames_sent
