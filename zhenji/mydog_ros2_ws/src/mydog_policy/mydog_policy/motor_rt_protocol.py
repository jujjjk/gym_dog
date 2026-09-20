"""Fixed-size local socket protocol for the motor real-time path.

The protocol deliberately contains no JSON or schema validation.  HTTP remains
available for configuration and monitoring; policy commands and state reads use
this small binary contract over an AF_UNIX stream socket.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"LZRT"
VERSION = 1

OP_COMMAND = 1
OP_GET_STATE = 2
OP_STOP = 3

FLAG_ENABLE_FIRST = 1 << 0
FLAG_STOP_FIRST = 1 << 1
FLAG_REQUIRE_TORQUE_LIMITS = 1 << 2
FLAG_REQUIRE_VERIFIED_LIMITS = 1 << 3

RESPONSE_COMMUNICATION_OK = 1 << 0
STATUS_OK = 0
STATUS_BAD_REQUEST = 1
STATUS_SAFETY_REJECTED = 2
STATUS_IO_ERROR = 3

MOTOR_COUNT = 12

# magic, version, opcode, flags, sequence, client monotonic timestamp
REQUEST_HEADER = struct.Struct("<4sBBHIQ")
# motor id, padding, position, velocity, feed-forward torque, kp, kd
COMMAND_ENTRY = struct.Struct("<B3xfffff")

# magic, version, status, flags, request sequence, cache sequence,
# server monotonic timestamp, cache age, SPI time, total time, board A/B seq
RESPONSE_HEADER = struct.Struct("<4sBBHIIQfffII")
# id, online, fault, mode, q, dq, torque, temperature, age, padding, seq, tick
STATE_ENTRY = struct.Struct("<BBBBffffH2xII")

COMMAND_REQUEST_SIZE = REQUEST_HEADER.size + MOTOR_COUNT * COMMAND_ENTRY.size
RESPONSE_SIZE = RESPONSE_HEADER.size + MOTOR_COUNT * STATE_ENTRY.size


@dataclass(frozen=True)
class MotorCommand:
    motor_id: int
    position: float
    speed: float
    torque: float
    kp: float
    kd: float


def pack_request(opcode: int, sequence: int, timestamp_ns: int, *, flags=0, commands=()):
    payload = bytearray(
        REQUEST_HEADER.pack(
            MAGIC,
            VERSION,
            int(opcode),
            int(flags),
            int(sequence) & 0xFFFFFFFF,
            int(timestamp_ns) & 0xFFFFFFFFFFFFFFFF,
        )
    )
    if opcode == OP_COMMAND:
        commands = tuple(commands)
        if len(commands) != MOTOR_COUNT:
            raise ValueError(f"real-time command requires {MOTOR_COUNT} motors")
        for item in commands:
            payload.extend(
                COMMAND_ENTRY.pack(
                    int(item.motor_id),
                    float(item.position),
                    float(item.speed),
                    float(item.torque),
                    float(item.kp),
                    float(item.kd),
                )
            )
    return bytes(payload)


def unpack_request_header(payload: bytes):
    magic, version, opcode, flags, sequence, timestamp_ns = REQUEST_HEADER.unpack(payload)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported motor real-time protocol")
    return {
        "opcode": opcode,
        "flags": flags,
        "sequence": sequence,
        "timestamp_ns": timestamp_ns,
    }


def unpack_commands(payload: bytes):
    if len(payload) != MOTOR_COUNT * COMMAND_ENTRY.size:
        raise ValueError("invalid motor command payload size")
    result = []
    for offset in range(0, len(payload), COMMAND_ENTRY.size):
        values = COMMAND_ENTRY.unpack_from(payload, offset)
        result.append(MotorCommand(values[0], *values[1:]))
    return result


def pack_response(*, status, flags, sequence, cache_sequence, timestamp_ns,
                  cache_age_ms, spi_ms, total_ms, board_a_seq, board_b_seq,
                  states):
    states = tuple(states)
    if len(states) != MOTOR_COUNT:
        raise ValueError(f"real-time response requires {MOTOR_COUNT} motor states")
    payload = bytearray(
        RESPONSE_HEADER.pack(
            MAGIC,
            VERSION,
            int(status),
            int(flags),
            int(sequence) & 0xFFFFFFFF,
            int(cache_sequence) & 0xFFFFFFFF,
            int(timestamp_ns) & 0xFFFFFFFFFFFFFFFF,
            float(cache_age_ms),
            float(spi_ms),
            float(total_ms),
            int(board_a_seq) & 0xFFFFFFFF,
            int(board_b_seq) & 0xFFFFFFFF,
        )
    )
    for item in states:
        payload.extend(
            STATE_ENTRY.pack(
                int(item.get("can_id", 0)),
                1 if item.get("online", False) else 0,
                int(item.get("error_code", item.get("fault_bits", 0))) & 0xFF,
                int(item.get("mode_state", 0)) & 0xFF,
                float(item.get("angle", 0.0)),
                float(item.get("speed", 0.0)),
                float(item.get("torque", 0.0)),
                float(item.get("temp", 0.0)),
                min(0xFFFF, max(0, int(item.get("age_ms", 0xFFFF)))),
                int(item.get("snapshot_seq", 0)) & 0xFFFFFFFF,
                int(item.get("board_tick_ms", 0)) & 0xFFFFFFFF,
            )
        )
    return bytes(payload)


def unpack_response(payload: bytes):
    if len(payload) != RESPONSE_SIZE:
        raise ValueError(f"invalid response size {len(payload)}, expected {RESPONSE_SIZE}")
    values = RESPONSE_HEADER.unpack_from(payload, 0)
    if values[0] != MAGIC or values[1] != VERSION:
        raise ValueError("unsupported motor real-time response")
    result = {
        "status": values[2],
        "flags": values[3],
        "sequence": values[4],
        "cache_sequence": values[5],
        "timestamp_ns": values[6],
        "cache_age_ms": values[7],
        "spi_ms": values[8],
        "total_ms": values[9],
        "board_a_seq": values[10],
        "board_b_seq": values[11],
        "states": [],
    }
    offset = RESPONSE_HEADER.size
    for _ in range(MOTOR_COUNT):
        item = STATE_ENTRY.unpack_from(payload, offset)
        offset += STATE_ENTRY.size
        result["states"].append({
            "can_id": item[0],
            "online": bool(item[1]),
            "error_code": item[2],
            "mode_state": item[3],
            "angle": item[4],
            "speed": item[5],
            "torque": item[6],
            "temp": item[7],
            "age_ms": item[8],
            "snapshot_seq": item[9],
            "board_tick_ms": item[10],
        })
    return result
