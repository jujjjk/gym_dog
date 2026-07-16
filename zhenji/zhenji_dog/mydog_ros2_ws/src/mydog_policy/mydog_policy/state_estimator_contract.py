"""Wire layout for the estimator-owned, policy-consumed state snapshot."""

from __future__ import annotations

import numpy as np


BASE_STATE_SIZE = 10
ESTIMATOR_META_SIZE = 7
LEGACY_FRAME_SIZE = BASE_STATE_SIZE + ESTIMATOR_META_SIZE
MOTOR_VECTOR_SIZE = 12

Q_REAL_SLICE = slice(17, 29)
DQ_REAL_SLICE = slice(29, 41)
TORQUE_SLICE = slice(41, 53)
TEMP_SLICE = slice(53, 65)
ONLINE_SLICE = slice(65, 77)
ERROR_CODE_SLICE = slice(77, 89)
AGE_MS_SLICE = slice(89, 101)
SHARED_FRAME_SIZE = AGE_MS_SLICE.stop


def append_shared_motor_snapshot(base_frame, motor_snapshot) -> np.ndarray:
    """Append one coherent HTTP motor snapshot to the estimator frame."""
    base = np.asarray(base_frame, dtype=np.float32).reshape(-1)
    if base.size != LEGACY_FRAME_SIZE:
        raise ValueError(
            f"base estimator frame must contain {LEGACY_FRAME_SIZE} floats"
        )
    vectors = [
        motor_snapshot.q_real,
        motor_snapshot.dq_real,
        motor_snapshot.torque,
        motor_snapshot.temp,
        np.asarray(motor_snapshot.online, dtype=np.float32),
        np.asarray(motor_snapshot.error_code, dtype=np.float32),
        motor_snapshot.age_ms,
    ]
    return np.concatenate(
        [base]
        + [
            np.asarray(value, dtype=np.float32).reshape(MOTOR_VECTOR_SIZE)
            for value in vectors
        ]
    ).astype(np.float32)


def unpack_shared_motor_snapshot(frame) -> dict | None:
    """Return shared real-order motor vectors, or None for a legacy frame."""
    state = np.asarray(frame, dtype=np.float32).reshape(-1)
    if state.size < SHARED_FRAME_SIZE:
        return None
    return {
        "q_real": state[Q_REAL_SLICE].copy(),
        "dq_real": state[DQ_REAL_SLICE].copy(),
        "torque": state[TORQUE_SLICE].copy(),
        "temp": state[TEMP_SLICE].copy(),
        "online": state[ONLINE_SLICE] > 0.5,
        "error_code": np.rint(state[ERROR_CODE_SLICE]).astype(np.int32),
        "age_ms": state[AGE_MS_SLICE].copy(),
    }
