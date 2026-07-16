# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from collections import deque
import threading
import time
import os

from lingzu_motor import LingZuMotorController, SPITransport, send_motion_control_batch

app = FastAPI(title="RS04 Control API")

# ---- static dir (use Path consistently) ----
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 共享 SPI 传输层，多电机按 motor_id 区分
STATE_REFRESH_HZ = float(os.environ.get("LINGZU_STATE_REFRESH_HZ", "50"))
STATE_READ_COUNT = int(os.environ.get("LINGZU_STATE_READ_COUNT", "1"))
STATE_INTER_READ_DELAY_S = float(os.environ.get("LINGZU_STATE_INTER_READ_DELAY_S", "0"))
STATE_CACHE_STALE_MS = float(os.environ.get("LINGZU_STATE_CACHE_STALE_MS", "200"))
SPI_MAX_SPEED_HZ = int(os.environ.get("LINGZU_SPI_MAX_SPEED_HZ", "4000000"))
SPI_PERSISTENT_OPEN = os.environ.get("LINGZU_SPI_PERSISTENT_OPEN", "1").lower() not in ("0", "false", "no")
DEBUG_TX = os.environ.get("LINGZU_DEBUG_TX", "0").lower() in ("1", "true", "yes")

_shared_transport = SPITransport(
    targets=((0, 0), (1, 0)),
    max_speed_hz=SPI_MAX_SPEED_HZ,
    persistent_open=SPI_PERSISTENT_OPEN,
)

MOTOR_IDS = [
    0x11, 0x12, 0x13,
    0x21, 0x22, 0x23,
    0x31, 0x32, 0x33,
    0x41, 0x42, 0x43,
]

motors: dict[int, LingZuMotorController] = {
    mid: LingZuMotorController(motor_id=mid, transport=_shared_transport, debug_tx=DEBUG_TX)
    for mid in MOTOR_IDS
}

_state_cache_lock = threading.Lock()
_state_cache: dict[str, dict] = {}
_state_cache_stamp = 0.0
_state_refresh_dt_ms = float("inf")
_state_refresh_stop = threading.Event()
_state_refresh_thread: threading.Thread | None = None
_control_timing_lock = threading.Lock()
_control_send_stamps = deque(maxlen=200)
_configured_motion_torque_limits: dict[int, float] = {}
_configured_motion_current_limits: dict[int, float] = {}
_verified_motion_safety_limits: dict[int, dict[str, float]] = {}


def _invalidate_motion_safety_limits(reason: str) -> None:
    if (
        _configured_motion_torque_limits
        or _configured_motion_current_limits
        or _verified_motion_safety_limits
    ):
        print(f"[SAFETY_LIMITS] invalidated: {reason}")
    _configured_motion_torque_limits.clear()
    _configured_motion_current_limits.clear()
    _verified_motion_safety_limits.clear()


def _record_control_send_timing() -> dict:
    now = time.perf_counter()
    with _control_timing_lock:
        if _control_send_stamps and (now - _control_send_stamps[-1]) > 0.5:
            _control_send_stamps.clear()
        _control_send_stamps.append(now)
        stamps = tuple(_control_send_stamps)
    if len(stamps) < 2:
        return {"actual_send_hz": 0.0, "send_period_p95_ms": 0.0, "send_period_max_ms": 0.0}
    periods = [b - a for a, b in zip(stamps, stamps[1:])]
    ordered = sorted(periods)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "actual_send_hz": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
        "send_period_p95_ms": ordered[p95_index] * 1000.0,
        "send_period_max_ms": max(periods) * 1000.0,
    }


def _refresh_all_states_legacy_unused():
    # A板代表：0x11；B板代表：0x31
    motors[0x11].refresh_board_snapshot()
    motors[0x31].refresh_board_snapshot()


def _refresh_all_states():
    t0 = time.perf_counter()
    snap_a = motors[0x11].refresh_board_snapshot(STATE_READ_COUNT, STATE_INTER_READ_DELAY_S)
    snap_b = motors[0x31].refresh_board_snapshot(STATE_READ_COUNT, STATE_INTER_READ_DELAY_S)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _update_state_cache(dt_ms=dt_ms)
    return {"board_a": snap_a, "board_b": snap_b, "dt_ms": dt_ms}


def _state_meta(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    age_ms = (now - _state_cache_stamp) * 1000.0 if _state_cache_stamp > 0.0 else float("inf")
    board_a = _state_cache.get("0x11") or {}
    board_b = _state_cache.get("0x31") or {}
    communication_ok = bool(age_ms <= STATE_CACHE_STALE_MS and _state_cache)
    meta = {
        "state_cache_age_ms": age_ms,
        "board_a_seq": int(board_a.get("snapshot_seq", 0)),
        "board_b_seq": int(board_b.get("snapshot_seq", 0)),
        "state_refresh_dt_ms": _state_refresh_dt_ms,
        "communication_ok": communication_ok,
    }
    if not communication_ok:
        meta["warning"] = f"state cache stale: {age_ms:.1f}ms"
    return meta


def _update_state_cache(dt_ms: float):
    global _state_cache, _state_cache_stamp, _state_refresh_dt_ms
    cache = {}
    for mid, motor in motors.items():
        cache[hex(mid)] = motor.get_state_dict()
    if any(not bool(item.get("online", False)) for item in cache.values()):
        _invalidate_motion_safety_limits("one or more motors offline")
    with _state_cache_lock:
        _state_cache = cache
        _state_cache_stamp = time.time()
        _state_refresh_dt_ms = float(dt_ms)


def _get_state_cache() -> tuple[dict[str, dict], dict]:
    with _state_cache_lock:
        cache = {k: dict(v) for k, v in _state_cache.items()}
        meta = _state_meta()
    for item in cache.values():
        item.update(meta)
    return cache, meta


def _state_refresh_loop():
    period = 1.0 / max(STATE_REFRESH_HZ, 1.0)
    while not _state_refresh_stop.is_set():
        t0 = time.perf_counter()
        try:
            _refresh_all_states()
        except Exception as exc:
            print(f"[STATE] refresh failed: {exc}")
        elapsed = time.perf_counter() - t0
        _state_refresh_stop.wait(max(0.0, period - elapsed))


@app.on_event("startup")
def _startup_state_refresh():
    global _state_refresh_thread
    print(
        f"[CONFIG] state_refresh_hz={STATE_REFRESH_HZ} "
        f"state_cache_stale_ms={STATE_CACHE_STALE_MS} state_read_count={STATE_READ_COUNT} "
        f"spi_max_speed_hz={SPI_MAX_SPEED_HZ} "
        f"spi_persistent_open={SPI_PERSISTENT_OPEN} debug_tx={DEBUG_TX}"
    )
    _state_refresh_stop.clear()
    try:
        _refresh_all_states()
    except Exception as exc:
        print(f"[STATE] initial refresh failed: {exc}")
    _state_refresh_thread = threading.Thread(target=_state_refresh_loop, name="state-refresh", daemon=True)
    _state_refresh_thread.start()


@app.on_event("shutdown")
def _shutdown_state_refresh():
    _state_refresh_stop.set()
    if _state_refresh_thread is not None:
        _state_refresh_thread.join(timeout=1.0)
    _shared_transport.close()


def _get_motor(motor_id: int):
    if motor_id not in motors:
        raise HTTPException(status_code=400, detail=f"motor_id {motor_id} not in {list(motors.keys())}")
    return motors[motor_id]


# --------------------------
# Basic APIs（均支持 motor_id / motor_ids）
# --------------------------
@app.get("/api")
def api_root():
    return {"ok": True, "msg": "RS04 API running", "motor_ids": MOTOR_IDS}


class MotorIdsBody(BaseModel):
    motor_id: int | None = None
    motor_ids: list[int] | None = None

class ZeroCmd(BaseModel):
    motor_id: int | None = None
    motor_ids: list[int] | None = None
    stop_first: bool = True
    set_motion_mode: bool = True

@app.post("/api/enable")
def api_enable(body: MotorIdsBody | None = None):
    ids = _resolve_motor_ids(body)
    for mid in ids:
        _get_motor(mid).enable()
    return {"ok": True, "motor_ids": ids}


@app.post("/api/stop")
def api_stop(clear_error: bool = False, body: MotorIdsBody | None = None):
    ids = _resolve_motor_ids(body)
    for mid in ids:
        _get_motor(mid).stop(clear_error=clear_error)
    return {"ok": True, "motor_ids": ids}


def _resolve_motor_ids(body: MotorIdsBody | None) -> list[int]:
    if body is None:
        return list(motors.keys())
    if body.motor_ids is not None and len(body.motor_ids) > 0:
        return body.motor_ids
    if body.motor_id is not None:
        return [body.motor_id]
    return list(motors.keys())



@app.post("/api/zero")
def api_zero(cmd: ZeroCmd | None = None):
    ids = _resolve_motor_ids(cmd)

    stop_first = True if cmd is None else cmd.stop_first
    set_motion_mode = True if cmd is None else cmd.set_motion_mode

    # 为了避免 PP 模式下标零被屏蔽，先停，再切回运控模式(0)
    if stop_first:
        for mid in ids:
            _get_motor(mid).stop(clear_error=False)
        time.sleep(0.05)

    if set_motion_mode:
        for mid in ids:
            _get_motor(mid).set_mode(0)   # 运控模式
        time.sleep(0.05)

    for mid in ids:
        _get_motor(mid).set_zero()
        time.sleep(0.02)

    _refresh_all_states()

    return {
        "ok": True,
        "action": "set_zero",
        "motor_ids": ids,
        "count": len(ids),
    }


@app.get("/api/state")
def api_state(motor_id: int | None = None):
    cache, meta = _get_state_cache()
    if motor_id is not None:
        item = cache.get(hex(motor_id))
        if item is None:
            item = dict(_get_motor(motor_id).get_state_dict())
            item.update(meta)
        return item
    out = dict(cache)
    out["__meta__"] = meta
    return out


@app.post("/api/state/refresh_once")
def api_state_refresh_once():
    result = _refresh_all_states()
    cache, meta = _get_state_cache()
    return {
        "ok": True,
        "state_refresh_dt_ms": result["dt_ms"],
        "state_cache_age_ms": meta["state_cache_age_ms"],
        "board_a_seq": meta["board_a_seq"],
        "board_b_seq": meta["board_b_seq"],
        "communication_ok": meta["communication_ok"],
    }

class SpeedExactCmd(BaseModel):
    motor_id: int = 0x11
    speed: float = 3.0
    accel: float = 2.0
    current_limit: float = 2.0


class MotionCmd(BaseModel):
    motor_id: int = 0x11
    position: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    kp: float = 4.0
    kd: float = 4.0
    enable_first: bool = True
    stop_first: bool = False

@app.post("/api/rs04/speed_mode_run_exact")
def rs04_speed_mode_run_exact(cmd: SpeedExactCmd):
    m = _get_motor(cmd.motor_id)
    _invalidate_motion_safety_limits("speed_mode_run_exact changed 0x7018")
    m.set_mode(2)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7018, cmd.current_limit)
    time.sleep(0.05)
    m.set_param_f32(0x7022, cmd.accel)
    time.sleep(0.05)
    m.set_param_f32(0x700A, cmd.speed)
    time.sleep(0.05)
    return {
        "ok": True,
        "mode": "speed_exact",
        "motor_id": cmd.motor_id,
        "speed": cmd.speed,
        "accel": cmd.accel,
        "current_limit": cmd.current_limit,
    }


@app.post("/api/rs04/motion_mode_run")
def rs04_motion_mode_run(cmd: MotionCmd):
    m = _get_motor(cmd.motor_id)

    if cmd.stop_first:
        m.stop(clear_error=False)
        time.sleep(0.05)

    # 运控模式 = 0
    m.set_mode(0)
    time.sleep(0.03)

    if cmd.enable_first:
        m.enable()
        time.sleep(0.03)

    m.motion_control(
        torque=cmd.torque,
        position=cmd.position,
        speed=cmd.speed,
        kp=cmd.kp,
        kd=cmd.kd,
    )

    return {
        "ok": True,
        "mode": "motion",
        "motor_id": cmd.motor_id,
        "position": cmd.position,
        "speed": cmd.speed,
        "torque": cmd.torque,
        "kp": cmd.kp,
        "kd": cmd.kd,
    }




class MotionBatchItem(BaseModel):
    motor_id: int
    position: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    kp: float = 4.0
    kd: float = 4.0

class MotionBatchCmd(BaseModel):
    items: list[MotionBatchItem]
    enable_first: bool = True
    stop_first: bool = False
    require_hardware_torque_limits: bool = False
    require_verified_hardware_safety_limits: bool = False


class MotionTorqueLimitItem(BaseModel):
    motor_id: int
    torque_limit_nm: float


class MotionTorqueLimitsCmd(BaseModel):
    items: list[MotionTorqueLimitItem]


class MotionSafetyLimitItem(BaseModel):
    motor_id: int
    torque_limit_nm: float
    current_limit_amp: float


class MotionSafetyLimitsCmd(BaseModel):
    items: list[MotionSafetyLimitItem]


@app.post("/api/rs04/configure_motion_torque_limits")
def rs04_configure_motion_torque_limits(cmd: MotionTorqueLimitsCmd):
    """Configure volatile RS01 0x700B limits before motion-mode control.

    The setting is lost when a motor is power-cycled, so the policy performs
    this handshake on every launch.  A duplicate or incomplete motor list is
    rejected to prevent a partially configured robot from starting.
    """
    items = list(cmd.items)
    _invalidate_motion_safety_limits("unverified torque-only configuration")
    ids = [int(item.motor_id) for item in items]
    if len(items) != len(MOTOR_IDS) or set(ids) != set(MOTOR_IDS):
        raise HTTPException(
            status_code=400,
            detail="torque-limit configuration must contain all 12 motors exactly once",
        )
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="duplicate motor_id")

    requested = {}
    for item in items:
        limit_nm = float(item.torque_limit_nm)
        if not 0.0 < limit_nm <= 17.0:
            raise HTTPException(
                status_code=400,
                detail=f"invalid torque limit for motor 0x{item.motor_id:02X}",
            )
        requested[int(item.motor_id)] = limit_nm

    try:
        for mid in MOTOR_IDS:
            _get_motor(mid).set_torque_limit(requested[mid])
    except Exception as exc:
        _configured_motion_torque_limits.clear()
        raise HTTPException(
            status_code=503,
            detail=f"failed to configure RS01 torque limits: {exc}",
        ) from exc

    _configured_motion_torque_limits.clear()
    _configured_motion_torque_limits.update(requested)
    return {
        "ok": True,
        "configured": True,
        "mode": "rs01_internal_torque_limit",
        "parameter_index": "0x700B",
        "count": len(requested),
        "limits": {
            hex(mid): _configured_motion_torque_limits[mid]
            for mid in MOTOR_IDS
        },
        "note": (
            "volatile writes dispatched; policy must repeat after every motor "
            "power cycle and rack-test the resulting torque response"
        ),
    }


@app.get("/api/rs04/configured_motion_torque_limits")
def rs04_configured_motion_torque_limits():
    configured = len(_configured_motion_torque_limits) == len(MOTOR_IDS)
    return {
        "ok": True,
        "configured": configured,
        "parameter_index": "0x700B",
        "count": len(_configured_motion_torque_limits),
        "limits": {
            hex(mid): value
            for mid, value in sorted(_configured_motion_torque_limits.items())
        },
    }


@app.post("/api/rs04/configure_verified_motion_safety_limits")
def rs04_configure_verified_motion_safety_limits(
    cmd: MotionSafetyLimitsCmd,
):
    """Write and read back RS01 0x700B and 0x7018 for all 12 motors."""
    items = list(cmd.items)
    ids = [int(item.motor_id) for item in items]
    if (
        len(items) != len(MOTOR_IDS)
        or set(ids) != set(MOTOR_IDS)
        or len(ids) != len(set(ids))
    ):
        raise HTTPException(
            status_code=400,
            detail="safety configuration must contain all 12 motors exactly once",
        )

    requested = {}
    for item in items:
        torque = float(item.torque_limit_nm)
        current = float(item.current_limit_amp)
        if not 0.0 < torque <= 17.0:
            raise HTTPException(
                status_code=400,
                detail=f"invalid torque limit for motor 0x{item.motor_id:02X}",
            )
        if not 0.0 < current <= 23.0:
            raise HTTPException(
                status_code=400,
                detail=f"invalid current limit for motor 0x{item.motor_id:02X}",
            )
        requested[int(item.motor_id)] = {
            "torque_limit_nm": torque,
            "current_limit_amp": current,
        }

    verified = {}
    try:
        for motor in motors.values():
            motor.stop(clear_error=False)
        time.sleep(0.02)
        # Torque is constrained and verified before raising the current ceiling.
        for mid in MOTOR_IDS:
            motor = _get_motor(mid)
            values = requested[mid]
            torque_actual = motor.write_and_verify_param_f32(
                0x700B,
                values["torque_limit_nm"],
                atol=0.05,
            )
            current_actual = motor.write_and_verify_param_f32(
                0x7018,
                values["current_limit_amp"],
                atol=0.05,
            )
            verified[mid] = {
                "torque_limit_nm": torque_actual,
                "current_limit_amp": current_actual,
            }
    except Exception as exc:
        _configured_motion_torque_limits.clear()
        _configured_motion_current_limits.clear()
        _verified_motion_safety_limits.clear()
        for motor in motors.values():
            try:
                motor.stop(clear_error=False)
            except Exception:
                pass
        raise HTTPException(
            status_code=503,
            detail=f"verified RS01 safety configuration failed: {exc}",
        ) from exc

    _configured_motion_torque_limits.clear()
    _configured_motion_torque_limits.update(
        {mid: values["torque_limit_nm"] for mid, values in verified.items()}
    )
    _configured_motion_current_limits.clear()
    _configured_motion_current_limits.update(
        {mid: values["current_limit_amp"] for mid, values in verified.items()}
    )
    _verified_motion_safety_limits.clear()
    _verified_motion_safety_limits.update(verified)
    return {
        "ok": True,
        "configured": True,
        "verified": True,
        "count": len(verified),
        "parameter_indices": ["0x700B", "0x7018"],
        "limits": {hex(mid): verified[mid] for mid in MOTOR_IDS},
    }


@app.get("/api/rs04/verified_motion_safety_limits")
def rs04_verified_motion_safety_limits():
    verified = len(_verified_motion_safety_limits) == len(MOTOR_IDS)
    return {
        "ok": True,
        "configured": verified,
        "verified": verified,
        "count": len(_verified_motion_safety_limits),
        "parameter_indices": ["0x700B", "0x7018"],
        "limits": {
            hex(mid): values
            for mid, values in sorted(_verified_motion_safety_limits.items())
        },
    }


@app.get("/api/rs04/read_param_f32")
def rs04_read_param_f32(motor_id: int, index: int):
    if not 0 <= int(index) <= 0xFFFF:
        raise HTTPException(status_code=400, detail="index must fit uint16")
    try:
        value = _get_motor(int(motor_id)).read_param_f32(int(index))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RS01 parameter read failed: {exc}",
        ) from exc
    return {
        "ok": True,
        "motor_id": hex(int(motor_id)),
        "index": hex(int(index)),
        "value": float(value),
    }

@app.post("/api/rs04/motion_mode_run_batch")
def rs04_motion_mode_run_batch(cmd: MotionBatchCmd):
    if cmd.stop_first:
        for item in cmd.items:
            _get_motor(item.motor_id).stop(clear_error=False)
        time.sleep(0.05)

    for item in cmd.items:
        _get_motor(item.motor_id).set_mode(0)
    time.sleep(0.03)

    if cmd.enable_first:
        for item in cmd.items:
            _get_motor(item.motor_id).enable()
        time.sleep(0.03)

    for item in cmd.items:
        _get_motor(item.motor_id).motion_control(
            torque=item.torque,
            position=item.position,
            speed=item.speed,
            kp=item.kp,
            kd=item.kd,
        )

    return {"ok": True, "mode": "motion_batch", "count": len(cmd.items)}

@app.post("/api/rs04/motion_batch_fast")
def rs04_motion_batch_fast(cmd: MotionBatchCmd):
    """
    高频策略控制专用接口。

    正常50 Hz周期只发送motion_control。启动站立的第一帧可以通过
    enable_first执行一次安全初始化：STOP、设置运控模式、在失能状态预写
    当前姿态目标、ENABLE，然后立即重发同一目标。

    enable_first之后的策略循环必须将该字段保持为false。
    """
    if cmd.require_hardware_torque_limits and (
        len(_configured_motion_torque_limits) != len(MOTOR_IDS)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "RS01 hardware torque limits are not configured; restart the "
                "5730 policy to repeat the 0x700B handshake"
            ),
        )
    if cmd.require_verified_hardware_safety_limits and (
        len(_verified_motion_safety_limits) != len(MOTOR_IDS)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "verified RS01 torque/current limits are not active; restart "
                "the policy after flashing both updated STM32 boards"
            ),
        )
    t0 = time.perf_counter()
    t_prepare = time.perf_counter()
    items = list(cmd.items)
    item_ids = [int(item.motor_id) for item in items]
    if cmd.enable_first and (
        len(items) != len(MOTOR_IDS)
        or set(item_ids) != set(MOTOR_IDS)
        or len(item_ids) != len(set(item_ids))
    ):
        raise HTTPException(
            status_code=400,
            detail="enable_first requires all 12 motors exactly once",
        )
    prepare_ms = (time.perf_counter() - t_prepare) * 1000.0
    t_spi = time.perf_counter()
    commands = [
        (_get_motor(item.motor_id), item.torque, item.position, item.speed, item.kp, item.kd)
        for item in items
    ]
    enable_applied = False
    stop_applied = False
    target_primed = False
    priming_spi_frames = 0

    try:
        if cmd.stop_first or cmd.enable_first:
            # enable_first is deliberately self-contained.  The verified
            # safety handshake stops all motors, so relying on an earlier
            # enable state is unsafe and caused motors to remain disabled.
            for item in items:
                _get_motor(item.motor_id).stop(clear_error=False)
            stop_applied = True

        if cmd.enable_first:
            for item in items:
                _get_motor(item.motor_id).set_mode(0)
            time.sleep(0.01)

            # Prime the live-pose target while disabled.  Some RS01 revisions
            # may ignore it until enabled, so send the exact frame again below.
            priming_spi_frames = send_motion_control_batch(commands)
            target_primed = True

            for item in items:
                _get_motor(item.motor_id).enable()
            time.sleep(0.01)
            enable_applied = True

        num_spi_frames = send_motion_control_batch(commands)
    except Exception as exc:
        if cmd.enable_first or cmd.stop_first:
            for item in items:
                try:
                    _get_motor(item.motor_id).stop(clear_error=False)
                except Exception:
                    pass
        raise HTTPException(
            status_code=503,
            detail=f"motion startup handshake failed; all motors stopped: {exc}",
        ) from exc
    spi_send_ms = (time.perf_counter() - t_spi) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    timing = _record_control_send_timing()
    batch_active = num_spi_frames < len(items)

    return {
        "ok": True,
        "mode": "motion_batch_fast",
        "count": len(items),
        "note": (
            "fast path packs up to six CAN commands into one SPI frame per STM32"
            if batch_active else
            "STM32 batch capability not detected; using safe legacy SPI fallback"
        ),
        "prepare_ms": prepare_ms,
        "spi_send_ms": spi_send_ms,
        "total_ms": total_ms,
        "num_spi_frames": num_spi_frames,
        "priming_spi_frames": priming_spi_frames,
        "num_can_frames": len(items),
        "batch_active": batch_active,
        "stop_applied": stop_applied,
        "target_primed": target_primed,
        "enable_applied": enable_applied,
        "hardware_torque_limits_active": (
            len(_configured_motion_torque_limits) == len(MOTOR_IDS)
        ),
        "verified_hardware_safety_limits_active": (
            len(_verified_motion_safety_limits) == len(MOTOR_IDS)
        ),
        **timing,
    }
# --------------------------
# Raw Hex (12 bytes) - debug helper
# --------------------------
class RawHexCmd(BaseModel):
    hex12: str

@app.post("/api/raw_hex")
def raw_hex(cmd: RawHexCmd):
    parts = cmd.hex12.strip().split()
    if len(parts) != 12:
        return {"ok": False, "error": "Need exactly 12 bytes"}
    vals = bytes(int(x, 16) for x in parts)
    ext_id = int.from_bytes(vals[:4], "big")
    data = vals[4:12]
    target_id = ext_id & 0xFF
    m = motors.get(target_id)
    if m is None:
        m = next(iter(motors.values()))
    m._send(ext_id, data)
    return {"ok": True}


# --------------------------
# RS04 mode-based APIs（body 中带 motor_id）
# --------------------------
class SpeedCmd(BaseModel):
    motor_id: int = 0x11
    speed: float = 2.0
    accel: float = 2.0


class PosCmd(BaseModel):
    motor_id: int = 0x11
    position: float = 0.0
    speed: float = 2.0
    accel: float = 1.0


class CurrentCmd(BaseModel):
    motor_id: int = 0x11
    iq: float = 0.0


@app.post("/api/rs04/speed_mode_run")
def rs04_speed_mode_run(cmd: SpeedCmd):
    m = _get_motor(cmd.motor_id)
    _invalidate_motion_safety_limits("speed_mode_run changed 0x7018")
    m.set_param_u8(0x7005, 2)
    time.sleep(0.08)
    m.enable()
    time.sleep(0.08)
    m.set_param_f32(0x7018, 2.0)
    time.sleep(0.08)
    m.set_param_f32(0x7022, cmd.accel)
    time.sleep(0.08)
    m.set_param_f32(0x700A, cmd.speed)
    time.sleep(0.08)
    return {
        "ok": True,
        "mode": "speed",
        "motor_id": cmd.motor_id,
        "speed": cmd.speed,
        "accel": cmd.accel,
        "current_limit": 2.0,
    }


@app.post("/api/rs04/pp_mode_run")
def rs04_pp_mode_run(cmd: PosCmd):
    m = _get_motor(cmd.motor_id)
    m.set_mode(1)
    time.sleep(0.05)
    m.set_param_f32(0x7024, cmd.speed)
    time.sleep(0.05)
    m.set_param_f32(0x7025, cmd.accel)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7016, cmd.position)
    return {
        "ok": True,
        "mode": "pp",
        "motor_id": cmd.motor_id,
        "position": cmd.position,
        "speed": cmd.speed,
        "accel": cmd.accel,
    }


@app.post("/api/rs04/current_mode_run")
def rs04_current_mode_run(cmd: CurrentCmd):
    m = _get_motor(cmd.motor_id)
    m.set_mode(3)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7006, cmd.iq)
    return {"ok": True, "mode": "current", "motor_id": cmd.motor_id, "iq": cmd.iq}
    
    
    
    
    
