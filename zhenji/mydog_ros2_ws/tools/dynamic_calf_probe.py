#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import requests


BASE_URL = "http://127.0.0.1:8000"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor-id", required=True, type=lambda x: int(x, 0))
    parser.add_argument(
        "--freqs",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 1.85],
    )
    parser.add_argument("--amplitude", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--discard", type=float, default=1.5)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=1.2)
    parser.add_argument(
        "--output-dir",
        default="/home/jetson/mydog_ros2_ws/log",
    )
    return parser.parse_args()


def read_state(motor_id):
    response = requests.get(
        f"{BASE_URL}/api/state",
        params={"motor_id": motor_id},
        timeout=1.0,
    )
    response.raise_for_status()
    return response.json()


def initialize_motor(motor_id, position, kp, kd):
    payload = {
        "motor_id": motor_id,
        "position": float(position),
        "speed": 0.0,
        "torque": 0.0,
        "kp": float(kp),
        "kd": float(kd),
        "enable_first": True,
        "stop_first": False,
    }

    response = requests.post(
        f"{BASE_URL}/api/rs04/motion_mode_run",
        json=payload,
        timeout=2.0,
    )
    response.raise_for_status()


def send_target(motor_id, position, kp, kd):
    payload = {
        "items": [
            {
                "motor_id": motor_id,
                "position": float(position),
                "speed": 0.0,
                "torque": 0.0,
                "kp": float(kp),
                "kd": float(kd),
            }
        ],
        "enable_first": False,
        "stop_first": False,
    }

    response = requests.post(
        f"{BASE_URL}/api/rs04/motion_batch_fast",
        json=payload,
        timeout=1.0,
    )
    response.raise_for_status()


def stop_motor(motor_id):
    response = requests.post(
        f"{BASE_URL}/api/stop",
        json={"motor_id": motor_id},
        timeout=1.0,
    )
    response.raise_for_status()


def fit_sine(t, y, frequency):
    omega = 2.0 * math.pi * frequency

    matrix = np.column_stack(
        [
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t),
        ]
    )

    coef, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)

    sin_coef = float(coef[0])
    cos_coef = float(coef[1])

    amplitude = math.hypot(sin_coef, cos_coef)
    phase = math.atan2(cos_coef, sin_coef)

    return amplitude, phase


def wrap_pi(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def main():
    args = parse_args()

    if not 0.001 <= args.amplitude <= 0.05:
        raise RuntimeError("amplitude必须在0.001～0.05 rad之间")

    if any(freq <= 0.0 or freq > 2.5 for freq in args.freqs):
        raise RuntimeError("频率必须在0～2.5 Hz之间")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = output_dir / (
        f"dynamic_probe_0x{args.motor_id:02X}_{stamp}.csv"
    )

    state = read_state(args.motor_id)

    if not bool(state.get("online", False)):
        raise RuntimeError(f"电机0x{args.motor_id:02X}不在线")

    q0 = float(state["angle"])

    print("=" * 70)
    print(f"motor_id       = 0x{args.motor_id:02X}")
    print(f"中心角度 q0    = {q0:+.4f} rad")
    print(f"振幅           = ±{args.amplitude:.4f} rad")
    print(f"测试频率       = {args.freqs}")
    print(f"控制频率       = {args.control_hz:.1f} Hz")
    print(f"Kp / Kd        = {args.kp:.1f} / {args.kd:.1f}")
    print(f"输出文件       = {output}")
    print("=" * 70)
    print("3秒后开始，请确认电机和腿部没有被人抓住。")
    time.sleep(3.0)

    fieldnames = [
        "time",
        "elapsed_sec",
        "stage",
        "frequency_hz",
        "motor_id",
        "center_rad",
        "target_rad",
        "angle_rad",
        "speed_rad_s",
        "torque_feedback",
        "online",
        "error_code",
        "age_ms",
        "loop_dt_ms",
    ]

    all_rows = []

    period = 1.0 / args.control_hz

    try:
        initialize_motor(
            args.motor_id,
            q0,
            args.kp,
            args.kd,
        )

        time.sleep(1.0)

        for frequency in args.freqs:
            print()
            print(f"开始测试：{frequency:.2f} Hz")

            stage_start = time.perf_counter()
            next_tick = stage_start
            last_tick = stage_start

            while True:
                now_perf = time.perf_counter()
                elapsed = now_perf - stage_start

                if elapsed >= args.duration:
                    break

                target = q0 + args.amplitude * math.sin(
                    2.0 * math.pi * frequency * elapsed
                )

                send_target(
                    args.motor_id,
                    target,
                    args.kp,
                    args.kd,
                )

                state = read_state(args.motor_id)

                now_wall = time.time()
                loop_dt_ms = (now_perf - last_tick) * 1000.0
                last_tick = now_perf

                row = {
                    "time": now_wall,
                    "elapsed_sec": elapsed,
                    "stage": "sine",
                    "frequency_hz": frequency,
                    "motor_id": f"0x{args.motor_id:02X}",
                    "center_rad": q0,
                    "target_rad": target,
                    "angle_rad": float(state.get("angle", 0.0)),
                    "speed_rad_s": float(state.get("speed", 0.0)),
                    "torque_feedback": float(state.get("torque", 0.0)),
                    "online": int(bool(state.get("online", False))),
                    "error_code": int(state.get("error_code", 0)),
                    "age_ms": float(state.get("age_ms", 999999)),
                    "loop_dt_ms": loop_dt_ms,
                }

                all_rows.append(row)

                next_tick += period
                sleep_time = next_tick - time.perf_counter()

                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

            # 每档频率结束后回到中心位置
            send_target(
                args.motor_id,
                q0,
                args.kp,
                args.kd,
            )
            time.sleep(2.0)

        send_target(
            args.motor_id,
            q0,
            args.kp,
            args.kd,
        )
        time.sleep(1.0)

    finally:
        try:
            send_target(
                args.motor_id,
                q0,
                args.kp,
                args.kd,
            )
            time.sleep(0.5)
        except Exception:
            pass

        try:
            stop_motor(args.motor_id)
        except Exception as exc:
            print("停止电机失败：", exc)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 86)
    print(
        f"{'频率':>7} "
        f"{'目标幅度':>10} "
        f"{'实际幅度':>10} "
        f"{'幅度比':>8} "
        f"{'滞后ms':>9} "
        f"{'RMSE':>9} "
        f"{'最大age':>9}"
    )

    for frequency in args.freqs:
        rows = [
            row
            for row in all_rows
            if abs(row["frequency_hz"] - frequency) < 1.0e-6
            and row["elapsed_sec"] >= args.discard
        ]

        if len(rows) < 20:
            print(f"{frequency:7.2f} 有效数据不足")
            continue

        t = np.asarray([row["elapsed_sec"] for row in rows])
        target = np.asarray([row["target_rad"] for row in rows])
        angle = np.asarray([row["angle_rad"] for row in rows])

        target_amp, target_phase = fit_sine(
            t,
            target,
            frequency,
        )
        actual_amp, actual_phase = fit_sine(
            t,
            angle,
            frequency,
        )

        phase_difference = wrap_pi(
            actual_phase - target_phase
        )

        lag_ms = (
            -phase_difference
            / (2.0 * math.pi * frequency)
            * 1000.0
        )

        amplitude_ratio = actual_amp / max(target_amp, 1.0e-9)
        rmse = float(np.sqrt(np.mean((target - angle) ** 2)))
        max_age = max(row["age_ms"] for row in rows)

        print(
            f"{frequency:7.2f} "
            f"{target_amp:10.4f} "
            f"{actual_amp:10.4f} "
            f"{amplitude_ratio:8.3f} "
            f"{lag_ms:9.1f} "
            f"{rmse:9.4f} "
            f"{max_age:9.1f}"
        )

    error_codes = sorted(
        set(int(row["error_code"]) for row in all_rows)
    )
    online_ratio = np.mean(
        [row["online"] for row in all_rows]
    )

    print("=" * 86)
    print("在线比例：", f"{online_ratio * 100.0:.1f}%")
    print("故障码：", error_codes)
    print("数据文件：", output)


if __name__ == "__main__":
    main()
