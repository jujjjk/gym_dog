#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import requests


BASE_URL = "http://127.0.0.1:8000"

# 真实电机编号，以及 q_policy = sign * q_real 的方向关系
LEG_CONFIG = {
    "FR": {"thigh": 0x12, "calf": 0x13, "sign": +1.0},
    "FL": {"thigh": 0x22, "calf": 0x23, "sign": -1.0},
    "RL": {"thigh": 0x32, "calf": 0x33, "sign": -1.0},
    "RR": {"thigh": 0x42, "calf": 0x43, "sign": +1.0},
}


def wrap_pi(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def fit_sine(t, y, frequency):
    omega = 2.0 * math.pi * frequency
    matrix = np.column_stack([
        np.sin(omega * t),
        np.cos(omega * t),
        np.ones_like(t),
    ])
    coef, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)

    amplitude = math.hypot(float(coef[0]), float(coef[1]))
    phase = math.atan2(float(coef[1]), float(coef[0]))
    center = float(coef[2])
    return amplitude, phase, center


def get_states():
    response = requests.get(
        f"{BASE_URL}/api/state",
        timeout=1.0,
    )
    response.raise_for_status()
    return response.json()


def initialize_pair(
    thigh_id,
    calf_id,
    thigh_target,
    calf_target,
    thigh_kp,
    thigh_kd,
    calf_kp,
    calf_kd,
):
    payload = {
        "items": [
            {
                "motor_id": thigh_id,
                "position": float(thigh_target),
                "speed": 0.0,
                "torque": 0.0,
                "kp": float(thigh_kp),
                "kd": float(thigh_kd),
            },
            {
                "motor_id": calf_id,
                "position": float(calf_target),
                "speed": 0.0,
                "torque": 0.0,
                "kp": float(calf_kp),
                "kd": float(calf_kd),
            },
        ],
        "enable_first": True,
        "stop_first": False,
    }

    response = requests.post(
        f"{BASE_URL}/api/rs04/motion_mode_run_batch",
        json=payload,
        timeout=2.0,
    )
    response.raise_for_status()


def send_pair(
    thigh_id,
    calf_id,
    thigh_target,
    calf_target,
    thigh_kp,
    thigh_kd,
    calf_kp,
    calf_kd,
):
    payload = {
        "items": [
            {
                "motor_id": thigh_id,
                "position": float(thigh_target),
                "speed": 0.0,
                "torque": 0.0,
                "kp": float(thigh_kp),
                "kd": float(thigh_kd),
            },
            {
                "motor_id": calf_id,
                "position": float(calf_target),
                "speed": 0.0,
                "torque": 0.0,
                "kp": float(calf_kp),
                "kd": float(calf_kd),
            },
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", required=True, choices=["FR", "FL", "RL", "RR"])
    parser.add_argument("--freqs", nargs="+", type=float, default=[0.5])
    parser.add_argument("--thigh-amplitude", type=float, default=0.01)
    parser.add_argument("--calf-ratio", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--discard", type=float, default=1.0)
    parser.add_argument("--control-hz", type=float, default=50.0)

    parser.add_argument("--thigh-kp", type=float, default=40.0)
    parser.add_argument("--thigh-kd", type=float, default=1.2)
    parser.add_argument("--calf-kp", type=float, default=70.0)
    parser.add_argument("--calf-kd", type=float, default=1.6)

    parser.add_argument(
        "--output-dir",
        default="/home/jetson/mydog_ros2_ws/log",
    )
    args = parser.parse_args()

    calf_amplitude = args.thigh_amplitude * args.calf_ratio

    if args.thigh_amplitude <= 0.0 or args.thigh_amplitude > 0.02:
        raise RuntimeError("thigh-amplitude必须在0～0.02rad之间")

    if calf_amplitude <= 0.0 or calf_amplitude > 0.04:
        raise RuntimeError("calf实际振幅必须在0～0.04rad之间")

    if any(f <= 0.0 or f > 2.0 for f in args.freqs):
        raise RuntimeError("频率必须在0～2.0Hz之间")

    config = LEG_CONFIG[args.leg]
    thigh_id = config["thigh"]
    calf_id = config["calf"]
    sign = config["sign"]

    states = get_states()
    thigh_state = states[hex(thigh_id)]
    calf_state = states[hex(calf_id)]

    if not thigh_state.get("online") or not calf_state.get("online"):
        raise RuntimeError("被测大腿或小腿电机不在线")

    if int(thigh_state.get("error_code", 0)) != 0:
        raise RuntimeError("大腿电机存在故障码")

    if int(calf_state.get("error_code", 0)) != 0:
        raise RuntimeError("小腿电机存在故障码")

    q0_thigh = float(thigh_state["angle"])
    q0_calf = float(calf_state["angle"])

    # 防止机器人并未处于弯腿站姿时误测
    if abs(q0_thigh) < 0.10 or abs(q0_calf) < 0.20:
        raise RuntimeError(
            f"当前不是正常站姿：thigh={q0_thigh:+.4f}, "
            f"calf={q0_calf:+.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = output_dir / f"dual_probe_{args.leg}_{stamp}.csv"

    print("=" * 72)
    print(f"测试腿          = {args.leg}")
    print(f"thigh motor     = 0x{thigh_id:02X}")
    print(f"calf motor      = 0x{calf_id:02X}")
    print(f"thigh中心角     = {q0_thigh:+.4f} rad")
    print(f"calf中心角      = {q0_calf:+.4f} rad")
    print(f"thigh振幅       = ±{args.thigh_amplitude:.4f} rad")
    print(f"calf振幅        = ±{calf_amplitude:.4f} rad")
    print(f"测试频率        = {args.freqs}")
    print(f"输出文件        = {output}")
    print("=" * 72)
    print("3秒后开始。确认机身有支撑、足端没有被卡死。")
    time.sleep(3.0)

    rows = []
    control_period = 1.0 / args.control_hz

    try:
        initialize_pair(
            thigh_id,
            calf_id,
            q0_thigh,
            q0_calf,
            args.thigh_kp,
            args.thigh_kd,
            args.calf_kp,
            args.calf_kd,
        )
        time.sleep(1.0)

        for frequency in args.freqs:
            print(f"开始：{frequency:.2f} Hz")

            stage_start = time.perf_counter()
            next_tick = stage_start
            previous_tick = stage_start

            while True:
                now_perf = time.perf_counter()
                elapsed = now_perf - stage_start

                if elapsed >= args.duration:
                    break

                wave = math.sin(2.0 * math.pi * frequency * elapsed)

                # 策略坐标中：
                # thigh正方向、calf负方向联动，主要产生足端上下运动。
                #
                # 真机左腿方向和策略相反，因此乘sign转换。
                thigh_delta_real = (
                    sign * args.thigh_amplitude * wave
                )
                calf_delta_real = (
                    sign * (-calf_amplitude) * wave
                )

                thigh_target = q0_thigh + thigh_delta_real
                calf_target = q0_calf + calf_delta_real

                send_pair(
                    thigh_id,
                    calf_id,
                    thigh_target,
                    calf_target,
                    args.thigh_kp,
                    args.thigh_kd,
                    args.calf_kp,
                    args.calf_kd,
                )

                states = get_states()
                thigh_state = states[hex(thigh_id)]
                calf_state = states[hex(calf_id)]

                if not thigh_state.get("online") or not calf_state.get("online"):
                    raise RuntimeError("测试中电机离线")

                if (
                    int(thigh_state.get("error_code", 0)) != 0
                    or int(calf_state.get("error_code", 0)) != 0
                ):
                    raise RuntimeError("测试中出现电机故障码")

                thigh_angle = float(thigh_state["angle"])
                calf_angle = float(calf_state["angle"])

                if abs(thigh_angle - q0_thigh) > 0.08:
                    raise RuntimeError("大腿角度偏离中心超过0.08rad，停止")

                if abs(calf_angle - q0_calf) > 0.10:
                    raise RuntimeError("小腿角度偏离中心超过0.10rad，停止")

                loop_dt_ms = (now_perf - previous_tick) * 1000.0
                previous_tick = now_perf

                rows.append({
                    "time": time.time(),
                    "elapsed_sec": elapsed,
                    "leg": args.leg,
                    "frequency_hz": frequency,
                    "thigh_motor_id": f"0x{thigh_id:02X}",
                    "calf_motor_id": f"0x{calf_id:02X}",
                    "thigh_center_rad": q0_thigh,
                    "calf_center_rad": q0_calf,
                    "thigh_target_rad": thigh_target,
                    "calf_target_rad": calf_target,
                    "thigh_angle_rad": thigh_angle,
                    "calf_angle_rad": calf_angle,
                    "thigh_speed_rad_s": float(thigh_state.get("speed", 0.0)),
                    "calf_speed_rad_s": float(calf_state.get("speed", 0.0)),
                    "thigh_torque_feedback": float(
                        thigh_state.get("torque", 0.0)
                    ),
                    "calf_torque_feedback": float(
                        calf_state.get("torque", 0.0)
                    ),
                    "thigh_age_ms": float(thigh_state.get("age_ms", 99999)),
                    "calf_age_ms": float(calf_state.get("age_ms", 99999)),
                    "loop_dt_ms": loop_dt_ms,
                })

                next_tick += control_period
                sleep_time = next_tick - time.perf_counter()

                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

            # 每档结束返回测试前站姿
            send_pair(
                thigh_id,
                calf_id,
                q0_thigh,
                q0_calf,
                args.thigh_kp,
                args.thigh_kd,
                args.calf_kp,
                args.calf_kd,
            )
            time.sleep(2.0)

    finally:
        # 出错或Ctrl+C时先返回测试前站姿。
        try:
            send_pair(
                thigh_id,
                calf_id,
                q0_thigh,
                q0_calf,
                args.thigh_kp,
                args.thigh_kd,
                args.calf_kp,
                args.calf_kd,
            )
            time.sleep(1.0)
        except Exception:
            pass

    if not rows:
        raise RuntimeError("没有采集到数据")

    fieldnames = list(rows[0].keys())

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 108)
    print(
        f"{'Hz':>5} "
        f"{'大腿幅度比':>11} "
        f"{'小腿幅度比':>11} "
        f"{'大腿滞后ms':>12} "
        f"{'小腿滞后ms':>12} "
        f"{'两关节滞后差':>13} "
        f"{'协调相位误差°':>15}"
    )

    for frequency in args.freqs:
        selected = [
            row for row in rows
            if abs(row["frequency_hz"] - frequency) < 1e-6
            and row["elapsed_sec"] >= args.discard
        ]

        if len(selected) < 20:
            print(f"{frequency:5.2f} 有效数据不足")
            continue

        t = np.asarray([r["elapsed_sec"] for r in selected])

        thigh_target = np.asarray([r["thigh_target_rad"] for r in selected])
        calf_target = np.asarray([r["calf_target_rad"] for r in selected])
        thigh_angle = np.asarray([r["thigh_angle_rad"] for r in selected])
        calf_angle = np.asarray([r["calf_angle_rad"] for r in selected])

        thigh_target_amp, thigh_target_phase, _ = fit_sine(
            t, thigh_target, frequency
        )
        calf_target_amp, calf_target_phase, _ = fit_sine(
            t, calf_target, frequency
        )

        thigh_amp, thigh_phase, _ = fit_sine(
            t, thigh_angle, frequency
        )
        calf_amp, calf_phase, _ = fit_sine(
            t, calf_angle, frequency
        )

        thigh_lag = (
            -wrap_pi(thigh_phase - thigh_target_phase)
            / (2.0 * math.pi * frequency)
            * 1000.0
        )
        calf_lag = (
            -wrap_pi(calf_phase - calf_target_phase)
            / (2.0 * math.pi * frequency)
            * 1000.0
        )

        thigh_ratio = thigh_amp / max(thigh_target_amp, 1e-9)
        calf_ratio = calf_amp / max(calf_target_amp, 1e-9)

        # 实际thigh和calf理论上应保持约180°反相。
        actual_relative_phase = wrap_pi(calf_phase - thigh_phase)
        coordination_error = abs(
            math.degrees(
                wrap_pi(actual_relative_phase - math.pi)
            )
        )

        print(
            f"{frequency:5.2f} "
            f"{thigh_ratio:11.3f} "
            f"{calf_ratio:11.3f} "
            f"{thigh_lag:12.1f} "
            f"{calf_lag:12.1f} "
            f"{abs(thigh_lag-calf_lag):13.1f} "
            f"{coordination_error:15.1f}"
        )

    print("=" * 108)
    print("数据文件：", output)
    print("被测关节已回到测试前站姿，但仍保持使能。")
    print("确认机身被支撑后，可执行：")
    print("curl -X POST http://127.0.0.1:8000/api/stop")


if __name__ == "__main__":
    main()
