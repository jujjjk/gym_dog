#!/usr/bin/env python3
"""Collect low-amplitude real motor-chain timing and response data.

The test holds every motor around its live position, then applies a small
common position step and sine excitation.  It uses the same AF_UNIX binary
command path as the policy node and always requests STOP in ``finally``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import statistics
import sys
import time

from mydog_policy.motor_rt_client import MotorRtClient


MOTOR_IDS = (0x11, 0x12, 0x13, 0x21, 0x22, 0x23,
             0x31, 0x32, 0x33, 0x41, 0x42, 0x43)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--socket", default="/tmp/lingzu_motor_rt.sock")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--amplitude-rad", type=float, default=0.03)
    parser.add_argument("--sine-hz", type=float, default=0.7)
    parser.add_argument("--kp", type=float, default=6.0)
    parser.add_argument("--kd", type=float, default=0.5)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    x = (len(ordered) - 1) * fraction
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - x) + ordered[hi] * (x - lo)


def main():
    args = parse_args()
    if not (10.0 <= args.rate_hz <= 100.0):
        raise SystemExit("rate-hz must be within [10, 100]")
    if not (0.0 < args.amplitude_rad <= 0.05):
        raise SystemExit("amplitude-rad must be within (0, 0.05]")
    if not (0.0 < args.kp <= 70.0 and 0.0 <= args.kd <= 2.0):
        raise SystemExit("safe probe requires kp <= 70 and kd <= 2")
    # Allow policy-scale stiffness only with a correspondingly tiny motion.
    # This bounds the initial position-error torque before any motor moves.
    if args.kp * args.amplitude_rad > 0.75:
        raise SystemExit("kp * amplitude-rad must not exceed 0.75 Nm")

    client = MotorRtClient(args.socket, timeout=0.08)
    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    state = client.get_state()
    states = state["states"]
    if len(states) != 12:
        raise RuntimeError(f"expected 12 motors, got {len(states)}")
    if tuple(item["can_id"] for item in states) != MOTOR_IDS:
        raise RuntimeError("motor ID/order mismatch")
    if not all(item["online"] for item in states):
        raise RuntimeError("one or more motors are offline")
    if any(item["error_code"] for item in states):
        raise RuntimeError("one or more motors report a fault")
    if not (state["flags"] & 1):
        raise RuntimeError("motor communication is not healthy")

    # Several samples protect the initial hold target from one stale snapshot.
    samples = [[float(item["angle"]) for item in states]]
    for _ in range(9):
        time.sleep(0.02)
        samples.append([float(item["angle"]) for item in client.get_state()["states"]])
    center = [statistics.median(row[j] for row in samples) for j in range(12)]

    phases = (
        ("hold", 1.0),
        ("step_pos", 1.0),
        ("center_1", 0.5),
        ("step_neg", 1.0),
        ("center_2", 0.5),
        ("sine", 2.0),
        ("final_hold", 0.5),
    )
    headers = [
        "sample", "host_monotonic_ns", "phase", "phase_time_s",
        "scheduled_time_s", "loop_dt_ms", "loop_lateness_ms", "rpc_ms",
        "server_spi_ms", "server_total_ms", "cache_age_ms",
        "cache_sequence", "board_a_seq", "board_b_seq",
    ]
    for motor_id in MOTOR_IDS:
        prefix = f"m{motor_id:02x}"
        headers.extend((
            f"{prefix}_target_rad", f"{prefix}_q_rad", f"{prefix}_dq_rad_s",
            f"{prefix}_torque_nm", f"{prefix}_age_ms", f"{prefix}_mode",
            f"{prefix}_fault", f"{prefix}_snapshot_seq",
            f"{prefix}_board_tick_ms",
        ))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    period = 1.0 / args.rate_hz
    rpc_values = []
    spi_values = []
    loop_values = []
    sample_index = 0
    last_loop = None
    enabled = False

    try:
        with open(args.output, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            test_start = time.monotonic()
            phase_start = test_start
            next_tick = test_start

            for phase, duration in phases:
                phase_start = time.monotonic()
                phase_end = phase_start + duration
                while time.monotonic() < phase_end and not stopping:
                    now = time.monotonic()
                    phase_time = now - phase_start
                    if phase == "step_pos":
                        offset = args.amplitude_rad
                    elif phase == "step_neg":
                        offset = -args.amplitude_rad
                    elif phase == "sine":
                        offset = args.amplitude_rad * math.sin(
                            2.0 * math.pi * args.sine_hz * phase_time
                        )
                    else:
                        offset = 0.0
                    targets = [value + offset for value in center]
                    commands = [
                        {
                            "motor_id": motor_id,
                            "position": target,
                            "speed": 0.0,
                            "torque": 0.0,
                            "kp": args.kp,
                            "kd": args.kd,
                        }
                        for motor_id, target in zip(MOTOR_IDS, targets)
                    ]

                    rpc_start = time.perf_counter()
                    response = client.send_motion(
                        commands,
                        enable_first=not enabled,
                        require_hardware_torque_limits=True,
                        require_verified_hardware_safety_limits=True,
                    )
                    rpc_ms = (time.perf_counter() - rpc_start) * 1000.0
                    enabled = True
                    after = time.monotonic()
                    loop_dt_ms = 0.0 if last_loop is None else (after - last_loop) * 1000.0
                    last_loop = after
                    lateness_ms = max(0.0, (now - next_tick) * 1000.0)

                    row = [
                        sample_index, time.monotonic_ns(), phase, phase_time,
                        now - test_start, loop_dt_ms, lateness_ms, rpc_ms,
                        response["spi_ms"], response["total_ms"],
                        response["cache_age_ms"], response["cache_sequence"],
                        response["board_a_seq"], response["board_b_seq"],
                    ]
                    for target, motor in zip(targets, response["states"]):
                        row.extend((
                            target, motor["angle"], motor["speed"],
                            motor["torque"], motor["age_ms"],
                            motor["mode_state"], motor["error_code"],
                            motor["snapshot_seq"], motor["board_tick_ms"],
                        ))
                    writer.writerow(row)
                    rpc_values.append(rpc_ms)
                    spi_values.append(float(response["spi_ms"]))
                    if loop_dt_ms > 0.0:
                        loop_values.append(loop_dt_ms)
                    sample_index += 1
                    next_tick += period
                    delay = next_tick - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    else:
                        # Do not issue a burst of catch-up motor commands.
                        next_tick = time.monotonic()
                next_tick = time.monotonic()
            stream.flush()
    finally:
        try:
            client.stop()
            print("STOP accepted for all 12 motors", flush=True)
        finally:
            client.close()

    print(f"output={args.output}")
    print(f"samples={sample_index}")
    for name, values in (("rpc_ms", rpc_values), ("spi_ms", spi_values),
                         ("loop_dt_ms", loop_values)):
        print(
            f"{name}: mean={statistics.fmean(values):.3f} "
            f"p50={percentile(values, 0.50):.3f} "
            f"p95={percentile(values, 0.95):.3f} "
            f"p99={percentile(values, 0.99):.3f} "
            f"max={max(values):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
