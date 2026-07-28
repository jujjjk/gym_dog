#!/usr/bin/env python3
"""Collect unloaded low-speed RS01 friction and reversal data."""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import statistics
import time

from collect_rs01_subset_identification import (
    ALL_MOTOR_IDS,
    JsonHttpClient,
    parse_float_list,
    parse_id_list,
    safety_item,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-offline", default="0x11,0x41")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--amplitude-rad", type=float, default=0.05)
    parser.add_argument("--speeds-rad-s", default="0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.2)
    return parser.parse_args()


def main():
    args = parse_args()
    expected_offline = set(parse_id_list(args.expected_offline))
    speeds = parse_float_list(args.speeds_rad_s)
    if not 10.0 <= args.rate_hz <= 100.0:
        raise SystemExit("rate-hz must be within [10, 100]")
    if not 0.0 < args.amplitude_rad <= 0.06:
        raise SystemExit("amplitude-rad must be within (0, 0.06]")
    if args.kp * args.amplitude_rad > 3.6:
        raise SystemExit("kp * amplitude-rad must not exceed 3.6 Nm")
    if not speeds or any(speed <= 0.0 or speed > 0.5 for speed in speeds):
        raise SystemExit("speeds must be within (0, 0.5] rad/s")

    client = JsonHttpClient(args.base_url)
    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    initial = client.request("GET", "/api/state")
    online_ids = tuple(
        mid for mid in ALL_MOTOR_IDS
        if bool(initial.get(hex(mid), {}).get("online", False))
    )
    offline_ids = set(ALL_MOTOR_IDS) - set(online_ids)
    if offline_ids != expected_offline:
        raise RuntimeError(
            f"offline set mismatch: expected={sorted(expected_offline)} "
            f"actual={sorted(offline_ids)}"
        )
    for mid in online_ids:
        state = initial[hex(mid)]
        if int(state.get("mode_state", 0)) != 0:
            raise RuntimeError(f"motor 0x{mid:02X} is not stopped")
        if int(state.get("error_code", 0)) != 0:
            raise RuntimeError(f"motor 0x{mid:02X} reports a fault")

    verified = client.request(
        "POST",
        "/api/rs04/configure_verified_diagnostic_safety_limits",
        {"items": [safety_item(mid) for mid in online_ids]},
    )
    if not verified.get("verified") or not verified.get("read_only"):
        raise RuntimeError("read-only diagnostic limit verification failed")

    samples = []
    for _ in range(10):
        state = client.request("GET", "/api/state")
        samples.append([float(state[hex(mid)]["angle"]) for mid in online_ids])
        time.sleep(0.02)
    centers = [
        statistics.median(row[index] for row in samples)
        for index in range(len(online_ids))
    ]

    phases = [("hold", 1.0, None)]
    for speed in speeds:
        period = 4.0 * args.amplitude_rad / speed
        duration = max(1.5 * period, math.ceil(4.0 / period) * period)
        phases.append((f"triangle_{speed:g}", duration, speed))
        phases.append((f"center_after_{speed:g}", 0.75, None))
    phases.append(("final_hold", 1.0, None))

    headers = [
        "sample", "host_request_ns", "host_response_ns", "phase",
        "phase_time_s", "target_rate_rad_s", "rpc_ms", "server_spi_ms",
        "server_total_ms", "cache_age_ms", "board_a_seq", "board_b_seq",
    ]
    for mid in online_ids:
        prefix = f"m{mid:02x}"
        headers.extend((
            f"{prefix}_target_rad", f"{prefix}_q_rad",
            f"{prefix}_dq_rad_s", f"{prefix}_torque_nm",
            f"{prefix}_age_ms", f"{prefix}_mode", f"{prefix}_fault",
            f"{prefix}_temperature_c",
        ))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    period_s = 1.0 / args.rate_hz
    sample_index = 0
    enabled = False
    try:
        with open(args.output, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for phase, duration, speed in phases:
                phase_start = time.monotonic()
                phase_end = phase_start + duration
                next_tick = phase_start
                while time.monotonic() < phase_end and not stopping:
                    now = time.monotonic()
                    phase_time = now - phase_start
                    if speed is None:
                        offset = 0.0
                        target_rate = 0.0
                    else:
                        frequency = speed / (4.0 * args.amplitude_rad)
                        angle = 2.0 * math.pi * frequency * phase_time
                        offset = (
                            2.0 * args.amplitude_rad / math.pi
                            * math.asin(math.sin(angle))
                        )
                        target_rate = speed if math.cos(angle) >= 0.0 else -speed
                    targets = [center + offset for center in centers]
                    commands = [
                        {
                            "motor_id": mid,
                            "position": target,
                            "speed": 0.0,
                            "torque": 0.0,
                            "kp": args.kp,
                            "kd": args.kd,
                        }
                        for mid, target in zip(online_ids, targets)
                    ]
                    request_ns = time.monotonic_ns()
                    response = client.request(
                        "POST",
                        "/api/rs04/diagnostic_motion_batch_fast",
                        {
                            "items": commands,
                            "enable_first": not enabled,
                            "stop_first": False,
                        },
                    )
                    response_ns = time.monotonic_ns()
                    enabled = True
                    state_by_id = {
                        int(item["can_id"]): item for item in response["states"]
                    }
                    row = [
                        sample_index, request_ns, response_ns, phase, phase_time,
                        target_rate, (response_ns - request_ns) / 1.0e6,
                        response["spi_send_ms"], response["total_ms"],
                        response["cache_age_ms"], response["board_a_seq"],
                        response["board_b_seq"],
                    ]
                    for mid, target in zip(online_ids, targets):
                        motor = state_by_id[mid]
                        row.extend((
                            target, motor["angle"], motor["speed"], motor["torque"],
                            motor["age_ms"], motor["mode_state"],
                            motor["error_code"], motor["temp"],
                        ))
                    writer.writerow(row)
                    sample_index += 1
                    next_tick += period_s
                    delay = next_tick - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    else:
                        next_tick = time.monotonic()
            stream.flush()
    finally:
        try:
            stopped = client.request("POST", "/api/stop", {})
            if not stopped.get("verified_stopped"):
                raise RuntimeError("STOP was not verified by motor state readback")
            print(
                f"STOP verified for all online motors "
                f"after {stopped['attempts']} attempt(s)",
                flush=True,
            )
        finally:
            client.close()

    print(f"output={args.output}")
    print(f"online_ids={[hex(mid) for mid in online_ids]}")
    print(f"samples={sample_index}")


if __name__ == "__main__":
    main()
