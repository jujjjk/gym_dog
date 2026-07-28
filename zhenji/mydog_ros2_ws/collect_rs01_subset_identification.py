#!/usr/bin/env python3
"""Identify the currently-online unloaded RS01 subset through the Jetson path.

The production 12-motor policy endpoint is not changed or relaxed.  This tool
uses a restricted diagnostic endpoint that requires every currently-online
motor to be included and to have verified torque/current limits.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import os
import signal
import statistics
import time
from urllib.parse import urlsplit


ALL_MOTOR_IDS = (
    0x11, 0x12, 0x13, 0x21, 0x22, 0x23,
    0x31, 0x32, 0x33, 0x41, 0x42, 0x43,
)


def parse_id_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip(), 0) for part in value.split(",") if part.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-offline", default="0x11,0x41")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--amplitude-rad", type=float, default=0.01)
    parser.add_argument("--frequencies-hz", default="0.25,0.5,0.7,1.0,1.5,2.0")
    parser.add_argument("--cycles-per-frequency", type=float, default=4.0)
    parser.add_argument("--min-sine-duration-s", type=float, default=4.0)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.2)
    return parser.parse_args()


class JsonHttpClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("base-url must be an http URL")
        self.prefix = parsed.path.rstrip("/")
        self.connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=timeout
        )

    def request(self, method: str, path: str, body=None):
        payload = None
        headers = {}
        if body is not None:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        self.connection.request(method, self.prefix + path, payload, headers)
        response = self.connection.getresponse()
        raw = response.read()
        if response.status >= 300:
            raise RuntimeError(
                f"HTTP {response.status} {path}: {raw.decode('utf-8', 'replace')}"
            )
        return json.loads(raw)

    def close(self):
        self.connection.close()


def safety_item(mid: int) -> dict:
    return {
        "motor_id": mid,
        "torque_limit_nm": 17.0,
        "current_limit_amp": 23.0,
    }


def main():
    args = parse_args()
    expected_offline = set(parse_id_list(args.expected_offline))
    frequencies = parse_float_list(args.frequencies_hz)
    if not 10.0 <= args.rate_hz <= 100.0:
        raise SystemExit("rate-hz must be within [10, 100]")
    if not 0.0 < args.amplitude_rad <= 0.02:
        raise SystemExit("amplitude-rad must be within (0, 0.02]")
    if args.kp * args.amplitude_rad > 1.25:
        raise SystemExit("kp * amplitude-rad must not exceed 1.25 Nm")
    if not frequencies or any(value <= 0.0 or value > 3.0 for value in frequencies):
        raise SystemExit("frequencies must be within (0, 3] Hz")

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
    if not verified.get("verified") or int(verified.get("count", 0)) != len(online_ids):
        raise RuntimeError("diagnostic safety limit verification failed")

    samples = []
    for _ in range(10):
        state = client.request("GET", "/api/state")
        samples.append([float(state[hex(mid)]["angle"]) for mid in online_ids])
        time.sleep(0.02)
    centers = [
        statistics.median(row[index] for row in samples)
        for index in range(len(online_ids))
    ]

    phases = [
        ("hold", 1.0, None),
        ("step_pos", 1.5, None),
        ("center_1", 0.75, None),
        ("step_neg", 1.5, None),
        ("center_2", 0.75, None),
    ]
    for frequency in frequencies:
        duration = max(
            args.min_sine_duration_s,
            args.cycles_per_frequency / frequency,
        )
        phases.append((f"sine_{frequency:g}hz", duration, frequency))
        phases.append((f"center_after_{frequency:g}hz", 0.5, None))
    phases.append(("final_hold", 0.75, None))

    headers = [
        "sample", "host_request_ns", "host_response_ns", "phase",
        "phase_time_s", "loop_dt_ms", "loop_lateness_ms", "rpc_ms",
        "server_spi_ms", "server_total_ms", "cache_age_ms",
        "board_a_seq", "board_b_seq",
    ]
    for mid in online_ids:
        prefix = f"m{mid:02x}"
        headers.extend((
            f"{prefix}_target_rad", f"{prefix}_q_rad",
            f"{prefix}_dq_rad_s", f"{prefix}_torque_nm",
            f"{prefix}_age_ms", f"{prefix}_mode", f"{prefix}_fault",
            f"{prefix}_snapshot_seq", f"{prefix}_board_tick_ms",
            f"{prefix}_temperature_c",
        ))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    period = 1.0 / args.rate_hz
    sample_index = 0
    last_response_time = None
    enabled = False
    try:
        with open(args.output, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for phase, duration, frequency in phases:
                phase_start = time.monotonic()
                phase_end = phase_start + duration
                next_tick = phase_start
                while time.monotonic() < phase_end and not stopping:
                    now = time.monotonic()
                    phase_time = now - phase_start
                    if phase == "step_pos":
                        offset = args.amplitude_rad
                    elif phase == "step_neg":
                        offset = -args.amplitude_rad
                    elif frequency is not None:
                        offset = args.amplitude_rad * math.sin(
                            2.0 * math.pi * frequency * phase_time
                        )
                    else:
                        offset = 0.0
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
                    loop_dt_ms = (
                        0.0 if last_response_time is None
                        else (response_ns - last_response_time) / 1.0e6
                    )
                    last_response_time = response_ns
                    state_by_id = {
                        int(item["can_id"]): item for item in response["states"]
                    }
                    row = [
                        sample_index, request_ns, response_ns, phase, phase_time,
                        loop_dt_ms, max(0.0, (now - next_tick) * 1000.0),
                        (response_ns - request_ns) / 1.0e6,
                        response["spi_send_ms"], response["total_ms"],
                        response["cache_age_ms"], response["board_a_seq"],
                        response["board_b_seq"],
                    ]
                    for mid, target in zip(online_ids, targets):
                        motor = state_by_id[mid]
                        row.extend((
                            target, motor["angle"], motor["speed"],
                            motor["torque"], motor["age_ms"],
                            motor["mode_state"], motor["error_code"],
                            motor["snapshot_seq"], motor["board_tick_ms"],
                            motor["temp"],
                        ))
                    writer.writerow(row)
                    sample_index += 1
                    next_tick += period
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
