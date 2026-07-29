#!/usr/bin/env python3
"""Read and save RS01 floating-point parameters without writing or enabling."""

from __future__ import annotations

import argparse
import json
import time

from collect_rs01_subset_identification import (
    ALL_MOTOR_IDS,
    JsonHttpClient,
    parse_id_list,
)


PARAMETERS = {
    "limit_torque_nm": 0x700B,
    "current_loop_kp": 0x7010,
    "current_loop_ki": 0x7011,
    "current_filter_gain": 0x7014,
    "csp_speed_limit_rad_s": 0x7017,
    "current_limit_a_peak": 0x7018,
    "vbus_v": 0x701C,
    "position_loop_kp": 0x701E,
    "speed_loop_kp": 0x701F,
    "speed_loop_ki": 0x7020,
    "speed_filter_gain": 0x7021,
    "speed_mode_accel_rad_s2": 0x7022,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-offline", default="0x11,0x41")
    args = parser.parse_args()

    expected_offline = set(parse_id_list(args.expected_offline))
    client = JsonHttpClient(args.base_url, timeout=3.0)
    try:
        state = client.request("GET", "/api/state")
        online_ids = tuple(
            mid for mid in ALL_MOTOR_IDS
            if bool(state.get(hex(mid), {}).get("online", False))
        )
        offline_ids = set(ALL_MOTOR_IDS) - set(online_ids)
        if offline_ids != expected_offline:
            raise RuntimeError(
                f"offline set mismatch: expected={sorted(expected_offline)} "
                f"actual={sorted(offline_ids)}"
            )
        for mid in online_ids:
            if int(state[hex(mid)].get("mode_state", 0)) != 0:
                raise RuntimeError(f"motor 0x{mid:02X} is not stopped")

        output = {
            "collected_unix_s": time.time(),
            "read_only": True,
            "online_motor_ids": [hex(mid) for mid in online_ids],
            "offline_motor_ids": [hex(mid) for mid in sorted(offline_ids)],
            "parameter_indices": {
                name: hex(index) for name, index in PARAMETERS.items()
            },
            "motors": {},
        }
        for mid in online_ids:
            values = {}
            for name, index in PARAMETERS.items():
                last_error = None
                for attempt in range(5):
                    try:
                        response = client.request(
                            "GET",
                            f"/api/rs04/read_param_f32?motor_id={mid}&index={index}",
                        )
                        values[name] = float(response["value"])
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        time.sleep(0.03 * (attempt + 1))
                if last_error is not None:
                    raise RuntimeError(
                        f"failed reading {name} from 0x{mid:02X}: {last_error}"
                    )
            output["motors"][hex(mid)] = values
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2)
            stream.write("\n")
        print(json.dumps(output, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
