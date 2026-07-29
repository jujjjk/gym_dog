#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare IsaacLab V4 golden CSV with ROS2 V4 migration dry-run CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_SUFFIX = ("hip", "thigh", "calf")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_SUFFIX)


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def f(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def vec(row, prefixes, names=JOINT_NAMES):
    for prefix in prefixes:
        values = []
        ok = True
        for i, name in enumerate(names):
            candidates = (
                f"{prefix}_{name}",
                f"{prefix}_{i}",
                f"{prefix}_policy_{name}",
            )
            found = None
            for key in candidates:
                if key in row:
                    found = f(row, key)
                    break
            if found is None:
                ok = False
                break
            values.append(found)
        if ok:
            return np.asarray(values, dtype=np.float64)
    return np.full(len(names), np.nan, dtype=np.float64)


def p95max(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 95)), float(np.max(arr))


def row_time(row, index):
    for key in ("relative_time", "elapsed_time", "time_rel"):
        if key in row:
            return f(row, key, float(index) * 0.02)
    if "time" in row:
        return f(row, "time", float(index) * 0.02)
    return float(index) * 0.02


def relative_times(rows):
    if not rows:
        return []
    raw = [row_time(row, i) for i, row in enumerate(rows)]
    t0 = raw[0]
    return [t - t0 for t in raw]


def nearest_row(rows, times, t):
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        return rows[0]
    if idx >= len(rows):
        return rows[-1]
    before = times[idx - 1]
    after = times[idx]
    return rows[idx - 1] if abs(t - before) <= abs(after - t) else rows[idx]


def summarize(name, values, unit=""):
    p95, mx = p95max(values)
    suffix = f" {unit}" if unit else ""
    print(f"{name:32s} p95={p95:.6f}{suffix} max={mx:.6f}{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_csv", required=True, help="IsaacLab V4 golden CSV")
    parser.add_argument("--ros_csv", required=True, help="ROS2 dry-run CSV from fanfan_cpg_vmc_v4_migration_node")
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    sim_rows = read_csv(Path(args.sim_csv).expanduser())
    ros_rows = read_csv(Path(args.ros_csv).expanduser())
    sim_times = relative_times(sim_rows)
    ros_times = relative_times(ros_rows)
    n = min(len(ros_rows), len(sim_rows))
    if args.max_rows > 0:
        n = min(n, args.max_rows)
    if n <= 0:
        raise RuntimeError("No rows to compare.")

    phase_diff = []
    leg_phase_diff = []
    swing_diff = []
    support_diff = []
    q_ref_diff = []
    q_cmd_diff = []
    kp_diff = []
    kd_diff = []
    clearance_diff = []

    for i in range(n):
        r = ros_rows[i]
        s = nearest_row(sim_rows, sim_times, ros_times[i])
        phase_diff.append(abs(f(s, "base_phase", f(s, "phase")) - f(r, "phase", f(r, "base_phase"))))
        for leg in LEG_ORDER:
            leg_phase_diff.append(abs(f(s, f"leg_phase_{leg}") - f(r, f"leg_phase_{leg}")))
            swing_diff.append(abs(f(s, f"swing_mask_{leg}") - f(r, f"swing_mask_{leg}")))
            support_diff.append(abs(f(s, f"support_mask_{leg}") - f(r, f"support_mask_{leg}")))
            clearance_diff.append(
                abs(f(s, f"fk_clearance_ref_{leg}", f(s, f"predicted_foot_height_{leg}")) - f(r, f"fk_clearance_ref_{leg}"))
            )

        sim_q_ref = vec(s, ("q_ref_policy", "q_ref", "q_cmd_raw"))
        ros_q_ref = vec(r, ("q_ref_policy",))
        sim_q_cmd = vec(s, ("q_cmd_final_policy", "q_cmd_final", "processed_actions"))
        ros_q_cmd = vec(r, ("q_cmd_final_policy",))
        sim_kp = vec(s, ("kp",), JOINT_NAMES)
        ros_kp = vec(r, ("kp",), JOINT_NAMES)
        sim_kd = vec(s, ("kd",), JOINT_NAMES)
        ros_kd = vec(r, ("kd",), JOINT_NAMES)

        if np.all(np.isfinite(sim_q_ref)) and np.all(np.isfinite(ros_q_ref)):
            q_ref_diff.extend(np.abs(sim_q_ref - ros_q_ref).tolist())
        if np.all(np.isfinite(sim_q_cmd)) and np.all(np.isfinite(ros_q_cmd)):
            q_cmd_diff.extend(np.abs(sim_q_cmd - ros_q_cmd).tolist())
        if np.all(np.isfinite(sim_kp)) and np.all(np.isfinite(ros_kp)):
            kp_diff.extend(np.abs(sim_kp - ros_kp).tolist())
        if np.all(np.isfinite(sim_kd)) and np.all(np.isfinite(ros_kd)):
            kd_diff.extend(np.abs(sim_kd - ros_kd).tolist())

    print(f"Compared rows: {n}")
    summarize("phase_abs_diff", phase_diff)
    summarize("leg_phase_abs_diff", leg_phase_diff)
    summarize("swing_mask_abs_diff", swing_diff)
    summarize("support_mask_abs_diff", support_diff)
    summarize("q_ref_abs_diff", q_ref_diff, "rad")
    summarize("q_cmd_abs_diff", q_cmd_diff, "rad")
    summarize("kp_abs_diff", kp_diff)
    summarize("kd_abs_diff", kd_diff)
    summarize("fk_clearance_ref_abs_diff", clearance_diff, "m")

    q_ref_p95, _ = p95max(q_ref_diff)
    phase_p95, _ = p95max(phase_diff)
    if math.isfinite(q_ref_p95) and q_ref_p95 > 0.08:
        print("[WARN] q_ref p95 diff is large; check default pose, phase, IK or hip signs.")
    if math.isfinite(phase_p95) and phase_p95 > 0.05:
        print("[WARN] phase p95 diff is large; check dt/start phase alignment.")


if __name__ == "__main__":
    main()
