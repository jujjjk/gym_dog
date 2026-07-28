#!/usr/bin/env python3
"""Sim2real CSV diagnosis: policy contract vs real motor tracking.

Columns use two semantic spaces on purpose:
  - policy_joint_name / *_policy_*  : ONNX training contract (FL,FR,RL,RR order)
  - joint_name / q_*_real           : real motor order (FR,FL,RL,RR) via mapper

Do not compare policy angles to real angles without JointSemanticMapper.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_policy_debug_csv import (  # noqa: E402
    JointStats,
    amp,
    as_float,
    load_rows,
    mean,
    pct,
    print_default_alignment,
    print_diagnosis,
    print_joint_table,
    print_type_table,
)


def segment_rows(rows):
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row.get("mode", "policy") or "policy"].append(row)
    return dict(by_mode)


def summarize_pd_gains(rows):
    by_type = defaultdict(lambda: {"kp": set(), "kd": set()})
    for row in rows:
        joint_type = row.get("joint_type", "")
        if not joint_type:
            continue
        kp = as_float(row, "kp")
        kd = as_float(row, "kd")
        if math.isfinite(kp):
            by_type[joint_type]["kp"].add(round(kp, 6))
        if math.isfinite(kd):
            by_type[joint_type]["kd"].add(round(kd, 6))
    if not by_type:
        return
    print("\nEffective real-motor PD gains")
    for joint_type in ("hip", "thigh", "calf"):
        values = by_type.get(joint_type)
        if not values:
            continue
        kp_text = ",".join(str(v) for v in sorted(values["kp"]))
        kd_text = ",".join(str(v) for v in sorted(values["kd"]))
        print(f"- {joint_type}: kp={kp_text} kd={kd_text}")


def summarize_target_clipping_and_velocity(rows):
    by_joint = defaultdict(list)
    by_type_velocity = defaultdict(list)
    for row in rows:
        name = row.get("policy_joint_name") or row.get("joint_name", "")
        if name:
            by_joint[name].append(row)
        joint_type = row.get("joint_type", "")
        if joint_type:
            by_type_velocity[joint_type].append(abs(as_float(row, "dq_current_real")))

    print("\nDeployment target clipping")
    found = False
    for name, items in sorted(by_joint.items()):
        clipped = 0
        max_delta = 0.0
        for row in items:
            requested = (
                as_float(row, "q_default_policy")
                + as_float(row, "gait_ref_policy")
                + as_float(row, "rl_action_contrib_policy")
            )
            deployed = as_float(row, "q_raw_target_policy_abs")
            delta = abs(requested - deployed)
            max_delta = max(max_delta, delta)
            clipped += int(delta > 1.0e-4)
        clip_pct = pct(clipped, len(items))
        if clip_pct > 0.0:
            found = True
            print(f"- {name}: clipped={clip_pct:.1f}% max_delta={max_delta:.3f} rad")
    if not found:
        print("- no policy target clipping detected")

    print("\nReal joint velocity stability")
    for joint_type in ("hip", "thigh", "calf"):
        values = by_type_velocity.get(joint_type, [])
        if not values:
            continue
        values = sorted(values)
        p95_index = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
        print(
            f"- {joint_type}: mean_abs={mean(values):.3f} "
            f"p95_abs={values[p95_index]:.3f} max_abs={max(values):.3f} rad/s"
        )


def settled_tail(rows, seconds=2.0):
    """Return only the settled end of a ramp/initialization segment."""
    timed_rows = [(as_float(row, "time"), row) for row in rows]
    timed_rows = [(t, row) for t, row in timed_rows if math.isfinite(t)]
    if not timed_rows:
        return rows
    end_time = max(t for t, _ in timed_rows)
    tail = [row for t, row in timed_rows if t >= end_time - max(seconds, 0.0)]
    return tail or rows


def summarize_gait_contribution(rows):
    if not rows:
        return
    has_gait = "gait_ref_policy" in rows[0]
    if not has_gait:
        print("\n[gait] CSV has no gait_ref_policy column; re-sync mydog_policy and re-record.")
        return

    by_joint = defaultdict(list)
    for row in rows:
        name = row.get("policy_joint_name") or row.get("joint_name", "")
        if name:
            by_joint[name].append(row)

    print("\nSim2real gait vs RL contribution (training policy space)")
    print(
        "policy_joint          type   gait_amp  rl_amp  raw_tgt_amp  final_amp  current_amp  track%"
    )
    print("-" * 95)
    calf_rows = []
    for name, items in sorted(by_joint.items()):
        joint_type = items[0].get("joint_type", "")
        if not joint_type and "_calf_" in name:
            joint_type = "calf"
        gait_amp = amp([as_float(r, "gait_ref_policy") for r in items])
        rl_amp = amp([as_float(r, "rl_action_contrib_policy") for r in items])
        raw_amp = amp([as_float(r, "q_raw_target_minus_default_policy") for r in items])
        final_amp = amp([as_float(r, "q_final_target_minus_default_policy") for r in items])
        current_amp = amp([as_float(r, "q_current_minus_default_policy") for r in items])
        track = 100.0 * current_amp / final_amp if final_amp > 1e-6 else 0.0
        print(
            f"{name:22s} {joint_type:5s} "
            f"{gait_amp:8.3f} {rl_amp:7.3f} {raw_amp:10.3f} "
            f"{final_amp:9.3f} {current_amp:11.3f} {track:6.1f}"
        )
        if joint_type == "calf" or name.endswith("_calf_joint"):
            calf_rows.append((name, gait_amp, rl_amp, final_amp, current_amp, track))

    if calf_rows:
        print("\nCalf verdict")
        for name, gait_amp, rl_amp, final_amp, current_amp, track in calf_rows:
            notes = []
            if gait_amp < 0.05 and rl_amp < 0.05:
                notes.append("策略+步态参考都几乎为 0（检查 /cmd_vel 是否持续发布、global_gait_phase 是否在走）")
            elif final_amp >= 0.08 and current_amp < 0.04:
                notes.append("目标有摆幅但真机反馈几乎不动（电机/力矩/Kp 或 semantic sign）")
            elif final_amp < 0.5 * max(gait_amp + rl_amp, 1e-6):
                notes.append("部署安全层吃掉摆幅（看 rate_limited / final_limited）")
            elif track < 50.0 and final_amp >= 0.08:
                notes.append(f"跟踪偏弱 track={track:.0f}%")
            else:
                notes.append("摆幅与跟踪正常范围内")
            print(f"  - {name}: " + "; ".join(notes))


def summarize_real_motor_calf(rows):
    calf_real = [r for r in rows if r.get("joint_type") == "calf"]
    if not calf_real:
        return
    print("\nReal motor calf (q_target_real vs q_current_real)")
    by_name = defaultdict(list)
    for row in calf_real:
        by_name[row.get("joint_name", "")].append(row)

    print("real_joint           motor  target_amp  current_amp  error_amp  rate%  final_lim%")
    print("-" * 88)
    for name, items in sorted(by_name.items()):
        tgt = amp([as_float(r, "q_target_real") for r in items])
        cur = amp([as_float(r, "q_current_real") for r in items])
        err = amp([as_float(r, "q_error_real") for r in items])
        rate_pct = pct(sum(as_float(r, "rate_limited") for r in items), len(items))
        final_pct = pct(sum(as_float(r, "final_limited_joint_mask") or as_float(r, "torque_limited") for r in items), len(items))
        motor = items[0].get("motor_id", "")
        print(
            f"{name:20s} {motor:5s} {tgt:10.3f} {cur:11.3f} {err:9.3f} "
            f"{rate_pct:5.1f} {final_pct:10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze sim2real policy debug CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--skip-sec", type=float, default=0.0)
    parser.add_argument("--min-cmd-x", type=float, default=0.15)
    parser.add_argument(
        "--walk-only",
        action="store_true",
        help="Only analyze walking rows (cmd_x >= --min-cmd-x).",
    )
    parser.add_argument(
        "--include-startup",
        action="store_true",
        help="Also print startup_stand default alignment.",
    )
    args = parser.parse_args()

    all_rows = load_rows(args.csv_path, args.skip_sec, None, None)
    if not all_rows:
        print("No rows loaded.")
        return 1

    segments = segment_rows(all_rows)
    print(f"csv={args.csv_path}")
    print(
        "segments: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(segments.items()))
    )

    if args.include_startup and segments.get("startup_stand"):
        startup_tail = settled_tail(segments["startup_stand"], seconds=2.0)
        print(
            "\n=== startup_stand settled tail "
            f"(last 2.0 s, rows={len(startup_tail)}) ==="
        )
        print_default_alignment(startup_tail)

    walk_rows = load_rows(
        args.csv_path,
        args.skip_sec,
        args.min_cmd_x if args.walk_only or args.min_cmd_x is not None else None,
        None,
    )
    if not walk_rows:
        print("No walking rows; try lowering --min-cmd-x or drop --walk-only.")
        walk_rows = all_rows

    print(f"\n=== walk analysis rows={len(walk_rows)} ===")
    summarize_pd_gains(walk_rows)
    summarize_target_clipping_and_velocity(walk_rows)
    summarize_gait_contribution(walk_rows)
    summarize_real_motor_calf(walk_rows)

    joints = defaultdict(JointStats)
    for row in walk_rows:
        name = row.get("joint_name", "")
        if name:
            joints[name].add(row)

    summaries = [stats.summary() for _, stats in sorted(joints.items())]
    print(f"\nrows_used={len(walk_rows)} joints={len(summaries)}")
    print_joint_table(summaries)
    print_type_table(summaries)
    print_diagnosis(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
