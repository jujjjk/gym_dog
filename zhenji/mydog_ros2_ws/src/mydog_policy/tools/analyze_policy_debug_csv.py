#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def as_int(row, key, default=0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def amp(values):
    if not values:
        return 0.0
    return max(values) - min(values)


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def pct(count, total):
    return 100.0 * count / max(total, 1)


class JointStats:
    def __init__(self):
        self.rows = []
        self.meta = {}

    def add(self, row):
        self.rows.append(row)
        for key in ("joint_name", "leg_name", "joint_type", "motor_id", "policy_joint_name"):
            self.meta[key] = row.get(key, "")

    @property
    def n(self):
        return len(self.rows)

    def values(self, key):
        return [as_float(row, key) for row in self.rows]

    def count_true(self, key):
        return sum(as_int(row, key) for row in self.rows)

    def limit_pct(self, key):
        return pct(self.count_true(key), self.n)

    def amplitude(self, key):
        return amp(self.values(key))

    def mean_abs(self, key):
        return mean([abs(v) for v in self.values(key)])

    def mean_value(self, key):
        return mean(self.values(key))

    def summary(self):
        raw_amp = self.amplitude("q_raw_target_policy_abs")
        smooth_amp = self.amplitude("q_smooth_target_policy_abs")
        final_amp = self.amplitude("q_final_target_policy_abs")
        current_amp = self.amplitude("q_current_policy_abs")

        raw_to_smooth_loss = max(0.0, raw_amp - smooth_amp)
        smooth_to_final_loss = max(0.0, smooth_amp - final_amp)
        raw_to_final_loss = max(0.0, raw_amp - final_amp)
        current_raw_ratio = current_amp / raw_amp if raw_amp > 1e-6 else 0.0
        current_final_ratio = current_amp / final_amp if final_amp > 1e-6 else 0.0

        torque_measured_abs_max = max([abs(v) for v in self.values("torque_measured")] or [0.0])
        torque_budget = mean(self.values("torque_budget_nm"))
        tau_est_mean = mean(self.values("tau_est_real"))
        motor_vel_cmd_abs_max = max([abs(v) for v in self.values("motor_vel_cmd")] or [0.0])

        return {
            "joint_name": self.meta.get("joint_name", ""),
            "joint_type": self.meta.get("joint_type", ""),
            "raw_amp": raw_amp,
            "smooth_amp": smooth_amp,
            "final_amp": final_amp,
            "current_amp": current_amp,
            "raw_to_smooth_amp_loss": raw_to_smooth_loss,
            "smooth_to_final_amp_loss": smooth_to_final_loss,
            "raw_to_final_amp_loss": raw_to_final_loss,
            "current_raw_ratio": current_raw_ratio,
            "current_final_ratio": current_final_ratio,
            "pre_limited_pct": self.limit_pct("pre_limited_joint_mask") or self.limit_pct("pre_limited"),
            "rate_limited_pct": self.limit_pct("rate_limited_joint_mask") or self.limit_pct("rate_limited"),
            "accel_limited_pct": self.limit_pct("accel_limited_joint_mask") or self.limit_pct("accel_limited"),
            "final_limited_pct": self.limit_pct("final_limited_joint_mask") or self.limit_pct("torque_limited"),
            "raw_action_sat_pct": pct(
                sum(abs(as_float(row, "action_raw_policy")) >= 0.95 for row in self.rows),
                self.n,
            ),
            "torque_measured_abs_max": torque_measured_abs_max,
            "torque_budget_nm": torque_budget,
            "tau_est_mean": tau_est_mean,
            "motor_vel_cmd_abs_max": motor_vel_cmd_abs_max,
            "raw_amp_small": raw_amp < 0.10,
        }


def load_rows(path, skip_sec, min_cmd_x, mode=None):
    rows = []
    first_time = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = as_float(row, "time")
            if first_time is None:
                first_time = t
            if skip_sec > 0.0 and t - first_time < skip_sec:
                continue
            if min_cmd_x is not None and abs(as_float(row, "cmd_x")) < min_cmd_x:
                continue
            if mode and row.get("mode", "") != mode:
                continue
            rows.append(row)
    return rows


def print_default_alignment(rows):
    if not rows:
        return
    joints = defaultdict(list)
    for row in rows:
        name = row.get("policy_joint_name") or row.get("joint_name", "")
        if name:
            joints[name].append(row)

    print("\nDefault Pose Alignment")
    print(
        "policy_joint        real_joint          motor  default current diff_mean abs_mean abs_max q_target q_real tgt-def-real"
    )
    print("-" * 123)
    summaries = []
    for name, items in sorted(joints.items()):
        default = mean([as_float(row, "q_default_policy") for row in items])
        current = mean([as_float(row, "q_current_policy_abs") for row in items])
        diff_values = [as_float(row, "q_current_minus_default_policy") for row in items]
        abs_values = [abs(v) for v in diff_values]
        diff_mean = mean(diff_values)
        abs_mean = mean(abs_values)
        abs_max = max(abs_values or [0.0])
        q_real = mean([as_float(row, "q_current_real") for row in items])
        q_target = mean([as_float(row, "q_target_real") for row in items])
        target_minus_default_real = mean(
            [as_float(row, "target_real_minus_policy_default_real") for row in items]
        )
        real_joint = items[0].get("joint_name", "")
        motor_id = items[0].get("motor_id", "")
        summaries.append((items[0].get("leg_name", ""), items[0].get("joint_type", ""), abs_mean))
        level = "ERROR" if abs_mean > 0.10 or abs_max > 0.10 else "WARN" if abs_mean > 0.05 or abs_max > 0.05 else "OK"
        print(
            f"{name:19s} {real_joint:18s} {motor_id:5s} "
            f"{default:+7.3f} {current:+7.3f} {diff_mean:+9.3f} "
            f"{abs_mean:8.3f} {abs_max:7.3f} {q_target:+8.3f} "
            f"{q_real:+7.3f} {target_minus_default_real:+12.3f} {level}"
        )

    print("\nDefault alignment by joint type")
    by_type = defaultdict(list)
    by_leg = defaultdict(list)
    for leg, joint_type, abs_mean in summaries:
        by_type[joint_type].append(abs_mean)
        by_leg[leg].append(abs_mean)
    for joint_type in ("hip", "thigh", "calf"):
        if by_type.get(joint_type):
            print(f"- {joint_type}: mean_abs_error={mean(by_type[joint_type]):.4f} rad")
    print("Default alignment by leg")
    for leg in ("FR", "FL", "RR", "RL"):
        if by_leg.get(leg):
            print(f"- {leg}: mean_abs_error={mean(by_leg[leg]):.4f} rad")


def print_joint_table(summaries):
    print(
        "joint                 type   raw  smooth final current "
        "loss_pre loss_final loss_all keep% cur/final% pre% rate% accel% final% tau_meas budget vcmd rawsat%"
    )
    print("-" * 148)
    for s in summaries:
        print(
            f"{s['joint_name']:20s} {s['joint_type']:5s} "
            f"{s['raw_amp']:5.3f} "
            f"{s['smooth_amp']:6.3f} "
            f"{s['final_amp']:5.3f} "
            f"{s['current_amp']:7.3f} "
            f"{s['raw_to_smooth_amp_loss']:8.3f} "
            f"{s['smooth_to_final_amp_loss']:10.3f} "
            f"{s['raw_to_final_amp_loss']:8.3f} "
            f"{100.0 * s['current_raw_ratio']:5.1f} "
            f"{100.0 * s['current_final_ratio']:10.1f} "
            f"{s['pre_limited_pct']:5.1f} "
            f"{s['rate_limited_pct']:5.1f} "
            f"{s['accel_limited_pct']:6.1f} "
            f"{s['final_limited_pct']:6.1f} "
            f"{s['torque_measured_abs_max']:8.2f} "
            f"{s['torque_budget_nm']:6.2f} "
            f"{s['motor_vel_cmd_abs_max']:4.2f} "
            f"{s['raw_action_sat_pct']:7.1f}"
        )


def print_type_table(summaries):
    groups = defaultdict(list)
    for s in summaries:
        groups[s["joint_type"]].append(s)

    print("\nBy joint type")
    print(
        "type    raw  smooth final current loss_pre loss_final loss_all keep% cur/final% pre% rate% accel% final%"
    )
    print("-" * 116)
    for joint_type in ("hip", "thigh", "calf"):
        items = groups.get(joint_type, [])
        if not items:
            continue
        avg = lambda key: mean([s[key] for s in items])
        print(
            f"{joint_type:6s} "
            f"{avg('raw_amp'):5.3f} "
            f"{avg('smooth_amp'):6.3f} "
            f"{avg('final_amp'):5.3f} "
            f"{avg('current_amp'):7.3f} "
            f"{avg('raw_to_smooth_amp_loss'):8.3f} "
            f"{avg('smooth_to_final_amp_loss'):10.3f} "
            f"{avg('raw_to_final_amp_loss'):8.3f} "
            f"{100.0 * avg('current_raw_ratio'):5.1f} "
            f"{100.0 * avg('current_final_ratio'):10.1f} "
            f"{avg('pre_limited_pct'):5.1f} "
            f"{avg('rate_limited_pct'):5.1f} "
            f"{avg('accel_limited_pct'):6.1f} "
            f"{avg('final_limited_pct'):6.1f}"
        )


def print_diagnosis(summaries):
    print("\nDiagnosis")
    for s in summaries:
        notes = []
        if s["raw_amp"] >= 0.20 and s["final_amp"] < 0.65 * s["raw_amp"]:
            notes.append("raw large but final small: deployment safety is eating amplitude")
        if s["raw_amp"] < 0.10:
            notes.append("raw amplitude is small: ONNX/obs is not asking this joint to move")
        if s["final_amp"] > 0.10 and s["current_amp"] < 0.65 * s["final_amp"]:
            notes.append(
                "final target is available but motor tracking is weak; check velocity feedforward, "
                "kd, current limit, power supply, or mechanical load"
            )
        if s["rate_limited_pct"] > 70.0:
            notes.append("rate_limited very high: target velocity limit too conservative")
        if s["accel_limited_pct"] > 70.0:
            notes.append("accel_limited very high: target acceleration limit too conservative")
        if s["final_limited_pct"] > 30.0:
            notes.append("final_limited high: torque safety is still heavily clamping")
        if (
            s["final_limited_pct"] > 30.0
            and s["torque_budget_nm"] > 0.0
            and s["torque_measured_abs_max"] < 0.6 * s["torque_budget_nm"]
        ):
            notes.append("measured torque is below budget but final limit is high: tau_est may be conservative")
        if notes:
            print(f"- {s['joint_name']}: " + "; ".join(notes))

    type_groups = defaultdict(list)
    for s in summaries:
        type_groups[s["joint_type"]].append(s)
    for joint_type in ("hip", "thigh", "calf"):
        items = type_groups.get(joint_type, [])
        if not items:
            continue
        raw = mean([s["raw_amp"] for s in items])
        current = mean([s["current_amp"] for s in items])
        final = mean([s["final_amp"] for s in items])
        print(
            f"- {joint_type}: current/raw={current / raw if raw > 1e-6 else 0.0:.2f}, "
            f"final/raw={final / raw if raw > 1e-6 else 0.0:.2f}, "
            f"current/final={current / final if final > 1e-6 else 0.0:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--skip-sec", type=float, default=0.0)
    parser.add_argument(
        "--min-cmd-x",
        type=float,
        default=None,
        help="Only analyze rows with abs(cmd_x) >= this value.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Only analyze rows with this mode, e.g. policy, stand_only, joint_probe.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_path, args.skip_sec, args.min_cmd_x, args.mode)
    if not rows:
        print("No rows loaded. Check csv_path, --skip-sec, or --min-cmd-x.")
        return

    joints = defaultdict(JointStats)
    for row in rows:
        name = row.get("joint_name", "")
        if name:
            joints[name].add(row)

    summaries = [stats.summary() for name, stats in sorted(joints.items())]
    print(f"rows_used={len(rows)} joints={len(summaries)}")
    if args.mode in ("stand_only", "joint_probe") or any(row.get("mode") for row in rows):
        print_default_alignment(rows)
    print_joint_table(summaries)
    print_type_table(summaries)
    print_diagnosis(summaries)


if __name__ == "__main__":
    main()
