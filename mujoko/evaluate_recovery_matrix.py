"""Screen policies for saturation exit and symmetric roll recovery."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from sim2sim import Sim, body_linear_velocity


SPEEDS = (0.12, 0.20, 0.35)
ROLLS_DEG = (3.0, 5.0, 8.0)


def attitude(quaternion_wxyz):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion_wxyz)
    matrix = matrix.reshape(3, 3)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    return float(roll), float(pitch)


def foot_geometries(sim):
    result = []
    for leg in ("FL", "FR", "RL", "RR"):
        body = mujoco.mj_name2id(
            sim.m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf"
        )
        candidates = [
            geom
            for geom in range(
                int(sim.m.body_geomadr[body]),
                int(sim.m.body_geomadr[body] + sim.m.body_geomnum[body]),
            )
            if sim.m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one foot geometry for {leg}")
        result.append(candidates[0])
    return result


def scenarios():
    for speed in SPEEDS:
        for degrees in ROLLS_DEG:
            for sign in (-1.0, 1.0):
                yield {
                    "name": f"roll_{sign * degrees:+.0f}_vx{speed:.2f}",
                    "speed": speed,
                    "kind": "roll",
                    "sign": sign,
                    "roll_deg": degrees,
                    "disturbance_time": 0.0,
                }
        for sign in (-1.0, 1.0):
            yield {
                "name": f"push_{sign:+.0f}_vx{speed:.2f}",
                "speed": speed,
                "kind": "push",
                "sign": sign,
                "roll_deg": 0.0,
                "disturbance_time": 2.0,
            }
    for sign in (-1.0, 1.0):
        yield {
            "name": f"saturated_{sign:+.0f}_vx0.20",
            "speed": 0.20,
            "kind": "saturated",
            "sign": sign,
            "roll_deg": 3.0,
            "disturbance_time": 0.0,
        }


def first_stable_recovery(times, roll, roll_rate, disturbance_time):
    # Recovery means re-entering the normal gait envelope.  A trotting body
    # has a non-zero instantaneous roll rate, so the earlier 0.30 rad/s test
    # incorrectly reported infinity even while roll stayed below three degrees.
    stable = (np.abs(roll) < np.deg2rad(3.0)) & (np.abs(roll_rate) < 1.00)
    required = 10  # 0.20 s at 50 Hz
    start = int(np.searchsorted(times, disturbance_time + 0.05))
    for index in range(start, len(times) - required + 1):
        if np.all(stable[index:index + required]):
            return float(times[index] - disturbance_time)
    return float("inf")


def first_symmetric_reentry(
        times, roll, diagonal_error, contact_mismatch, disturbance_time):
    """First stable diagonal-cycle window after the disturbance."""
    required = 20  # 0.40 s at the 50 Hz policy rate
    start = int(np.searchsorted(times, disturbance_time + 0.05))
    for index in range(start, len(times) - required + 1):
        window = slice(index, index + required)
        if (
            np.max(np.abs(roll[window])) < np.deg2rad(3.0)
            and np.mean(diagonal_error[window]) < 0.025
            and np.mean(contact_mismatch[window]) < 0.20
        ):
            return float(times[index] - disturbance_time)
    return float("inf")


def evaluate_scenario(policy, model, scenario, duration):
    sim = Sim(model, policy, (scenario["speed"], 0.0, 0.0))
    feet = foot_geometries(sim)
    sign = float(scenario["sign"])
    initial_roll = np.deg2rad(float(scenario["roll_deg"])) * sign
    if initial_roll != 0.0:
        sim.d.qpos[3:7] = [
            np.cos(initial_roll / 2.0),
            np.sin(initial_roll / 2.0),
            0.0,
            0.0,
        ]
    if scenario["kind"] == "saturated":
        pattern = np.asarray([
            1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1
        ], dtype=np.float32)
        sim.action[:] = 0.95 * sign * pattern
        sim.gait_phase = 0.37 if sign > 0.0 else 0.83
    mujoco.mj_forward(sim.m, sim.d)

    physics_steps = int(round(duration / sim.m.opt.timestep))
    pushed = False
    samples = []
    for step in range(physics_steps):
        time_s = step * sim.m.opt.timestep
        if (
            scenario["kind"] == "push"
            and not pushed
            and time_s >= scenario["disturbance_time"]
        ):
            # Equal magnitude in both directions, matching the training impulse.
            sim.d.qvel[1] += 0.22 * sign
            sim.d.qvel[3] += 0.25 * sign
            pushed = True
        control_step = step % sim.decimation == 0
        if control_step:
            sim.policy()
        raw_torque = sim.step()
        if not control_step:
            continue
        roll, pitch = attitude(sim.d.qpos[3:7])
        body_velocity = body_linear_velocity(sim.d.qpos[3:7], sim.d.qvel[:3])
        foot_height = np.asarray([sim.d.geom_xpos[index, 2] for index in feet])
        contact = np.zeros(4, dtype=bool)
        for contact_index in range(sim.d.ncon):
            item = sim.d.contact[contact_index]
            for foot_index, geom in enumerate(feet):
                if item.geom1 == geom or item.geom2 == geom:
                    contact[foot_index] = True
        samples.append((
            time_s,
            roll,
            pitch,
            float(sim.d.qvel[3]),
            float(body_velocity[0]),
            float(body_velocity[1]),
            float(np.max(np.abs(sim.policy_action))),
            float(np.mean(np.abs(sim.policy_action) > 0.95)),
            bool(np.any(np.abs(sim.policy_action) > 0.95)),
            float(np.max(np.abs(raw_torque))),
            float(abs(foot_height[0] - foot_height[3])
                  + abs(foot_height[1] - foot_height[2])) / 2.0,
            float((contact[0] != contact[3]) + (contact[1] != contact[2])) / 2.0,
            float(sim.d.qpos[2]),
        ))

    values = np.asarray(samples, dtype=np.float64)
    time_s = values[:, 0]
    roll = values[:, 1]
    disturbance_time = float(scenario["disturbance_time"])
    post = time_s >= disturbance_time
    post_grace = time_s >= disturbance_time + 0.25
    settled = time_s >= disturbance_time + 2.0
    recovery = first_stable_recovery(
        time_s, roll, values[:, 3], disturbance_time
    )
    symmetric_reentry = first_symmetric_reentry(
        time_s, roll, values[:, 10], values[:, 11], disturbance_time
    )
    opposite = sign * roll < 0.0
    reverse_overshoot = np.max(np.abs(roll[post & opposite]), initial=0.0)
    fallen = bool(np.min(values[:, 12]) < 0.20 or np.max(np.abs(roll)) > 0.80)
    return {
        "policy": str(Path(policy).resolve()),
        "scenario": scenario["name"],
        "kind": scenario["kind"],
        "cmd_vx": scenario["speed"],
        "direction": int(sign),
        "initial_roll_deg": scenario["roll_deg"] * sign,
        "fallen": int(fallen),
        "peak_roll_deg": np.rad2deg(np.max(np.abs(roll[post]))),
        "peak_roll_after_025_deg": np.rad2deg(np.max(np.abs(roll[post_grace]))),
        "recovery_time_s": recovery,
        "symmetric_reentry_time_s": symmetric_reentry,
        "reverse_overshoot_deg": np.rad2deg(reverse_overshoot),
        "steady_vx": np.mean(values[settled, 4]),
        "steady_vy_abs": np.mean(np.abs(values[settled, 5])),
        "action_max": np.max(values[:, 6]),
        "action_element_over_095_ratio": np.mean(values[:, 7]),
        "action_cycle_over_095_ratio": np.mean(values[:, 8]),
        "torque_abs_max": np.max(values[:, 9]),
        "settled_diagonal_height_error": np.mean(values[settled, 10]),
        "settled_contact_mismatch_ratio": np.mean(values[settled, 11]),
        "min_height": np.min(values[:, 12]),
    }


def mirror_gaps(rows):
    grouped = {}
    for row in rows:
        key = (
            row["kind"],
            float(row["cmd_vx"]),
            abs(float(row["initial_roll_deg"])),
        )
        grouped.setdefault(key, {})[int(row["direction"])] = row
    result = []
    for key, pair in grouped.items():
        if -1 not in pair or 1 not in pair:
            continue
        left, right = pair[-1], pair[1]
        result.append({
            "key": key,
            "roll_gap_deg": abs(
                left["peak_roll_after_025_deg"]
                - right["peak_roll_after_025_deg"]
            ),
            "overshoot_gap_deg": abs(
                left["reverse_overshoot_deg"]
                - right["reverse_overshoot_deg"]
            ),
            "recovery_gap_s": abs(
                left["recovery_time_s"] - right["recovery_time_s"]
            ),
            "steady_vx_gap": abs(left["steady_vx"] - right["steady_vx"]),
        })
    return result


def qualifies(rows):
    if any(row["fallen"] for row in rows):
        return False
    if max(row["peak_roll_after_025_deg"] for row in rows) >= 6.0:
        return False
    if max(row["recovery_time_s"] for row in rows) >= 1.0:
        return False
    if max(row["reverse_overshoot_deg"] for row in rows) >= 3.0:
        return False
    if max(row["action_cycle_over_095_ratio"] for row in rows) >= 0.05:
        return False
    if max(row["symmetric_reentry_time_s"] for row in rows) >= 2.0:
        return False
    if any(row["steady_vx"] < 0.45 * row["cmd_vx"] for row in rows):
        return False
    gaps = mirror_gaps(rows)
    return all(
        gap["roll_gap_deg"] < 1.5
        and gap["overshoot_gap_deg"] < 1.5
        and gap["recovery_gap_s"] < 0.5
        and gap["steady_vx_gap"] < 0.06
        for gap in gaps
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+", type=Path)
    parser.add_argument(
        "--model", type=Path, default=root / "models/fanfan_scene.xml"
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for policy in args.policies:
        for scenario in scenarios():
            rows.append(evaluate_scenario(
                policy, args.model, scenario, args.duration
            ))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for policy in args.policies:
        selected = [r for r in rows if r["policy"] == str(policy.resolve())]
        recovery_max = max(r["recovery_time_s"] for r in selected)
        reentry_max = max(r["symmetric_reentry_time_s"] for r in selected)
        gaps = mirror_gaps(selected)
        print(
            f"{policy}: falls={sum(r['fallen'] for r in selected)}, "
            f"roll_after025_max={max(r['peak_roll_after_025_deg'] for r in selected):.3f}deg, "
            f"recovery_max={recovery_max:.3f}s, "
            f"overshoot_max={max(r['reverse_overshoot_deg'] for r in selected):.3f}deg, "
            f"sat_cycle_max={max(r['action_cycle_over_095_ratio'] for r in selected):.3%}, "
            f"symmetry_reentry_max={reentry_max:.3f}s, "
            f"mirror_roll_gap_max={max((g['roll_gap_deg'] for g in gaps), default=float('inf')):.3f}deg, "
            f"qualified={qualifies(selected)}"
        )
    print(f"csv={args.csv.resolve()}")


if __name__ == "__main__":
    main()
