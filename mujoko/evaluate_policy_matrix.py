"""Deterministic continuous-command MuJoCo evaluation with CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from sim2sim import Sim, body_linear_velocity


COMMANDS = (
    ("stand", 0.00, 0.00, 0.00),
    ("forward_low", 0.10, 0.00, 0.00),
    ("forward", 0.40, 0.00, 0.00),
    ("backward", -0.10, 0.00, 0.00),
    ("left", 0.00, 0.06, 0.00),
    ("right", 0.00, -0.06, 0.00),
    ("yaw_left", 0.00, 0.00, 0.60),
    ("yaw_right", 0.00, 0.00, -0.60),
    ("diagonal", 0.25, 0.06, 0.00),
    ("arc", 0.25, 0.00, 0.50),
)


def attitude(quaternion_wxyz):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion_wxyz)
    matrix = matrix.reshape(3, 3)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return float(roll), float(pitch), float(yaw)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate(policy, model, segment_s, warmup_s):
    sim = Sim(model, policy)
    foot_geoms = []
    for leg in ("FL", "FR", "RL", "RR"):
        body_id = mujoco.mj_name2id(
            sim.m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf"
        )
        first = int(sim.m.body_geomadr[body_id])
        count = int(sim.m.body_geomnum[body_id])
        candidates = [
            geom_id for geom_id in range(first, first + count)
            if sim.m.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one foot sphere for {leg}, got {candidates}")
        foot_geoms.append(candidates[0])
    physics_dt = float(sim.m.opt.timestep)
    steps_per_segment = int(round(segment_s / physics_dt))
    warmup_steps = int(round(warmup_s / physics_dt))
    rows = []
    previous_target = sim.target.copy()
    previous_target_velocity = np.zeros(12, dtype=np.float32)

    for command_index, (name, vx, vy, yaw_rate) in enumerate(COMMANDS):
        sim.command[:] = (vx, vy, yaw_rate)
        values = {
            key: [] for key in (
                "vx_error", "vy_error", "yaw_error", "roll", "pitch",
                "vx", "vy", "yaw_rate",
                "height", "raw_torque_mean", "raw_torque_max",
                "torque_sat", "target_speed", "target_accel",
                "sagittal_diagonal_error",
                "mid_swing_clearance", "diagonal_foot_height_error",
                "contact_phase_mismatch", "left_right_clearance_error",
            )
        }
        fallen = False
        start_position = sim.d.qpos[:2].copy()
        start_yaw = attitude(sim.d.qpos[3:7])[2]

        for local_step in range(steps_per_segment):
            if local_step % sim.decimation == 0:
                sim.policy()
                target_velocity = (
                    sim.target - previous_target
                ) / (physics_dt * sim.decimation)
                target_acceleration = (
                    target_velocity - previous_target_velocity
                ) / (physics_dt * sim.decimation)
                previous_target = sim.target.copy()
                previous_target_velocity = target_velocity.copy()
            raw_torque = sim.step()

            if local_step < warmup_steps or local_step % sim.decimation != 0:
                continue
            quat = sim.d.qpos[3:7]
            linear = body_linear_velocity(quat, sim.d.qvel[:3])
            angular = sim.d.qvel[3:6]
            roll, pitch, _ = attitude(quat)
            absolute_torque = np.abs(raw_torque)
            target_delta = sim.target - sim.default
            diagonal_error = np.mean(np.abs([
                target_delta[1] - target_delta[10],
                target_delta[2] - target_delta[11],
                target_delta[4] - target_delta[7],
                target_delta[5] - target_delta[8],
            ]))
            values["vx_error"].append(abs(float(linear[0]) - vx))
            values["vy_error"].append(abs(float(linear[1]) - vy))
            values["yaw_error"].append(abs(float(angular[2]) - yaw_rate))
            values["vx"].append(float(linear[0]))
            values["vy"].append(float(linear[1]))
            values["yaw_rate"].append(float(angular[2]))
            values["roll"].append(abs(roll))
            values["pitch"].append(abs(pitch))
            values["height"].append(float(sim.d.qpos[2]))
            values["raw_torque_mean"].append(float(absolute_torque.mean()))
            values["raw_torque_max"].append(float(absolute_torque.max()))
            values["torque_sat"].append(float(np.mean(
                absolute_torque >= 0.98 * sim.limits
            )))
            values["target_speed"].append(float(np.max(np.abs(target_velocity))))
            values["target_accel"].append(float(np.max(np.abs(target_acceleration))))
            values["sagittal_diagonal_error"].append(float(diagonal_error))
            foot_height = sim.d.geom_xpos[foot_geoms, 2]
            phase = np.asarray([
                (sim.gait_phase + sim.gait["phase_offsets"][leg]) % 1.0
                for leg in ("FL", "FR", "RL", "RR")
            ])
            stance_ratio = float(sim.gait["stance_ratio"])
            swing_progress = np.clip(
                (phase - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0
            )
            mid_swing = (
                (phase >= stance_ratio)
                & (swing_progress >= 0.20)
                & (swing_progress <= 0.80)
            )
            if np.any(mid_swing):
                values["mid_swing_clearance"].extend(
                    foot_height[mid_swing].tolist()
                )
            values["diagonal_foot_height_error"].append(float(
                abs(foot_height[0] - foot_height[3])
                + abs(foot_height[1] - foot_height[2])
            ) / 2.0)
            observed_contact = foot_height <= 0.0205
            desired_contact = phase < stance_ratio
            values["contact_phase_mismatch"].append(float(np.mean(
                observed_contact != desired_contact
            )))
            values["left_right_clearance_error"].append(float(abs(
                0.5 * (foot_height[0] + foot_height[2])
                - 0.5 * (foot_height[1] + foot_height[3])
            )))
            fallen = fallen or bool(
                sim.d.qpos[2] < 0.20 or abs(roll) > 0.80 or abs(pitch) > 0.80
            )

        displacement = sim.d.qpos[:2] - start_position
        final_yaw = attitude(sim.d.qpos[3:7])[2]
        yaw_delta = float(np.arctan2(
            np.sin(final_yaw - start_yaw), np.cos(final_yaw - start_yaw)
        ))
        tracking = (
            np.mean(values["vx_error"]) / 0.20
            + np.mean(values["vy_error"]) / 0.08
            + np.mean(values["yaw_error"]) / 0.60
        )
        stability = (
            np.mean(values["roll"]) / 0.20
            + np.mean(values["pitch"]) / 0.20
            + max(0.0, 0.25 - min(values["height"])) / 0.05
        )
        safety = (
            20.0 * np.mean(values["torque_sat"])
            + max(0.0, percentile(values["raw_torque_max"], 99) - 13.0)
        )
        coordination = (
            max(0.0, 0.050 - percentile(values["mid_swing_clearance"], 10)) / 0.02
            + np.mean(values["diagonal_foot_height_error"]) / 0.025
            + np.mean(values["contact_phase_mismatch"])
            + np.mean(values["left_right_clearance_error"]) / 0.02
        )
        score = float(
            tracking + stability + safety + coordination
            + (500.0 if fallen else 0.0)
        )
        rows.append({
            "policy": str(Path(policy).resolve()),
            "command_index": command_index,
            "command": name,
            "cmd_vx": vx,
            "cmd_vy": vy,
            "cmd_yaw": yaw_rate,
            "segment_s": segment_s,
            "fallen": int(fallen),
            "score": score,
            "vx_mae": np.mean(values["vx_error"]),
            "vy_mae": np.mean(values["vy_error"]),
            "yaw_rate_mae": np.mean(values["yaw_error"]),
            "vx_mean": np.mean(values["vx"]),
            "vy_mean": np.mean(values["vy"]),
            "yaw_rate_mean": np.mean(values["yaw_rate"]),
            "roll_abs_mean": np.mean(values["roll"]),
            "roll_abs_p95": percentile(values["roll"], 95),
            "pitch_abs_mean": np.mean(values["pitch"]),
            "pitch_abs_p95": percentile(values["pitch"], 95),
            "min_height": min(values["height"]),
            "raw_torque_abs_mean": np.mean(values["raw_torque_mean"]),
            "raw_torque_abs_p99_max": percentile(values["raw_torque_max"], 99),
            "torque_saturation_ratio": np.mean(values["torque_sat"]),
            "target_speed_abs_max": max(values["target_speed"]),
            "target_accel_abs_max": max(values["target_accel"]),
            "sagittal_diagonal_error_mean": np.mean(
                values["sagittal_diagonal_error"]
            ),
            "mid_swing_clearance_p10": percentile(
                values["mid_swing_clearance"], 10
            ),
            "mid_swing_clearance_mean": np.mean(
                values["mid_swing_clearance"]
            ),
            "mid_swing_clearance_p90": percentile(
                values["mid_swing_clearance"], 90
            ),
            "diagonal_foot_height_error_mean": np.mean(
                values["diagonal_foot_height_error"]
            ),
            "contact_phase_mismatch_ratio": np.mean(
                values["contact_phase_mismatch"]
            ),
            "left_right_clearance_error_mean": np.mean(
                values["left_right_clearance_error"]
            ),
            "dx": float(displacement[0]),
            "dy": float(displacement[1]),
            "yaw_delta": yaw_delta,
        })
    return rows


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=root / "models/fanfan_scene.xml")
    parser.add_argument("--segment-duration", type=float, default=15.0)
    parser.add_argument("--warmup-duration", type=float, default=2.0)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    all_rows = []
    for policy in args.policies:
        all_rows.extend(evaluate(
            policy, args.model, args.segment_duration, args.warmup_duration
        ))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    scores = {}
    for row in all_rows:
        scores.setdefault(row["policy"], []).append(float(row["score"]))
    for policy, values in scores.items():
        print(f"{policy}: mean_score={np.mean(values):.6f}, max_score={max(values):.6f}")
    print(f"csv={args.csv.resolve()}")


if __name__ == "__main__":
    main()
