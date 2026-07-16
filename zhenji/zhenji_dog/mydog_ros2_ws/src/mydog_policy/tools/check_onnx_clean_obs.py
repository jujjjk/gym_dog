#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline clean-observation ONNX check.

Example:
    cd ~/mydog_ros2_ws
    source install/setup.bash

    python3 src/mydog_policy/tools/check_onnx_clean_obs.py \
      --onnx_path /home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_yaw_clean_5100.onnx \
      --cmd_x 0.06 \
      --steps 20
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np


DEFAULT_ONNX_PATH = (
    "/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/"
    "fanfan_yaw_clean_5100.onnx"
)
DEFAULT_CSV_DIR = "/home/jetson/mydog_ros2_ws/log"

DISPLAY_JOINTS = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]
DISPLAY_FULL_NAMES = [f"{name}_joint" for name in DISPLAY_JOINTS]
FOCUS_FULL_NAMES = [
    "FL_thigh_joint",
    "RR_thigh_joint",
    "RL_hip_joint",
    "FL_hip_joint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fanfan_yaw_clean ONNX on ideal clean observations and report "
            "whether raw actions are already asymmetric."
        )
    )
    parser.add_argument("--onnx_path", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--cmd_x", type=float, default=0.06)
    parser.add_argument("--cmd_y", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        print("Install it with:")
        print("  pip install onnxruntime")
        sys.exit(2)
    return ort


def load_deployment_config(session):
    metadata = session.get_modelmeta().custom_metadata_map
    raw_config = metadata.get("fanfan_deployment_config")
    if not raw_config:
        print("ERROR: ONNX is missing fanfan_deployment_config metadata.")
        print("Cannot safely construct deployment-compatible observations.")
        sys.exit(2)
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid fanfan_deployment_config JSON: {exc}")
        sys.exit(2)


def is_static_int(value):
    return isinstance(value, int) and value > 0


def concrete_shape(shape):
    concrete = []
    for dim in shape:
        if is_static_int(dim):
            concrete.append(int(dim))
        else:
            concrete.append(1)
    return tuple(concrete)


def numpy_dtype(onnx_type):
    if onnx_type in ("tensor(double)", "tensor(float64)"):
        return np.float64
    if onnx_type in ("tensor(int64)",):
        return np.int64
    if onnx_type in ("tensor(int32)",):
        return np.int32
    return np.float32


def find_obs_input(inputs, obs_dim):
    for item in inputs:
        shape = list(item.shape)
        if len(shape) >= 1 and shape[-1] == obs_dim:
            return item
        if len(shape) >= 2 and shape[1] == obs_dim:
            return item
    return inputs[0]


def resolve_obs_dim(obs_input, config):
    meta_dim = int(config.get("dimensions", {}).get("observations", -1))
    shape = list(obs_input.shape)
    static_last = shape[-1] if shape else None
    if is_static_int(static_last):
        graph_dim = int(static_last)
        if meta_dim > 0 and meta_dim != graph_dim:
            print(
                "ERROR: ONNX graph and metadata observation dimensions differ: "
                f"graph={graph_dim}, metadata={meta_dim}"
            )
            sys.exit(2)
        return graph_dim
    if meta_dim in (36, 48, 50, 52):
        return meta_dim
    print(
        "ERROR: cannot infer fixed observation dimension from input shape "
        f"{shape} or metadata."
    )
    sys.exit(2)


def obs_feed_array(obs, obs_input):
    shape = list(obs_input.shape)
    if len(shape) == 1:
        actual = obs.reshape(obs.shape[0])
    elif len(shape) == 2:
        actual = obs.reshape(1, obs.shape[0])
    else:
        print(f"ERROR: unsupported observation input rank: shape={shape}")
        sys.exit(2)

    if len(shape) >= 1 and is_static_int(shape[-1]) and int(shape[-1]) != obs.shape[0]:
        print("ERROR: ONNX input dimension and constructed obs dimension mismatch.")
        print(f"  expected input shape: {shape}")
        print(f"  actual obs shape: {actual.shape}")
        sys.exit(2)
    return actual.astype(np.float32)


def build_clean_obs(config, obs_dim, cmd, last_action):
    obs_config = config["observations"]
    lin_vel_scale = float(obs_config["lin_vel_scale"])
    ang_vel_scale = float(obs_config["ang_vel_scale"])
    dof_pos_scale = float(obs_config["dof_pos_scale"])
    dof_vel_scale = float(obs_config["dof_vel_scale"])
    command_scale = np.asarray(obs_config["command_scale"], dtype=np.float32).reshape(3)
    obs_clip = abs(float(obs_config["clip"]))

    obs = np.zeros(obs_dim, dtype=np.float32)

    # Keep this aligned with ObsBuilder36.build_obs() in mydog_policy/obs_builder.py:
    # [0:3] base_lin_vel, [3:6] base_ang_vel, [6:9] projected_gravity,
    # [9:12] commands, [12:24] joint_pos_error, [24:36] joint_vel,
    # [36:48] last_action, [48:50] gait phase, [50:52] heading error.
    obs[0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32) * lin_vel_scale
    obs[3:6] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32) * ang_vel_scale
    obs[6:9] = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    obs[9:12] = np.asarray(cmd, dtype=np.float32) * command_scale
    obs[12:24] = np.zeros(12, dtype=np.float32) * dof_pos_scale
    obs[24:36] = np.zeros(12, dtype=np.float32) * dof_vel_scale
    if obs_dim >= 48:
        obs[36:48] = np.asarray(last_action, dtype=np.float32).reshape(12)
    if obs_dim >= 50:
        obs[48:50] = np.asarray([0.0, 1.0], dtype=np.float32)
    if obs_dim >= 52:
        obs[50:52] = np.asarray([0.0, 1.0], dtype=np.float32)

    return np.clip(obs, -obs_clip, obs_clip).astype(np.float32)


def output_action(raw_output, output_transform):
    action = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if action.shape[0] < 12:
        print(f"ERROR: ONNX output has fewer than 12 values: shape={action.shape}")
        sys.exit(2)
    action = action[:12]
    if output_transform == "tanh":
        action = np.tanh(action)
    elif output_transform not in ("identity", "none"):
        print(f"ERROR: unsupported output_transform={output_transform!r}")
        sys.exit(2)
    return action.astype(np.float32)


def display_order_action(action, joint_names):
    by_name = {name: float(action[i]) for i, name in enumerate(joint_names)}
    missing = [name for name in DISPLAY_FULL_NAMES if name not in by_name]
    if missing:
        print(f"ERROR: ONNX joint_names missing required joints: {missing}")
        sys.exit(2)
    return np.asarray([by_name[name] for name in DISPLAY_FULL_NAMES], dtype=np.float32)


def print_step_table(step, action_display):
    print("")
    print(f"step {step}")
    print(f"{'step':>4s}  {'joint_name':<10s}  {'action_raw':>11s}")
    print("-" * 31)
    for joint_name, value in zip(DISPLAY_JOINTS, action_display):
        print(f"{step:4d}  {joint_name:<10s}  {float(value):+11.6f}")


def print_statistics(actions_display):
    arr = np.asarray(actions_display, dtype=np.float32)
    print("")
    print("Final statistics")
    print(
        f"{'joint_name':<10s}  {'mean':>10s}  {'min':>10s}  {'max':>10s}  "
        f"{'abs_gt_0.80_ratio':>18s}  {'abs_gt_0.95_ratio':>18s}"
    )
    print("-" * 84)
    for i, joint_name in enumerate(DISPLAY_JOINTS):
        values = arr[:, i]
        print(
            f"{joint_name:<10s}  "
            f"{float(np.mean(values)):>+10.6f}  "
            f"{float(np.min(values)):>+10.6f}  "
            f"{float(np.max(values)):>+10.6f}  "
            f"{float(np.mean(np.abs(values) > 0.80)):>18.3f}  "
            f"{float(np.mean(np.abs(values) > 0.95)):>18.3f}"
        )


def print_focus_and_judgment(actions_display):
    arr = np.asarray(actions_display, dtype=np.float32)
    index_by_full = {full: i for i, full in enumerate(DISPLAY_FULL_NAMES)}

    print("")
    print("Focus joints")
    print(f"{'joint_name':<10s}  {'mean':>10s}  {'min':>10s}  {'max':>10s}")
    print("-" * 47)
    focus_stats = {}
    for full_name in FOCUS_FULL_NAMES:
        short_name = full_name.replace("_joint", "")
        values = arr[:, index_by_full[full_name]]
        focus_stats[full_name] = values
        print(
            f"{short_name:<10s}  "
            f"{float(np.mean(values)):>+10.6f}  "
            f"{float(np.min(values)):>+10.6f}  "
            f"{float(np.max(values)):>+10.6f}"
        )

    thigh_biased = (
        float(np.mean(focus_stats["FL_thigh_joint"])) > 0.70
        or float(np.mean(focus_stats["RR_thigh_joint"])) > 0.70
        or float(np.mean(focus_stats["FL_thigh_joint"] > 0.80)) > 0.50
        or float(np.mean(focus_stats["RR_thigh_joint"] > 0.80)) > 0.50
    )
    hip_biased = (
        float(np.mean(focus_stats["RL_hip_joint"])) < -0.60
        or float(np.mean(focus_stats["FL_hip_joint"])) < -0.60
        or float(np.mean(focus_stats["RL_hip_joint"] < -0.70)) > 0.50
        or float(np.mean(focus_stats["FL_hip_joint"] < -0.70)) > 0.50
    )

    print("")
    if thigh_biased or hip_biased:
        print(
            "WARNING: ONNX itself outputs strong asymmetric/raw-biased actions "
            "under clean observation. Model/checkpoint may be biased."
        )
    else:
        print(
            "Clean obs output is not strongly biased. Real robot observation "
            "pipeline is more suspicious."
        )


def write_csv(actions_display):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DEFAULT_CSV_DIR, f"onnx_clean_obs_check_{timestamp}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + DISPLAY_JOINTS)
        for step, action in enumerate(actions_display):
            writer.writerow([step] + [f"{float(v):.9f}" for v in action])
    return path


def main():
    args = parse_args()
    if args.steps <= 0:
        print("ERROR: --steps must be positive.")
        sys.exit(2)

    ort = import_onnxruntime()
    if not os.path.exists(args.onnx_path):
        print(f"ERROR: ONNX file does not exist: {args.onnx_path}")
        sys.exit(2)

    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CPUExecutionProvider"],
    )
    config = load_deployment_config(session)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        print("ERROR: ONNX session has no inputs or no outputs.")
        sys.exit(2)

    joint_names = list(config.get("joint_names", []))
    if len(joint_names) != 12:
        print(f"ERROR: expected 12 joint_names in metadata, got {len(joint_names)}")
        sys.exit(2)

    obs_dim = int(config.get("dimensions", {}).get("observations", -1))
    obs_input = find_obs_input(inputs, obs_dim)
    obs_dim = resolve_obs_dim(obs_input, config)
    output_name = outputs[0].name
    output_transform = str(
        config.get("control", {}).get("output_transform", "identity")
    ).lower()
    command_scale = np.asarray(
        config["observations"]["command_scale"], dtype=np.float32
    ).reshape(3)

    print("ONNX clean observation check")
    print(f"onnx_path: {args.onnx_path}")
    print("")
    print("Inputs:")
    for item in inputs:
        marker = "  <-- observation" if item.name == obs_input.name else ""
        print(f"  name={item.name!r} shape={item.shape} type={item.type}{marker}")
    print("Outputs:")
    for item in outputs:
        marker = "  <-- action output" if item.name == output_name else ""
        print(f"  name={item.name!r} shape={item.shape} type={item.type}{marker}")
    print("")
    print(f"observation_dim: {obs_dim}")
    print(f"metadata joint_names policy order: {joint_names}")
    print(f"display order: {DISPLAY_JOINTS}")
    print(f"command_scale from metadata: {command_scale.tolist()}")
    print(
        "clean command obs: "
        f"cmd=({args.cmd_x:.6f}, {args.cmd_y:.6f}, {args.cmd_yaw:.6f}) -> "
        f"obs_cmd={(np.asarray([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32) * command_scale).tolist()}"
    )
    print("clean base_lin_vel is [0, 0, 0] before lin_vel_scale.")
    print("clean base_ang_vel is [0, 0, 0] before ang_vel_scale.")
    print("clean projected_gravity is [0, 0, -1].")
    print("clean joint_pos_minus_default, joint_vel and initial last_action are zeros.")
    if obs_dim >= 50:
        print("clean gait_phase obs is fixed to [sin(0), cos(0)] = [0, 1].")
    if obs_dim >= 52:
        print("clean heading error obs is fixed to [sin(0), cos(0)] = [0, 1].")
    if len(inputs) > 1:
        print("")
        print(
            "Recurrent/history-style extra inputs detected. They will be "
            "initialized to zero for this offline check."
        )
    else:
        print("Single-frame observation input detected; no hidden state inputs.")

    extra_feeds = {}
    for item in inputs:
        if item.name == obs_input.name:
            continue
        extra_feeds[item.name] = np.zeros(
            concrete_shape(item.shape),
            dtype=numpy_dtype(item.type),
        )

    cmd = np.asarray([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    actions_display = []

    for step in range(args.steps):
        obs = build_clean_obs(config, obs_dim, cmd, last_action)
        obs_feed = obs_feed_array(obs, obs_input)
        feed = dict(extra_feeds)
        feed[obs_input.name] = obs_feed
        actual_shape = tuple(obs_feed.shape)
        expected_shape = tuple(obs_input.shape)
        if step == 0:
            print("")
            print(f"expected observation input shape: {expected_shape}")
            print(f"actual observation input shape: {actual_shape}")

        raw_outputs = session.run([item.name for item in outputs], feed)
        action = output_action(raw_outputs[0], output_transform)
        action_display = display_order_action(action, joint_names)
        print_step_table(step, action_display)
        actions_display.append(action_display)

        # Deployment feeds previous action in policy order, not display order.
        last_action = action.copy()

    csv_path = write_csv(actions_display)
    print_statistics(actions_display)
    print_focus_and_judgment(actions_display)
    print("")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
