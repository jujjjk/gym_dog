"""Export an RS01 dog policy to ONNX plus a deployment contract JSON.

The actor is rebuilt from the checkpoint rather than from a live simulator, so
this script needs neither a GPU nor Isaac Gym.

The exported graph is ``action = tanh(actor(obs))``, i.e. the bounded action the
environment actually uses. The consumer only has to apply

    q_target[j] = q_default[j] + action_scale[j] * action[j]

with the per-joint values recorded in the JSON sidecar.

Usage:
    python3 legged_gym/scripts/export_dog_onnx.py \
        --task dog_rs01_straight_walk \
        --load_run Jul25_18-00-00_rs01_straight_walk \
        --checkpoint 3000
"""

import argparse
import importlib
import json
import os
import sys
import types
import xml.etree.ElementTree as ET

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

JOINT_TYPES = ("hip", "thigh", "calf")

OBSERVATION_LAYOUT = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("commands", 3),
    ("dof_pos_error", 12),
    ("dof_vel", 12),
    ("previous_actions", 12),
    ("gait_phase_sin_cos", 2),
    ("heading_error_sin_cos", 2),
)


def _import_config_only():
    """Import the config classes without triggering the Isaac Gym imports."""
    envs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs")
    for name, path in (
        ("legged_gym.envs", envs_dir),
        ("legged_gym.envs.dog", os.path.join(envs_dir, "dog")),
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [path]
            sys.modules[name] = stub
    return importlib.import_module(
        "legged_gym.envs.dog.dog_rs01_straight_config"
    )


TASKS = {
    "dog_rs01_straight_stand": ("DogRs01StraightStandCfg", "DogRs01StraightStandCfgPPO"),
    "dog_rs01_straight_walk": ("DogRs01StraightWalkCfg", "DogRs01StraightWalkCfgPPO"),
}


def _class_to_dict(instance):
    result = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        value = getattr(instance, key)
        if callable(value):
            continue
        if isinstance(value, type):
            result[key] = _class_to_dict(value)
        else:
            result[key] = value
    return result


def _matched(mapping, name):
    values = [value for key, value in mapping.items() if key in name]
    if len(values) != 1:
        raise ValueError(f"Expected one config match for {name}, got {values}")
    return values[0]


def _action_scales(cfg, names):
    scales = []
    for name in names:
        if "hip" in name:
            scales.append(float(cfg.control.hip_action_scale))
            continue
        for kind in ("thigh", "calf"):
            override = getattr(cfg.control, f"{kind}_action_scale", None)
            if f"_{kind}_" in name and override is not None:
                scales.append(float(override))
                break
        else:
            if name.startswith(("RL_", "RR_")):
                scales.append(float(cfg.control.rear_action_scale))
            else:
                scales.append(float(cfg.control.action_scale))
    return scales


class BoundedActor(torch.nn.Module):
    """The environment's action contract: tanh applied to the actor output."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations):
        return torch.tanh(self.actor(observations))


def resolve_checkpoint(experiment_name, load_run, checkpoint):
    root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    if not os.path.isdir(root):
        raise SystemExit(f"No log directory at {root}")
    if load_run in (None, "-1"):
        runs = sorted(
            entry for entry in os.listdir(root)
            if entry != "exported"
            and os.path.isdir(os.path.join(root, entry))
        )
        if not runs:
            raise SystemExit(f"No runs under {root}")
        load_run = runs[-1]
    run_dir = os.path.join(root, load_run)
    if checkpoint in (None, -1, "-1"):
        models = sorted(
            entry for entry in os.listdir(run_dir)
            if entry.startswith("model_") and entry.endswith(".pt")
        )
        if not models:
            raise SystemExit(f"No model_*.pt under {run_dir}")
        models.sort(key=lambda entry: int(entry[len("model_"):-len(".pt")]))
        name = models[-1]
    else:
        name = f"model_{int(checkpoint)}.pt"
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        raise SystemExit(f"No checkpoint at {path}")
    return run_dir, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dog_rs01_straight_walk",
                        choices=sorted(TASKS))
    parser.add_argument("--load_run", default=None,
                        help="run directory name, default: newest")
    parser.add_argument("--checkpoint", default=None,
                        help="iteration number, default: highest")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    straight = _import_config_only()
    cfg_name, ppo_name = TASKS[args.task]
    cfg = getattr(straight, cfg_name)
    train_cfg = getattr(straight, ppo_name)

    run_dir, checkpoint_path = resolve_checkpoint(
        train_cfg.runner.experiment_name, args.load_run, args.checkpoint
    )

    from rsl_rl.modules import ActorCritic

    policy_kwargs = _class_to_dict(train_cfg.policy)
    actor_critic = ActorCritic(
        cfg.env.num_observations,
        cfg.env.num_observations,
        cfg.env.num_actions,
        **policy_kwargs,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    actor_critic.load_state_dict(state["model_state_dict"])
    actor_critic.eval()

    model = BoundedActor(actor_critic.actor).eval()
    example = torch.zeros(1, cfg.env.num_observations)
    with torch.no_grad():
        reference = model(example)

    output_dir = args.output_dir or os.path.join(run_dir, "exported")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.task}_{os.path.basename(checkpoint_path)[:-3]}"
    onnx_path = os.path.join(output_dir, f"{stem}.onnx")
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    names = list(cfg.control.policy_joint_order)
    urdf_path = cfg.asset.file.replace(
        "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
    )
    urdf = ET.parse(urdf_path).getroot()
    urdf_limits = {}
    for joint in urdf.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        urdf_limits[joint.get("name")] = {
            "lower": float(limit.get("lower")),
            "upper": float(limit.get("upper")),
            "effort": float(limit.get("effort")),
            "velocity": float(limit.get("velocity")),
        }

    obs_scales = cfg.normalization.obs_scales
    contract = {
        "schema_version": 1,
        "task": args.task,
        "checkpoint": os.path.realpath(checkpoint_path),
        "onnx": os.path.realpath(onnx_path),
        "graph": {
            "input": "observations",
            "output": "actions",
            "output_transform": "tanh applied inside the graph",
            "num_observations": cfg.env.num_observations,
            "num_actions": cfg.env.num_actions,
        },
        "joint_names": names,
        "joint_order_note": (
            "This is the runtime Isaac Gym DOF order, confirmed by "
            "validate_dog_straight.py. It is not the URDF declaration order."
        ),
        "default_joint_angles": [
            float(cfg.init_state.default_joint_angles[name]) for name in names
        ],
        "action": {
            "formula": "q_target = q_default + action_scale * action",
            "action_scale": _action_scales(cfg, names),
            "target_clamp": {
                kind: list(cfg.control.target_position_limits_by_joint[kind])
                for kind in JOINT_TYPES
            } if hasattr(cfg.control, "target_position_limits_by_joint") else None,
        },
        "control": {
            "sim_dt": float(cfg.sim.dt),
            "decimation": int(cfg.control.decimation),
            "control_hz": 1.0 / (cfg.sim.dt * cfg.control.decimation),
            "stiffness": [_matched(cfg.control.stiffness, name) for name in names],
            "damping": [_matched(cfg.control.damping, name) for name in names],
            "torque_limits_applied": [
                float(_matched(cfg.control.torque_limits_by_joint, name))
                for name in names
            ],
        },
        "rs01": {
            "rated_torque_nm": straight.RS01_RATED_TORQUE_NM,
            "project_soft_torque_nm": straight.RS01_SOFT_TORQUE_NM,
            "hardware_peak_torque_nm": straight.RS01_PEAK_TORQUE_NM,
        },
        "observations": {
            "layout": [
                {"field": field, "width": width}
                for field, width in OBSERVATION_LAYOUT
            ],
            "clip": float(cfg.normalization.clip_observations),
            "lin_vel_scale": float(obs_scales.lin_vel),
            "ang_vel_scale": float(obs_scales.ang_vel),
            "dof_pos_scale": float(obs_scales.dof_pos),
            "dof_vel_scale": float(obs_scales.dof_vel),
            "command_scale": [
                float(obs_scales.lin_vel),
                float(obs_scales.lin_vel),
                float(obs_scales.ang_vel),
            ],
        },
        "commands": {
            "lin_vel_x": list(cfg.commands.ranges.lin_vel_x),
            "lin_vel_y": list(cfg.commands.ranges.lin_vel_y),
            "ang_vel_yaw": list(cfg.commands.ranges.ang_vel_yaw),
            "heading": list(cfg.commands.ranges.heading),
        },
        "gait": {
            "period_s": float(cfg.rewards.gait_period),
            "stance_ratio": float(cfg.rewards.gait_stance_ratio),
            "diagonal_pairs": [["FL", "RR"], ["FR", "RL"]],
        },
        "urdf": {
            "file": os.path.realpath(urdf_path),
            "limits": {name: urdf_limits[name] for name in names},
        },
        "unresolved": [
            "Real motor IDs, motor signs and zero offsets are not part of this "
            "export. The policy is trained and exported purely in URDF joint "
            "semantics; any real-robot mapping must be added downstream."
        ],
    }
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)
        handle.write("\n")

    print(f"[export] checkpoint    {checkpoint_path}")
    print(f"[export] onnx          {onnx_path}")
    print(f"[export] contract      {json_path}")
    print(f"[export] zero-obs action (first 6): "
          f"{[round(float(value), 6) for value in reference[0][:6]]}")

    try:
        import onnxruntime
    except ImportError:
        print("[export] onnxruntime not installed; skipping numeric parity check")
        return 0

    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    probe = torch.randn(4, cfg.env.num_observations)
    with torch.no_grad():
        expected = model(probe).numpy()
    actual = session.run(["actions"], {"observations": probe.numpy()})[0]
    deviation = float(abs(expected - actual).max())
    print(f"[export] max |torch - onnx| over 4 random inputs: {deviation:.3e}")
    if deviation > 1.0e-5:
        print("[export] ISSUE: ONNX output does not match PyTorch")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
