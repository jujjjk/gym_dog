"""Export the RS01 Go2 Kp40 actor and its complete Sim2Sim contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Isaac Gym must be imported before torch in this workspace.
import isaacgym  # noqa: F401
import torch


ROOT = Path(__file__).resolve().parents[2]
GYM_ROOT = ROOT / "unitree_rl_gym"
sys.path.insert(0, str(GYM_ROOT))

from legged_gym.envs.rs01_go2_straight.rs01_go2_kp40_config import (  # noqa: E402
    Rs01Go2Kp40Cfg,
    Rs01Go2Kp40CfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_sim2sim_config import (  # noqa: E402
    Rs01Go2Sim2SimAdaptCfg,
    Rs01Go2Sim2SimAdaptCfgPPO,
    Rs01Go2Sim2SimCalfRepairCfg,
    Rs01Go2Sim2SimCalfRepairCfgPPO,
    Rs01Go2Sim2SimKd050Cfg,
    Rs01Go2Sim2SimKd050CfgPPO,
    Rs01Go2Sim2SimRobustCfg,
    Rs01Go2Sim2SimRobustCfgPPO,
    Rs01Go2MatchedTransferCfg,
    Rs01Go2MatchedTransferCfgPPO,
)
from rsl_rl.modules import ActorCritic  # noqa: E402


METADATA_KEY = "rs01_go2_deployment_config"
TASKS = {
    "rs01_go2_straight_kp40": (
        Rs01Go2Kp40Cfg,
        Rs01Go2Kp40CfgPPO,
    ),
    "rs01_go2_sim2sim_adapt": (
        Rs01Go2Sim2SimAdaptCfg,
        Rs01Go2Sim2SimAdaptCfgPPO,
    ),
    "rs01_go2_sim2sim_calf_repair": (
        Rs01Go2Sim2SimCalfRepairCfg,
        Rs01Go2Sim2SimCalfRepairCfgPPO,
    ),
    "rs01_go2_sim2sim_kd050": (
        Rs01Go2Sim2SimKd050Cfg,
        Rs01Go2Sim2SimKd050CfgPPO,
    ),
    "rs01_go2_sim2sim_robust": (
        Rs01Go2Sim2SimRobustCfg,
        Rs01Go2Sim2SimRobustCfgPPO,
    ),
    "rs01_go2_sim2sim_matched_transfer": (
        Rs01Go2MatchedTransferCfg,
        Rs01Go2MatchedTransferCfgPPO,
    ),
}


def class_to_dict(instance):
    result = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        value = getattr(instance, key)
        if callable(value):
            continue
        result[key] = class_to_dict(value) if isinstance(value, type) else value
    return result


def matching_value(mapping, joint_name):
    matches = [float(value) for key, value in mapping.items() if key in joint_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one joint-type value for {joint_name}, got {matches}"
        )
    return matches[0]


def movable_joint_names(urdf_path):
    root = ET.parse(urdf_path).getroot()
    return [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") not in ("fixed", "floating")
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motor_values(cfg, joint_names, values):
    return [
        float(values[cfg.rs01_actuator.joint_to_motor_id[name]])
        for name in joint_names
    ]


def build_contract(task_name, cfg, checkpoint, onnx_path):
    urdf_path = Path(
        cfg.asset.file.replace(
            "{LEGGED_GYM_ROOT_DIR}",
            str(GYM_ROOT),
        )
    ).resolve()
    joint_names = movable_joint_names(urdf_path)
    if len(joint_names) != cfg.env.num_actions:
        raise ValueError(
            f"URDF has {len(joint_names)} movable joints, expected "
            f"{cfg.env.num_actions}"
        )

    obs_scales = cfg.normalization.obs_scales
    actuator = cfg.rs01_actuator
    action_scale_by_joint = getattr(
        cfg.control,
        "action_scale_by_joint",
        None,
    )
    if action_scale_by_joint is None:
        action_scales = [
            float(cfg.control.action_scale) for _ in joint_names
        ]
    else:
        action_scales = [
            matching_value(action_scale_by_joint, name)
            for name in joint_names
        ]

    return {
        "schema_version": 1,
        "task": task_name,
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx_path.resolve()),
        "dimensions": {
            "observations": int(cfg.env.num_observations),
            "actions": int(cfg.env.num_actions),
        },
        "joint_names": joint_names,
        "joint_order_note": (
            "URDF declaration order, equal to the Isaac Gym runtime DOF order "
            "for dog_rs01.urdf: FR, FL, RR, RL and hip, thigh, calf."
        ),
        "default_joint_angles_rad": [
            float(cfg.init_state.default_joint_angles[name])
            for name in joint_names
        ],
        "initial_state": {
            "base_position_m": list(cfg.init_state.pos),
            "base_quaternion_xyzw": list(cfg.init_state.rot),
        },
        "observations": {
            "layout": [
                ["base_linear_velocity_body_m_s", 3],
                ["base_angular_velocity_body_rad_s", 3],
                ["projected_gravity_body", 3],
                ["command_scaled", 3],
                ["joint_position_error_rad_scaled", 12],
                ["joint_velocity_rad_s_scaled", 12],
                ["previous_action", 12],
                ["gait_phase_sin_cos", 2],
                ["wrapped_heading_error_scaled", 1],
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
            "straight_heading_error_scale": float(
                cfg.commands.straight_heading_observation_scale
            ),
        },
        "commands": {
            "default": [float(cfg.commands.playback_speed_mps), 0.0, 0.0],
            "ranges": {
                "lin_vel_x": list(cfg.commands.ranges.lin_vel_x),
                "lin_vel_y": list(cfg.commands.ranges.lin_vel_y),
                "ang_vel_yaw": list(cfg.commands.ranges.ang_vel_yaw),
            },
        },
        "gait": {
            "period_s": float(cfg.rewards.gait_period_s),
            "stance_ratio": float(cfg.rewards.gait_stance_ratio),
            "phase_offsets": {
                "FL": 0.0,
                "FR": 0.5,
                "RL": 0.5,
                "RR": 0.0,
            },
            "contact_threshold_n": float(
                cfg.rewards.foot_contact_force_threshold
            ),
        },
        "control": {
            "physics_dt_s": float(cfg.sim.dt),
            "policy_dt_s": float(cfg.sim.dt * cfg.control.decimation),
            "decimation": int(cfg.control.decimation),
            "action_output": (
                "raw actor mean; no tanh; clipped to normalization.clip_actions"
            ),
            "action_clip": float(cfg.normalization.clip_actions),
            "action_scale_rad": action_scales,
            "kp_nm_per_rad": [
                matching_value(cfg.control.stiffness, name)
                for name in joint_names
            ],
            "kd_nm_per_rad_s": [
                matching_value(cfg.control.damping, name)
                for name in joint_names
            ],
            "target_rate_limit_rad_s": [
                matching_value(actuator.target_rate_limit_rad_s, name)
                for name in joint_names
            ],
            "target_acceleration_limit_rad_s2": [
                matching_value(
                    actuator.target_acceleration_limit_rad_s2,
                    name,
                )
                for name in joint_names
            ],
            "response_gain": motor_values(
                cfg, joint_names, actuator.response_gain
            ),
            "time_constant_s": motor_values(
                cfg, joint_names, actuator.time_constant_s
            ),
            "observed_closed_loop_delay_s": motor_values(
                cfg, joint_names, actuator.observed_closed_loop_delay_s
            ),
            "coulomb_friction_nm": motor_values(
                cfg, joint_names, actuator.coulomb_friction_nm
            ),
            "friction_smoothing_rad_s": float(
                actuator.friction_smoothing_rad_s
            ),
            "continuous_torque_nm": float(actuator.continuous_torque_nm),
            "peak_torque_limit_nm": float(actuator.peak_torque_limit_nm),
        },
        "simulator": {
            "source": "Isaac Gym / PhysX",
            "target": "MuJoCo",
            "urdf": str(urdf_path),
            # Bind the policy to this exact calibrated RS01 machine model.
            # The converter and runner reject a stale/other-machine scene.
            "urdf_sha256": sha256_file(urdf_path),
            "self_collision": False,
            "ground_friction": 1.0,
            "physx": {
                "physics_step_s": float(cfg.sim.dt),
                "contact_substeps": int(cfg.sim.substeps),
                "contact_substep_s": float(
                    cfg.sim.dt / cfg.sim.substeps
                ),
                "solver_type": int(cfg.sim.physx.solver_type),
                "position_iterations": int(
                    cfg.sim.physx.num_position_iterations
                ),
                "velocity_iterations": int(
                    cfg.sim.physx.num_velocity_iterations
                ),
                "contact_offset_m": float(
                    cfg.sim.physx.contact_offset
                ),
                "rest_offset_m": float(cfg.sim.physx.rest_offset),
                "max_depenetration_velocity_m_s": float(
                    cfg.sim.physx.max_depenetration_velocity
                ),
            },
            "mujoco": {
                # Keep the identified RS01 motor update at 5 ms, but solve
                # contact with the same two internal 2.5 ms substeps as the
                # training task.
                "integration_timestep_s": float(
                    cfg.sim.dt / cfg.sim.substeps
                ),
                "integration_substeps_per_motor_step": int(
                    cfg.sim.substeps
                ),
                "integrator": "implicitfast",
                "solver": "Newton",
                # Calibrated from the new 11.731736 kg dog_rs01 zero-action
                # stand. The 6.5 ms contact time constant leaves less than
                # 0.04 mm steady foot penetration and keeps the base-height
                # mismatch near 1.4 mm without an impact-like rigid stop.
                "contact_solref": [0.0065, 1.0],
                "contact_solimp": [
                    0.99,
                    0.999,
                    0.0001,
                    0.5,
                    2.0,
                ],
                "contact_margin_m": 0.0,
                "contact_gap_m": 0.0,
                "contact_dimension": 3,
                "friction": [1.0, 0.005, 0.0001],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="rs01_go2_straight_kp40",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cfg, train_cfg = TASKS[args.task]
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    actor_critic = ActorCritic(
        cfg.env.num_observations,
        cfg.env.num_observations,
        cfg.env.num_actions,
        **class_to_dict(train_cfg.policy),
    )
    state = torch.load(checkpoint, map_location="cpu")
    actor_critic.load_state_dict(state["model_state_dict"], strict=True)
    actor = actor_critic.actor.eval()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, cfg.env.num_observations)
    torch.onnx.export(
        actor,
        example,
        output,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    contract = build_contract(args.task, cfg, checkpoint, output)
    import onnx

    model = onnx.load(output)
    entry = model.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = json.dumps(contract, separators=(",", ":"))
    onnx.save(model, output)
    output.with_suffix(".json").write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output),
        providers=["CPUExecutionProvider"],
    )
    torch.manual_seed(1)
    probe = torch.randn(8, cfg.env.num_observations)
    with torch.no_grad():
        expected = actor(probe).numpy()
    actual = session.run(
        ["actions"],
        {"observations": probe.numpy()},
    )[0]
    max_error = float(abs(expected - actual).max())
    if max_error > 1.0e-5:
        raise RuntimeError(f"ONNX parity error is too large: {max_error:.3e}")

    print(f"checkpoint={checkpoint}")
    print(f"onnx={output}")
    print(f"contract={output.with_suffix('.json')}")
    print(f"observations={cfg.env.num_observations}")
    print(f"actions={cfg.env.num_actions}")
    print(f"max_torch_onnx_error={max_error:.3e}")


if __name__ == "__main__":
    main()
