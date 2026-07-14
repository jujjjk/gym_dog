"""Export actor plus the complete deployment contract derived from training cfg."""
from pathlib import Path
import argparse, importlib, json, os, sys, xml.etree.ElementTree as ET
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
# Isaac Gym must initialize its binary bindings before torch in this project.
import isaacgym
import torch

class Actor(torch.nn.Sequential):
    def __init__(self, observations, cfg=None):
        super().__init__(torch.nn.Linear(observations,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                         torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),
                         torch.nn.Linear(128,12))
        self.cfg = cfg

    def forward(self, observations):
        control = getattr(self.cfg, "control", None)
        longitudinal_gain = float(getattr(
            control, "command_feedback_longitudinal_gain", 0.0
        ))
        lateral_gain = float(getattr(
            control, "command_feedback_lateral_gain", 0.0
        ))
        yaw_gain = float(getattr(
            control, "command_feedback_yaw_gain", 0.0
        ))
        heading_gain = float(getattr(
            control, "command_feedback_heading_gain", 0.0
        ))
        heading_damping = float(getattr(
            control, "command_feedback_heading_damping", 0.0
        ))
        diagonal_x_scale = float(getattr(
            control, "command_feedback_diagonal_longitudinal_scale", 1.0
        ))
        if any(gain != 0.0 for gain in (
            longitudinal_gain, lateral_gain, yaw_gain, heading_gain
        )):
            lin_scale = float(self.cfg.normalization.obs_scales.lin_vel)
            ang_scale = float(self.cfg.normalization.obs_scales.ang_vel)
            vx = observations[:, 0] / lin_scale
            vy = observations[:, 1] / lin_scale
            yaw_rate = observations[:, 5] / ang_scale
            cmd_x = observations[:, 9] / lin_scale
            cmd_y = observations[:, 10] / lin_scale
            cmd_yaw = observations[:, 11] / ang_scale
            effective_x = cmd_x + longitudinal_gain * (cmd_x - vx)
            effective_y = cmd_y + lateral_gain * (cmd_y - vy)
            effective_yaw = cmd_yaw + yaw_gain * (cmd_yaw - yaw_rate)
            diagonal = (
                (torch.abs(cmd_x) > 0.05)
                & (torch.abs(cmd_y) > 0.05)
                & (torch.abs(cmd_yaw) < 0.05)
            )
            effective_x = torch.where(
                diagonal, diagonal_x_scale * effective_x, effective_x
            )
            heading_error = torch.atan2(
                observations[:, 50], observations[:, 51]
            )
            heading_hold = (
                torch.sqrt(cmd_x * cmd_x + cmd_y * cmd_y) > 0.04
            ) & (torch.abs(cmd_yaw) < 0.05)
            heading_correction = (
                heading_gain * heading_error - heading_damping * yaw_rate
            )
            effective_yaw = torch.where(
                heading_hold, heading_correction, effective_yaw
            )
            ranges = self.cfg.commands.ranges
            effective_x = effective_x.clamp(
                ranges.lin_vel_x[0], ranges.lin_vel_x[1]
            )
            effective_y = effective_y.clamp(
                ranges.lin_vel_y[0], ranges.lin_vel_y[1]
            )
            effective_yaw = effective_yaw.clamp(
                ranges.ang_vel_yaw[0], ranges.ang_vel_yaw[1]
            )
            observations = torch.cat((
                observations[:, :9],
                (effective_x * lin_scale).unsqueeze(1),
                (effective_y * lin_scale).unsqueeze(1),
                (effective_yaw * ang_scale).unsqueeze(1),
                observations[:, 12:],
            ), dim=1)
        feedback_gain = float(getattr(
            control, "straight_vy_feedback_gain", 0.0
        ))
        forward_boost = float(getattr(
            control, "straight_vx_feedback_boost", 0.0
        ))
        sagittal_blend = float(getattr(
            control, "straight_vy_feedback_sagittal_blend", 1.0
        ))
        base_raw = super().forward(observations)
        actor_observations = observations
        if feedback_gain != 0.0:
            lin_scale = float(self.cfg.normalization.obs_scales.lin_vel)
            yaw_scale = float(self.cfg.normalization.obs_scales.ang_vel)
            straight_feedback = (
                (torch.abs(observations[:, 9]) > 0.03 * lin_scale)
                & (torch.abs(observations[:, 10]) < 0.02 * lin_scale)
                & (torch.abs(observations[:, 11]) < 0.05 * yaw_scale)
            )
            corrected_vy_command = torch.clamp(
                observations[:, 10] - feedback_gain * observations[:, 1],
                -0.12 * lin_scale,
                0.12 * lin_scale,
            )
            corrected_vy_command = torch.where(
                straight_feedback,
                corrected_vy_command,
                observations[:, 10],
            )
            corrected_vx_command = torch.clamp(
                observations[:, 9]
                + forward_boost * torch.abs(observations[:, 1]),
                -0.12 * lin_scale,
                0.46 * lin_scale,
            )
            corrected_vx_command = torch.where(
                straight_feedback,
                corrected_vx_command,
                observations[:, 9],
            )
            actor_observations = torch.cat((
                observations[:, :9],
                corrected_vx_command.unsqueeze(1),
                corrected_vy_command.unsqueeze(1),
                observations[:, 11:],
            ), dim=1)
        if feedback_gain != 0.0:
            corrected_raw = super().forward(actor_observations)
            names = list(control.policy_joint_order)
            blend = base_raw.new_tensor([
                1.0 if "hip" in name else sagittal_blend for name in names
            ])
            raw = base_raw + blend * (corrected_raw - base_raw)
        else:
            raw = base_raw
        if bool(getattr(control, "enforce_policy_symmetry", False)):
            mirrored_observations = observations.clone()
            mirrored_observations[:, 1] *= -1.0
            mirrored_observations[:, 7] *= -1.0
            mirrored_observations[:, 10] *= -1.0
            mirrored_observations[:, 3] *= -1.0
            mirrored_observations[:, 5] *= -1.0
            mirrored_observations[:, 11] *= -1.0
            leg_mirror = torch.tensor(
                [1, 0, 3, 2], dtype=torch.long,
                device=observations.device,
            )
            joint_sign = observations.new_tensor([-1.0, 1.0, 1.0])
            for start in (12, 24, 36):
                block = observations[:, start:start + 12].reshape(-1, 4, 3)
                block = block.index_select(1, leg_mirror) * joint_sign
                mirrored_observations[:, start:start + 12] = block.reshape(-1, 12)
            mirrored_observations[:, 48:50] *= -1.0
            mirrored_observations[:, 50] *= -1.0
            mirrored_raw = super().forward(mirrored_observations)
            mirrored_actions = mirrored_raw.reshape(-1, 4, 3)
            mirrored_actions = (
                mirrored_actions.index_select(1, leg_mirror) * joint_sign
            ).reshape(-1, 12)
            raw = 0.5 * (raw + mirrored_actions)
        bias_cfg = getattr(
            control,
            "straight_action_bias_by_joint",
            None,
        )
        project = bool(getattr(control, "project_straight_diagonal_actions", False))
        if bias_cfg is None and not project:
            return raw
        names = list(control.policy_joint_order)
        lin_scale = float(self.cfg.normalization.obs_scales.lin_vel)
        yaw_scale = float(self.cfg.normalization.obs_scales.ang_vel)
        straight = (
            (torch.abs(observations[:, 9]) > 0.03 * lin_scale)
            & (torch.abs(observations[:, 10]) < 0.02 * lin_scale)
            & (torch.abs(observations[:, 11]) < 0.05 * yaw_scale)
        ).unsqueeze(1)
        bounded = torch.tanh(raw)
        if bias_cfg is not None:
            bias = raw.new_tensor([
                float(bias_cfg.get(name, 0.0)) for name in names
            ])
            corrected = torch.clamp(bounded + bias, -0.999999, 0.999999)
            bounded = torch.where(straight, corrected, bounded)

        if project:
            scales = []
            for name in names:
                if "hip" in name:
                    scales.append(float(control.hip_action_scale))
                elif name.startswith(("RL_", "RR_")):
                    scales.append(float(control.rear_action_scale))
                else:
                    scales.append(float(control.action_scale))
            scales = raw.new_tensor(scales)
            physical = bounded * scales
            name_to_index = {name: index for index, name in enumerate(names)}
            pair = {"FL": "RR", "RR": "FL", "FR": "RL", "RL": "FR"}
            projected = []
            for index, name in enumerate(names):
                leg, joint, _ = name.split("_", 2)
                other = name_to_index[f"{pair[leg]}_{joint}_joint"]
                if joint == "hip":
                    sign = 1.0 if leg in ("FL", "FR") else -1.0
                    value = sign * 0.5 * (
                        sign * physical[:, index]
                        - sign * physical[:, other]
                    )
                else:
                    value = 0.5 * (
                        physical[:, index] + physical[:, other]
                    )
                projected.append(value / scales[index])
            projected = torch.stack(projected, dim=1).clamp(-0.999999, 0.999999)
            bounded = torch.where(straight, projected, bounded)

        # The deployment loop applies tanh to actor output, so map the bounded
        # transformed action back to the actor's raw-action contract.
        return 0.5 * (
            torch.log(1.0 + bounded) - torch.log(1.0 - bounded)
        )

def matched(mapping,name):
    values=[value for key,value in mapping.items() if key in name]
    if len(values)!=1: raise ValueError(f"Expected one cfg match for {name}, got {values}")
    return values[0]

def torque_limits(cfg, names, effort):
    per_joint = getattr(cfg.control, "torque_limits_by_joint", None)
    if per_joint is not None:
        return [float(matched(per_joint, name)) for name in names]
    override = getattr(cfg.control, "torque_limit_override", None)
    if override is not None:
        return [float(override) for _ in names]
    return [effort[name] for name in names]

def deployment_config(cfg, checkpoint, gym_root):
    names=list(cfg.control.policy_joint_order)
    urdf_path=Path(cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}",str(gym_root)))
    root=ET.parse(urdf_path).getroot()
    effort={j.get("name"):float(j.find("limit").get("effort")) for j in root.findall("joint") if j.find("limit") is not None}
    scales=[]
    for name in names:
        if "hip" in name: scales.append(cfg.control.hip_action_scale)
        elif name.startswith(("RL_","RR_")): scales.append(cfg.control.rear_action_scale)
        else: scales.append(cfg.control.action_scale)
    return {
        "schema_version":1,"task":cfg.__name__,"checkpoint":str(checkpoint.resolve()),
        "dimensions":{"observations":cfg.env.num_observations,"actions":cfg.env.num_actions},
        "joint_names":names,
        "default_joint_angles":[cfg.init_state.default_joint_angles[n] for n in names],
        "initial_state":{"base_position":list(cfg.init_state.pos),"base_quaternion_xyzw":list(cfg.init_state.rot)},
        "control":{"sim_dt":cfg.sim.dt,"decimation":cfg.control.decimation,
                   "stiffness":[matched(cfg.control.stiffness,n) for n in names],
                   "damping":[matched(cfg.control.damping,n) for n in names],
                   "filter_policy_actions":getattr(cfg.control,"filter_policy_actions",False),
                   "policy_action_filter_alpha":getattr(cfg.control,"policy_action_filter_alpha",1.0),
                   "policy_action_rate_limits":[
                       matched(cfg.control.policy_action_rate_limits,n) for n in names
                   ] if hasattr(cfg.control,"policy_action_rate_limits") else None,
                   "policy_action_accel_limits":[
                       matched(cfg.control.policy_action_accel_limits,n) for n in names
                   ] if hasattr(cfg.control,"policy_action_accel_limits") else None,
                   "action_scale":scales,
                   "torque_limits":torque_limits(cfg,names,effort),
                   "straight_vy_feedback_gain":getattr(cfg.control,"straight_vy_feedback_gain",0.0),
                   "straight_vx_feedback_boost":getattr(cfg.control,"straight_vx_feedback_boost",0.0),
                   "straight_vy_feedback_sagittal_blend":getattr(cfg.control,"straight_vy_feedback_sagittal_blend",1.0),
                   "command_feedback_longitudinal_gain":getattr(cfg.control,"command_feedback_longitudinal_gain",0.0),
                   "command_feedback_lateral_gain":getattr(cfg.control,"command_feedback_lateral_gain",0.0),
                   "command_feedback_yaw_gain":getattr(cfg.control,"command_feedback_yaw_gain",0.0),
                   "command_feedback_heading_gain":getattr(cfg.control,"command_feedback_heading_gain",0.0),
                   "command_feedback_heading_damping":getattr(cfg.control,"command_feedback_heading_damping",0.0),
                   "command_feedback_diagonal_longitudinal_scale":getattr(cfg.control,"command_feedback_diagonal_longitudinal_scale",1.0),
                   "enforce_policy_symmetry":getattr(cfg.control,"enforce_policy_symmetry",False),
                   "output_transform":"tanh"},
        "observations":{"clip":cfg.normalization.clip_observations,
                        "lin_vel_scale":cfg.normalization.obs_scales.lin_vel,
                        "ang_vel_scale":cfg.normalization.obs_scales.ang_vel,
                        "dof_pos_scale":cfg.normalization.obs_scales.dof_pos,
                        "dof_vel_scale":cfg.normalization.obs_scales.dof_vel,
                        "command_scale":[cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.ang_vel],
                        "layout":["base_lin_vel","base_ang_vel","projected_gravity","commands","dof_pos_error","dof_vel","previous_actions","gait_phase_sin_cos","heading_error_sin_cos"]},
        "commands":{"default":[0.0,0.0,0.0],"heading_command":cfg.commands.heading_command,
                    "observe_heading_error":getattr(cfg.commands,"observe_heading_error",False),
                    "ranges":{"lin_vel_x":list(cfg.commands.ranges.lin_vel_x),
                              "lin_vel_y":list(cfg.commands.ranges.lin_vel_y),
                              "ang_vel_yaw":list(cfg.commands.ranges.ang_vel_yaw)},
                    "default_heading":sum(cfg.commands.ranges.heading)/2 if cfg.commands.heading_command else 0.0,"heading_gain":0.5},
        "gait":{"period":cfg.rewards.gait_period,"stance_ratio":cfg.rewards.gait_stance_ratio,
                "thigh_amplitude":cfg.rewards.gait_thigh_amplitude,"calf_amplitude":cfg.rewards.gait_calf_amplitude,
                "gate_with_command":getattr(cfg.control,"gate_gait_with_command",False),
                "command_gate_sigma":getattr(cfg.control,"gait_command_gate_sigma",0.0004),
                "phase_offsets":{"FL":0.0,"FR":0.5,"RL":0.5,"RR":0.0}},
        "episode_length_s":cfg.env.episode_length_s,
    }

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--config-class",default="legged_gym.envs.fanfan.fanfan_config:FanfanRoughCfg")
    p.add_argument("--gym-root",type=Path,default=Path(__file__).resolve().parents[1]/"unitree_rl_gym")
    p.add_argument("--straight-vy-feedback-gain",type=float)
    p.add_argument("--straight-vx-feedback-boost",type=float)
    p.add_argument("--straight-vy-feedback-sagittal-blend",type=float);a=p.parse_args()
    sys.path.insert(0,str(a.gym_root));module_name,class_name=a.config_class.split(":",1);cfg=getattr(importlib.import_module(module_name),class_name)
    if a.straight_vy_feedback_gain is not None:cfg.control.straight_vy_feedback_gain=a.straight_vy_feedback_gain
    if a.straight_vx_feedback_boost is not None:cfg.control.straight_vx_feedback_boost=a.straight_vx_feedback_boost
    if a.straight_vy_feedback_sagittal_blend is not None:cfg.control.straight_vy_feedback_sagittal_blend=a.straight_vy_feedback_sagittal_blend
    state=torch.load(a.checkpoint,map_location="cpu")["model_state_dict"]
    actor=Actor(cfg.env.num_observations,cfg).eval();actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("actor.")})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(actor,torch.zeros(1,cfg.env.num_observations),a.output,input_names=["observations"],output_names=["raw_actions"],dynamic_axes={"observations":{0:"batch"},"raw_actions":{0:"batch"}},opset_version=17)
    manifest=deployment_config(cfg,a.checkpoint,a.gym_root)
    import onnx
    model=onnx.load(a.output);entry=model.metadata_props.add();entry.key="fanfan_deployment_config";entry.value=json.dumps(manifest,separators=(",",":"));onnx.save(model,a.output)
    sidecar=a.output.with_suffix(".json");sidecar.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Exported {a.output} with cfg metadata and {sidecar}")
