"""Export actor plus the complete deployment contract derived from training cfg."""
from pathlib import Path
import argparse, importlib, json, os, sys, xml.etree.ElementTree as ET
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
# Isaac Gym must initialize its binary bindings before torch in this project.
import isaacgym
import torch

class Actor(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(50,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                         torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),
                         torch.nn.Linear(128,12))

def matched(mapping,name):
    values=[value for key,value in mapping.items() if key in name]
    if len(values)!=1: raise ValueError(f"Expected one cfg match for {name}, got {values}")
    return values[0]

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
        "schema_version":1,"task":"fanfan","checkpoint":str(checkpoint.resolve()),
        "dimensions":{"observations":cfg.env.num_observations,"actions":cfg.env.num_actions},
        "joint_names":names,
        "default_joint_angles":[cfg.init_state.default_joint_angles[n] for n in names],
        "initial_state":{"base_position":list(cfg.init_state.pos),"base_quaternion_xyzw":list(cfg.init_state.rot)},
        "control":{"sim_dt":cfg.sim.dt,"decimation":cfg.control.decimation,
                   "stiffness":[matched(cfg.control.stiffness,n) for n in names],
                   "damping":[matched(cfg.control.damping,n) for n in names],
                   "action_scale":scales,"torque_limits":[effort[n] for n in names],
                   "output_transform":"tanh"},
        "observations":{"clip":cfg.normalization.clip_observations,
                        "lin_vel_scale":cfg.normalization.obs_scales.lin_vel,
                        "ang_vel_scale":cfg.normalization.obs_scales.ang_vel,
                        "dof_pos_scale":cfg.normalization.obs_scales.dof_pos,
                        "dof_vel_scale":cfg.normalization.obs_scales.dof_vel,
                        "command_scale":[cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.ang_vel],
                        "layout":["base_lin_vel","base_ang_vel","projected_gravity","commands","dof_pos_error","dof_vel","previous_actions","gait_phase_sin_cos"]},
        "commands":{"default":[sum(cfg.commands.ranges.lin_vel_x)/2,0.0,0.0],"heading_command":cfg.commands.heading_command},
        "gait":{"period":cfg.rewards.gait_period,"stance_ratio":cfg.rewards.gait_stance_ratio,
                "thigh_amplitude":cfg.rewards.gait_thigh_amplitude,"calf_amplitude":cfg.rewards.gait_calf_amplitude,
                "phase_offsets":{"FL":0.0,"FR":0.5,"RL":0.5,"RR":0.0}},
        "episode_length_s":cfg.env.episode_length_s,
    }

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--gym-root",type=Path,default=Path(__file__).resolve().parents[1]/"unitree_rl_gym");a=p.parse_args()
    sys.path.insert(0,str(a.gym_root));cfg=importlib.import_module("legged_gym.envs.fanfan.fanfan_config").FanfanRoughCfg
    state=torch.load(a.checkpoint,map_location="cpu")["model_state_dict"]
    actor=Actor().eval();actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("actor.")})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(actor,torch.zeros(1,cfg.env.num_observations),a.output,input_names=["observations"],output_names=["raw_actions"],dynamic_axes={"observations":{0:"batch"},"raw_actions":{0:"batch"}},opset_version=17)
    manifest=deployment_config(cfg,a.checkpoint,a.gym_root)
    import onnx
    model=onnx.load(a.output);entry=model.metadata_props.add();entry.key="fanfan_deployment_config";entry.value=json.dumps(manifest,separators=(",",":"));onnx.save(model,a.output)
    sidecar=a.output.with_suffix(".json");sidecar.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Exported {a.output} with cfg metadata and {sidecar}")
