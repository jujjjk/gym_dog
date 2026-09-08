"""Compare the independent bridge against actual Torch training methods."""
import json
from pathlib import Path
import isaacgym  # must precede torch
import torch
import numpy as np
from export_v13 import Rs01OmniV13DirectionCfg
from sim2sim_v13 import V13Sim, roll_pitch_yaw, quaternion_rotation_matrix
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_env import Rs01OmniV13Robot
from legged_gym.envs.rs01_go2_straight.rs01_odometry import Rs01TorchLegOdometry


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'artifacts/rs01_v13_a6850_sim2sim'
    sim=V13Sim(out/'scene.xml',out/'model_6850.onnx',[.2,0,0])
    r=object.__new__(Rs01OmniV13Robot)
    r.cfg=Rs01OmniV13DirectionCfg(); r.device='cpu'; r.num_envs=1; r.add_noise=False
    r._direction_ready=r._omni_reference_ready=r._wide_phase_ready=True
    r._v2_command_ready=r._rs01_observation_estimator_ready=True
    r.obs_scales=r.cfg.normalization.obs_scales
    r.commands_scale=torch.tensor(sim.obs_cfg['command_scale'])
    r.default_dof_pos=torch.tensor(sim.default[None],dtype=torch.float)
    r._straight_path_observation_state=lambda: (torch.zeros(1),torch.zeros(1))
    r.omni_desired_position_xy=torch.zeros(1,2); r.omni_estimated_position_xy=torch.zeros(1,2)
    c=r.cfg.rs01_odometry
    odom=Rs01TorchLegOdometry(1,'cpu',nominal_base_height=c.nominal_base_height_m,
        foot_radius=c.foot_radius_m,height_margin=c.height_margin_m,
        vertical_speed_threshold=c.vertical_speed_threshold_m_s,
        velocity_residual_threshold=c.velocity_residual_threshold_m_s,
        filter_alpha=c.filter_alpha,no_contact_decay=c.no_contact_decay,
        previous_stance_score_bonus=c.previous_stance_score_bonus,
        strict_diagonal_pairs=getattr(c,'strict_diagonal_pairs',False))
    tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float)
    errors=dict(observation_max=0.,direction_target_max=0.,frequency_max=0.,odometry_velocity_max=0.,odometry_confidence_max=0.)
    for step in range(180):
        if step in (25,50,75,100,125,150):
            sim.set_command({25:[0,.2,0],50:[0,0,.3],75:[-.2,0,0],100:[0,0,0],125:[.3,.1,.25],150:[.6,0,0]}[step])
        r.commands=tensor([list(sim.command)+[0.]])
        r.direction_turning=torch.tensor([sim.turning]); r.gait_enable=tensor([sim.gait_enable])
        r.rpy=tensor([roll_pitch_yaw(sim.data.qpos[3:7])])
        r.omni_desired_heading_rad=tensor([sim.heading_target])
        r.wide_gait_phase=tensor([sim.phase])
        r.estimated_base_lin_vel=tensor([sim.last_estimated_linear_velocity])
        r.estimated_odom_confidence=tensor([sim.confidence])
        r.base_lin_vel=tensor([sim.base_velocity_body()[0]])
        r.base_ang_vel=tensor([sim.base_velocity_body()[1]])
        r.projected_gravity=tensor([quaternion_rotation_matrix(sim.data.qpos[3:7]).T @ np.array([0,0,-1])])
        r.dof_pos=tensor([sim.data.qpos[sim.qpos_indices]])
        r.dof_vel=tensor([sim.data.qvel[sim.qvel_indices]])
        r.actions=tensor([sim.action])
        r.compute_observations()
        errors['observation_max']=max(errors['observation_max'],float(np.max(np.abs(r.obs_buf.numpy()[0]-sim.observation()))))
        errors['direction_target_max']=max(errors['direction_target_max'],float(np.max(np.abs(r._direction_velocity_target().numpy()[0]-sim.effective_command()))))
        errors['frequency_max']=max(errors['frequency_max'],abs(r._command_gait_frequency().item()-sim.frequency()))
        sim.control_step()
        result=odom.estimate(tensor([sim.data.qpos[sim.qpos_indices]]),tensor([sim.data.qvel[sim.qvel_indices]]),tensor([sim.base_velocity_body()[1]]))
        errors['odometry_velocity_max']=max(errors['odometry_velocity_max'],float(np.max(np.abs(result['base_linear_velocity'].numpy()[0]-sim.last_estimated_linear_velocity))))
        errors['odometry_confidence_max']=max(errors['odometry_confidence_max'],abs(result['confidence'].item()-sim.confidence))
    print(json.dumps(errors,indent=2))
    assert max(errors.values()) < 1e-4, errors
    (out/'parity.json').write_text(json.dumps(errors,indent=2)+'\n')


if __name__=='__main__': main()
