"""Direction fine tuning: unchanged AMP/CTS student and RS01 plant."""
from .amp_expand_config import Urdf4AMPExpandXCfg,Urdf4AMPExpandXPPO


class Urdf4AMPDirectionCfg(Urdf4AMPExpandXCfg):
    class commands(Urdf4AMPExpandXCfg.commands):
        resampling_time=12.
        reference_probability=.72
        stand_probability=.08
        march_probability=0.
    class expand(Urdf4AMPExpandXCfg.expand):
        initial_stage=0
        frontier_probability=.20
        stages=[[-.18,.25,.10,.40],[-.18,.40,.10,.40],[-.18,.60,.10,.40]]
    class amp_cts(Urdf4AMPExpandXCfg.amp_cts):
        reward_revision='v5_direction_external_pose'
        stage='omni_direction'
        reference_reset_probability=0.  # Explicit cold starts, no hidden pose/history teleport.
        teacher_fraction=.5
    class rewards(Urdf4AMPExpandXCfg.rewards):
        heading_gain=0.  # Heading correction is already in the commanded twist.
    class direction:
        position_noise_m=.003
        yaw_noise_deg=.2
        velocity_noise_m_s=.005
        pose_latency_steps=2
        curriculum_heading_rad=.035
        curriculum_cross_m=.03
        gains=dict(filter_tau_s=.5,yaw_damping=0.,cross_damping=.2,cross_gain=.8,
            lateral_limit_m_s=.04,lateral_acceleration=.15,straight_along_gain=0.,
            straight_speed_damping=0.,arc_left_bias=[0.,-.02,0.],
            arc_right_bias=[-.05,-.015,.03],transverse_cross_gain=1.2,
            nonforward_heading_integral_gain=.25,output_yaw_limit_rad_s=.55,
            left_command_scale=.90,turn_position_gain=1.0,nonforward_yaw_limit_rad_s=.30)


class Urdf4AMPDirectionPPO(Urdf4AMPExpandXPPO):
    class algorithm(Urdf4AMPExpandXPPO.algorithm):
        learning_rate=3.e-5
        entropy_coef=.0005
    class runner(Urdf4AMPExpandXPPO.runner):
        cts_direction_enabled=True
        experiment_name='urdf4_amp_cts_direction'
        seed_trust_coef=300.
        seed_action_deadband=.05
        save_interval=100
