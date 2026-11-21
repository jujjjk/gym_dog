"""New-mass V3 all-direction AMP+CTS. No old task/data are overwritten."""
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR
from .amp_direction_config import Urdf4AMPDirectionCfg,Urdf4AMPDirectionPPO
ROOT=Path(LEGGED_GYM_ROOT_DIR).parent


class Urdf4AMPV3Cfg(Urdf4AMPDirectionCfg):
    class env(Urdf4AMPDirectionCfg.env):
        num_envs=4096
        episode_length_s=30.
    class normalization(Urdf4AMPDirectionCfg.normalization):
        clip_actions=4.
    class commands(Urdf4AMPDirectionCfg.commands):
        resampling_time=8.
        reference_probability=.65
        stand_probability=.10
        march_probability=0.
        ramp_s=2.
        fast_ramp_s=3.
        class ranges:
            lin_vel_x=[-.35,.60]
            lin_vel_y=[-.20,.20]
            ang_vel_yaw=[-.80,.80]
            heading=[0.,0.]
    class expand(Urdf4AMPDirectionCfg.expand):
        x_only=False
        initial_stage=0
        max_stage=2
        stages=[[-.20,.30,.10,.40],[-.35,.50,.20,.60],[-.35,.60,.20,.80]]
        frontier_probability=.25
        curriculum_window=50
        curriculum_min_updates=200
        curriculum_required_windows=2
        curriculum_max_phase_error=.25
        cold_settle_s=1.
    class amp_cts(Urdf4AMPDirectionCfg.amp_cts):
        reference=str(ROOT/'artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/gym_verified')
        receipt=str(ROOT/'artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/validation_receipt.json')
        reference_bank=str(ROOT/'artifacts/urdf4_amp_v3_training/no_legacy_reference_bank.pt')
        reference_reset_probability=0.
        variable_cadence=True
        prior_archive=str(ROOT/'artifacts/urdf4_amp_v3_training/reference_prior.pt')
        reward_revision='v6_reference_prior_new_mass_v3_omni_direction'
        stage='v3_new_mass_omni'
        teacher_fraction=.5
    class domain_rand(Urdf4AMPDirectionCfg.domain_rand):
        friction_range=[.6,1.1]
        added_mass_range=[-.60,.60]
        base_com_offset_m=[.008,.008,.008]
        randomize_link_mass=True
        link_mass_scale_range=[.97,1.03]
        randomize_armature=True
        armature_range_kg_m2=[0.,.001]
        randomize_gains=True
        stiffness_scale_range=[.95,1.05]
        damping_scale_range=[.9,1.1]
        randomize_motor_strength=True
        motor_strength_range=[.95,1.05]
        rs01_response_gain_scale_range=[.95,1.05]
        rs01_time_constant_scale_range=[.9,1.1]
        rs01_delay_step_offset_range=[-1,1]
    class push(Urdf4AMPDirectionCfg.push):
        enabled=True
        peak_force_n=[8.,18.]
        duration_s=[.10,.20]
        interval_s=[6.,10.]
        initial_strength=.25
        ramp_s=120.
    class rewards(Urdf4AMPDirectionCfg.rewards):
        heading_gain=0.
        class scales:
            command_tracking=1.
            gait_quality=-1.
            raw_torque_over_peak=-.5
            collision=-1.
            action_rate=-.001
            termination=-10.
    class direction(Urdf4AMPDirectionCfg.direction):
        gains=dict(filter_tau_s=.3,yaw_damping=.1,cross_damping=.2,cross_gain=1.2,
            heading_gain=3.,lateral_limit_m_s=.08,lateral_acceleration=.3,
            yaw_acceleration=.8,straight_along_gain=0.,straight_speed_damping=0.,
            arc_left_bias=[0.,0.,0.],arc_right_bias=[0.,0.,0.],
            transverse_cross_gain=1.2,nonforward_heading_integral_gain=.25,
            output_yaw_limit_rad_s=1.1,output_lateral_limit_m_s=.30,
            forward_heading_integral_gain=.25,
            left_command_scale=1.,turn_position_gain=1.,nonforward_yaw_limit_rad_s=.25)


class Urdf4AMPV3PPO(Urdf4AMPDirectionPPO):
    class policy(Urdf4AMPDirectionPPO.policy):
        command_frame_min=[-.90,-.60,-.30]
        command_frame_max=[1.40,.60,.30]
    class algorithm(Urdf4AMPDirectionPPO.algorithm):
        learning_rate=3.e-5
        entropy_coef=.0005
    class runner(Urdf4AMPDirectionPPO.runner):
        cts_v3_enabled=True
        cts_direction_enabled=False
        cts_expand_enabled=False
        experiment_name='urdf4_amp_cts_v3'
        max_iterations=30000
        save_interval=100
        amp_style_weight=3.
        amp_disc_steps=2
        amp_warmup_iterations=20
        seed_trust_coef=0.
        seed_weight_index=1119
        training_objective='v3_omni_direction_robust'
        v3_preflight=str(ROOT/'artifacts/urdf4_amp_v3_training/preflight.json')
