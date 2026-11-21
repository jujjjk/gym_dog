"""Continuous-command AMP/CTS continuation on the unchanged RS01 plant."""
from .amp_cts_config import Urdf4AMPCTSCfg,Urdf4AMPCTSPPO

class Urdf4AMPExpandCfg(Urdf4AMPCTSCfg):
    class env(Urdf4AMPCTSCfg.env):
        # One private PPO reference weight; the student still receives 1020.
        num_observations=1121
        num_privileged_obs=1121
    class commands(Urdf4AMPCTSCfg.commands):
        resampling_time=4.
        reference_probability=.25
        stand_probability=.08
        march_probability=.02
        single_axis_probability=.70
        class ranges(Urdf4AMPCTSCfg.commands.ranges):
            lin_vel_x=[-.25,.40]
            lin_vel_y=[-.15,.15]
            ang_vel_yaw=[-.60,.60]
    class expand:
        robust_training=True
        nominal_stride=4
        max_stage=1
        stages=[[-.18,.28,.10,.40],[-.25,.40,.15,.60],[-.30,.60,.25,1.0]]
        frontier_probability=.30
        command_acceleration=[.4,.3,.8]
        cold_settle_s=1.
        recovery_grace_s=1.
        curriculum_window=50
        curriculum_min_updates=200
        curriculum_required_windows=2
        curriculum_min_samples_per_direction=200
        curriculum_linear_error_m_s=.04
        curriculum_yaw_error_rad_s=.10
        curriculum_illegal_fraction=.001
        curriculum_failure_fraction=.001
        curriculum_max_phase_error=.25
        curriculum_min_diagonal_fraction=.05
    class amp_cts(Urdf4AMPCTSCfg.amp_cts):
        reward_revision='v4_continuous_robust_bounded'
        stage='expand_robust'
        reference_reset_probability=.4
    class domain_rand(Urdf4AMPCTSCfg.domain_rand):
        randomize_friction=True
        friction_range=[.6,1.0]
        randomize_base_mass=True
        added_mass_range=[-.58,.58]
        randomize_base_com=True
        base_com_offset_m=[.005,.005,.005]
        randomize_rs01_actuator=True
        rs01_response_gain_scale_range=[.95,1.05]
        rs01_time_constant_scale_range=[.9,1.1]
        rs01_friction_scale_range=[1.,1.]
        rs01_delay_step_offset_range=[-1,1]
    class push(Urdf4AMPCTSCfg.push):
        enabled=True
        peak_force_n=[5.,10.]
        duration_s=[.1,.2]
        interval_s=[6.,10.]
        first_delay_s=[3.,6.]
        initial_strength=.3
        ramp_s=120.
    class imu(Urdf4AMPCTSCfg.imu):
        enabled=True
        clean_env_fraction=0.
        latency_range_s=[0.,.010]
        mount_roll_pitch_deg=.3
        mount_yaw_deg=.3
        gyro_noise_std_rad_s=.004
        gyro_bias_rad_s=.003
        gyro_bias_walk_rad_s_sqrt_s=.0001
        gyro_bias_limit_rad_s=.005
        gyro_scale_error=.01
        gravity_noise_std=.003
        attitude_drift_std_deg=.2
        attitude_drift_limit_deg=.5
        spike_probability=0.
        stale_frame_probability=0.
    class joint_sensor(Urdf4AMPCTSCfg.joint_sensor):
        enabled=True
        latency_range_s=[0.,.005]
        position_noise_rad=.003
        velocity_noise_rad_s=.10
        zero_offset_rad=.003
    class noise(Urdf4AMPCTSCfg.noise):
        add_noise=True
    class rewards(Urdf4AMPCTSCfg.rewards):
        class scales:
            command_tracking=1.5
            gait_quality=-1.
            raw_torque_over_peak=-.5
            collision=-1.
            action_rate=-.001
            termination=-10.

class Urdf4AMPExpandPPO(Urdf4AMPCTSPPO):
    class policy(Urdf4AMPCTSPPO.policy):
        command_frame_min=[-.36,-.20,-.10]
        command_frame_max=[.56,.20,.10]
    class algorithm(Urdf4AMPCTSPPO.algorithm):
        learning_rate=5.e-5
        entropy_coef=.001
    class runner(Urdf4AMPCTSPPO.runner):
        cts_expand_enabled=True
        training_objective='expand_robust'
        experiment_name='urdf4_amp_cts_expand'
        max_iterations=30000
        seed_trust_coef=10.
        seed_action_deadband=.10
        seed_weight_index=1119
        encoder_learning_rate=1.e-4


class Urdf4AMPExpandXCfg(Urdf4AMPExpandCfg):
    """Expand straight forward speed; preserve every other command limit."""
    class commands(Urdf4AMPExpandCfg.commands):
        reference_probability=.60
        class ranges(Urdf4AMPExpandCfg.commands.ranges):
            lin_vel_x=[-.18,.60]
            lin_vel_y=[-.10,.10]
            ang_vel_yaw=[-.40,.40]
    class expand(Urdf4AMPExpandCfg.expand):
        x_only=True
        initial_stage=1
        max_stage=2
        stages=[[-.18,.28,.10,.40],[-.18,.40,.10,.40],[-.18,.60,.10,.40]]
    class amp_cts(Urdf4AMPExpandCfg.amp_cts):
        reward_revision='v4_continuous_robust_bounded_xonly'


class Urdf4AMPExpandXPPO(Urdf4AMPExpandPPO):
    class runner(Urdf4AMPExpandPPO.runner):
        experiment_name='urdf4_amp_cts_expand_x'
        # Preserve the validated low-speed gait. This existing hinge is masked
        # outside the old nominal command box; pushes/high speeds remain free.
        seed_trust_coef=1000.
        seed_action_deadband=.05
