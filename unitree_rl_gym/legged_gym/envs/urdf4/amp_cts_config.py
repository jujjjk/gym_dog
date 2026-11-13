"""V2-covered omnidirectional style acquisition; nominal RS01 plant."""
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR
from .urdf4_omni_config import Urdf4OmniCfg, Urdf4OmniCfgPPO
ROOT = Path(LEGGED_GYM_ROOT_DIR).parent

class Urdf4AMPCTSCfg(Urdf4OmniCfg):
    class env(Urdf4OmniCfg.env):
        frame_size = 51  # original 48 + heading sin/cos-1 + gait-enable
        history_length = 20
        public_obs_size = 51 * 20
        teacher_state_size = 99  # original privileged 96 + same three public fields
        num_observations = public_obs_size + teacher_state_size + 1
        num_privileged_obs = num_observations  # packed input; critic selects its own fields
        num_envs = 4096
    class commands(Urdf4OmniCfg.commands):
        resampling_time = 5.
        stand_probability = .10
        march_probability = .0
        reference_probability = .90
        single_axis_probability = .50
        class ranges:
            lin_vel_x = [-.18, .28]
            lin_vel_y = [-.1, .1]
            ang_vel_yaw = [-.4, .4]
            heading = [0., 0.]
    class command_curriculum(Urdf4OmniCfg.command_curriculum):
        enabled = False
    class init_state(Urdf4OmniCfg.init_state):
        reset_dof_position_noise_rad = .02
        reset_linear_velocity_m_s = 0.
        reset_angular_velocity_rad_s = 0.
        randomize_yaw = False
    class domain_rand(Urdf4OmniCfg.domain_rand):
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        randomize_base_com = False
        randomize_link_mass = False
        randomize_armature = False
        randomize_gains = False
        randomize_motor_strength = False
        randomize_rs01_actuator = False
        push_robots = False
    class push(Urdf4OmniCfg.push):
        enabled = False
    class imu(Urdf4OmniCfg.imu):
        enabled = False
    class joint_sensor(Urdf4OmniCfg.joint_sensor):
        enabled = False
    class noise(Urdf4OmniCfg.noise):
        add_noise = False
    class gait(Urdf4OmniCfg.gait):
        stance_ratio = .68
        swing_clearance_m = .035
        expert_command_scale = [.28, .10, .40]
        fast_stance_ratio = .54  # retain double support, allow more swing return time
    class rewards(Urdf4OmniCfg.rewards):
        # Deliberately do not inherit the old reward scale list.
        # Each objective has one owner: commands, reference geometry, or safety.
        class scales:
            command_tracking = .5
            phase_reference = 1.
            illegal_support = -1.
            raw_torque_over_peak = -.5
            collision = -1.
            action_rate = -.002
            termination = -10.
        command_linear_sigma_m_s = .08
        command_yaw_sigma_rad_s = .20
        heading_gain = .75
        heading_correction_limit_rad_s = .20
        reference_foot_sigma_m = .025
    class amp_cts:
        reference = str(ROOT / 'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2')
        receipt = str(ROOT / 'artifacts/urdf4_amp_isaac_validation_20261113/validation_receipt.json')
        reward_revision = 'v2_style_cycle_v2'
        stage = 'v2_style'
        teacher_fraction = .75
        student_only = False

class Urdf4AMPCTSPPO(Urdf4OmniCfgPPO):
    class policy(Urdf4OmniCfgPPO.policy):
        init_noise_std = .15
        latent_size = 16
        encoder_hidden_dims = [256, 128]
    class algorithm(Urdf4OmniCfgPPO.algorithm):
        learning_rate = 3.e-4
        entropy_coef = .001
        num_learning_epochs = 4
        num_mini_batches = 4
    class runner(Urdf4OmniCfgPPO.runner):
        cts_amp_enabled = True
        amp_enabled = False
        experiment_name = 'urdf4_amp_cts_style'
        max_iterations = 8000
        num_steps_per_env = 24
        save_interval = 100
        amp_style_weight = 2.
        amp_warmup_iterations = 20
        amp_disc_steps = 2
        amp_feature_weights = [1.] * 46
        amp_window_frames = 6
        amp_window_stride = 5  # six samples span 0.5 s, one full gait cycle
        encoder_learning_rate = 3.e-4
        encoder_batches = 4
        encoder_batch_size = 2048
