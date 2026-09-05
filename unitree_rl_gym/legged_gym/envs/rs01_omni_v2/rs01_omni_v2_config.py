"""Clean random-initialized omnidirectional task for the measured RS01."""

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_estimator_parity_config import (
    Rs01Go2EstimatorParityCfg,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_sim2sim_config import (
    Rs01Go2Sim2SimAdaptCfg,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_straight_config import (
    Rs01Go2StraightCfg,
)


class Rs01OmniV2Cfg(LeggedRobotCfg):
    """An isolated task: reuse RS01 hardware, not the old reward lineage."""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 48 proprioceptive/command values + phase sin/cos + heading sin/cos
        # + lateral trajectory state + explicit gait enable.
        num_observations = 55
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20

    class terrain(Rs01Go2StraightCfg.terrain):
        pass

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 4.0
        heading_command = False

        # Initial reset is always in-place march. These probabilities remain
        # available for validation and future non-transitioned initialization.
        mode_probabilities = [0.05, 0.20, 0.25, 0.10, 0.15, 0.10, 0.15]
        moving_mode_probabilities = [0.32, 0.14, 0.20, 0.18, 0.16]
        moving_to_march_probability = 0.35
        moving_to_stand_probability = 0.10

        forward_velocity_range_m_s = [0.08, 0.20]
        backward_speed_range_m_s = [0.06, 0.12]
        lateral_speed_range_m_s = [0.04, 0.08]
        yaw_speed_range_rad_s = [0.12, 0.35]
        combined_forward_speed_range_m_s = [0.06, 0.16]
        combined_lateral_speed_range_m_s = [0.03, 0.06]
        combined_yaw_speed_range_rad_s = [0.10, 0.25]

        observe_straight_heading_sin_cos = True
        observe_straight_path_state = True
        straight_path_lateral_position_scale = 2.0
        straight_path_lateral_velocity_scale = 2.0
        use_rs01_estimated_observations = True

        class ranges:
            # Sampling is stratified in Rs01OmniV2Robot._resample_commands.
            lin_vel_x = [-0.12, 0.20]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.35, 0.35]
            heading = [0.0, 0.0]

    class init_state(Rs01Go2StraightCfg.init_state):
        reset_heading_noise_rad = 0.10
        reset_yaw_rate_noise_rad_s = 0.08
        reset_path_lateral_error_noise_m = 0.0

    class control(Rs01Go2Sim2SimAdaptCfg.control):
        pass

    class rs01_actuator(Rs01Go2Sim2SimAdaptCfg.rs01_actuator):
        pass

    class rs01_odometry(Rs01Go2EstimatorParityCfg.rs01_odometry):
        pass

    class asset(Rs01Go2StraightCfg.asset):
        name = "rs01_omni_v2"

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Learn the task on the measured nominal machine first. A later,
        # separate robustness task may introduce narrow randomization.
        randomize_friction = False
        friction_range = [1.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        randomize_rs01_actuator = False
        rs01_independent_motor_randomization = False
        rs01_independent_delay_randomization = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        foot_contact_force_threshold = 2.0
        foot_contact_release_force_threshold = 2.0
        gait_period_s = 0.60
        gait_stance_ratio = 0.65
        all_feet_contact_grace_s = 0.12
        foot_collision_radius_m = 0.016
        swing_clearance_m = 0.014
        phase_support_sigma = 0.10

        planar_tracking_sigma = 0.010
        yaw_tracking_sigma = 0.040
        trajectory_lateral_scale_m = 0.10
        heading_error_scale_rad = 0.20
        stance_foot_slip_scale_m_s = 0.20
        roll_scale_rad = 0.12
        rear_torque_balance_ema_s = 0.60
        calf_velocity_soft_limit_rad_s = 8.0
        action_saturation_soft_limit = 0.90
        soft_dof_pos_limit = 0.90
        max_contact_force = 200.0

        class scales(LeggedRobotCfg.rewards.scales):
            # Task objectives.
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_planar_velocity = 2.0
            tracking_yaw_velocity = 0.75
            trajectory_lateral_error = -0.15
            omni_heading_error = -0.15

            # Minimal diagonal-trot contract. No joint trajectory or equal-load
            # reference is imposed.
            phase_support_tracking = 1.0
            phase_swing_clearance = -0.25
            prolonged_all_feet_contact = -0.50
            same_axle_flight = -0.75
            flight = -1.0

            # Hardware feasibility and light regularization only.
            stance_foot_slip = -0.05
            raw_torque_over_peak = -0.50
            motor_saturation = -0.10
            action_saturation = -0.10
            stand_still = -0.15
            lin_vel_z = -0.50
            ang_vel_xy = -0.03
            torques = -0.00005
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.005
            dof_pos_limits = -2.0
            termination = 0.0

    class normalization(Rs01Go2StraightCfg.normalization):
        pass

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class viewer(Rs01Go2StraightCfg.viewer):
        pass

    class sim(Rs01Go2StraightCfg.sim):
        pass


class Rs01OmniV2CfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.20

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1.0e-4
        schedule = "fixed"
        entropy_coef = 0.0

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_omni_v2"
        max_iterations = 3000
        save_interval = 50
        action_std_value = 0.20
        freeze_action_std = True
        actor_output_init_scale = 0.01
        load_optimizer = False
        adapt_observation_input = False
        reference_policy_coef = 0.0
