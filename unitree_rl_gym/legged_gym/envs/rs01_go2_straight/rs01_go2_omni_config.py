"""One-stage omnidirectional training config for the measured RS01."""

from .rs01_go2_estimator_parity_config import (
    Rs01Go2EstimatorParityCfg,
    Rs01Go2EstimatorParityCfgPPO,
)


class Rs01Go2OmniDiagonalCfg(Rs01Go2EstimatorParityCfg):
    """Multidirectional commands inside the currently measured RS01 envelope."""

    class commands(Rs01Go2EstimatorParityCfg.commands):
        curriculum = False
        resampling_time = 4.0
        heading_command = False

        # Explicit strata make both A/B runs see every important motion type.
        mode_probabilities = [0.10, 0.30, 0.10, 0.15, 0.15, 0.20]
        forward_velocity_range_m_s = [0.10, 0.23]
        backward_velocity_range_m_s = [-0.15, -0.08]
        lateral_velocity_range_m_s = [-0.10, 0.10]
        yaw_velocity_range_rad_s = [-0.45, 0.45]
        combined_forward_range_m_s = [-0.15, 0.23]
        combined_lateral_range_m_s = [-0.08, 0.08]
        combined_yaw_range_rad_s = [-0.35, 0.35]
        walking_speed_threshold_m_s = 0.04
        walking_yaw_threshold_rad_s = 0.08
        stand_command_threshold = 0.02

        class ranges:
            # Used by logging/config introspection; sampling is stratified by
            # Rs01Go2OmniDiagonalRobot._resample_commands.
            lin_vel_x = [-0.15, 0.23]
            lin_vel_y = [-0.10, 0.10]
            ang_vel_yaw = [-0.45, 0.45]
            heading = [0.0, 0.0]

    class init_state(Rs01Go2EstimatorParityCfg.init_state):
        # The trajectory reference latches the randomized initial yaw, exactly
        # as an onboard command integrator would.  Yaw-rate perturbation still
        # exposes recovery without inventing an unobservable world heading.
        reset_heading_noise_rad = 0.10
        reset_yaw_rate_noise_rad_s = 0.08
        reset_path_lateral_error_noise_m = 0.0

    class rewards(Rs01Go2EstimatorParityCfg.rewards):
        # Roughly 0.05 m/s planar error retains 78% of tracking reward; a
        # stationary policy at the maximum forward command retains only 0.5%.
        planar_tracking_sigma = 0.010
        yaw_tracking_sigma = 0.040
        trajectory_lateral_scale_m = 0.10
        heading_error_scale_rad = 0.20
        stance_foot_slip_scale_m_s = 0.20

        class scales(Rs01Go2EstimatorParityCfg.rewards.scales):
            # Replace every straight-only objective with body-frame command
            # tracking plus a rotating trajectory/heading reference.
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_forward_velocity = 0.0
            yaw_rate = 0.0
            lateral_velocity = 0.0
            lateral_path_recovery = 0.0
            heading_recovery = 0.0
            near_heading_yaw_rate = 0.0
            tracking_planar_velocity = 2.0
            tracking_yaw_velocity = 0.75
            trajectory_lateral_error = -0.25
            omni_heading_error = -0.25

            # One phase/contact term is enough to enforce FL+RR / FR+RL.
            # Equal load sharing and left-right symmetry would conflict with
            # lateral motion and turning, so they are explicitly disabled.
            phase_support_tracking = 1.0
            diagonal_contact_sync = 0.0
            left_right_foot_force_balance = 0.0
            rear_motor_torque_balance = 0.0
            phase_swing_clearance = -0.35
            rear_swing_clearance = 0.0
            same_axle_flight = -1.0
            flight = -1.0
            prolonged_all_feet_contact = -0.75

            lateral_foot_slip = 0.0
            stance_foot_slip = -0.10
            roll_posture = -0.15
            stand_still = -0.20

            # Keep executable-action and motor limits, but avoid letting their
            # regularization outweigh directional tracking.
            raw_torque_over_peak = -1.0
            motor_saturation = -0.25
            action_saturation = -0.25
            calf_velocity_excess = -0.10
            torques = -0.0001
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -10.0


class Rs01Go2OmniDiagonalCfgPPO(Rs01Go2EstimatorParityCfgPPO):
    class policy(Rs01Go2EstimatorParityCfgPPO.policy):
        init_noise_std = 0.25

    class algorithm(Rs01Go2EstimatorParityCfgPPO.algorithm):
        learning_rate = 2.0e-4
        schedule = "adaptive"
        entropy_coef = 0.005

    class runner(Rs01Go2EstimatorParityCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_omni_diagonal"
        max_iterations = 3000
        save_interval = 50
        action_std_value = 0.25
        freeze_action_std = False
        load_optimizer = False
        adapt_observation_input = False
        # A straight-only reference policy would penalize every useful lateral
        # and yaw action. Warm-start transfers weights, not behavioral shackles.
        reference_policy_coef = 0.0
