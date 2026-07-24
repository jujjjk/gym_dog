"""Guarded straight/yaw correction from the selected 76-observation model."""

from .dog_smooth_straight_v2_config import (
    DogRs01SmoothStraightV2Cfg,
    DogRs01SmoothStraightV2CfgPPO,
)


class DogRs01StraightGuardedCfg(DogRs01SmoothStraightV2Cfg):
    """Closed-loop path/yaw correction with selected-gait preservation."""

    class rewards(DogRs01SmoothStraightV2Cfg.rewards):
        straight_heading_recovery_gain_s = 2.0
        straight_heading_recovery_max_rate_rad_s = 0.80

        # The fixed-reference policy loss is the primary gait-preservation
        # guard. These terms retain explicit physical guards as a second line.
        class scales(DogRs01SmoothStraightV2Cfg.rewards.scales):
            tracking_lin_vel = 6.0
            normalized_command_tracking = 5.0
            absolute_longitudinal_tracking_error = -2.0
            commanded_smooth_straight_progress = 10.0

            exact_diagonal_swing = 10.0
            scheduled_diagonal_pair_lift = 12.0
            touchdown_pair_support = 5.0
            diagonal_load_transfer = 5.0
            diagonal_contact_sync_all = -24.0
            diagonal_foot_height_sync_all = -80.0
            diagonal_stride_sync_shortfall = -8.0
            diagonal_joint_sync = -1.0
            non_diagonal_swing = -18.0
            phase_contact_mismatch = -5.0
            flight = -24.0
            all_feet_contact = -1.5
            excessive_foot_contact_time = -0.3

            # The correction head observes instantaneous lateral velocity,
            # yaw rate, phase and heading error. Do not optimize unobserved
            # absolute lateral position as the dominant objective.
            straight_path_recovery_velocity = 0.0
            straight_heading_recovery_rate = -6.0
            straight_path_lateral_displacement = -0.2
            straight_path_lateral_velocity = -8.0
            straight_lateral_speed = -2.0
            straight_heading_error = -3.0
            yaw_rate = -0.5
            body_angular_velocity = -1.5
            body_angular_acceleration = -0.4
            straight_contact_lateral_force = -2.0
            straight_contact_yaw_moment = -2.0
            orientation = -2.0

            action_rate = -0.04
            torque_clip = -0.5
            motor_thermal_overload = -0.5


class DogRs01StraightGuardedCfgPPO(DogRs01SmoothStraightV2CfgPPO):
    """Short, trust-region continuation from structural model_9200."""

    class algorithm(DogRs01SmoothStraightV2CfgPPO.algorithm):
        learning_rate = 5.0e-5
        entropy_coef = 0.0
        clip_param = 0.10
        max_grad_norm = 0.5
        schedule = "fixed"

    class runner(DogRs01SmoothStraightV2CfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_pair_sync_correction_v4"
        resume = True
        load_run = "Jul24_15-09-38_rs01_direct12_structural_rebuild_v1"
        checkpoint = 9200
        load_optimizer = False
        adapt_observation_input = False
        symmetry_coef = 0.0

        # Freeze the full model_9200 locomotion actor. Only a bounded
        # phase/state/torque correction head is trainable. Motor torque is a
        # deployable proxy for foot loading; no foot-force observation is used.
        phase_residual_policy = True
        residual_feature_indices = [
            1, 5, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ]
        residual_hidden_dim = 64
        residual_action_scale = 0.03
        reference_policy_coef = 0.2
        reference_action_deadband = 0.02
        reference_action_hinge_coef = 8.0

        # Keep the continuation short and retain dense checkpoints for strict
        # regression selection against model_9200.
        max_iterations = 400
        save_interval = 25
