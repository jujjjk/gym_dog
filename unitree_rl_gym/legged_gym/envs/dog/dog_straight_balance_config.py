"""Straight-path and body-balance continuation from selected model_7000."""

from .dog_hip_torque_config import (
    DogRs01HipTorqueCfg,
    DogRs01HipTorqueCfgPPO,
)


class DogRs01StraightBalanceCfg(DogRs01HipTorqueCfg):
    """Correct lateral/heading drift without losing the compact diagonal gait."""

    class rewards(DogRs01HipTorqueCfg.rewards):
        straight_path_lateral_deadband_m = 0.020
        straight_path_lateral_penalty_width_m = 0.180

        class scales(DogRs01HipTorqueCfg.rewards.scales):
            # Positive credit is jointly gated by useful forward motion,
            # world-path side slip, heading, attitude, and angular quietness.
            # Standing or twisting in place therefore cannot exploit positive
            # reward clipping.
            straight_balanced_progress = 16.0

            # World-path correction.  model_7000 measured about 0.31 m mean
            # absolute lateral displacement and 0.36 rad absolute heading
            # error in deterministic evaluation.
            straight_path_lateral_displacement = -8.0
            straight_path_lateral_velocity = -36.0
            straight_path_lateral_acceleration = -1.5
            straight_lateral_speed = -30.0
            straight_heading_error = -16.0
            yaw_rate = -12.0

            # Keep the heavy trunk level and remove the ground-force source of
            # residual yaw/roll rather than masking it with large hip motion.
            translation_roll = -36.0
            orientation = -12.0
            ang_vel_xy = -6.0
            body_angular_velocity = -5.0
            body_angular_acceleration = -2.2
            handoff_body_twist = -8.0
            straight_contact_lateral_force = -18.0
            straight_contact_yaw_moment = -26.0
            straight_contact_side_load_balance = -12.0

            # Preserve the selected compact-hip, low-torque diagonal gait.
            compact_hip_low_torque_forward = 12.0
            hip_joint_excursion = -7.0
            hip_target_excursion = -3.0
            motor_torque_usage = -4.0
            sagittal_motor_saturation = -4.0
            smooth_low_torque_forward = 12.0
            smooth_diagonal_handoff = 10.0
            exact_diagonal_swing = 8.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            tracking_lin_vel = 8.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 13.0


class DogRs01StraightBalanceCfgPPO(DogRs01HipTorqueCfgPPO):
    """Low-rate path/balance polish from the selected model_7000."""

    class algorithm(DogRs01HipTorqueCfgPPO.algorithm):
        learning_rate = 1.0e-5
        entropy_coef = 5.0e-5
        schedule = "fixed"

    class runner(DogRs01HipTorqueCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_straight_balance_v1"
        resume = True
        load_run = "Jul23_19-03-58_rs01_direct12_hip_compact_torque_v1"
        checkpoint = 7000
        load_optimizer = False
        # Retain physical diagonal symmetry while allowing the policy to
        # compensate the measured left/right RS01 motor differences.
        symmetry_coef = 0.60
        max_iterations = 1000
        save_interval = 25
