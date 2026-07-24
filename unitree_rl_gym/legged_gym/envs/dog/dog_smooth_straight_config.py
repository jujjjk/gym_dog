"""Speed, balance, symmetry and path polish from selected safe-v2 model_8700."""

from .dog_safe_torque_path_v2_config import (
    DogRs01SafeTorquePathV2Cfg,
    DogRs01SafeTorquePathV2CfgPPO,
)


class DogRs01SmoothStraightCfg(DogRs01SafeTorquePathV2Cfg):
    """Favor a smooth commanded-speed trot over minimum motor torque."""

    class control(DogRs01SafeTorquePathV2Cfg.control):
        # Keep the measured RS01 actuator chain and 6 Nm thermal reference,
        # but retain transient support authority for the heavy trunk.
        continuous_derating_start_ratio = 1.05
        continuous_derating_full_ratio = 1.70
        continuous_derating_curriculum_iterations = 0

    class commands(DogRs01SafeTorquePathV2Cfg.commands):
        class ranges(DogRs01SafeTorquePathV2Cfg.commands.ranges):
            # A moderately faster envelope without rewarding uncontrolled speed.
            lin_vel_x = [0.13, 0.18]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(DogRs01SafeTorquePathV2Cfg.rewards):
        # Start with two-frame contact tolerance, then finish under the same
        # strict one-frame diagonal contract used by play and evaluation.
        non_diagonal_termination_curriculum = [
            {"until_iteration": 700, "steps": 2},
            {"until_iteration": 1.0e12, "steps": 1},
        ]
        continuous_torque_penalty_curriculum_iterations = 0

        class scales(DogRs01SafeTorquePathV2Cfg.rewards.scales):
            # Match the command instead of maximizing unbounded progress.
            tracking_lin_vel = 18.0
            normalized_command_tracking = 18.0
            absolute_longitudinal_tracking_error = -16.0
            commanded_smooth_straight_progress = 18.0
            command_velocity_progress = 12.0
            forward_progress_with_diagonal_swing = 10.0
            forward_diagonal_pair_swing = 12.0
            straight_balanced_progress = 16.0
            safe_torque_straight_progress = 5.0
            compact_symmetric_forward = 10.0

            # Suppress periodic trunk motion and correct world-frame drift.
            straight_path_lateral_displacement = -14.0
            straight_path_lateral_velocity = -48.0
            straight_path_lateral_acceleration = -2.0
            straight_lateral_speed = -38.0
            straight_heading_error = -26.0
            yaw_rate = -18.0
            body_angular_velocity = -7.0
            body_angular_acceleration = -3.0
            handoff_body_twist = -10.0
            translation_roll = -42.0
            orientation = -15.0
            ang_vel_xy = -7.0
            lin_vel_z = -20.0
            straight_contact_lateral_force = -22.0
            straight_contact_yaw_moment = -32.0
            straight_contact_side_load_balance = -16.0

            # Coordinate physical feet/joints; measured motors may still use
            # different direct outputs to produce the same physical motion.
            exact_diagonal_swing = 14.0
            scheduled_diagonal_pair_lift = 5.0
            touchdown_pair_support = 4.0
            diagonal_load_transfer = 5.0
            diagonal_contact_sync_all = -18.0
            diagonal_foot_height_sync_all = -60.0
            diagonal_stride_sync_shortfall = -8.0
            straight_diagonal_contact_sync = -5.0
            straight_diagonal_target_mirror = -0.50
            straight_diagonal_joint_mirror = -0.35
            straight_diagonal_torque_mirror = -0.10
            straight_policy_side_balance = -0.05
            straight_torque_side_balance = -0.12

            # Smooth target reversals and impacts without making the heavy
            # robot too soft to support itself.
            action_rate = -0.10
            policy_action_rate = -0.10
            dof_acc = -1.5e-6
            final_target_acceleration = -0.08
            mechanical_power = -0.0025

            # Torque is now a safety/efficiency secondary objective.
            motor_continuous_usage = -0.50
            motor_continuous_overload = -1.20
            motor_thermal_overload = -1.50
            motor_thermal_peak = -1.00
            sustained_torque = -0.80
            sustained_torque_max = -0.90
            motor_torque_usage = -2.50
            sagittal_motor_saturation = -2.50
            torque_clip = -1.20
            torque_near_limit = -0.45
            peak_torque = -0.60
            torques = -1.5e-5


class DogRs01SmoothStraightCfgPPO(DogRs01SafeTorquePathV2CfgPPO):
    """Conservative continuation from the selected model_8700."""

    class algorithm(DogRs01SafeTorquePathV2CfgPPO.algorithm):
        learning_rate = 7.5e-6
        entropy_coef = 3.0e-5
        schedule = "fixed"

    class runner(DogRs01SafeTorquePathV2CfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_smooth_straight_v1"
        resume = True
        load_run = "Jul24_10-48-23_rs01_direct12_safe6nm_path_v2"
        checkpoint = 8700
        load_optimizer = False
        symmetry_coef = 0.72
        max_iterations = 1000
        save_interval = 25
