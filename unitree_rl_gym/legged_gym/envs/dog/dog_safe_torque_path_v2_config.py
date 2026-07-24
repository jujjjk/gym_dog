"""Recoverable 6 Nm torque curriculum, restarted from selected model_7750."""

from .dog_safe_torque_path_config import (
    DogRs01SafeTorquePathCfg,
    DogRs01SafeTorquePathCfgPPO,
)


class DogRs01SafeTorquePathV2Cfg(DogRs01SafeTorquePathCfg):
    """Reduce sustained torque without collapsing complete gait cycles."""

    class control(DogRs01SafeTorquePathCfg.control):
        # Motor thermal behavior is much slower than the failed v1's 2 s
        # approximation. Peak torque remains transient and eventually derates
        # to the 6 Nm continuous rating, but not inside one or two steps.
        continuous_torque_thermal_time_constant_s = 8.0
        continuous_torque_initial_thermal_ratio = 0.65
        continuous_derating_start_ratio = 1.00
        continuous_derating_full_ratio = 1.50
        continuous_derating_curriculum_iterations = 1000

    class rewards(DogRs01SafeTorquePathCfg.rewards):
        # Training must observe complete diagonal cycles before deployment's
        # one-frame rule is restored. Evaluation always ignores this curriculum
        # and uses the inherited strict one-frame termination.
        non_diagonal_swing_termination_steps = 1
        non_diagonal_termination_curriculum = [
            {"until_iteration": 500, "steps": 3},
            {"until_iteration": 1100, "steps": 2},
            {"until_iteration": 1.0e12, "steps": 1},
        ]
        continuous_torque_penalty_initial_blend = 0.25
        continuous_torque_penalty_curriculum_iterations = 700

        class scales(DogRs01SafeTorquePathCfg.rewards.scales):
            # v1's combined torque penalties drove total locomotion reward to
            # zero before the policy could adapt. Keep a strong continuous-
            # rating objective, but let forward/support gradients survive.
            safe_torque_straight_progress = 16.0
            motor_continuous_usage = -1.5
            motor_continuous_overload = -3.0
            motor_thermal_overload = -4.0
            motor_thermal_peak = -2.5
            sustained_torque = -1.3
            sustained_torque_max = -1.5
            motor_torque_usage = -4.0
            sagittal_motor_saturation = -4.0
            torque_clip = -1.8
            torque_near_limit = -0.7
            peak_torque = -0.85
            torques = -2.0e-5
            mechanical_power = -0.0035

            # Correct drift without overwhelming the contact/gait objective.
            straight_path_lateral_displacement = -10.0
            straight_path_lateral_velocity = -40.0
            straight_path_lateral_acceleration = -1.5
            straight_lateral_speed = -32.0
            straight_heading_error = -18.0
            yaw_rate = -12.0
            straight_balanced_progress = 18.0
            body_angular_velocity = -5.0
            body_angular_acceleration = -2.2
            translation_roll = -36.0
            orientation = -12.0

            # Preserve complete physical diagonal cycles while torque is
            # redistributed among the four stance/swing legs.
            exact_diagonal_swing = 12.0
            forward_diagonal_pair_swing = 16.0
            forward_progress_with_diagonal_swing = 18.0
            scheduled_diagonal_pair_lift = 4.0
            touchdown_pair_support = 3.0
            diagonal_load_transfer = 3.0
            tracking_lin_vel = 10.0
            command_velocity_progress = 22.0
            normalized_command_tracking = 15.0
            compact_symmetric_forward = 14.0


class DogRs01SafeTorquePathV2CfgPPO(DogRs01SafeTorquePathCfgPPO):
    """Longer, recoverable continuation from pre-failure model_7750."""

    class algorithm(DogRs01SafeTorquePathCfgPPO.algorithm):
        learning_rate = 1.0e-5
        entropy_coef = 5.0e-5
        schedule = "fixed"

    class runner(DogRs01SafeTorquePathCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_safe6nm_path_v2"
        resume = True
        load_run = "Jul23_21-29-50_rs01_direct12_compact_hip_coord_v1"
        checkpoint = 7750
        load_optimizer = False
        symmetry_coef = 0.65
        max_iterations = 1500
        save_interval = 25
