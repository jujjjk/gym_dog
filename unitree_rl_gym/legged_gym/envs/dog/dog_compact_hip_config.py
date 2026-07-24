"""Compact, coordinated hip continuation from selected model_7700."""

from .dog_straight_balance_config import (
    DogRs01StraightBalanceCfg,
    DogRs01StraightBalanceCfgPPO,
)


class DogRs01CompactHipCfg(DogRs01StraightBalanceCfg):
    """Reduce visible hip sweep while preserving straight diagonal balance."""

    class control(DogRs01StraightBalanceCfg.control):
        # model_7700 still measures about 0.258 rad hip peak-to-peak.  Reduce
        # direct hip target authority by 11% without changing the other eight
        # motor outputs or adding a compensating controller.
        hip_action_scale = 0.16
        target_position_limits_by_joint = {
            "hip": [-0.25, 0.25],
            "thigh": [-1.20, 0.45],
            "calf": [0.45, 1.75],
        }

    class rewards(DogRs01StraightBalanceCfg.rewards):
        # Tighten the balance band gradually rather than demanding a rigid hip.
        hip_excursion_soft_limit_rad = 0.050
        hip_excursion_penalty_width_rad = 0.070
        hip_target_soft_limit_rad = 0.060
        hip_target_penalty_width_rad = 0.060
        hip_peak_soft_limit_rad = 0.065
        hip_peak_penalty_width_rad = 0.070
        compact_hip_headroom_limit_rad = 0.12

        class scales(DogRs01StraightBalanceCfg.rewards.scales):
            # Joint positive objective prevents low-hip-motion standing from
            # exploiting positive reward clipping.
            compact_symmetric_forward = 14.0

            # Target both mean motion and the worst individual hip.  Physical
            # angle/rate coordination is used instead of forcing identical
            # actions onto independently measured RS01 motors.
            hip_joint_excursion = -10.0
            hip_peak_excursion = -6.0
            hip_target_excursion = -5.0
            hip_policy_action_rate = -0.35
            hip_velocity = -0.007
            hip_diagonal_motion_mismatch = -3.0
            hip_trunk_twist_coupling = -3.0
            hip_symmetry = -0.8

            # Preserve the previously selected straight path and quiet trunk.
            straight_balanced_progress = 16.0
            straight_path_lateral_displacement = -8.0
            straight_path_lateral_velocity = -36.0
            straight_heading_error = -16.0
            body_angular_velocity = -5.0
            body_angular_acceleration = -2.2
            handoff_body_twist = -8.0
            translation_roll = -36.0
            orientation = -12.0

            # Do not trade compact hips for torque saturation or broken gait.
            motor_torque_usage = -4.0
            sagittal_motor_saturation = -4.0
            compact_hip_low_torque_forward = 12.0
            smooth_low_torque_forward = 12.0
            smooth_diagonal_handoff = 10.0
            exact_diagonal_swing = 8.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            tracking_lin_vel = 8.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 13.0


class DogRs01CompactHipCfgPPO(DogRs01StraightBalanceCfgPPO):
    """Very-low-rate compact-hip polish from selected model_7700."""

    class algorithm(DogRs01StraightBalanceCfgPPO.algorithm):
        learning_rate = 7.5e-6
        entropy_coef = 3.0e-5
        schedule = "fixed"

    class runner(DogRs01StraightBalanceCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_compact_hip_coord_v1"
        resume = True
        load_run = "Jul23_20-19-01_rs01_direct12_straight_balance_v1"
        checkpoint = 7700
        load_optimizer = False
        # Slightly strengthen policy equivariance while leaving room for the
        # measured per-motor gain/tau/friction differences.
        symmetry_coef = 0.65
        max_iterations = 1000
        save_interval = 25
