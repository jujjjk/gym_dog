"""Full-actor continuation that reduces torque chatter and path snake motion."""

from .dog_straight_guarded_config import (
    DogRs01StraightGuardedCfg,
    DogRs01StraightGuardedCfgPPO,
)


class DogRs01TorqueStraightV5Cfg(DogRs01StraightGuardedCfg):
    """Preserve the diagonal gait while fixing its actuator and path defects."""

    class commands(DogRs01StraightGuardedCfg.commands):
        # First make 0.16 m/s safe and repeatable. Faster commands can be
        # reintroduced only after the strict 30 s regression passes.
        stand_probability = 0.05
        inject_straight_path_recovery_velocity = True
        straight_path_observation_gain_s = 1.25
        straight_path_observation_max_velocity_m_s = 0.08

        class ranges(DogRs01StraightGuardedCfg.commands.ranges):
            lin_vel_x = [0.12, 0.16]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(DogRs01StraightGuardedCfg.rewards):
        # DogRs01Robot now routes every invalid diagonal pattern through this
        # single debounced counter. Keep enough time for the measured motor
        # delay, then tighten once long cycles have formed.
        non_diagonal_swing_termination_steps = 4
        non_diagonal_swing_termination_steps_test = 3
        non_diagonal_termination_curriculum = [
            {"until_iteration": 500, "steps": 4},
            {"until_iteration": 1.0e12, "steps": 3},
        ]

        # Use the explicit scales below immediately. A fresh simulation step
        # counter must not silently replace them with an old torque curriculum.
        torque_curriculum = False
        straight_path_recovery_gain_s = 1.25
        straight_path_recovery_max_velocity_m_s = 0.08
        straight_heading_recovery_gain_s = 1.5
        straight_heading_recovery_max_rate_rad_s = 0.50
        torque_near_limit_ratio = 0.72
        peak_torque_soft_ratio = 0.82

        class scales(DogRs01StraightGuardedCfg.rewards.scales):
            # Track commanded speed instead of buying reward by lunging at
            # 0.24--0.25 m/s.
            tracking_lin_vel = 10.0
            normalized_command_tracking = 8.0
            absolute_longitudinal_tracking_error = -10.0
            commanded_smooth_straight_progress = 5.0

            # Keep the learned diagonal structure and require a supported
            # handoff before the next pair leaves.
            exact_diagonal_swing = 12.0
            scheduled_diagonal_pair_lift = 12.0
            touchdown_pair_support = 7.0
            diagonal_load_transfer = 7.0
            diagonal_contact_sync_all = -24.0
            diagonal_foot_height_sync_all = -70.0
            diagonal_stride_sync_shortfall = -8.0
            diagonal_joint_sync = -1.0
            non_diagonal_swing = -20.0
            phase_contact_mismatch = -6.0
            flight = -26.0
            all_feet_contact = -1.0
            excessive_foot_contact_time = -0.3

            # Suppress the measured 2 Hz yaw/lateral oscillation at its source
            # while supplying a compatible return-to-path velocity target.
            straight_path_recovery_velocity = -6.0
            straight_heading_recovery_rate = -10.0
            straight_path_lateral_displacement = -3.0
            straight_path_lateral_velocity = -12.0
            straight_lateral_speed = -5.0
            straight_heading_error = -6.0
            straight_yaw_error = -4.0
            yaw_rate = -1.5
            body_angular_velocity = -3.0
            body_angular_acceleration = -0.8
            straight_contact_lateral_force = -4.0
            straight_contact_yaw_moment = -5.0
            orientation = -2.5

            # Penalize both the size and the 50 Hz variation of the unclipped
            # PD request. Keep transient peak authority for the 11.73 kg body,
            # but stop treating persistent 14--17 Nm clipping as normal.
            action_rate = -0.08
            policy_action_rate = -0.08
            final_target_acceleration = -0.10
            raw_torque_rate = -0.35
            torque_clip = -1.20
            torque_near_limit = -0.55
            peak_torque = -0.75
            motor_continuous_usage = -0.70
            motor_continuous_overload = -1.40
            motor_thermal_overload = -0.80
            motor_thermal_peak = -0.60
            sustained_torque = -0.70
            sustained_torque_max = -0.80
            motor_torque_usage = -2.00
            sagittal_motor_saturation = -2.00
            pd_position_error_over_limit = -0.25
            mechanical_power = -0.0025
            torques = -1.0e-5


class DogRs01TorqueStraightV5CfgPPO(DogRs01StraightGuardedCfgPPO):
    """Conservative full-actor fine-tuning from the standard model_9200."""

    class algorithm(DogRs01StraightGuardedCfgPPO.algorithm):
        learning_rate = 7.5e-6
        entropy_coef = 0.0
        clip_param = 0.08
        max_grad_norm = 0.35
        schedule = "fixed"

    class runner(DogRs01StraightGuardedCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_torque_straight_v5"
        resume = True
        load_run = "Jul24_15-09-38_rs01_direct12_structural_rebuild_v1"
        checkpoint = 9200
        load_optimizer = False
        adapt_observation_input = False
        symmetry_coef = 0.0

        # Train the complete actor: a +/-0.03 residual cannot undo the
        # 67--82 Nm raw PD requests measured in the base policy. The frozen
        # reference loss still prevents sudden gait destruction.
        phase_residual_policy = False
        reference_policy_coef = 0.35
        reference_action_deadband = 0.035
        reference_action_hinge_coef = 3.0

        max_iterations = 1000
        save_interval = 25
