"""Torque-limited continuation from smooth_straight_v1 model_9200 (52-obs).

v5/v6 resumed the 76-obs structural rebuild and kept frequent contact resets;
this stage returns to the better 52-obs gait and teaches peak/continuous torque
discipline without truncating every short diagonal handoff.
"""

from .dog_smooth_straight_config import (
    DogRs01SmoothStraightCfg,
    DogRs01SmoothStraightCfgPPO,
)


class DogRs01TorqueStraightV7Cfg(DogRs01SmoothStraightCfg):
    """Keep model_9200 gait, then reduce PD saturation and path snake motion."""

    class control(DogRs01SmoothStraightCfg.control):
        # RS01: 6 Nm continuous, 14/16/17 Nm peak. Soften peak authority as
        # thermal RMS approaches the continuous envelope, but do not slam the
        # resumed 9200 policy into full derating on the first rollout.
        continuous_torque_limits_by_joint = {
            "hip": 6.0, "thigh": 6.0, "calf": 6.0
        }
        apply_continuous_torque_derating = True
        continuous_torque_thermal_time_constant_s = 12.0
        continuous_torque_initial_thermal_ratio = 0.75
        continuous_derating_start_ratio = 0.90
        continuous_derating_full_ratio = 1.40
        continuous_derating_curriculum_iterations = 800
        preserve_thermal_state_across_resets = True

    class commands(DogRs01SmoothStraightCfg.commands):
        # Stabilize commanded 0.16 m/s before expanding the speed envelope.
        stand_probability = 0.05

        class ranges(DogRs01SmoothStraightCfg.commands.ranges):
            lin_vel_x = [0.12, 0.16]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(DogRs01SmoothStraightCfg.rewards):
        # Parent debounce only. Contact flicker of 40--60 ms must not end an
        # episode; keep enough frames for the measured actuator delay.
        enable_rear_pair_air_termination = False
        enable_overlapping_diagonal_termination = False
        non_diagonal_swing_grace_steps = 15
        non_diagonal_swing_termination_steps = 4
        non_diagonal_swing_termination_steps_test = 3
        non_diagonal_termination_curriculum = [
            {"until_iteration": 600, "steps": 5},
            {"until_iteration": 1200, "steps": 4},
            {"until_iteration": 1.0e12, "steps": 3},
        ]
        flight_termination_grace_steps = 15
        flight_termination_steps = 4
        terminate_on_calf_angle = False

        torque_curriculum = False
        continuous_torque_penalty_initial_blend = 0.30
        continuous_torque_penalty_curriculum_iterations = 700
        motor_torque_ema_alpha = 0.985
        torque_near_limit_ratio = 0.72
        peak_torque_soft_ratio = 0.82
        straight_path_recovery_gain_s = 1.25
        straight_path_recovery_max_velocity_m_s = 0.08
        straight_heading_recovery_gain_s = 1.5
        straight_heading_recovery_max_rate_rad_s = 0.50

        class scales(DogRs01SmoothStraightCfg.rewards.scales):
            # Track the command; stop buying reward by lunging at ~0.25 m/s.
            tracking_lin_vel = 12.0
            normalized_command_tracking = 10.0
            absolute_longitudinal_tracking_error = -10.0
            commanded_smooth_straight_progress = 8.0
            command_velocity_progress = 8.0
            forward_progress_with_diagonal_swing = 8.0
            forward_diagonal_pair_swing = 10.0
            straight_balanced_progress = 10.0
            safe_torque_straight_progress = 10.0
            compact_symmetric_forward = 8.0

            # Preserve the learned diagonal handoff from model_9200.
            exact_diagonal_swing = 12.0
            scheduled_diagonal_pair_lift = 8.0
            touchdown_pair_support = 6.0
            diagonal_load_transfer = 6.0
            diagonal_contact_sync_all = -20.0
            diagonal_foot_height_sync_all = -60.0
            diagonal_stride_sync_shortfall = -8.0
            non_diagonal_swing = -18.0
            flight = -24.0
            all_feet_contact = -1.0

            # Suppress 2 Hz yaw/lateral oscillation and allow path recovery.
            straight_path_recovery_velocity = -4.0
            straight_heading_recovery_rate = -8.0
            straight_path_lateral_displacement = -6.0
            straight_path_lateral_velocity = -20.0
            straight_path_lateral_acceleration = -1.5
            straight_lateral_speed = -16.0
            straight_heading_error = -12.0
            yaw_rate = -4.0
            body_angular_velocity = -4.0
            body_angular_acceleration = -1.2
            handoff_body_twist = -8.0
            straight_contact_lateral_force = -10.0
            straight_contact_yaw_moment = -14.0
            orientation = -8.0

            # Teach the policy to ask for deployable torques before clipping.
            action_rate = -0.08
            policy_action_rate = -0.08
            final_target_acceleration = -0.08
            raw_torque_rate = -0.40
            torque_clip = -1.50
            torque_near_limit = -0.60
            peak_torque = -0.85
            motor_continuous_usage = -1.20
            motor_continuous_overload = -2.40
            motor_thermal_overload = -2.50
            motor_thermal_peak = -1.80
            sustained_torque = -1.20
            sustained_torque_max = -1.40
            motor_torque_usage = -3.00
            sagittal_motor_saturation = -3.00
            mechanical_power = -0.003
            torques = -2.5e-5


class DogRs01TorqueStraightV7CfgPPO(DogRs01SmoothStraightCfgPPO):
    """Full-actor fine-tune from the selected 52-obs model_9200."""

    class algorithm(DogRs01SmoothStraightCfgPPO.algorithm):
        learning_rate = 8.0e-6
        entropy_coef = 1.0e-5
        clip_param = 0.10
        max_grad_norm = 0.40
        schedule = "fixed"

    class runner(DogRs01SmoothStraightCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_torque_from9200_v7"
        resume = True
        load_run = "Jul24_13-19-42_rs01_direct12_smooth_straight_v1"
        checkpoint = 9200
        load_optimizer = False
        adapt_observation_input = False
        symmetry_coef = 0.0

        # Keep the 9200 gait while the actor redistributes stance load below
        # peak clip. Residual-only heads cannot undo 60--80 Nm PD requests.
        phase_residual_policy = False
        reference_policy_coef = 0.20
        reference_action_deadband = 0.04
        reference_action_hinge_coef = 1.5

        max_iterations = 1200
        save_interval = 25
