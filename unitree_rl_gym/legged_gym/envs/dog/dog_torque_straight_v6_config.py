"""RS01-rated-torque continuation with tolerant contact-state termination."""

from .dog_torque_straight_v5_config import (
    DogRs01TorqueStraightV5Cfg,
    DogRs01TorqueStraightV5CfgPPO,
)


class DogRs01TorqueStraightV6Cfg(DogRs01TorqueStraightV5Cfg):
    """Retain model_9600 straightness while making its motor demand deployable."""

    class control(DogRs01TorqueStraightV5Cfg.control):
        # RS01 manual: 6 Nm rated continuous, 17 Nm peak. Keep the measured
        # real-hardware PD gains and let short peaks carry this 11.73 kg dog,
        # but progressively remove peak authority as the running I^2 load
        # approaches the continuous envelope.
        continuous_torque_limits_by_joint = {
            "hip": 6.0, "thigh": 6.0, "calf": 6.0
        }
        apply_continuous_torque_derating = True
        continuous_torque_thermal_time_constant_s = 12.0
        continuous_torque_initial_thermal_ratio = 0.80
        continuous_derating_start_ratio = 0.85
        continuous_derating_full_ratio = 1.15
        continuous_derating_curriculum_iterations = 900
        preserve_thermal_state_across_resets = True

    class rewards(DogRs01TorqueStraightV5Cfg.rewards):
        # Do not reset on a 40--60 ms contact/actuator mismatch. Persistent
        # same-side/single-foot patterns are still penalized every frame and
        # terminate after 120 ms in evaluation.
        non_diagonal_swing_grace_steps = 20
        non_diagonal_swing_termination_steps = 8
        non_diagonal_swing_termination_steps_test = 6
        non_diagonal_termination_curriculum = [
            {"until_iteration": 700, "steps": 10},
            {"until_iteration": 1300, "steps": 8},
            {"until_iteration": 1.0e12, "steps": 6},
        ]
        flight_termination_grace_steps = 20
        flight_termination_steps = 4
        # URDF limits, target limits and the physical 17 Nm clamp remain
        # active. A one-frame calf threshold crossing should not end a cycle.
        terminate_on_calf_angle = False

        # Torque penalties ramp in while physical derating is introduced.
        # Thermal state is never cleared by gait resets.
        continuous_torque_penalty_initial_blend = 0.35
        continuous_torque_penalty_curriculum_iterations = 700
        motor_torque_ema_alpha = 0.985

        class scales(DogRs01TorqueStraightV5Cfg.rewards.scales):
            # Symmetric tracking, not a separate over-speed-only punishment.
            tracking_lin_vel = 12.0
            normalized_command_tracking = 10.0
            absolute_longitudinal_tracking_error = -12.0
            commanded_smooth_straight_progress = 5.0

            # Preserve the model_9600 diagonal/load symmetry while allowing
            # brief threshold ambiguity during physical touchdown.
            exact_diagonal_swing = 14.0
            scheduled_diagonal_pair_lift = 13.0
            touchdown_pair_support = 9.0
            diagonal_load_transfer = 9.0
            non_diagonal_swing = -22.0
            flight = -28.0

            # Directly distinguish a 7 Nm request from the old 60--80 Nm PD
            # requests. Positive progress is also conditional on low request.
            low_request_straight_progress = 14.0
            raw_continuous_torque_usage = -3.5
            raw_continuous_torque_peak = -1.5
            raw_torque_rate = -0.55
            torque_clip = -2.0
            torque_near_limit = -0.8
            peak_torque = -1.0
            sustained_torque = -2.2
            sustained_torque_max = -2.5
            motor_torque_usage = -3.5
            sagittal_motor_saturation = -3.5
            motor_continuous_usage = -1.5
            motor_continuous_overload = -3.0
            motor_thermal_overload = -4.0
            motor_thermal_peak = -3.0
            safe_torque_straight_progress = 10.0
            torques = -4.0e-5
            mechanical_power = -0.004


class DogRs01TorqueStraightV6CfgPPO(DogRs01TorqueStraightV5CfgPPO):
    """Full-actor adaptation from the measured best V5 checkpoint."""

    class algorithm(DogRs01TorqueStraightV5CfgPPO.algorithm):
        learning_rate = 1.0e-5
        entropy_coef = 2.0e-5
        clip_param = 0.10
        max_grad_norm = 0.40
        schedule = "fixed"

    class runner(DogRs01TorqueStraightV5CfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_rated_torque_v6"
        resume = True
        load_run = "Jul24_20-04-18_rs01_direct12_torque_straight_v5"
        checkpoint = 9600
        load_optimizer = False
        adapt_observation_input = False
        phase_residual_policy = False
        symmetry_coef = 0.0

        # V5's strong frozen anchor preserved its 60--80 Nm PD requests.
        # Keep a weak guard against abrupt gait loss, but allow the complete
        # actor to redistribute stance load and reduce target error.
        reference_policy_coef = 0.06
        reference_action_deadband = 0.06
        reference_action_hinge_coef = 0.50

        max_iterations = 1800
        save_interval = 25
