"""Direct 12-output RS01 diagonal-walk training task.

There is no CPG joint offset, action projection, or identified-gain
compensation. The phase clock is observation/reward context only. Policy
targets still pass through the measured RS01 delay/tau/friction dynamics and
the real controller's rate, acceleration, position, and torque safety limits.
"""

from .dog_config import (
    DogRs01TrotCfg as _Direct12Cfg,
    DogRs01TrotCfgPPO as _Direct12CfgPPO,
)

WALK_PERIOD_S = 0.5
WALK_STANCE_RATIO = 0.66


class DogRs01TrotCfg(_Direct12Cfg):
    __doc__ = "Direct 12-joint policy shaped into a forward diagonal walk."

    class control(_Direct12Cfg.control):
        use_rs01_diagonal_cpg = False
        use_continuous_gait_scaling = False
        compensate_identified_position_gain_in_gait = False
        project_straight_diagonal_actions = False
        enforce_swing_calf_reference = False
        enforce_stance_leg_extension = False
        gate_swing_on_opposite_diagonal_support = False
        use_contact_aware_phase_transfer = False
        use_active_diagonal_load_transfer = False
        use_fast_swing_profile = False
        filter_policy_actions = False
        gait_target_phase_lead = 0.0
        gait_transition_ramp_s = 0.0
        action_scale = 0.22
        rear_action_scale = 0.22
        hip_action_scale = 0.22
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        final_target_rate_limits_initial = {"hip": 2.0, "thigh": 2.6, "calf": 3.2}
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {"hip": 60.0, "thigh": 78.0, "calf": 96.0}
        final_target_accel_limits_final = final_target_accel_limits_initial
        use_real_actuator_model = True
        torque_limits_by_joint = {"hip": 14.0, "thigh": 16.0, "calf": 17.0}
        training_torque_limit_ranges = {
            "hip": [12.0, 14.0],
            "thigh": [15.0, 16.0],
            "calf": [16.0, 17.0],
        }

    class commands(_Direct12Cfg.commands):
        stand_probability = 0.05
        pure_sagittal_probability = 0.95
        pure_yaw_probability = 0.0
        pure_lateral_probability = 0.0
        resampling_time = 4.0

        class ranges(_Direct12Cfg.commands.ranges):
            lin_vel_x = [0.12, 0.24]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(_Direct12Cfg.domain_rand):
        gait_stance_ratio_range = [WALK_STANCE_RATIO, WALK_STANCE_RATIO]
        gait_low_speed_period_range = [WALK_PERIOD_S, WALK_PERIOD_S]
        gait_high_speed_period_range = [WALK_PERIOD_S, WALK_PERIOD_S]
        gait_calf_amplitude_max_range = [0.0, 0.0]
        randomize_gait_phase_on_reset = True
        randomize_motor_strength = True
        randomize_friction = True
        randomize_base_mass = True
        randomize_base_com = True
        push_robots = False

    class rewards(_Direct12Cfg.rewards):
        gait_period = WALK_PERIOD_S
        gait_stance_ratio = WALK_STANCE_RATIO
        gait_thigh_amplitude = 0.0
        gait_swing_thigh_lift_amplitude = 0.0
        gait_calf_amplitude = 0.0
        gait_lateral_hip_amplitude = 0.0
        base_height_target = 0.309475
        swing_height_target = 0.025
        diagonal_pair_lift_start_height = 0.012
        diagonal_pair_lift_target_height = 0.025
        swing_clearance_minimum = 0.018
        foot_contact_force_threshold = 2.0
        # Centralized contact hysteresis. Equal thresholds preserve the
        # audited 2 N behavior while keeping the interface debounce-ready.
        foot_contact_release_force_threshold = 2.0
        max_foot_contact_time_s = 0.43
        foot_contact_time_penalty_saturation_s = 0.18
        max_all_feet_contact_time_s = 0.1
        all_feet_contact_penalty_saturation_s = 0.1
        enable_non_diagonal_swing_termination = True
        enable_rear_pair_air_termination = False
        enable_overlapping_diagonal_termination = False
        enable_flight_termination = False
        non_diagonal_swing_grace_steps = 10
        non_diagonal_swing_termination_steps = 1
        non_diagonal_termination_curriculum = [
            {"until_iteration": 500, "steps": 5},
            {"until_iteration": 1400, "steps": 3},
            {"until_iteration": 2800, "steps": 2},
            {"until_iteration": 1000000000000.0, "steps": 1},
        ]

        class scales(_Direct12Cfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_ang_vel = 0.5
            command_velocity_progress = 18.0
            normalized_command_tracking = 12.0
            absolute_longitudinal_tracking_error = -6.0
            backward_velocity = -18.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            phase_foot_velocity_tracking = 5.0
            exact_diagonal_swing = 6.0
            scheduled_diagonal_pair_lift = 3.0
            diagonal_support = 0.0
            diagonal_stride_sync_all = 0.0
            diagonal_gait = 1.0
            touchdown_pair_support = 2.0
            diagonal_load_transfer = 2.0
            touchdown_pair_support_shortfall = -8.0
            diagonal_load_transfer_error = -6.0
            diagonal_support_shortfall = -14.0
            diagonal_stride_sync_shortfall = -2.0
            phase_contact_mismatch = -8.0
            phase_foot_force_tracking = -5.0
            swing_contact = -10.0
            single_foot_swing = -18.0
            non_diagonal_swing = -35.0
            rear_pair_air = -45.0
            overlapping_diagonal_air = -45.0
            flight = -60.0
            all_feet_contact = -14.0
            excessive_foot_contact_time = -1.5


class DogRs01TrotCfgPPO(_Direct12CfgPPO):
    __doc__ = "Fresh direct-output run; incompatible with CPG-residual checkpoints."

    class policy(_Direct12CfgPPO.policy):
        init_noise_std = 0.3

    class algorithm(_Direct12CfgPPO.algorithm):
        entropy_coef = 0.002
        schedule = "fixed"
        learning_rate = 0.0001

    class runner(_Direct12CfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_forward_diagonal_v3"
        resume = False
        load_optimizer = False
        actor_output_init_scale = None
        max_iterations = 5000
        save_interval = 50
