"""Archived RS01 full 12-DoF policy experiment (no CPG joint offset).

Contract:
  target_q = default_stand + action * action_scale
  then the measured RS01 actuator chain (delay, FOPDT, friction, limits).

Phase sin/cos remain in the observation only so diagonal-walk *rewards* can
shape timing.  They do not inject any joint reference.
"""

from .dog_config import (
    DogRs01TrotCfg as _LegacyDogRs01TrotCfg,
    DogRs01TrotCfgPPO as _LegacyDogRs01TrotCfgPPO,
)


# Reward-clock only (not a joint CPG).
# duty > 0.5 → mandatory four-foot overlap so one diagonal finishes before
# the other may start (no overlapping swings, no rear-pair hop).
WALK_PERIOD_S = 0.60
WALK_STANCE_RATIO = 0.68


class DogRs01Full12Cfg(_LegacyDogRs01TrotCfg):
    """Go2-style 12-action control + RS01 real actuator + diagonal rewards."""

    class init_state(_LegacyDogRs01TrotCfg.init_state):
        pos = [0.0, 0.0, 0.316]
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.32987297,
            "FR_calf_joint": 1.31853104,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": -0.32987297,
            "FL_calf_joint": 1.31853104,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.32987297,
            "RR_calf_joint": 1.31853104,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.32987297,
            "RL_calf_joint": 1.31853104,
        }

    class asset(_LegacyDogRs01TrotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/../dog_urdf/urdf/"
            "dog_rs01.urdf"
        )
        name = "dog_rs01"

    class control(_LegacyDogRs01TrotCfg.control):
        # ---- full 12-DoF policy, no open-loop joint gait ----
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
        gait_target_phase_lead = 0.0
        gait_transition_ramp_s = 0.0

        # Equal authority on all 12 joints (Go2-style).
        action_scale = 0.22
        rear_action_scale = 0.22
        hip_action_scale = 0.22

        # Identification-like PD; trunk is heavier than Fanfan.
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}

        # Manual peak 17 N·m; keep hip/thigh a bit below peak.
        torque_limits_by_joint = {"hip": 14.0, "thigh": 16.0, "calf": 17.0}

        # Real software target board from rs01shujv (not ideal / not 315 rpm).
        final_target_rate_limits_initial = {
            "hip": 2.0, "thigh": 2.6, "calf": 3.2
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {
            "hip": 60.0, "thigh": 78.0, "calf": 96.0
        }
        final_target_accel_limits_final = final_target_accel_limits_initial

        # Measured actuator chain stays ON for training.
        use_real_actuator_model = True
        training_torque_limit_ranges = {
            "hip": [12.0, 14.0],
            "thigh": [14.0, 16.0],
            "calf": [15.0, 17.0],
        }

    class commands(_LegacyDogRs01TrotCfg.commands):
        stand_probability = 0.10
        pure_sagittal_probability = 0.90
        pure_yaw_probability = 0.0
        pure_lateral_probability = 0.0
        resampling_time = 4.0

        class ranges(_LegacyDogRs01TrotCfg.commands.ranges):
            lin_vel_x = [0.10, 0.25]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(_LegacyDogRs01TrotCfg.domain_rand):
        # Phase clock for rewards only; no joint CPG amplitude randomization.
        gait_stance_ratio_range = [WALK_STANCE_RATIO, WALK_STANCE_RATIO]
        gait_low_speed_period_range = [WALK_PERIOD_S, WALK_PERIOD_S]
        gait_high_speed_period_range = [WALK_PERIOD_S, WALK_PERIOD_S]
        gait_calf_amplitude_max_range = [0.0, 0.0]
        gait_backward_scale_range = [1.0, 1.0]
        randomize_gait_phase_on_reset = True
        randomize_motor_strength = True
        randomize_friction = True
        randomize_base_mass = True
        randomize_base_com = True
        push_robots = False

    class rewards(_LegacyDogRs01TrotCfg.rewards):
        base_height_target = 0.309475
        gym_settled_base_height_m = 0.309475
        mujoco_settled_base_height_m = 0.307120

        # CRITICAL: zero joint gait seed so target = default + action*scale.
        gait_period = WALK_PERIOD_S
        gait_stance_ratio = WALK_STANCE_RATIO
        gait_thigh_amplitude = 0.0
        gait_swing_thigh_lift_amplitude = 0.0
        gait_calf_amplitude = 0.0
        gait_lateral_hip_amplitude = 0.0

        swing_height_target = 0.022
        diagonal_pair_lift_start_height = 0.010
        diagonal_pair_lift_target_height = 0.022
        swing_clearance_minimum = 0.016
        # Slightly higher so a soft toe still counts as planted support.
        foot_contact_force_threshold = 3.0

        # Stance ≈ 0.41 s; DS overlap ≈ 0.60*(0.68-0.5) ≈ 108 ms.
        max_foot_contact_time_s = 0.60
        foot_contact_time_penalty_saturation_s = 0.20
        max_all_feet_contact_time_s = 0.14
        all_feet_contact_penalty_saturation_s = 0.14

        continuous_torque_limits_by_joint = {
            "hip": 6.0, "thigh": 6.0, "calf": 6.0
        }

        # Hard gait legality for sequential diagonal walk.
        enable_non_diagonal_swing_termination = True
        enable_flight_termination = True
        enable_rear_pair_air_termination = True
        enable_overlapping_diagonal_termination = True
        flight_termination_grace_steps = 8
        diagonal_sequence_grace_steps = 8
        non_diagonal_swing_grace_steps = 8
        # Do not tolerate multi-frame rear-bound / overlap during training.
        non_diagonal_swing_termination_steps = 1
        non_diagonal_termination_curriculum = [
            {"until_iteration": 200, "steps": 2},
            {"until_iteration": 1.0e12, "steps": 1},
        ]

        class scales(_LegacyDogRs01TrotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            command_velocity_progress = 8.0
            normalized_command_tracking = 8.0
            # Reward only one finished diagonal at a time.
            exact_diagonal_swing = 16.0
            scheduled_diagonal_pair_lift = 12.0
            touchdown_pair_support = 14.0
            touchdown_pair_support_shortfall = -18.0
            diagonal_load_transfer = 16.0
            diagonal_load_transfer_error = -16.0
            diagonal_support = 18.0
            diagonal_support_shortfall = -26.0
            non_diagonal_swing = -40.0
            rear_pair_air = -50.0
            overlapping_diagonal_air = -50.0
            flight = -50.0
            all_feet_contact = -6.0
            stand_height = 1.0
            stand_posture = 0.8
            base_height = -30.0
            orientation = -5.0


class DogRs01Full12CfgPPO(_LegacyDogRs01TrotCfgPPO):
    """Full-policy run; do not resume a CPG-residual checkpoint."""

    class policy(_LegacyDogRs01TrotCfgPPO.policy):
        # Need real joint exploration without a CPG seed.
        init_noise_std = 0.8

    class algorithm(_LegacyDogRs01TrotCfgPPO.algorithm):
        entropy_coef = 0.008
        schedule = "fixed"
        learning_rate = 1.0e-4

    class runner(_LegacyDogRs01TrotCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_full12_seq_diag"
        resume = False
        load_optimizer = False
        # Normal actor init (NOT near-zero residual init).
        actor_output_init_scale = None
        max_iterations = 5000
        save_interval = 50
