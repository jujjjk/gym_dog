"""Structural direct-12 rebuild from model_9200 with observable actuator state."""

from .dog_config import (
    FRICTION_BY_JOINT_NM,
    GAIN_BY_JOINT,
    REVERSAL_BY_JOINT_RAD,
    TAU_BY_JOINT_S,
)
from .dog_smooth_straight_config import (
    DogRs01SmoothStraightCfg,
    DogRs01SmoothStraightCfgPPO,
)


def _fixed_midpoints(ranges):
    return {
        key: [0.5 * (float(value[0]) + float(value[1]))] * 2
        for key, value in ranges.items()
    }


class DogRs01SmoothStraightV2Cfg(DogRs01SmoothStraightCfg):
    """Signed, observable and minimally shaped direct motor control task."""

    class env(DogRs01SmoothStraightCfg.env):
        # Original 52 observations remain in exactly the same order. Append
        # the two fields directly returned by all 12 RS01 motors: reported
        # torque and temperature. Thermal RMS and active derating limits stay
        # inside the safety controller rather than the learned policy input.
        num_observations = 76
        observe_actuator_state = True

    class control(DogRs01SmoothStraightCfg.control):
        # Keep 12 independent direct targets and the identified actuator
        # chain. Start from the midpoint of each measured motor range so the
        # first stage is learnable rather than partially observable.
        actuator_time_constant_ranges_by_joint = _fixed_midpoints(
            TAU_BY_JOINT_S
        )
        actuator_position_gain_ranges_by_joint = _fixed_midpoints(
            GAIN_BY_JOINT
        )
        command_delay_range_s = [0.0423, 0.0423]
        command_delay_slow_probability = 0.0
        command_delay_slow_range_s = [0.0423, 0.0423]
        training_torque_limit_ranges = {
            "hip": [14.0, 14.0],
            "thigh": [16.0, 16.0],
            "calf": [17.0, 17.0],
        }
        # Slow observation-only temperature plant, anchored to the measured
        # 28.6--32.6 C identification logs. Real deployment replaces this
        # channel with each motor's type-2 feedback temperature.
        motor_temperature_initial_c = 30.0
        motor_temperature_ambient_c = 25.0
        motor_temperature_rise_at_rated_c = 55.0
        motor_temperature_time_constant_s = 180.0
        motor_temperature_protection_c = 103.0

    class domain_rand(DogRs01SmoothStraightCfg.domain_rand):
        # Nominal identified hardware first. Robustness randomization belongs
        # in a later continuation after long stable cycles exist.
        randomize_friction = False
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor_strength = False
        motor_strength_range = [1.0, 1.0]
        rear_calf_strength_range = [1.0, 1.0]
        kp_multiplier_range = [1.0, 1.0]
        kd_multiplier_range = [1.0, 1.0]
        joint_zero_offset_ranges = {
            "hip": [0.0, 0.0],
            "thigh": [0.0, 0.0],
            "calf": [0.0, 0.0],
        }
        effective_reversal_gap_ranges_by_joint = _fixed_midpoints(
            REVERSAL_BY_JOINT_RAD
        )
        coulomb_friction_ranges_by_joint = _fixed_midpoints(
            FRICTION_BY_JOINT_NM
        )
        gait_stance_ratio_range = [0.70, 0.70]
        randomize_gait_phase_on_reset = False

    class noise(DogRs01SmoothStraightCfg.noise):
        # Establish the nominal closed-loop gait before sensor robustness.
        add_noise = False

    class rewards(DogRs01SmoothStraightCfg.rewards):
        # Preserve signed differences between mildly and severely bad states.
        only_positive_rewards = False

        # Training tolerates two transient 50 Hz threshold crossings and
        # resets on the third; playback/test resets on the second. This still
        # rejects a persistent wrong pattern without chopping every learning
        # rollout at one delayed-contact sample.
        non_diagonal_swing_termination_steps = 2
        non_diagonal_swing_termination_steps_test = 2
        non_diagonal_termination_curriculum = [
            {"until_iteration": 1.0e12, "steps": 3},
        ]

        body_headroom_roll_rate_rad_s = 0.90
        body_headroom_pitch_rate_rad_s = 0.85
        body_headroom_yaw_rate_rad_s = 1.25
        body_headroom_roll_accel_rad_s2 = 20.0
        body_headroom_pitch_accel_rad_s2 = 20.0
        body_headroom_yaw_accel_rad_s2 = 30.0
        smooth_progress_load_transfer_floor = 0.25
        diagonal_load_transfer_sigma = 0.16

        # Replace the inherited ~100 correlated terms with a small signed
        # objective. Positive locomotion/contact terms and negative physical
        # errors now remain distinguishable to PPO.
        class scales:
            termination = -12.0
            stand_height = 2.0
            stand_posture = 0.5

            tracking_lin_vel = 6.0
            normalized_command_tracking = 5.0
            commanded_smooth_straight_progress = 10.0
            exact_diagonal_swing = 6.0
            touchdown_pair_support = 3.0
            diagonal_load_transfer = 4.0

            non_diagonal_swing = -10.0
            phase_contact_mismatch = -2.0
            flight = -20.0
            all_feet_contact = -1.5
            excessive_foot_contact_time = -0.3

            body_angular_velocity = -1.5
            body_angular_acceleration = -0.4
            straight_path_lateral_displacement = -3.0
            straight_path_lateral_velocity = -4.0
            straight_heading_error = -3.0
            orientation = -2.0

            action_rate = -0.03
            torque_clip = -0.4
            motor_thermal_overload = -0.4


class DogRs01SmoothStraightV2CfgPPO(DogRs01SmoothStraightCfgPPO):
    """Migrate model_9200 to the observable signed-reward task."""

    class algorithm(DogRs01SmoothStraightCfgPPO.algorithm):
        learning_rate = 1.0e-5
        entropy_coef = 5.0e-5
        schedule = "fixed"

    class runner(DogRs01SmoothStraightCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_structural_rebuild_v1"
        resume = True
        load_run = "Jul24_13-19-42_rs01_direct12_smooth_straight_v1"
        checkpoint = 9200
        load_optimizer = False
        adapt_observation_input = True
        # Physical symmetry remains in the reward. Do not force equal mirrored
        # motor commands onto identified motors with unequal gain and delay.
        symmetry_coef = 0.0
        max_iterations = 1500
        save_interval = 25
