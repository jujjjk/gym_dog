"""6 Nm continuous-torque and path-balance polish from model_7750."""

from .dog_compact_hip_config import (
    DogRs01CompactHipCfg,
    DogRs01CompactHipCfgPPO,
)


class DogRs01SafeTorquePathCfg(DogRs01CompactHipCfg):
    """Use 17 Nm only transiently while improving straight balanced motion."""

    class control(DogRs01CompactHipCfg.control):
        # Manual contract: 6 Nm continuous, peak limits only for transients.
        continuous_torque_limits_by_joint = {
            "hip": 6.0, "thigh": 6.0, "calf": 6.0
        }
        apply_continuous_torque_derating = True
        continuous_torque_thermal_time_constant_s = 2.0
        continuous_torque_initial_thermal_ratio = 0.55
        continuous_derating_start_ratio = 0.90
        continuous_derating_full_ratio = 1.15
        continuous_derating_curriculum_iterations = 400

    class commands(DogRs01CompactHipCfg.commands):
        # The heavy robot must learn an efficient gait near 6 Nm instead of
        # chasing the old top speed through repeated peak torque.
        class ranges(DogRs01CompactHipCfg.commands.ranges):
            lin_vel_x = [0.08, 0.14]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(DogRs01CompactHipCfg.rewards):
        motor_torque_ema_alpha = 0.99

        class scales(DogRs01CompactHipCfg.rewards.scales):
            # Positive credit exists only for useful, straight progress with
            # thermal torque headroom.
            safe_torque_straight_progress = 18.0

            # Directly target the manual's 6 Nm continuous region. The peak
            # 14/16/17 Nm clips remain available only while thermally cold.
            motor_continuous_usage = -3.0
            motor_continuous_overload = -6.0
            motor_thermal_overload = -8.0
            motor_thermal_peak = -5.0
            sustained_torque = -2.0
            sustained_torque_max = -2.2
            motor_torque_usage = -5.0
            sagittal_motor_saturation = -5.0
            torque_clip = -2.0
            torque_near_limit = -0.9
            peak_torque = -1.0
            torques = -2.5e-5
            mechanical_power = -0.004

            # Further reduce path drift while preserving body balance.
            straight_path_lateral_displacement = -12.0
            straight_path_lateral_velocity = -48.0
            straight_path_lateral_acceleration = -2.0
            straight_lateral_speed = -38.0
            straight_heading_error = -22.0
            yaw_rate = -14.0
            straight_balanced_progress = 18.0
            body_angular_velocity = -5.5
            body_angular_acceleration = -2.5
            translation_roll = -40.0
            orientation = -14.0
            straight_contact_lateral_force = -20.0
            straight_contact_yaw_moment = -28.0
            straight_contact_side_load_balance = -14.0

            # Retain compact hips, physical diagonal symmetry, and no-flight
            # locomotion while the policy relearns lower-torque support.
            compact_symmetric_forward = 14.0
            hip_joint_excursion = -10.0
            hip_peak_excursion = -6.0
            hip_diagonal_motion_mismatch = -3.0
            exact_diagonal_swing = 9.0
            forward_diagonal_pair_swing = 15.0
            forward_progress_with_diagonal_swing = 17.0
            tracking_lin_vel = 9.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 14.0


class DogRs01SafeTorquePathCfgPPO(DogRs01CompactHipCfgPPO):
    """Low-rate continuous-torque adaptation from selected model_7750."""

    class algorithm(DogRs01CompactHipCfgPPO.algorithm):
        learning_rate = 7.5e-6
        entropy_coef = 3.0e-5
        schedule = "fixed"

    class runner(DogRs01CompactHipCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_safe6nm_path_v1"
        resume = True
        load_run = "Jul23_21-29-50_rs01_direct12_compact_hip_coord_v1"
        checkpoint = 7750
        load_optimizer = False
        symmetry_coef = 0.65
        max_iterations = 1000
        save_interval = 25
