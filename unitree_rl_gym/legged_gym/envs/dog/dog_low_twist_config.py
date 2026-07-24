"""Low-twist, torque-desaturated fine-tuning from body-stable model_5300."""

from .dog_body_stable_config import (
    DogRs01BodyStableCfg,
    DogRs01BodyStableCfgPPO,
)


class DogRs01LowTwistCfg(DogRs01BodyStableCfg):
    """Smooth physical-diagonal handoff with bounded sagittal motor demand."""

    class control(DogRs01BodyStableCfg.control):
        # Direct independent outputs remain intact. Only the two motor types
        # observed to saturate receive 9% less target-position authority.
        hip_action_scale = 0.22
        thigh_action_scale = 0.20
        calf_action_scale = 0.20

    class commands(DogRs01BodyStableCfg.commands):
        # Keep the useful 0.15 m/s region while no longer training the heavy
        # robot to chase 0.24 m/s through persistent torque saturation.
        class ranges(DogRs01BodyStableCfg.commands.ranges):
            lin_vel_x = [0.12, 0.20]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(DogRs01BodyStableCfg.domain_rand):
        # 100 ms per handoff instead of 80 ms; full cycle remains 0.50 s.
        gait_stance_ratio_range = [0.70, 0.70]

    class rewards(DogRs01BodyStableCfg.rewards):
        gait_stance_ratio = 0.70
        max_foot_contact_time_s = 0.46
        max_all_feet_contact_time_s = 0.12

        class scales(DogRs01BodyStableCfg.rewards.scales):
            # Joint positive objective: useful forward motion is valuable only
            # when both motor headroom and body stability remain available.
            smooth_low_torque_forward = 12.0
            smooth_diagonal_handoff = 10.0

            # Directly target the phase-0/0.5 twisting and sagittal saturation.
            handoff_body_twist = -7.0
            sagittal_motor_saturation = -3.0
            body_angular_velocity = -4.0
            body_angular_acceleration = -2.0
            yaw_rate = -10.0
            ang_vel_xy = -5.0

            # Increase desaturation pressure without reducing the hard RS01
            # safety clips or inventing additional actuator capability.
            torque_clip = -1.5
            torque_near_limit = -0.55
            peak_torque = -0.70
            sustained_torque = -1.0
            sustained_torque_max = -1.1
            torques = -1.5e-5
            mechanical_power = -0.003

            # Retain the acquired diagonal gait and commanded progress.
            tracking_lin_vel = 8.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 13.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            exact_diagonal_swing = 7.0


class DogRs01LowTwistCfgPPO(DogRs01BodyStableCfgPPO):
    """Very-low-rate continuation from selected body-stable model_5300."""

    class algorithm(DogRs01BodyStableCfgPPO.algorithm):
        learning_rate = 2.0e-5
        entropy_coef = 1.0e-4
        schedule = "fixed"

    class runner(DogRs01BodyStableCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_low_twist_desat_v1"
        resume = True
        load_run = "Jul23_16-47-04_rs01_direct12_body_stable_v1"
        checkpoint = 5300
        load_optimizer = False
        symmetry_coef = 0.75
        max_iterations = 1000
        save_interval = 25
