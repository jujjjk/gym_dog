"""Symmetry and dynamic-balance fine-tuning for the direct-12 RS01 policy.

This stage resumes model_2900 from the forward-diagonal run. It preserves the
direct action contract and measured per-motor dynamics; symmetry is imposed on
physical motion and contact loads, with only a moderate mirrored-policy loss
so unequal identified motors may still receive slightly different commands.
"""

from .dog_cpg_fixed_config import DogRs01TrotCfg, DogRs01TrotCfgPPO


class DogRs01BalanceCfg(DogRs01TrotCfg):
    """Strict diagonal locomotion with straight, level, balanced propulsion."""

    class rewards(DogRs01TrotCfg.rewards):
        # model_2900 has already completed the gait curriculum. Keep the
        # deployment contract strict from the first continuation iteration.
        non_diagonal_swing_termination_steps = 1
        non_diagonal_termination_curriculum = [
            {"until_iteration": 1.0e12, "steps": 1},
        ]

        # Avoid restarting the torque curriculum at stage one when a new
        # environment process loads iteration 2900.
        torque_curriculum = False

        class scales(DogRs01TrotCfg.rewards.scales):
            # Preserve the acquired forward gait.
            tracking_lin_vel = 7.0
            command_velocity_progress = 18.0
            normalized_command_tracking = 12.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            exact_diagonal_swing = 6.0

            # Straight motion and trunk balance. These operate on physical
            # velocities/forces after the measured RS01 actuator chain.
            tracking_ang_vel = 2.0
            yaw_rate = -3.0
            straight_lateral_speed = -20.0
            straight_heading_error = -8.0
            translation_roll = -18.0
            orientation = -6.0
            ang_vel_xy = -1.5
            lin_vel_z = -16.0

            # Remove lateral force and yaw moment at their source instead of
            # allowing the trunk to compensate with a permanent lean.
            straight_contact_lateral_force = -8.0
            straight_contact_yaw_moment = -10.0
            straight_contact_side_load_balance = -5.0
            straight_diagonal_contact_sync = -2.0

            # Physical diagonal agreement remains more important than equal
            # raw actions because the 12 identified motors are not identical.
            diagonal_contact_sync_all = -16.0
            diagonal_foot_height_sync_all = -40.0
            diagonal_stride_sync_shortfall = -5.0
            diagonal_joint_sync = -1.4
            hip_symmetry = -0.5
            straight_diagonal_target_mirror = -0.30
            straight_diagonal_joint_mirror = -0.20
            straight_diagonal_torque_mirror = -0.05

            # model_2900 still requests about 14 Nm average raw torque. Apply
            # the final safety shaping immediately and reduce impact energy.
            torques = -1.0e-5
            torque_clip = -1.0
            torque_near_limit = -0.35
            peak_torque = -0.45
            sustained_torque = -0.70
            sustained_torque_max = -0.80
            mechanical_power = -0.002
            action_magnitude = -0.005
            action_rate = -0.025
            policy_action_rate = -0.025


class DogRs01BalanceCfgPPO(DogRs01TrotCfgPPO):
    """Low-rate symmetry fine-tuning from the selected model_2900."""

    class algorithm(DogRs01TrotCfgPPO.algorithm):
        learning_rate = 5.0e-5
        entropy_coef = 5.0e-4
        schedule = "fixed"

    class runner(DogRs01TrotCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_symmetry_balance_v1"
        resume = True
        load_run = "Jul23_13-59-23_rs01_direct12_forward_diagonal_v3"
        checkpoint = 2900
        load_optimizer = False
        # Moderate equivariance: enough to remove left/right policy bias,
        # without erasing compensation for measured per-motor differences.
        symmetry_coef = 0.5
        max_iterations = 1500
        save_interval = 25
