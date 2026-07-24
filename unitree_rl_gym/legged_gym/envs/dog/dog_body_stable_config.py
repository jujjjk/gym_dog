"""Whole-body oscillation suppression from the selected balance model_4300."""

from .dog_balance_config import DogRs01BalanceCfg, DogRs01BalanceCfgPPO


class DogRs01BodyStableCfg(DogRs01BalanceCfg):
    """Preserve the diagonal gait while reducing roll/pitch/yaw oscillation."""

    class rewards(DogRs01BalanceCfg.rewards):
        class scales(DogRs01BalanceCfg.rewards.scales):
            # Keep enough locomotion credit that the low-motion solution is
            # commanded walking, not standing to avoid stability penalties.
            tracking_lin_vel = 8.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 13.0
            forward_diagonal_pair_swing = 14.0
            forward_progress_with_diagonal_swing = 16.0
            exact_diagonal_swing = 6.0

            # Penalize instantaneous motion amplitude and rapid direction
            # reversal. This closes the loophole where signed yaw averages to
            # zero while the trunk visibly twists left/right every half-step.
            body_angular_velocity = -3.0
            body_angular_acceleration = -1.5
            yaw_rate = -8.0
            ang_vel_xy = -4.0
            orientation = -10.0
            translation_roll = -28.0
            lin_vel_z = -20.0

            # Reduce the contact impulses that create trunk twisting.
            straight_contact_lateral_force = -14.0
            straight_contact_yaw_moment = -20.0
            straight_contact_side_load_balance = -8.0
            straight_lateral_speed = -24.0
            straight_heading_error = -10.0

            # Preserve physical diagonal coordination while avoiding stiff,
            # bang-bang joint reversals.
            diagonal_foot_height_sync_all = -55.0
            diagonal_stride_sync_shortfall = -6.0
            straight_diagonal_contact_sync = -3.0
            straight_diagonal_target_mirror = -0.35
            straight_diagonal_joint_mirror = -0.25
            straight_diagonal_torque_mirror = -0.08
            action_rate = -0.08
            policy_action_rate = -0.08
            dof_acc = -1.2e-6
            final_target_acceleration = -0.06
            mechanical_power = -0.0025


class DogRs01BodyStableCfgPPO(DogRs01BalanceCfgPPO):
    """Conservative polish from symmetry/balance model_4300."""

    class algorithm(DogRs01BalanceCfgPPO.algorithm):
        learning_rate = 3.0e-5
        entropy_coef = 2.0e-4
        schedule = "fixed"

    class runner(DogRs01BalanceCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_body_stable_v1"
        resume = True
        load_run = "Jul23_15-44-26_rs01_direct12_symmetry_balance_v1"
        checkpoint = 4300
        load_optimizer = False
        symmetry_coef = 0.75
        max_iterations = 1000
        save_interval = 25
