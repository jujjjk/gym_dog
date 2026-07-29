"""Nominal straight-line migration from the correctly mapped model_930."""

from .rs01_go2_sim2sim_config import (
    Rs01Go2Heading52Cfg,
    Rs01Go2Heading52CfgPPO,
)


class Rs01Go2Model930DriftRepairCfg(Rs01Go2Heading52Cfg):
    """Remove model_930's one-sided drift without changing its 52-D contract."""

    class commands(Rs01Go2Heading52Cfg.commands):
        # Match the tethered real-robot capture exactly.  This first migration
        # asks one narrow question: can the correctly mapped nominal robot
        # remove the learned left/right DC bias while preserving its gait?
        playback_speed_mps = 0.23
        resampling_time = 1000.0

        class ranges(Rs01Go2Heading52Cfg.commands.ranges):
            lin_vel_x = [0.23, 0.23]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(Rs01Go2Heading52Cfg.init_state):
        # Two-sided perturbations prevent PPO from replacing one fixed bias
        # with the opposite bias.  They remain smaller than the previous
        # heading52 recovery stage so this is a gait-preserving migration.
        reset_heading_noise_rad = 0.20
        reset_yaw_rate_noise_rad_s = 0.10

    class noise(Rs01Go2Heading52Cfg.noise):
        # Establish nominal straightness first. Sensor noise and dynamics
        # spread are restored progressively only after this stage passes.
        add_noise = False

    class domain_rand(Rs01Go2Heading52Cfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_rs01_actuator = False
        push_robots = False
        rs01_independent_motor_randomization = False
        rs01_independent_delay_randomization = False

    class rewards(Rs01Go2Heading52Cfg.rewards):
        # All new error terms are dimensionless after division by these
        # physical scales. Values are based on the 0.23 m/s tether capture:
        # vy RMS 0.054 m/s, wz RMS 0.394 rad/s, roll RMS 0.071 rad.
        lateral_velocity_scale_mps = 0.08
        near_heading_window_rad = 0.20
        yaw_rate_scale_rad_s = 0.60
        roll_scale_rad = 0.12
        lateral_foot_slip_scale_mps = 0.20
        rear_torque_balance_ema_s = 0.60

        class scales(Rs01Go2Heading52Cfg.rewards.scales):
            # Split the inherited xy tracking term into forward tracking and
            # an explicit vy cost so speed and drift have separate gradients.
            tracking_lin_vel = 0.0
            tracking_forward_velocity = 1.0
            lateral_velocity = -0.35

            # heading_recovery tracks a restoring yaw-rate target.  The extra
            # near-heading term suppresses residual wz only near straight and
            # therefore does not fight useful correction at large error.
            heading_recovery = -0.80
            near_heading_yaw_rate = -0.15
            roll_posture = -0.20

            # Normalize left/right load and rear-motor usage before applying
            # modest costs. Rear torque balance uses a one-cycle EMA because
            # RL and RR are deliberately half a gait cycle apart.
            left_right_foot_force_balance = -0.30
            rear_motor_torque_balance = -0.20
            lateral_foot_slip = -0.25

            # Keep the existing executable-action and phase contracts active.
            action_saturation = -0.75
            diagonal_contact_sync = -1.00
            phase_support_tracking = 1.00


class Rs01Go2Model930DriftRepairCfgPPO(Rs01Go2Heading52CfgPPO):
    class policy(Rs01Go2Heading52CfgPPO.policy):
        init_noise_std = 0.04

    class algorithm(Rs01Go2Heading52CfgPPO.algorithm):
        learning_rate = 1.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Heading52CfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.04
        freeze_action_std = True
        load_optimizer = False
        # model_930 already has 52 observations; no column migration is needed.
        adapt_observation_input = False
        # Keep a light executed-action anchor.  It preserves the accepted gait
        # but leaves enough authority to remove the large left/right DC bias.
        reference_policy_coef = 0.03
        reference_action_deadband = 0.10
        reference_action_hinge_coef = 0.40
        reference_action_transform = "clip"
        reference_action_clip = 1.0
