"""Conservative rear-leg coordination polish derived from model_550."""

from .rs01_go2_straight_config import (
    Rs01Go2StraightCfg,
    Rs01Go2StraightCfgPPO,
)


class Rs01Go2RearCoordCfg(Rs01Go2StraightCfg):
    """Preserve the learned front-leg gait while repairing rear timing."""

    class commands(Rs01Go2StraightCfg.commands):
        # Stay close to the command used for the accepted model_550 playback.
        # This is a polish stage, not another speed expansion.
        class ranges(Rs01Go2StraightCfg.commands.ranges):
            lin_vel_x = [0.30, 0.45]

    class domain_rand(Rs01Go2StraightCfg.domain_rand):
        # Establish nominal coordination before reintroducing Sim2Real spread.
        randomize_friction = False

    class rewards(Rs01Go2StraightCfg.rewards):
        class scales(Rs01Go2StraightCfg.rewards.scales):
            # Contact XOR measures takeoff/touchdown disagreement inside each
            # diagonal without forcing equal front/rear height every frame.
            diagonal_contact_sync = -0.75
            # The accepted checkpoint already clears both front feet well.
            # Add gradient only to scheduled RL/RR clearance shortfall.
            rear_swing_clearance = -0.50
            # The 550 pilot improved gait by spending more time on the 17 Nm
            # clip.  Prevent the polish stage from continuing that trade.
            raw_torque_over_peak = -0.50
            motor_saturation = -0.50


class Rs01Go2RearCoordCfgPPO(Rs01Go2StraightCfgPPO):
    class policy(Rs01Go2StraightCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(Rs01Go2StraightCfgPPO.algorithm):
        learning_rate = 2.0e-4
        schedule = "fixed"

    class runner(Rs01Go2StraightCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 25
        action_std_value = 0.15
        freeze_action_std = True
        # A fresh low-rate optimizer avoids carrying the aggressive adaptive
        # momentum that raised yaw and saturation between 500 and 550.
        load_optimizer = False
        # Freeze a copy of the loaded model_550 as the executed-action anchor.
        reference_policy_coef = 0.15
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 1.0
