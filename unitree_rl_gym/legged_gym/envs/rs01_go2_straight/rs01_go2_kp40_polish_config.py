"""Conservative Kp40 refinement from the selected model_730."""

from .rs01_go2_kp40_config import (
    Rs01Go2Kp40Cfg,
    Rs01Go2Kp40CfgPPO,
)


class Rs01Go2Kp40PolishCfg(Rs01Go2Kp40Cfg):
    """Improve torque headroom and diagonal timing without changing motion."""

    class rewards(Rs01Go2Kp40Cfg.rewards):
        class scales(Rs01Go2Kp40Cfg.rewards.scales):
            # Model_730 still has 33.2 Nm raw P95 and 11.0% peak saturation.
            # These are small increases from the pilot weights, not a new
            # reward architecture.
            raw_torque_over_peak = -1.0
            motor_saturation = -1.0
            # Exact desired-contact matching is 56.6%; apply a modest increase
            # to pair-event synchronization while leaving phase timing intact.
            diagonal_contact_sync = -1.0


class Rs01Go2Kp40PolishCfgPPO(Rs01Go2Kp40CfgPPO):
    class policy(Rs01Go2Kp40CfgPPO.policy):
        init_noise_std = 0.07

    class algorithm(Rs01Go2Kp40CfgPPO.algorithm):
        learning_rate = 5.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Kp40CfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.07
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        # Dynamics are unchanged from model_730, so use it as a stronger
        # executed-action anchor than during the original Kp40 adaptation.
        reference_policy_coef = 0.15
        reference_action_deadband = 0.06
        reference_action_hinge_coef = 1.0
