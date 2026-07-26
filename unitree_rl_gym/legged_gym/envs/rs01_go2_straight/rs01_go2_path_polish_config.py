"""Closed-loop straight-heading polish derived from rear-coordinated model_600."""

from .rs01_go2_rear_coord_config import (
    Rs01Go2RearCoordCfg,
    Rs01Go2RearCoordCfgPPO,
)


class Rs01Go2PathPolishCfg(Rs01Go2RearCoordCfg):
    """Add the minimum observable state required to stop accumulated yaw."""

    class env(Rs01Go2RearCoordCfg.env):
        # The appended scalar is wrapped desired-minus-current heading.
        num_observations = 51

    class commands(Rs01Go2RearCoordCfg.commands):
        observe_straight_heading_error = True
        # +/-0.5 rad maps to the clipped +/-1 observation range.
        straight_heading_observation_scale = 2.0

    class init_state(Rs01Go2RearCoordCfg.init_state):
        # Small two-sided perturbations teach recovery instead of memorizing
        # the model_600's one-direction nominal drift. Disabled in playback.
        reset_heading_noise_rad = 0.08

    class rewards(Rs01Go2RearCoordCfg.rewards):
        heading_recovery_gain_rad_s_per_rad = 1.5
        heading_recovery_max_rate_rad_s = 0.60

        class scales(Rs01Go2RearCoordCfg.rewards.scales):
            # Do not fight useful correction with an unconditional yaw-rate
            # cost. Track a restoring yaw-rate target instead.
            yaw_rate = 0.0
            heading_recovery = -0.25


class Rs01Go2PathPolishCfgPPO(Rs01Go2RearCoordCfgPPO):
    class policy(Rs01Go2RearCoordCfgPPO.policy):
        init_noise_std = 0.12

    class algorithm(Rs01Go2RearCoordCfgPPO.algorithm):
        # The first 25 updates removed most of the one-sided drift, while 50
        # updates crossed through zero and built drift in the other direction.
        # Use a finer continuation step around the accepted model_625.
        learning_rate = 5.0e-5
        schedule = "fixed"

    class runner(Rs01Go2RearCoordCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        # Straight-path quality is non-monotonic during this small polish.
        # Dense checkpoints let evaluation select the minimum-drift policy
        # instead of assuming that the final update is best.
        save_interval = 5
        action_std_value = 0.12
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        reference_policy_coef = 0.20
        reference_action_deadband = 0.06
        reference_action_hinge_coef = 1.5
