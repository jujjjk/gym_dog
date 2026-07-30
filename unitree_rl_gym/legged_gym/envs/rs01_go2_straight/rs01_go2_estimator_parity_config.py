"""54-D migration using the same leg/path estimator as the real RS01."""

from .rs01_go2_path54_sim2sim_config import (
    Rs01Go2Path54Sim2SimTransferCfg,
    Rs01Go2Path54Sim2SimTransferCfgPPO,
)


class Rs01Go2EstimatorParityCfg(Rs01Go2Path54Sim2SimTransferCfg):
    """Replace actor root-state velocity/path inputs with deployable estimates."""

    class commands(Rs01Go2Path54Sim2SimTransferCfg.commands):
        use_rs01_estimated_observations = True

    class init_state(Rs01Go2Path54Sim2SimTransferCfg.init_state):
        # A real dead-reckoning path always starts at zero.  The inherited
        # synthetic +/-0.15 m offset was observable only through simulator
        # truth and therefore cannot be reproduced on the robot.
        reset_path_lateral_error_noise_m = 0.0
        # The real node latches the current yaw after stable-ready.  Initial
        # world-heading offsets would ask the simulator to recover information
        # that the real node intentionally defines as its new zero.
        reset_heading_noise_rad = 0.0
        reset_yaw_rate_noise_rad_s = 0.0

    class rs01_odometry:
        # Exact values used by Rs01NewMachineLegOdometry on the Jetson.
        nominal_base_height_m = 0.307
        foot_radius_m = 0.016
        height_margin_m = 0.030
        vertical_speed_threshold_m_s = 0.25
        velocity_residual_threshold_m_s = 0.35
        filter_alpha = 0.35
        no_contact_decay = 0.90
        previous_stance_score_bonus = 0.08
        # A trot path update is valid only when one full diagonal support pair
        # agrees kinematically.  Same-side and one-foot estimates caused the
        # real robot to integrate false lateral displacement.
        strict_diagonal_pairs = True
        path_update_min_confidence = 0.5


class Rs01Go2EstimatorParityCfgPPO(
    Rs01Go2Path54Sim2SimTransferCfgPPO
):
    class policy(Rs01Go2Path54Sim2SimTransferCfgPPO.policy):
        init_noise_std = 0.02

    class algorithm(Rs01Go2Path54Sim2SimTransferCfgPPO.algorithm):
        # The 54-D layout and model_1850 gait orbit are unchanged.  This stage
        # only adapts to a changed observation source, so use a smaller step.
        learning_rate = 2.0e-6
        schedule = "fixed"

    class runner(Rs01Go2Path54Sim2SimTransferCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.02
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = False
        reference_policy_coef = 0.05
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 0.50
        reference_action_transform = "clip"
        reference_action_clip = 1.0
