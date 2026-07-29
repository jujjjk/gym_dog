"""Narrow Sim2Sim transfer from the accepted 54-D model_1725 path policy."""

from .rs01_go2_path54_config import (
    Rs01Go2Model1425Path54Cfg,
    Rs01Go2Model1425Path54CfgPPO,
)


class Rs01Go2Path54Sim2SimTransferCfg(Rs01Go2Model1425Path54Cfg):
    """Expose model_1725 to measured cross-simulator dynamics spread."""

    class noise(Rs01Go2Model1425Path54Cfg.noise):
        # First close dynamics parity. Sensor noise belongs to the later
        # Sim2Real stage and would obscure this experiment.
        add_noise = False

    class domain_rand(Rs01Go2Model1425Path54Cfg.domain_rand):
        # The nominal path stage disabled all spread and consequently used
        # materially less torque than the matched MuJoCo scene. Keep these
        # ranges narrow and centred on the measured new-machine parameters.
        randomize_friction = True
        friction_range = [0.90, 1.10]
        randomize_base_mass = True
        added_mass_range = [-0.20, 0.20]
        randomize_rs01_actuator = True
        rs01_response_gain_scale_range = [0.97, 1.03]
        rs01_time_constant_scale_range = [0.95, 1.05]
        rs01_friction_scale_range = [0.95, 1.05]
        rs01_delay_step_offset_range = [-1, 1]
        rs01_independent_motor_randomization = True
        rs01_independent_delay_randomization = True
        push_robots = False


class Rs01Go2Path54Sim2SimTransferCfgPPO(
    Rs01Go2Model1425Path54CfgPPO
):
    class policy(Rs01Go2Model1425Path54CfgPPO.policy):
        init_noise_std = 0.025

    class algorithm(Rs01Go2Model1425Path54CfgPPO.algorithm):
        learning_rate = 5.0e-6
        schedule = "fixed"

    class runner(Rs01Go2Model1425Path54CfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.025
        freeze_action_std = True
        load_optimizer = False
        # model_1725 already has the exact 54-D layout.
        adapt_observation_input = False
        # Preserve its accepted path/gait orbit while allowing small
        # left-right corrections for the independent motor/contact spread.
        reference_policy_coef = 0.05
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 0.50
        reference_action_transform = "clip"
        reference_action_clip = 1.0
