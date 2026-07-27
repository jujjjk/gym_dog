"""RS01 Kp40 actuator-feasibility continuation from path model_635."""

from .rs01_go2_path_polish_config import (
    Rs01Go2PathPolishCfg,
    Rs01Go2PathPolishCfgPPO,
)


class Rs01Go2Kp40Cfg(Rs01Go2PathPolishCfg):
    """Keep the accepted gait structure while adapting to softer real gains."""

    class control(Rs01Go2PathPolishCfg.control):
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}
        # With model_635 at Kp40, 0.14 produced only 0.024 m/s. A controlled
        # sweep found 0.18 to be the smallest scale that restored meaningful
        # motion while keeping 17 Nm saturation near 8%.
        action_scale = 0.18

    class commands(Rs01Go2PathPolishCfg.commands):
        # First adapt the gait at a moderate speed. Do not ask the softer
        # support loop to reproduce the old 0.35 m/s policy immediately.
        playback_speed_mps = 0.23

        class ranges(Rs01Go2PathPolishCfg.commands.ranges):
            lin_vel_x = [0.18, 0.28]

    class rewards(Rs01Go2PathPolishCfg.rewards):
        # At 0.23 m/s a stationary policy receives only 2.9% of full speed
        # reward, while a 0.03 m/s error still receives 94%.
        tracking_sigma = 0.015

        class scales(Rs01Go2PathPolishCfg.rewards.scales):
            # The unadapted Kp40/scale0.18 sweep already reduced saturation to
            # 8%; retain that headroom during gait recovery.
            raw_torque_over_peak = -0.75
            motor_saturation = -0.75


class Rs01Go2Kp40CfgPPO(Rs01Go2PathPolishCfgPPO):
    class policy(Rs01Go2PathPolishCfgPPO.policy):
        init_noise_std = 0.10

    class algorithm(Rs01Go2PathPolishCfgPPO.algorithm):
        learning_rate = 1.0e-4
        schedule = "fixed"

    class runner(Rs01Go2PathPolishCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 10
        action_std_value = 0.10
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        # Dynamics and action scale changed, so the old actor is only a light
        # gait prior rather than a tight action anchor.
        reference_policy_coef = 0.05
        reference_action_deadband = 0.10
        reference_action_hinge_coef = 0.50
