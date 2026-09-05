"""Isolated hold/path/contact repair; same RS01 actuator as V9."""

from .rs01_omni_v9_config import Rs01OmniV9Speed14Cfg, Rs01OmniV9Speed14CfgPPO


class Rs01OmniV10RecoveryCfg(Rs01OmniV9Speed14Cfg):
    class env(Rs01OmniV9Speed14Cfg.env):
        # Preserve all 55 existing columns; append longitudinal path state.
        num_observations = 57

    class asset(Rs01OmniV9Speed14Cfg.asset):
        name = "rs01_omni_v10_recovery"

    class rewards(Rs01OmniV9Speed14Cfg.rewards):
        pose_position_scale_m = 0.10
        pose_heading_scale_rad = 0.20

        class scales(Rs01OmniV9Speed14Cfg.rewards.scales):
            trajectory_lateral_error = 0.0
            omni_heading_error = 0.0
            pose_error = -0.30


class Rs01OmniV10RecoveryStrongCfg(Rs01OmniV10RecoveryCfg):
    class asset(Rs01OmniV10RecoveryCfg.asset):
        name = "rs01_omni_v10_recovery_strong"

    class rewards(Rs01OmniV10RecoveryCfg.rewards):
        class scales(Rs01OmniV10RecoveryCfg.rewards.scales):
            pose_error = -0.60


class Rs01OmniV10RecoveryCfgPPO(Rs01OmniV9Speed14CfgPPO):
    class runner(Rs01OmniV9Speed14CfgPPO.runner):
        experiment_name = "rs01_omni_v10_recovery"
        adapt_observation_input = True
        load_optimizer = False


class Rs01OmniV10RecoveryStrongCfgPPO(Rs01OmniV10RecoveryCfgPPO):
    class runner(Rs01OmniV10RecoveryCfgPPO.runner):
        experiment_name = "rs01_omni_v10_recovery_strong"
