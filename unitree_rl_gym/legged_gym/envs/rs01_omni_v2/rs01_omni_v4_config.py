"""Positive diagonal-discovery A/B tasks after the v3 behavior pilot."""

from .rs01_omni_v3_config import (
    Rs01OmniV3Contact1Cfg,
    Rs01OmniV3Contact1CfgPPO,
    Rs01OmniV3Contact15Cfg,
    Rs01OmniV3Contact15CfgPPO,
)


class Rs01OmniV4Contact1Cfg(Rs01OmniV3Contact1Cfg):
    """Reward real two-foot diagonal support and remove the fall shortcut."""

    class asset(Rs01OmniV3Contact1Cfg.asset):
        name = "rs01_omni_v4_contact1"

    class rewards(Rs01OmniV3Contact1Cfg.rewards):
        class scales(Rs01OmniV3Contact1Cfg.rewards.scales):
            tracking_command_velocity = 4.0
            phase_contact_error = -0.75
            phase_two_contact_quality = 1.5
            alive = 1.0
            # Reward scales are multiplied by the 0.02 s policy dt. This is
            # an effective -1.0 event cost for non-timeout termination.
            termination = -50.0


class Rs01OmniV4Contact15Cfg(Rs01OmniV3Contact15Cfg):
    """Same v4 repair, retaining the stronger -1.5 four-foot A/B arm."""

    class asset(Rs01OmniV3Contact15Cfg.asset):
        name = "rs01_omni_v4_contact15"

    class rewards(Rs01OmniV3Contact15Cfg.rewards):
        class scales(Rs01OmniV3Contact15Cfg.rewards.scales):
            tracking_command_velocity = 4.0
            phase_contact_error = -0.75
            phase_two_contact_quality = 1.5
            alive = 1.0
            termination = -50.0


class Rs01OmniV4Contact1CfgPPO(Rs01OmniV3Contact1CfgPPO):
    class policy(Rs01OmniV3Contact1CfgPPO.policy):
        # 0.20 normalized exploration moves a calf target by only 0.028 rad,
        # which did not discover a 14 mm swing. Keep exploration bounded and
        # fixed, but make coherent foot lift reachable during scratch training.
        init_noise_std = 0.35

    class runner(Rs01OmniV3Contact1CfgPPO.runner):
        experiment_name = "rs01_omni_v4_contact1"
        action_std_value = 0.35
        freeze_action_std = True


class Rs01OmniV4Contact15CfgPPO(Rs01OmniV3Contact15CfgPPO):
    class policy(Rs01OmniV3Contact15CfgPPO.policy):
        init_noise_std = 0.35

    class runner(Rs01OmniV3Contact15CfgPPO.runner):
        experiment_name = "rs01_omni_v4_contact15"
        action_std_value = 0.35
        freeze_action_std = True
