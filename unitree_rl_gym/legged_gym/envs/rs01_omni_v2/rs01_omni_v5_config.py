"""Dense diagonal-support discovery with a controlled odd-contact A/B."""

from .rs01_omni_v4_config import (
    Rs01OmniV4Contact1Cfg,
    Rs01OmniV4Contact1CfgPPO,
)


class Rs01OmniV5Odd05Cfg(Rs01OmniV4Contact1Cfg):
    """Close the three-foot shortcut without prescribing joint trajectories."""

    class asset(Rs01OmniV4Contact1Cfg.asset):
        name = "rs01_omni_v5_odd05"

    class rewards(Rs01OmniV4Contact1Cfg.rewards):
        # A wider kernel supplies a continuous load-transfer signal before a
        # swing foot has fully cleared the 2 N contact threshold.
        phase_support_sigma = 0.25

        class scales(Rs01OmniV4Contact1Cfg.rewards.scales):
            tracking_command_velocity = 6.0
            phase_support_tracking = 2.0
            phase_two_contact_quality = 1.0
            prolonged_all_feet_contact = -1.0
            odd_feet_contact = -0.5


class Rs01OmniV5Odd10Cfg(Rs01OmniV5Odd05Cfg):
    """Same v5 task, changing only the odd-support penalty strength."""

    class asset(Rs01OmniV5Odd05Cfg.asset):
        name = "rs01_omni_v5_odd10"

    class rewards(Rs01OmniV5Odd05Cfg.rewards):
        class scales(Rs01OmniV5Odd05Cfg.rewards.scales):
            odd_feet_contact = -1.0


class Rs01OmniV5Odd05CfgPPO(Rs01OmniV4Contact1CfgPPO):
    class runner(Rs01OmniV4Contact1CfgPPO.runner):
        experiment_name = "rs01_omni_v5_odd05"


class Rs01OmniV5Odd10CfgPPO(Rs01OmniV4Contact1CfgPPO):
    class runner(Rs01OmniV4Contact1CfgPPO.runner):
        experiment_name = "rs01_omni_v5_odd10"
