"""Anti-static A/B tasks derived from the measured RS01 omni v2 task."""

from .rs01_omni_v2_config import Rs01OmniV2Cfg, Rs01OmniV2CfgPPO


class Rs01OmniV3Contact1Cfg(Rs01OmniV2Cfg):
    """Common anti-static repair with a -1.0 prolonged four-foot cost."""

    class commands(Rs01OmniV2Cfg.commands):
        # Reset strata: 20% stand, 20% march, and 60% moving. Moving modes
        # retain the v2 relative distribution.
        mode_probabilities = [
            0.200,
            0.200,
            0.192,
            0.084,
            0.120,
            0.108,
            0.096,
        ]

    class asset(Rs01OmniV2Cfg.asset):
        name = "rs01_omni_v3_contact1"

    class rewards(Rs01OmniV2Cfg.rewards):
        # Preserve broad early gradients. The progress factor below makes a
        # stationary policy score zero for every nonzero command.
        command_planar_tracking_sigma = 0.010
        command_yaw_tracking_sigma = 0.040

        class scales(Rs01OmniV2Cfg.rewards.scales):
            # Replace separately additive planar/yaw rewards with one command
            # objective. This removes the free yaw reward during translation
            # and the free planar reward during turning.
            tracking_planar_velocity = 0.0
            tracking_yaw_velocity = 0.0
            tracking_command_velocity = 3.0

            # Directly charge wrong swing/support feet. Correct four-foot
            # handoff is allowed but receives no positive phase payment.
            phase_support_tracking = 0.0
            phase_contact_error = -1.5
            prolonged_all_feet_contact = -1.0


class Rs01OmniV3Contact15Cfg(Rs01OmniV3Contact1Cfg):
    """Same repair, changing only prolonged four-foot contact to -1.5."""

    class asset(Rs01OmniV3Contact1Cfg.asset):
        name = "rs01_omni_v3_contact15"

    class rewards(Rs01OmniV3Contact1Cfg.rewards):
        class scales(Rs01OmniV3Contact1Cfg.rewards.scales):
            prolonged_all_feet_contact = -1.5


class Rs01OmniV3Contact1CfgPPO(Rs01OmniV2CfgPPO):
    class runner(Rs01OmniV2CfgPPO.runner):
        experiment_name = "rs01_omni_v3_contact1"


class Rs01OmniV3Contact15CfgPPO(Rs01OmniV2CfgPPO):
    class runner(Rs01OmniV2CfgPPO.runner):
        experiment_name = "rs01_omni_v3_contact15"
