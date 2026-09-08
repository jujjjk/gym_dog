"""Single-variable contact-gate ablation; unchanged V10 RS01 mechanics."""

from .rs01_omni_v10_config import Rs01OmniV10RecoveryCfg, Rs01OmniV10RecoveryCfgPPO


class Rs01OmniV11HardGateCfg(Rs01OmniV10RecoveryCfg):
    class asset(Rs01OmniV10RecoveryCfg.asset):
        name = "rs01_omni_v11_hard_gate"

    class rewards(Rs01OmniV10RecoveryCfg.rewards):
        tracking_contact_gate = True

        class scales(Rs01OmniV10RecoveryCfg.rewards.scales):
            # In V10 the legal-contact gate makes support_tracking exactly
            # equal to two_contact_quality. Preserve their TOTAL payoff.
            phase_support_tracking = 0.0
            phase_two_contact_quality = 1.75


class Rs01OmniV11ContinuousCfg(Rs01OmniV11HardGateCfg):
    class asset(Rs01OmniV11HardGateCfg.asset):
        name = "rs01_omni_v11_continuous"

    class rewards(Rs01OmniV11HardGateCfg.rewards):
        # Keep all gait/contact penalties; only stop zeroing velocity payoff.
        tracking_contact_gate = False


class Rs01OmniV11HardGateCfgPPO(Rs01OmniV10RecoveryCfgPPO):
    class runner(Rs01OmniV10RecoveryCfgPPO.runner):
        experiment_name = "rs01_omni_v11_hard_gate"


class Rs01OmniV11ContinuousCfgPPO(Rs01OmniV11HardGateCfgPPO):
    class runner(Rs01OmniV11HardGateCfgPPO.runner):
        experiment_name = "rs01_omni_v11_continuous"
