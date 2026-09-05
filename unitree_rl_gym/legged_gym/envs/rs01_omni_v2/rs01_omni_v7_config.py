"""Delay-aware structured exploration amplitude A/B for RS01."""

from .rs01_omni_v6_config import (
    Rs01OmniV6Seed08Cfg,
    Rs01OmniV6Seed08CfgPPO,
)


class Rs01OmniV7Seed14Cfg(Rs01OmniV6Seed08Cfg):
    class asset(Rs01OmniV6Seed08Cfg.asset):
        name = "rs01_omni_v7_seed14"

    class control(Rs01OmniV6Seed08Cfg.control):
        structured_exploration_amplitude = 1.40
        structured_exploration_profile = "plateau"
        structured_exploration_phase_lead = 0.12


class Rs01OmniV7Seed18Cfg(Rs01OmniV7Seed14Cfg):
    class asset(Rs01OmniV7Seed14Cfg.asset):
        name = "rs01_omni_v7_seed18"

    class control(Rs01OmniV7Seed14Cfg.control):
        structured_exploration_amplitude = 1.80


class Rs01OmniV7Seed14CfgPPO(Rs01OmniV6Seed08CfgPPO):
    class runner(Rs01OmniV6Seed08CfgPPO.runner):
        experiment_name = "rs01_omni_v7_seed14"


class Rs01OmniV7Seed18CfgPPO(Rs01OmniV6Seed08CfgPPO):
    class runner(Rs01OmniV6Seed08CfgPPO.runner):
        experiment_name = "rs01_omni_v7_seed18"
