"""Experimental command-conditioned AMP; unchanged real RS01 plant."""
from pathlib import Path
from .rs01_omni_v30_config import Rs01OmniV30Cfg, Rs01OmniV30CfgPPO

REFERENCE = str(Path(__file__).resolve().parents[4] / 'artifacts/rs01_reference/design_10s_20261005')
COMMANDS = [[0,0,0],[.25,0,0],[-.18,0,0],[0,.12,0],[0,-.12,0],
            [0,0,.45],[0,0,-.45],[.18,.08,0],[.18,-.08,0],[.15,.06,.3]]


class Rs01AMPCfg(Rs01OmniV30Cfg):
    class amp:
        commands = COMMANDS
        # Recorded templates step at one tempo; plant recordings follow the demonstrator's
        # command-dependent clock. The runner refuses a reference whose tempo this clock misses.
        command_dependent_clock = False
    class commands(Rs01OmniV30Cfg.commands):
        resampling_time = 5.
    class rewards(Rs01OmniV30Cfg.rewards):
        gait_period_s = .5
        gait_stance_ratio = .65
        coordination_weight = 0.
        class scales(Rs01OmniV30Cfg.rewards.scales):
            phase_swing_clearance = 0.


class Rs01AMPCfgPPO(Rs01OmniV30CfgPPO):
    class runner(Rs01OmniV30CfgPPO.runner):
        experiment_name = 'rs01_amp_style'
        amp_enabled = True
        amp_reference = REFERENCE
        amp_allow_kinematic = True  # User explicitly requested this designed style.
        amp_style_weight = 4.
        amp_warmup_iterations = 200
        adapt_observation_input = False  # Strict 61D load; AMP state must also restore.
        load_optimizer = False


class Rs01AMPRetimeCfg(Rs01AMPCfg):
    class rewards(Rs01AMPCfg.rewards):
        gait_period_s = 1. / 1.5
        gait_stance_ratio = .60


class Rs01AMPControlCfgPPO(Rs01AMPCfgPPO):
    class runner(Rs01AMPCfgPPO.runner):
        experiment_name = 'rs01_amp_control'


class Rs01AMPRetimeCfgPPO(Rs01AMPCfgPPO):
    class runner(Rs01AMPCfgPPO.runner):
        experiment_name = 'rs01_amp_retime'
        amp_reference = str(Path(REFERENCE).parent / 'amp_retime_20261006')
        amp_allow_reference_warmstart = True
