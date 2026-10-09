"""Experimental cadence task; pilot did NOT outperform32750 on stability.

Retained for reproducible research, not promoted for long training/deployment.
Unchanged guard5, rewards and real RS01 plant.
"""
from .rs01_v22_phase_guard import (
    Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard5PPO)


def lateral_frequency(original, commands, cap):
    # Continuous command blend; no phase reset or simulator-only contact input.
    pure = (1.-commands[:,0].abs()/.1-commands[:,2].abs()/.2).clamp(0.,1.)
    moving = ((commands[:,1].abs()-.08)/.12).clamp(0.,1.)
    return original + pure*moving*(original.clamp(max=cap)-original)


class Rs01V22PhaseLateralCfg(Rs01V22PhaseGuard5Cfg):
    class lateral_cadence:
        maximum_hz = 2.1


class Rs01V22PhaseLateralPPO(Rs01V22PhaseGuard5PPO):
    class runner(Rs01V22PhaseGuard5PPO.runner):
        experiment_name = 'rs01_v22_phase_lateral'


class Rs01V22PhaseLateralRobot(Rs01V22PhaseGuardRobot):
    def _command_gait_frequency(self):
        original = super()._command_gait_frequency()
        cap = self.cfg.lateral_cadence.maximum_hz
        if not 1./self.cfg.rewards.gait_period_s <= cap <= self.cfg.rewards.wide_gait_max_frequency_hz:
            raise ValueError('Lateral cadence cap must lie within the existing cadence range')
        return lateral_frequency(original, self.commands, cap)
