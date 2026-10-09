"""Targeted scratch acquisition repair; V22 motor, clock and reward weights stay."""
import math
import torch
from .rs01_v22_phase_scratch import (
    Rs01V22PhaseScratchCfg, Rs01V22PhaseScratchPPO, Rs01V22PhaseScratchRobot)
from .rs01_omni_v19_env import huber_positive
from ..rs01_go2_straight.rs01_go2_straight_env import Rs01Go2StraightRobot


def lift_credit(no_lift_cycles, maximum_cycles, ramp_cycles, floor):
    # Unlike a worst-foot/max gate, every recovered foot gives learning credit.
    stale = ((no_lift_cycles - maximum_cycles) / ramp_cycles).clamp(0., 1.)
    return floor + (1. - floor) * (1. - stale).mean(1).square(), stale.mean(1)


def attitude_cost(roll_cost, pitch, deadband, scale):
    # One attitude envelope, not a second additive orientation penalty.
    return torch.maximum(roll_cost, huber_positive((pitch.abs() - deadband) / scale))


class Rs01V22PhaseLiftCfg(Rs01V22PhaseScratchCfg):
    class phase_exchange(Rs01V22PhaseScratchCfg.phase_exchange):
        require_lift_event = True

    class phase_acquisition:
        tracking_floor = .05
        pitch_deadband_rad = math.radians(5.)
        pitch_scale_rad = math.radians(10.)


class Rs01V22PhaseLiftPPO(Rs01V22PhaseScratchPPO):
    class runner(Rs01V22PhaseScratchPPO.runner):
        experiment_name = 'rs01_v22_phase_lift'
        # Exploration only; policy action mapping and motor limits stay intact.
        action_std_value = .8


class Rs01V22PhaseLiftRobot(Rs01V22PhaseScratchRobot):
    def _lift_credit(self):
        if not self._exchange_ready:
            return torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)
        cfg = self.cfg.phase_exchange
        return lift_credit(self.no_lift_cycles, cfg.maximum_no_lift_cycles,
                           cfg.penalty_ramp_cycles, self.cfg.phase_acquisition.tracking_floor)

    def _reward_tracking_command_velocity(self):
        accuracy = super()._reward_tracking_command_velocity()
        credit, _ = self._lift_credit()
        # Explicit stand remains legal; zero-command MARCH must exchange feet.
        credit = torch.where(self._walking_command_gate() > .5, credit, 1.)
        self.v22_lift_tracking_gate = credit.detach()
        reward = accuracy * credit
        self.v11_tracking_reward = reward.detach()
        return reward

    def _reward_prolonged_all_feet_contact(self):
        original = Rs01Go2StraightRobot._reward_prolonged_all_feet_contact(self)
        _, stale_mean = self._lift_credit()
        return torch.maximum(original, stale_mean * self._walking_command_gate())

    def _reward_phase_two_contact_quality(self):
        reward = super()._reward_phase_two_contact_quality()
        cfg = self.cfg.phase_acquisition
        old_roll_cost = self.v17_balance_cost
        combined = attitude_cost(old_roll_cost, self.rpy[:, 1],
                                 cfg.pitch_deadband_rad, cfg.pitch_scale_rad)
        # Parent already charged roll. Charge ONLY the extra envelope cost.
        reward -= self.cfg.rewards.balance_prior_weight * (combined - old_roll_cost) * self._walking_command_gate()
        self.v17_balance_cost = combined.detach()
        self.v11_contact_quality_reward = reward.detach()
        return reward
