"""Omnidirectional inward-target guard; real RS01 dynamics remain unchanged.

Replaces V20 projection, never composes with it. Apply after action clipping
and scaling, BEFORE the existing rate/acceleration limits, delay and motor model.
The same mapping is required in MuJoCo/real deployment of any trained actor.
This bounds requested targets, not actual joint positions under external load.
"""
import math
import torch
from .rs01_v22_phase_lift import (
    Rs01V22PhaseLiftCfg, Rs01V22PhaseLiftPPO, Rs01V22PhaseLiftRobot)


def map_inward_targets(target, hip_ids, sides, full_range, inward_limit):
    """Left inward is negative, right inward positive; outward unchanged.

    Linear compression preserves the full inward action resolution. Inputs are
    already clipped by the parent RS01 controller. No command/phase switch can
    release the guard or introduce a target discontinuity.
    """
    hip = target[:, hip_ids]
    result = target.clone()
    result[:, hip_ids] = torch.where(
        hip * sides < 0., hip * (inward_limit / full_range), hip)
    return result


class Rs01V22PhaseGuard5Cfg(Rs01V22PhaseLiftCfg):
    class inward_guard:
        target_limit_rad = math.radians(5.)


class Rs01V22PhaseGuard3Cfg(Rs01V22PhaseGuard5Cfg):
    class inward_guard(Rs01V22PhaseGuard5Cfg.inward_guard):
        target_limit_rad = math.radians(3.)


class Rs01V22PhaseGuard5PPO(Rs01V22PhaseLiftPPO):
    class runner(Rs01V22PhaseLiftPPO.runner):
        experiment_name = 'rs01_v22_phase_guard5'


class Rs01V22PhaseGuard3PPO(Rs01V22PhaseLiftPPO):
    class runner(Rs01V22PhaseLiftPPO.runner):
        experiment_name = 'rs01_v22_phase_guard3'


class Rs01V22PhaseGuardRobot(Rs01V22PhaseLiftRobot):
    def _project_rs01_policy_target(self, target):
        # Deliberately do NOT call super(): that would double-compress inward
        # actions for straight commands and make train/deployment semantics differ.
        if not hasattr(self, 'inward_guard_hip_ids'):
            self.inward_guard_hip_ids = [
                self.dof_names.index(leg + '_hip_joint')
                for leg in ('FL', 'FR', 'RL', 'RR')]
            ids = self.inward_guard_hip_ids
            if not torch.allclose(self.default_dof_pos[:, ids],
                                  torch.zeros_like(self.default_dof_pos[:, ids])):
                raise ValueError('Inward guard requires zero default hip angles')
            self.inward_guard_sides = target.new_tensor([1., -1., 1., -1.])
            self.inward_guard_ranges = (self.rs01_action_scale_rad[ids]
                                       * self.cfg.normalization.clip_actions)
            limit = self.cfg.inward_guard.target_limit_rad
            if not math.isfinite(limit) or not bool(
                    ((self.inward_guard_ranges >= limit) & (limit > 0.)).all()):
                raise ValueError('Require 0 < inward limit <= every hip action range')
        return map_inward_targets(
            target, self.inward_guard_hip_ids, self.inward_guard_sides,
            self.inward_guard_ranges, self.cfg.inward_guard.target_limit_rad)
