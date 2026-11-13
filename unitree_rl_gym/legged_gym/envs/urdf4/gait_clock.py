"""Diagonal-trot timing shared by rewards, critic observations and playback.

Pure tensor helpers with no simulator dependency.  One scalar clock per
environment drives four legs: FL+RR share phase 0, FR+RL lead by half a cycle.
A leg is scheduled in stance while its phase is below ``stance_ratio``.
"""

import math

import torch

# Fraction-of-cycle offsets.  Equal offsets within a diagonal are what makes
# this a trot; changing them changes the gait family.
LEG_PHASE_OFFSETS = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}


def leg_phase(clock_phase, leg_offsets):
    """[num_envs] clock and [num_legs] offsets -> [num_envs, num_legs] in [0, 1)."""
    return torch.remainder(clock_phase.unsqueeze(1) + leg_offsets.unsqueeze(0), 1.0)


def scheduled_stance(phase_by_leg, stance_ratio):
    return phase_by_leg < stance_ratio


class CycleMean:
    """Per-robot mean of a per-step signal over the last ``steps`` policy steps.

    With ``steps`` equal to one gait period the periodic within-stride motion
    cancels and what remains is the steady offset.
    """

    def __init__(self, num_envs, width, steps, device):
        self.window = torch.zeros(num_envs, int(steps), width, device=device)
        self.head = 0

    def reset(self, env_ids):
        # A new episode has accumulated no error yet.
        self.window[env_ids] = 0.0

    def push(self, value):
        self.head = (self.head + 1) % self.window.shape[1]
        self.window[:, self.head] = value

    def mean(self):
        return self.window.mean(dim=1)


def swing_height_target(phase_by_leg, stance_ratio, foot_radius_m, clearance_m):
    """Foot-centre height requested during scheduled swing (flat ground).

    A half-sine from touchdown height to ``clearance_m`` and back, so takeoff
    and touchdown both ask for zero clearance and no step discontinuity.
    """
    progress = torch.clamp(
        (phase_by_leg - stance_ratio) / max(1.0 - stance_ratio, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    return foot_radius_m + clearance_m * torch.sin(math.pi * progress)
