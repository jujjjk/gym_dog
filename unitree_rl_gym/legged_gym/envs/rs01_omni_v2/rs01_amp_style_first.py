"""Scratch AMP without a 'must already trot' reward gate. Old experiments stay intact."""
import torch
from .rs01_amp_seeded_env import Rs01AMPSeededRobot
from .rs01_amp_seeded_config import (
    Rs01AMPSeededCfg, Rs01AMPSeededSlowCfg, Rs01AMPSeededPPO, Rs01AMPSeededSlowPPO)
from .rs01_amp_env import Rs01AMPRobot


def support_failure(contact, diagonal_a, diagonal_b):
    count = contact.sum(1)
    diagonal = ((contact == diagonal_a[None]).all(1)
                | (contact == diagonal_b[None]).all(1))
    # Three feet are a valid transient load transfer, not a forbidden gait.
    return ((count < 2) | ((count == 2) & ~diagonal)).float()


class Rs01AMPStyleFirstRobot(Rs01AMPSeededRobot):
    def _reward_illegal_support(self):
        return support_failure(self.get_foot_contact_mask(),
                               self.diagonal_a_contact_mask, self.diagonal_b_contact_mask)

    def compute_reward(self):
        # Keep transition capture, but deliberately skip Core's min-four-lifts gate.
        Rs01AMPRobot.compute_reward(self)
        # Style is learned from transitions, not an external contact/clearance clock.
        # Soft upright attenuation only; no all-feet-lift or exact-contact prerequisite.
        self.amp_gate = ((-self.projected_gravity[:, 2] - .8) / .2).clamp(0., 1.)


class StyleFirstRewards(Rs01AMPSeededCfg.rewards):
    class scales:
        tracking_command_velocity = 2.
        orientation = -1.
        illegal_support = -.5
        collision = -1.
        dof_pos_limits = -2.
        raw_torque_over_peak = -.5
        action_rate = -.002
        termination = -50.


class Rs01AMPStyleFirstCfg(Rs01AMPSeededCfg):
    class rewards(StyleFirstRewards):
        pass


class Rs01AMPStyleFirstSlowCfg(Rs01AMPSeededSlowCfg):
    class rewards(StyleFirstRewards):
        gait_period_s = 1. / 1.5


class Rs01AMPStyleFirstPPO(Rs01AMPSeededPPO):
    class runner(Rs01AMPSeededPPO.runner):
        experiment_name = 'rs01_amp_style_first'


class Rs01AMPStyleFirstSlowPPO(Rs01AMPSeededSlowPPO):
    class runner(Rs01AMPSeededSlowPPO.runner):
        experiment_name = 'rs01_amp_style_first_slow'
