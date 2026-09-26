"""Heading correction applies to straight walking only; every other command runs uncorrected."""
import numpy as np
import pytest
from mydog_policy.rs01_model23500_core import Rs01Model23500Core


def actor(command, weight=1.):
    core = Rs01Model23500Core.__new__(Rs01Model23500Core)
    core.command = np.asarray(command, dtype=float)
    core.heading_correction_weight = weight
    return core


@pytest.mark.parametrize('command', [[.4, 0, 0], [-.2, 0, 0], [.001, 0, 0]])
def test_straight_commands_keep_full_heading_error(command):
    a = actor(command)
    assert a.heading_correction_active()
    assert a._direction_heading_error(.3) == pytest.approx(.3)
    a.heading_correction_weight = .5
    assert a._direction_heading_error(.3) == pytest.approx(.15)


@pytest.mark.parametrize('command', [[0, 0, 0], [0, .2, 0], [0, 0, .3], [.3, .2, 0], [.3, 0, .3], [.3, .2, -.3]])
def test_march_lateral_turn_and_combined_commands_are_uncorrected(command):
    a = actor(command)
    assert not a.heading_correction_active()
    assert a._direction_heading_error(.3) == 0.
    assert a._direction_heading_error(-1.) == 0.


def test_no_command_yet_is_uncorrected():
    core = Rs01Model23500Core.__new__(Rs01Model23500Core)
    assert not core.heading_correction_active()
    assert core._direction_heading_error(.2) == 0.
