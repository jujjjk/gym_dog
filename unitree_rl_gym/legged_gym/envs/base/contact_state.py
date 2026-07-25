"""Shared foot-contact state helpers.

The environment updates this state exactly once per policy step.  Rewards,
phase logic, termination and telemetry consume the resulting mask instead of
applying their own force thresholds.
"""

import torch


def validate_contact_thresholds(enter_force_n, release_force_n):
    enter_force_n = float(enter_force_n)
    release_force_n = float(release_force_n)
    if enter_force_n <= 0.0:
        raise ValueError("foot contact enter threshold must be positive")
    if release_force_n < 0.0 or release_force_n > enter_force_n:
        raise ValueError(
            "foot contact release threshold must satisfy "
            f"0 <= release <= enter, got {release_force_n} > {enter_force_n}"
        )
    return enter_force_n, release_force_n


def update_contact_mask(
    vertical_force_n,
    previous_contact,
    enter_force_n,
    release_force_n,
):
    """Return debounced contact state with optional force hysteresis."""
    enter_force_n, release_force_n = validate_contact_thresholds(
        enter_force_n, release_force_n
    )
    entering = vertical_force_n >= enter_force_n
    remaining = vertical_force_n >= release_force_n
    return torch.where(previous_contact, remaining, entering)


def update_consecutive_true_count(condition, previous_count):
    """Count consecutive true policy frames and clear immediately on false."""
    return torch.where(
        condition,
        previous_count + 1,
        torch.zeros_like(previous_count),
    )
