#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-contract helpers shared by the sim-to-real parity node.

The action filter mirrors ``gym_dog/mujoko/sim2sim.py`` and the training
``FanfanRobot.step`` path:

    bounded actor action
    -> alpha filter
    -> action-rate limit
    -> action-acceleration limit
    -> filtered action

The PD limiter reproduces post-PD signed torque clipping while retaining the
position-PD command interface used by the real motors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _vector(value: Any, size: int, *, name: str, default: float) -> np.ndarray:
    """Convert a scalar/list metadata value to a finite float32 vector."""
    if value is None:
        return np.full(size, default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        arr = np.full(size, float(arr), dtype=np.float32)
    else:
        arr = arr.reshape(-1)
        if arr.size != size:
            raise ValueError(f"{name} must contain {size} values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr.astype(np.float32)


@dataclass(frozen=True)
class ActionFilterConfig:
    enabled: bool
    alpha: float
    rate_limits: np.ndarray
    accel_limits: np.ndarray

    @classmethod
    def from_control(cls, control: Mapping[str, Any], size: int = 12) -> "ActionFilterConfig":
        enabled = bool(control.get("filter_policy_actions", False))
        alpha = float(control.get("policy_action_filter_alpha", 1.0))
        if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("policy_action_filter_alpha must be finite and in [0, 1]")
        rate_limits = _vector(
            control.get("policy_action_rate_limits"),
            size,
            name="policy_action_rate_limits",
            default=1.0e9,
        )
        accel_limits = _vector(
            control.get("policy_action_accel_limits"),
            size,
            name="policy_action_accel_limits",
            default=1.0e9,
        )
        if np.any(rate_limits <= 0.0) or np.any(accel_limits <= 0.0):
            raise ValueError("policy action rate/acceleration limits must be positive")
        return cls(enabled, alpha, rate_limits, accel_limits)


class ContractPolicyActionFilter:
    """Stateful action filter matching the training and MuJoCo reference path."""

    def __init__(self, config: ActionFilterConfig, size: int = 12):
        self.config = config
        self.size = int(size)
        self.action = np.zeros(self.size, dtype=np.float32)
        self.action_velocity = np.zeros(self.size, dtype=np.float32)

    @classmethod
    def from_control(
        cls,
        control: Mapping[str, Any],
        size: int = 12,
    ) -> "ContractPolicyActionFilter":
        return cls(ActionFilterConfig.from_control(control, size=size), size=size)

    def reset(self, action: np.ndarray | None = None) -> None:
        if action is None:
            self.action.fill(0.0)
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(self.size)
            if not np.all(np.isfinite(arr)):
                raise ValueError("reset action must be finite")
            self.action[:] = arr
        self.action_velocity.fill(0.0)

    def step(self, bounded_action: np.ndarray, dt: float) -> np.ndarray:
        desired_policy_action = np.asarray(bounded_action, dtype=np.float32).reshape(self.size)
        if not np.all(np.isfinite(desired_policy_action)):
            raise ValueError("bounded policy action must be finite")

        if not self.config.enabled:
            self.action[:] = desired_policy_action
            self.action_velocity.fill(0.0)
            return self.action.copy()

        dt = max(float(dt), 1.0e-6)
        desired = self.action + self.config.alpha * (desired_policy_action - self.action)
        desired_velocity = np.clip(
            (desired - self.action) / dt,
            -self.config.rate_limits,
            self.config.rate_limits,
        )
        velocity_delta = np.clip(
            desired_velocity - self.action_velocity,
            -self.config.accel_limits * dt,
            self.config.accel_limits * dt,
        )
        next_velocity = self.action_velocity + velocity_delta
        next_action = self.action + next_velocity * dt

        crossed = (desired - self.action) * (desired - next_action) < 0.0
        next_action = np.where(crossed, desired, next_action)
        next_velocity = (next_action - self.action) / dt

        self.action[:] = next_action.astype(np.float32)
        self.action_velocity[:] = next_velocity.astype(np.float32)
        return self.action.copy()


class PDTorqueEquivalentLimiter:
    """Convert signed post-PD torque clipping back to a safe position target.

    For the motor law

        tau = kp * (q_target - q) + kd * (qd_target - dq) + tau_ff

    this class clips ``tau`` and solves the same equation for ``q_target``.
    It therefore changes the position target only as much as required to
    reproduce the clipped training torque, rather than applying independent
    absolute-value position-error limits twice.
    """

    def limit(
        self,
        q_raw: np.ndarray,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limits: np.ndarray | float,
        qd_target: np.ndarray | None = None,
        torque_ff: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
        q_current = np.asarray(q_current, dtype=np.float32).reshape(12)
        dq_current = np.asarray(dq_current, dtype=np.float32).reshape(12)
        kp = np.abs(np.asarray(kp, dtype=np.float32).reshape(12))
        kd = np.abs(np.asarray(kd, dtype=np.float32).reshape(12))
        limits = _vector(torque_limits, 12, name="torque_limits", default=1.0e9)
        qd_target = (
            np.zeros(12, dtype=np.float32)
            if qd_target is None
            else np.asarray(qd_target, dtype=np.float32).reshape(12)
        )
        torque_ff = (
            np.zeros(12, dtype=np.float32)
            if torque_ff is None
            else np.asarray(torque_ff, dtype=np.float32).reshape(12)
        )

        for name, arr in (
            ("q_raw", q_raw),
            ("q_current", q_current),
            ("dq_current", dq_current),
            ("kp", kp),
            ("kd", kd),
            ("qd_target", qd_target),
            ("torque_ff", torque_ff),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
        if np.any(kp <= 1.0e-6):
            raise ValueError("kp must be positive")
        if np.any(limits <= 0.0):
            raise ValueError("torque_limits must be positive")

        velocity_error = qd_target - dq_current
        raw_delta = q_raw - q_current
        tau_raw = kp * raw_delta + kd * velocity_error + torque_ff
        tau_safe = np.clip(tau_raw, -limits, limits)
        q_safe = q_current + (
            tau_safe - kd * velocity_error - torque_ff
        ) / kp
        safe_delta = q_safe - q_current
        limited_mask = np.abs(tau_raw - tau_safe) > 1.0e-6
        tau_reconstructed = kp * safe_delta + kd * velocity_error + torque_ff

        return q_safe.astype(np.float32), {
            "enabled": True,
            "limited_count": int(np.count_nonzero(limited_mask)),
            "limited_mask": limited_mask,
            "torque_budget": float(np.max(limits)),
            "torque_limits": limits.astype(np.float32),
            "tau_raw_signed": tau_raw.astype(np.float32),
            "tau_safe_signed": tau_safe.astype(np.float32),
            "tau_reconstructed_signed": tau_reconstructed.astype(np.float32),
            # Compatibility with the legacy CSV/debug helpers.
            "tau_est": np.abs(tau_safe).astype(np.float32),
            "tau_est_max": float(np.max(np.abs(tau_safe))),
            "raw_delta": raw_delta.astype(np.float32),
            "safe_delta": safe_delta.astype(np.float32),
            "err_limit": np.abs(safe_delta).astype(np.float32),
            "err_limit_min": float(np.min(np.abs(safe_delta))),
            "err_limit_max": float(np.max(np.abs(safe_delta))),
        }

    def limit_with_torque_feedforward(
        self,
        q_raw: np.ndarray,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limits: np.ndarray | float,
        qd_target: np.ndarray | None = None,
        torque_ff: np.ndarray | None = None,
        torque_ff_limit: float = 17.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Saturate the complete motor torque while preserving ``q_raw``.

        The RS04 motion command has both a position-PD target and a torque
        feed-forward field.  The previous parity path converted a clipped
        torque back into a different position target.  That changes the
        learned trajectory, especially when the derivative term is large.

        This path first keeps the learned position target and uses the
        feed-forward field to cancel the excess PD torque.  Only if the
        protocol feed-forward range is insufficient is the position target
        moved, and then only by the remaining amount needed to meet the
        training torque limit.
        """
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
        q_current = np.asarray(q_current, dtype=np.float32).reshape(12)
        dq_current = np.asarray(dq_current, dtype=np.float32).reshape(12)
        kp = np.abs(np.asarray(kp, dtype=np.float32).reshape(12))
        kd = np.abs(np.asarray(kd, dtype=np.float32).reshape(12))
        limits = _vector(torque_limits, 12, name="torque_limits", default=1.0e9)
        qd_target = (
            np.zeros(12, dtype=np.float32)
            if qd_target is None
            else np.asarray(qd_target, dtype=np.float32).reshape(12)
        )
        torque_ff = (
            np.zeros(12, dtype=np.float32)
            if torque_ff is None
            else np.asarray(torque_ff, dtype=np.float32).reshape(12)
        )
        ff_limit = float(torque_ff_limit)

        for name, arr in (
            ("q_raw", q_raw),
            ("q_current", q_current),
            ("dq_current", dq_current),
            ("kp", kp),
            ("kd", kd),
            ("qd_target", qd_target),
            ("torque_ff", torque_ff),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
        if np.any(kp <= 1.0e-6):
            raise ValueError("kp must be positive")
        if np.any(limits <= 0.0):
            raise ValueError("torque_limits must be positive")
        if not np.isfinite(ff_limit) or ff_limit <= 0.0:
            raise ValueError("torque_ff_limit must be positive and finite")

        velocity_error = qd_target - dq_current
        pd_torque = kp * (q_raw - q_current) + kd * velocity_error
        requested_torque = pd_torque + torque_ff
        target_torque = np.clip(requested_torque, -limits, limits)

        # Keep q_raw whenever the feed-forward channel can cancel the excess
        # torque.  The command torque replaces, rather than adds to, the
        # caller's nominal feed-forward value.
        torque_ff_needed = target_torque - pd_torque
        torque_ff_cmd = np.clip(
            torque_ff_needed,
            -ff_limit,
            ff_limit,
        ).astype(np.float32)
        q_cmd = q_raw.copy()

        # If the RS04 torque field clips before the requested total torque is
        # reached, make the smallest possible q-target correction.  This is
        # only the unavoidable protocol fallback; normal cycles send q_raw.
        residual_torque = target_torque - (
            kp * (q_cmd - q_current)
            + kd * velocity_error
            + torque_ff_cmd
        )
        q_cmd += residual_torque / kp

        final_torque = (
            kp * (q_cmd - q_current)
            + kd * velocity_error
            + torque_ff_cmd
        )
        target_adjusted_mask = np.abs(q_cmd - q_raw) > 1.0e-6
        torque_limited_mask = np.abs(requested_torque - target_torque) > 1.0e-6
        return q_cmd.astype(np.float32), {
            "enabled": True,
            "limited_count": int(np.count_nonzero(torque_limited_mask)),
            "limited_mask": torque_limited_mask,
            "target_adjusted_count": int(np.count_nonzero(target_adjusted_mask)),
            "target_adjusted_mask": target_adjusted_mask,
            "torque_budget": float(np.max(limits)),
            "torque_limits": limits.astype(np.float32),
            "tau_raw_signed": requested_torque.astype(np.float32),
            "tau_pd_signed": pd_torque.astype(np.float32),
            "tau_safe_signed": target_torque.astype(np.float32),
            "tau_final_signed": final_torque.astype(np.float32),
            "torque_ff_needed": torque_ff_needed.astype(np.float32),
            "torque_ff_cmd": torque_ff_cmd.astype(np.float32),
            "tau_reconstructed_signed": final_torque.astype(np.float32),
            "tau_est": np.abs(final_torque).astype(np.float32),
            "tau_est_max": float(np.max(np.abs(final_torque))),
            "raw_delta": (q_raw - q_current).astype(np.float32),
            "safe_delta": (q_cmd - q_current).astype(np.float32),
            "err_limit": np.abs(q_cmd - q_current).astype(np.float32),
            "err_limit_min": float(np.min(np.abs(q_cmd - q_current))),
            "err_limit_max": float(np.max(np.abs(q_cmd - q_current))),
        }

    def limit_with_hardware_torque_saturation(
        self,
        q_raw: np.ndarray,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limits: np.ndarray | float,
        qd_target: np.ndarray | None = None,
        torque_ff: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Describe RS01-internal torque saturation without changing commands.

        RS01 motion mode evaluates the PD law continuously inside the motor and
        exposes ``limit_torque`` (private-protocol index 0x700B).  Configuring
        that parameter to the training limit is the closest hardware analogue
        of clipping the simulated PD torque every physics step.  In contrast,
        cancelling excess PD torque with one fixed feed-forward value per
        20 ms policy frame under-drives the joint as soon as its state changes.

        This method therefore preserves the learned position target and the
        caller's feed-forward command.  ``tau_safe_signed`` is diagnostic: it
        is the torque expected after the already-configured motor limit.
        """
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
        q_current = np.asarray(q_current, dtype=np.float32).reshape(12)
        dq_current = np.asarray(dq_current, dtype=np.float32).reshape(12)
        kp = np.abs(np.asarray(kp, dtype=np.float32).reshape(12))
        kd = np.abs(np.asarray(kd, dtype=np.float32).reshape(12))
        limits = _vector(torque_limits, 12, name="torque_limits", default=1.0e9)
        qd_target = (
            np.zeros(12, dtype=np.float32)
            if qd_target is None
            else np.asarray(qd_target, dtype=np.float32).reshape(12)
        )
        torque_ff = (
            np.zeros(12, dtype=np.float32)
            if torque_ff is None
            else np.asarray(torque_ff, dtype=np.float32).reshape(12)
        )

        for name, arr in (
            ("q_raw", q_raw),
            ("q_current", q_current),
            ("dq_current", dq_current),
            ("kp", kp),
            ("kd", kd),
            ("qd_target", qd_target),
            ("torque_ff", torque_ff),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
        if np.any(kp <= 1.0e-6):
            raise ValueError("kp must be positive")
        if np.any(limits <= 0.0):
            raise ValueError("torque_limits must be positive")

        velocity_error = qd_target - dq_current
        requested_torque = (
            kp * (q_raw - q_current)
            + kd * velocity_error
            + torque_ff
        )
        target_torque = np.clip(requested_torque, -limits, limits)
        limited_mask = np.abs(requested_torque - target_torque) > 1.0e-6
        false_mask = np.zeros(12, dtype=bool)

        return q_raw.copy(), {
            "enabled": True,
            "hardware_torque_limit_required": True,
            "limited_count": int(np.count_nonzero(limited_mask)),
            "limited_mask": limited_mask,
            "target_adjusted_count": 0,
            "target_adjusted_mask": false_mask,
            "torque_budget": float(np.max(limits)),
            "torque_limits": limits.astype(np.float32),
            "tau_raw_signed": requested_torque.astype(np.float32),
            "tau_pd_signed": (
                requested_torque - torque_ff
            ).astype(np.float32),
            "tau_safe_signed": target_torque.astype(np.float32),
            "tau_final_signed": target_torque.astype(np.float32),
            "torque_ff_needed": np.zeros(12, dtype=np.float32),
            "torque_ff_cmd": torque_ff.astype(np.float32),
            "tau_reconstructed_signed": target_torque.astype(np.float32),
            "tau_est": np.abs(target_torque).astype(np.float32),
            "tau_est_max": float(np.max(np.abs(target_torque))),
            "raw_delta": (q_raw - q_current).astype(np.float32),
            "safe_delta": (q_raw - q_current).astype(np.float32),
            "err_limit": np.abs(q_raw - q_current).astype(np.float32),
            "err_limit_min": float(np.min(np.abs(q_raw - q_current))),
            "err_limit_max": float(np.max(np.abs(q_raw - q_current))),
        }
