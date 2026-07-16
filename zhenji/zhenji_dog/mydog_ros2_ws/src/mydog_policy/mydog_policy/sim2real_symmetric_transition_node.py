#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transactional real-machine deployment for symmetric-transition checkpoint 5530.

The ONNX graph already contains:
- longitudinal/lateral/yaw velocity feedback;
- heading correction;
- exact left/right policy symmetrization.

This node must therefore feed the exported 52-dimensional observation contract
without adding a second external feedback or symmetry transform.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy

from .mydog_policy_node import MydogPolicyNode
from .sim2real_parity_fixed_node import MydogPolicyParityFixedNode
from .symmetric_transition_contract import (
    COMMAND_FEEDBACK,
    MODEL_TASK,
    TORQUE_LIMITS_POLICY,
    validate_metadata,
)


class MydogSymmetricTransition5530Node(MydogPolicyParityFixedNode):
    """Deploy model 5530 with continuous command-transition semantics."""

    def __init__(self):
        super().__init__()

        if self.deployment_config is None:
            raise RuntimeError(
                "symmetric-transition deployment requires ONNX metadata"
            )
        validate_metadata(self.deployment_config)

        if not self.has_parameter("transition_reset_vx_delta"):
            self.declare_parameter("transition_reset_vx_delta", 0.025)
        if not self.has_parameter("transition_reset_vy_delta"):
            self.declare_parameter("transition_reset_vy_delta", 0.025)
        if not self.has_parameter("transition_reset_yaw_delta"):
            self.declare_parameter("transition_reset_yaw_delta", 0.08)
        if not self.has_parameter("hold_last_lin_vel_on_estimator_mismatch"):
            self.declare_parameter(
                "hold_last_lin_vel_on_estimator_mismatch",
                True,
            )
        if not self.has_parameter("estimator_mismatch_velocity_decay"):
            self.declare_parameter(
                "estimator_mismatch_velocity_decay",
                0.98,
            )

        self.transition_reset_delta = np.array(
            [
                float(self.get_parameter("transition_reset_vx_delta").value),
                float(self.get_parameter("transition_reset_vy_delta").value),
                float(self.get_parameter("transition_reset_yaw_delta").value),
            ],
            dtype=np.float32,
        )
        if np.any(self.transition_reset_delta <= 0.0):
            raise RuntimeError("transition reset deltas must be positive")

        self.hold_last_lin_vel_on_estimator_mismatch = bool(
            self.get_parameter(
                "hold_last_lin_vel_on_estimator_mismatch"
            ).value
        )
        self.estimator_mismatch_velocity_decay = float(
            np.clip(
                float(
                    self.get_parameter(
                        "estimator_mismatch_velocity_decay"
                    ).value
                ),
                0.0,
                1.0,
            )
        )

        model_limits_policy = np.asarray(
            self.deployment_config["control"]["torque_limits"],
            dtype=np.float32,
        ).reshape(12)
        expected_policy = np.asarray(TORQUE_LIMITS_POLICY, dtype=np.float32)
        if not np.allclose(model_limits_policy, expected_policy, atol=1.0e-6):
            raise RuntimeError("10/10/13 Nm model torque profile mismatch")

        expected_real = self.obs_builder.mapper.policy_values_to_real_order(
            expected_policy
        ).astype(np.float32)
        if not np.allclose(
            self.contract_torque_limits_real,
            expected_real,
            atol=1.0e-6,
        ):
            raise RuntimeError("policy-to-real torque-limit mapping mismatch")

        active_real = self._active_torque_limits_real()
        if np.any(active_real > expected_real + 1.0e-6):
            raise RuntimeError("active torque limit exceeds ONNX contract")

        if abs(float(self.policy_hz) - 50.0) > 1.0e-6:
            raise RuntimeError("model 5530 deployment requires policy_hz=50")
        if abs(float(self.model_gait_phase_period) - 0.45) > 1.0e-7:
            raise RuntimeError("model 5530 gait period must be 0.45 s")

        # Gym resets the desired heading to the current yaw whenever a new
        # command segment is sampled, while gait phase and action filter remain
        # continuous. Reproduce that behavior only when the target command
        # actually changes; repeated ROS publications must not reset heading.
        self._last_transition_anchor_cmd = np.asarray(
            self.cmd_target,
            dtype=np.float32,
        ).copy()

        # Make heading integration deterministic and transactional. Observation
        # construction reads the current heading target; successful control
        # commits advance it by exactly 1/50 s, matching simulation.
        self.obs_builder._get_heading_observation = (
            self._heading_observation_without_integration
        )

        self._last_good_scaled_lin_vel = np.zeros(3, dtype=np.float32)

        control = self.deployment_config["control"]
        feedback_text = ", ".join(
            f"{name}={float(control[name]):.3f}"
            for name in COMMAND_FEEDBACK
        )
        self.get_logger().warn(
            "[SYMMETRIC_TRANSITION_5530] exact metadata validated | "
            f"task={MODEL_TASK}, gait=0.45s, feedback=[{feedback_text}], "
            "strict_symmetry=true"
        )
        self.get_logger().warn(
            "[SYMMETRIC_TRANSITION_5530] command changes reset heading target "
            "without resetting gait phase or policy-action filter"
        )
        self.get_logger().warn(
            "[SYMMETRIC_TRANSITION_5530] ONNX owns velocity feedback and "
            "left/right symmetry; runtime adds neither a second feedback loop "
            "nor an action mirror"
        )
        self.get_logger().warn(
            "[SYMMETRIC_TRANSITION_5530] active torque limits real order="
            f"{active_real.tolist()}"
        )

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _reset_heading_target_to_current_yaw(self, reason: str) -> None:
        yaw = getattr(self.obs_builder, "heading_yaw", None)
        if yaw is None or not np.isfinite(float(yaw)):
            self.obs_builder.heading_target = None
            target_text = "pending"
        else:
            self.obs_builder.heading_target = float(yaw)
            target_text = f"{float(yaw):+.4f}rad"
        self.obs_builder._heading_stamp = time.monotonic()
        self.get_logger().info(
            "[SYMMETRIC_TRANSITION_5530] heading anchor reset | "
            f"reason={reason}, target={target_text}"
        )

    def _heading_observation_without_integration(self):
        yaw = getattr(self.obs_builder, "heading_yaw", None)
        if yaw is None or not np.isfinite(float(yaw)):
            return (
                np.array([0.0, 1.0], dtype=np.float32),
                float(self.obs_builder.cmd[2]),
            )

        if self.obs_builder.heading_target is None:
            self.obs_builder.heading_target = float(yaw)

        error = self._wrap_pi(
            float(self.obs_builder.heading_target) - float(yaw)
        )
        return (
            np.array([math.sin(error), math.cos(error)], dtype=np.float32),
            float(self.obs_builder.cmd[2]),
        )

    def _advance_contract_phase(self) -> None:
        # MydogPolicyParityFixedNode commits this method only after a successful
        # motor send, so heading target and gait phase remain synchronized.
        super()._advance_contract_phase()

        if self.obs_builder.heading_command:
            return

        yaw = getattr(self.obs_builder, "heading_yaw", None)
        if self.obs_builder.heading_target is None:
            if yaw is None or not np.isfinite(float(yaw)):
                return
            self.obs_builder.heading_target = float(yaw)

        nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)
        self.obs_builder.heading_target = self._wrap_pi(
            float(self.obs_builder.heading_target)
            + float(self.cmd[2]) * nominal_dt
        )

    def _is_significant_transition(
        self,
        previous_anchor: np.ndarray,
        new_target: np.ndarray,
    ) -> bool:
        delta = np.abs(
            np.asarray(new_target, dtype=np.float32)
            - np.asarray(previous_anchor, dtype=np.float32)
        )
        return bool(np.any(delta >= self.transition_reset_delta))

    def cmd_callback(self, msg):
        # Bypass MydogPolicyParityNode.cmd_callback because it resets phase and
        # action filter on zero->motion. Checkpoint 5530 was trained with
        # continuous phase/filter state across command transitions.
        was_fresh = bool(self.command_is_fresh())
        previous_anchor = self._last_transition_anchor_cmd.copy()

        MydogPolicyNode.cmd_callback(self, msg)

        new_target = np.asarray(self.cmd_target, dtype=np.float32).copy()

        if not was_fresh:
            # A genuinely new command stream is a new deployment episode.
            self._reset_contract_state(reset_phase=True)
            self._reset_rear_torque_boost_window()
            self._reset_heading_target_to_current_yaw("fresh_command_stream")
            self._last_transition_anchor_cmd = new_target
            return

        if self._is_significant_transition(previous_anchor, new_target):
            # Match Gym resampling: re-anchor desired heading, but preserve gait
            # phase, previous action and the action-filter state.
            self._reset_rear_torque_boost_window()
            self._reset_heading_target_to_current_yaw("command_transition")
            self._last_transition_anchor_cmd = new_target

    def finish_startup_stand(self, current_real):
        super().finish_startup_stand(current_real)
        self._reset_rear_torque_boost_window()
        self._reset_heading_target_to_current_yaw(
            "startup_stand_complete"
        )
        self._last_transition_anchor_cmd = np.asarray(
            self.cmd_target,
            dtype=np.float32,
        ).copy()

    def _guard_estimator_sync(self, obs: np.ndarray, info: dict):
        # Model 5530 contains strong velocity feedback. Replacing one stale
        # velocity frame with zero creates an artificial large tracking error.
        # Hold and gently decay the last synchronized scaled observation instead.
        original_obs = np.asarray(obs, dtype=np.float32).copy()
        guarded_obs, guarded_info, status = super()._guard_estimator_sync(
            obs,
            info,
        )

        sync_ok = bool(status.get("ok", False)) and (
            status.get("reason", "ok") == "ok"
        )
        if sync_ok:
            self._last_good_scaled_lin_vel[:] = original_obs[0:3]
            return guarded_obs, guarded_info, status

        if not self.hold_last_lin_vel_on_estimator_mismatch:
            return guarded_obs, guarded_info, status

        self._last_good_scaled_lin_vel *= (
            self.estimator_mismatch_velocity_decay
        )
        guarded_obs = np.asarray(guarded_obs, dtype=np.float32).copy()
        guarded_obs[0:3] = self._last_good_scaled_lin_vel

        guarded_info = dict(guarded_info)
        lin_scale = max(
            float(self.obs_builder.lin_vel_scale),
            1.0e-6,
        )
        guarded_info["base_lin_vel"] = (
            self._last_good_scaled_lin_vel / lin_scale
        ).astype(np.float32)
        status = dict(status)
        status["reason"] = (
            str(status.get("reason", "mismatch"))
            + "_held_last_velocity"
        )
        return guarded_obs, guarded_info, status


def main(args=None):
    rclpy.init(args=args)
    node = MydogSymmetricTransition5530Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
