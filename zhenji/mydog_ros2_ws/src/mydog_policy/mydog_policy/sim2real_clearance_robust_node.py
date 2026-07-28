#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact real-machine deployment for clearance-robust checkpoint 5730.

The ONNX graph owns command feedback and exact left/right policy symmetry. This
node owns the exported reference-gait path, including motion-dependent calf
amplitude, the 0.9 s command-transition boost and projected-gravity tilt boost.
Stateful phase, action filtering, heading target and transition age are committed
only after a successful motor send.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy

from .clearance_robust_contract import (
    COMMAND_FEEDBACK,
    MODEL_TASK,
    TORQUE_LIMITS_POLICY,
    clearance_gait_offset,
    validate_metadata,
)
from .mydog_policy_node import MydogPolicyNode
from .sim2real_parity_fixed_node import MydogPolicyParityFixedNode


class MydogClearanceRobust5730Node(MydogPolicyParityFixedNode):
    """Deploy model 5730 with the exact Gym/MuJoCo gait-reference contract."""

    def __init__(self):
        super().__init__()

        if self.deployment_config is None:
            raise RuntimeError(
                "clearance-robust deployment requires ONNX metadata"
            )
        validate_metadata(self.deployment_config)

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
            raise RuntimeError("model 5730 deployment requires policy_hz=50")
        if abs(float(self.model_gait_phase_period) - 0.45) > 1.0e-7:
            raise RuntimeError("model 5730 gait period must be 0.45 s")

        self._last_transition_anchor_cmd = np.asarray(
            self.cmd_target,
            dtype=np.float32,
        ).copy()
        self._command_transition_age_s = 10.0
        self._last_clearance_diagnostics = {}

        # Reproduce simulation's fixed-step heading integration transactionally.
        self.obs_builder._get_heading_observation = (
            self._heading_observation_without_integration
        )
        self._last_good_scaled_lin_vel = np.zeros(3, dtype=np.float32)

        control = self.deployment_config["control"]
        gait = self.deployment_config["gait"]
        feedback_text = ", ".join(
            f"{name}={float(control[name]):.3f}"
            for name in COMMAND_FEEDBACK
        )
        self.get_logger().warn(
            "[CLEARANCE_ROBUST_5730] exact metadata validated | "
            f"task={MODEL_TASK}, gait=0.45s, feedback=[{feedback_text}], "
            "strict_symmetry=true"
        )
        self.get_logger().warn(
            "[CLEARANCE_ROBUST_5730] dynamic gait enabled | "
            f"straight={float(gait['calf_amplitude']):+.3f}, "
            f"lateral={float(gait['motion_calf_amplitude']):+.3f}, "
            f"yaw={float(gait['yaw_calf_amplitude']):+.3f}, "
            f"transition={float(gait['clearance_transition_duration']):.2f}s, "
            f"tilt_max={float(gait['clearance_tilt_boost_max']):.2f}x"
        )
        self.get_logger().warn(
            "[CLEARANCE_ROBUST_5730] active torque limits real order="
            f"{active_real.tolist()} (global cap 13 Nm; model profile 10/10/13)"
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
            "[CLEARANCE_ROBUST_5730] heading anchor reset | "
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
        # Called only after a successful send (or a valid no-send preview).
        super()._advance_contract_phase()
        nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)
        self._command_transition_age_s += nominal_dt

        if self.obs_builder.heading_command:
            return
        yaw = getattr(self.obs_builder, "heading_yaw", None)
        if self.obs_builder.heading_target is None:
            if yaw is None or not np.isfinite(float(yaw)):
                return
            self.obs_builder.heading_target = float(yaw)
        self.obs_builder.heading_target = self._wrap_pi(
            float(self.obs_builder.heading_target)
            + float(self.cmd[2]) * nominal_dt
        )

    def _is_significant_transition(
        self,
        previous_anchor: np.ndarray,
        new_target: np.ndarray,
    ) -> bool:
        # sim2sim.py uses np.array_equal: every actual command change starts a
        # new clearance window, while identical 20 Hz republishes do not.
        previous = np.asarray(previous_anchor, dtype=np.float32).reshape(3)
        current = np.asarray(new_target, dtype=np.float32).reshape(3)
        return not np.array_equal(previous, current)

    def _reset_clearance_transition(self, command: np.ndarray, reason: str) -> None:
        self._command_transition_age_s = 0.0
        self.get_logger().info(
            "[CLEARANCE_ROBUST_5730] clearance transition reset | "
            f"reason={reason}, cmd={np.asarray(command).tolist()}"
        )

    def cmd_callback(self, msg):
        # Preserve phase, heading and action-filter state across command changes,
        # matching sim2sim.py --demo-matrix. A command change only starts the
        # temporary clearance window.
        was_fresh = bool(self.command_is_fresh())
        previous_anchor = self._last_transition_anchor_cmd.copy()

        MydogPolicyNode.cmd_callback(self, msg)
        new_target = np.asarray(self.cmd_target, dtype=np.float32).copy()

        if not was_fresh:
            self._reset_contract_state(reset_phase=True)
            self._reset_rear_torque_boost_window()
            self._reset_heading_target_to_current_yaw("fresh_command_stream")
            if float(np.linalg.norm(new_target)) > 0.05:
                self._reset_clearance_transition(
                    new_target,
                    "fresh_command_stream",
                )
            else:
                self._command_transition_age_s = 10.0
            self._last_transition_anchor_cmd = new_target
            return

        if self._is_significant_transition(previous_anchor, new_target):
            self._reset_rear_torque_boost_window()
            self._reset_clearance_transition(new_target, "command_transition")
            self._last_transition_anchor_cmd = new_target

    def finish_startup_stand(self, current_real):
        super().finish_startup_stand(current_real)
        self._reset_rear_torque_boost_window()
        self._reset_heading_target_to_current_yaw("startup_stand_complete")
        current_target = np.asarray(self.cmd_target, dtype=np.float32).copy()
        if float(np.linalg.norm(current_target)) > 0.05:
            self._reset_clearance_transition(
                current_target,
                "startup_stand_complete",
            )
        else:
            self._command_transition_age_s = 10.0
        self._last_transition_anchor_cmd = current_target

    def deployment_gait_offset(self, phase: float) -> np.ndarray:
        """Exact NumPy counterpart of 5730 Gym/MuJoCo reference-gait logic."""
        gait = self.deployment_config["gait"]
        result, diagnostics = clearance_gait_offset(
            phase,
            self.cmd,
            self._command_transition_age_s,
            self.obs_builder.projected_gravity,
            gait,
            self.deployment_config["joint_names"],
        )
        self._last_clearance_diagnostics = diagnostics
        self._contract_gait_gate = float(diagnostics["gait_gate"])
        return result

    def _guard_estimator_sync(self, obs: np.ndarray, info: dict):
        # Do not replace one stale velocity frame with zero; that would create a
        # false tracking error inside the ONNX feedback wrapper.
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

        self._last_good_scaled_lin_vel *= self.estimator_mismatch_velocity_decay
        guarded_obs = np.asarray(guarded_obs, dtype=np.float32).copy()
        guarded_obs[0:3] = self._last_good_scaled_lin_vel
        guarded_info = dict(guarded_info)
        lin_scale = max(float(self.obs_builder.lin_vel_scale), 1.0e-6)
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
    node = MydogClearanceRobust5730Node()
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
