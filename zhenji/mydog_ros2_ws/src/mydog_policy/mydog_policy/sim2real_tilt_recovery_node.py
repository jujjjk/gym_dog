#!/usr/bin/env python3
"""Real-machine node for checkpoint 5650 with a recovery-state exit guard."""

from __future__ import annotations

import math

import numpy as np
import rclpy

from .sim2real_symmetric_transition_node import (
    MydogSymmetricTransition5530Node,
)
from .tilt_recovery_contract import (
    COMMAND_FEEDBACK,
    MODEL_TASK,
    validate_metadata,
)


class MydogTiltRecovery5530Node(MydogSymmetricTransition5530Node):
    """Deploy 5650 and explicitly leave the abnormal previous-action loop."""

    MODEL_TASK = MODEL_TASK
    COMMAND_FEEDBACK = COMMAND_FEEDBACK
    validate_deployment_metadata = staticmethod(validate_metadata)

    def __init__(self):
        super().__init__()

        self.declare_parameter("recovery_trigger_tilt_deg", 3.0)
        self.declare_parameter("recovery_upright_tilt_deg", 2.0)
        self.declare_parameter("recovery_upright_cycles", 10)
        self.recovery_trigger_tilt_rad = math.radians(float(
            self.get_parameter("recovery_trigger_tilt_deg").value
        ))
        self.recovery_upright_tilt_rad = math.radians(float(
            self.get_parameter("recovery_upright_tilt_deg").value
        ))
        self.recovery_upright_cycles = int(
            self.get_parameter("recovery_upright_cycles").value
        )
        if not (
            0.0 < self.recovery_upright_tilt_rad
            < self.recovery_trigger_tilt_rad
            < self.max_tilt_rad
        ):
            raise RuntimeError(
                "recovery tilt thresholds must satisfy "
                "0 < upright < trigger < max_tilt"
            )
        if self.recovery_upright_cycles < 1:
            raise RuntimeError("recovery_upright_cycles must be positive")

        self._recovery_guard_active = False
        self._recovery_upright_count = 0
        self._recovery_reset_count = 0
        self.get_logger().warn(
            "[TILT_RECOVERY_5650] guard active: trigger="
            f"{math.degrees(self.recovery_trigger_tilt_rad):.1f}deg, "
            f"upright={math.degrees(self.recovery_upright_tilt_rad):.1f}deg "
            f"for {self.recovery_upright_cycles} cycles"
        )

    def _guard_estimator_sync(self, obs: np.ndarray, info: dict):
        guarded_obs, guarded_info, status = super()._guard_estimator_sync(
            obs,
            info,
        )
        guarded_obs = np.asarray(guarded_obs, dtype=np.float32)
        tilt_rad = self._tilt_from_observation(guarded_obs)

        if tilt_rad >= self.recovery_trigger_tilt_rad:
            self._recovery_guard_active = True
            self._recovery_upright_count = 0
        elif self._recovery_guard_active:
            if tilt_rad <= self.recovery_upright_tilt_rad:
                self._recovery_upright_count += 1
            else:
                self._recovery_upright_count = 0

            if self._recovery_upright_count >= self.recovery_upright_cycles:
                # The observation for this cycle was already built. Clear both
                # its previous-action block and the transactional runtime state,
                # while keeping gait phase and heading continuous. This breaks
                # the self-sustaining saturated-action loop seen on hardware.
                self._reset_contract_state(reset_phase=False)
                guarded_obs = guarded_obs.copy()
                guarded_obs[36:48] = 0.0
                self._recovery_guard_active = False
                self._recovery_upright_count = 0
                self._recovery_reset_count += 1
                self.get_logger().warn(
                    "[TILT_RECOVERY_5650] upright recovery complete; "
                    "previous action and policy filter reset, gait phase kept | "
                    f"count={self._recovery_reset_count}"
                )

        return guarded_obs, guarded_info, status


def main(args=None):
    rclpy.init(args=args)
    node = MydogTiltRecovery5530Node()
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
