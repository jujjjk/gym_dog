#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sim-to-real node that follows the exported training control contract.

This node intentionally lives beside the legacy ``mydog_policy_node`` so the
old deployment path remains available for A/B testing.  It preserves the
existing real/policy semantic mapper and changes only the policy-action,
reference-gait, phase and torque-protection stages.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy

from .mydog_policy_node import MydogPolicyNode
from .policy_contract import ContractPolicyActionFilter, PDTorqueEquivalentLimiter


class MydogPolicyParityNode(MydogPolicyNode):
    """Deployment node aligned with ``gym_dog`` training and MuJoCo control."""

    def __init__(self):
        super().__init__()

        # Explicit deployment-only phase compensation.  The underlying
        # fixed-step phase still advances exactly like simulation; only the
        # phase exposed to the ONNX observation and reference gait is shifted.
        if not self.has_parameter("gait_phase_lead_sec"):
            self.declare_parameter("gait_phase_lead_sec", 0.0)
        self.gait_phase_lead_sec = float(
            self.get_parameter("gait_phase_lead_sec").value
        )

        if self.policy is None or self.deployment_config is None:
            raise RuntimeError(
                "sim2real parity mode requires an ONNX model with "
                "fanfan_deployment_config metadata"
            )
        if not self.use_model_pd_gains:
            raise RuntimeError("sim2real parity mode requires use_model_pd_gains=true")

        control = self.deployment_config["control"]
        self.contract_action_filter = ContractPolicyActionFilter.from_control(control)
        self.contract_torque_limiter = PDTorqueEquivalentLimiter()
        if not self.has_parameter("motion_torque_ff_limit_nm"):
            self.declare_parameter("motion_torque_ff_limit_nm", 17.0)
        self.motion_torque_ff_limit_nm = float(
            self.get_parameter("motion_torque_ff_limit_nm").value
        )
        if (
            not np.isfinite(self.motion_torque_ff_limit_nm)
            or self.motion_torque_ff_limit_nm <= 0.0
        ):
            raise RuntimeError("motion_torque_ff_limit_nm must be positive")

        torque_limits_policy = np.asarray(
            control.get("torque_limits", np.full(12, 1.0e9, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(12)
        if not np.all(np.isfinite(torque_limits_policy)) or np.any(torque_limits_policy <= 0.0):
            # Old models may omit this field. In that case the ROS safety budget
            # remains the only active limit.
            torque_limits_policy = np.full(12, 1.0e9, dtype=np.float32)
        self.contract_torque_limits_real = self.obs_builder.mapper.policy_values_to_real_order(
            torque_limits_policy
        ).astype(np.float32)

        # Enforce the training contract.  These switches remain in the legacy
        # node, but must not alter the parity path.
        self.action_mode = "pure_rl"
        self.enable_cmd_smoothing = False
        self.enable_policy_action_cmd_gate = False
        self.policy_action_cmd_gate_max_scale = 1.0
        self.reset_gait_phase_on_command_start = False
        self.enable_target_smoothing = False
        self.enable_torque_error_limit = False
        self.enable_velocity_ff = False
        self.enable_rear_leg_posture_bias = False

        self.model_gait_phase_period = float(self.deployment_config["gait"]["period"])
        self.contract_gait_period_scale = float(self.deployment_gait_phase_period_scale)
        if (
            not np.isfinite(self.contract_gait_period_scale)
            or self.contract_gait_period_scale < 1.0
            or self.contract_gait_period_scale > 1.5
        ):
            raise RuntimeError(
                "sim2real parity gait period scale must be finite and in [1.0, 1.5]"
            )
        # Keep the policy-action/observation/PD contract unchanged while allowing
        # one explicit actuator-bandwidth adaptation: a slower reference phase.
        self.gait_phase_period = (
            self.model_gait_phase_period * self.contract_gait_period_scale
        )
        if (
            not np.isfinite(self.gait_phase_lead_sec)
            or self.gait_phase_lead_sec < 0.0
            or self.gait_phase_lead_sec > 0.10
        ):
            raise RuntimeError(
                "gait_phase_lead_sec must be finite and in [0.00, 0.10] seconds"
            )
        self.contract_phase_lead_cycles = (
            self.gait_phase_lead_sec / max(self.gait_phase_period, 1.0e-6)
        )
        self.obs_builder.gait_phase_period = self.gait_phase_period
        self._contract_phase = 0.0
        self._contract_gait_gate = 0.0
        self._walking_active_prev = False

        # ObsBuilder normally derives phase from wall-clock time. Replace that
        # instance method with a fixed-policy-step source, matching simulation.
        self.obs_builder.get_gait_phase = self._contract_get_gait_phase
        self._reset_contract_state(reset_phase=True)

        self.get_logger().warn(
            "[PARITY] active: metadata action filter, fixed-step gait phase, "
            "training gait gate, and signed post-PD torque-equivalent limiting"
        )
        self.get_logger().info(
            f"[PARITY] action_filter enabled={self.contract_action_filter.config.enabled} "
            f"alpha={self.contract_action_filter.config.alpha:.3f} "
            f"model_gait_period={self.model_gait_phase_period:.3f}s "
            f"gait_period_scale={self.contract_gait_period_scale:.3f} "
            f"active_gait_period={self.gait_phase_period:.3f}s "
            f"active_gait_frequency={1.0 / self.gait_phase_period:.3f}Hz "
            f"phase_lead={self.gait_phase_lead_sec:.3f}s "
            f"phase_lead_cycles={self.contract_phase_lead_cycles:.4f} "
            f"policy_hz={self.policy_hz:.3f}"
        )

    def _effective_contract_phase(self) -> float:
        """Return the phase seen by both ONNX and the reference gait."""
        return float(
            (self._contract_phase + self.contract_phase_lead_cycles) % 1.0
        )

    def _reset_contract_state(self, *, reset_phase: bool) -> None:
        self.contract_action_filter.reset()
        self.obs_builder.set_last_action(np.zeros(12, dtype=np.float32))
        if reset_phase:
            self._contract_phase = 0.0
        self.obs_builder.last_gait_phase = self._effective_contract_phase()

    def _contract_get_gait_phase(self) -> float:
        effective_phase = self._effective_contract_phase()
        self.obs_builder.last_gait_phase = effective_phase
        return effective_phase

    def _advance_contract_phase(self) -> None:
        nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)
        self._contract_phase = (
            self._contract_phase + nominal_dt / max(self.gait_phase_period, 1.0e-6)
        ) % 1.0
        self.obs_builder.last_gait_phase = self._effective_contract_phase()

    def cmd_callback(self, msg):
        was_zero = bool(np.all(np.abs(self.cmd) < self.zero_cmd_stand_threshold))
        was_stale = not self.command_is_fresh()
        super().cmd_callback(msg)
        is_zero = bool(np.all(np.abs(self.cmd_target) < self.zero_cmd_stand_threshold))
        if (was_zero and not is_zero) or (was_stale and not is_zero):
            # Treat a fresh walking command like the beginning of a simulation
            # episode: phase=0 and previous_action=0.
            self._reset_contract_state(reset_phase=True)
            self.get_logger().warn(
                "[PARITY] fresh walking command: reset fixed-step phase and filtered action"
            )

    def finish_startup_stand(self, current_real):
        super().finish_startup_stand(current_real)
        self._reset_contract_state(reset_phase=True)

    def deployment_gait_gate(self) -> float:
        gait = self.deployment_config["gait"]
        if not bool(gait.get("gate_with_command", False)):
            return 1.0
        cmd = np.asarray(self.cmd, dtype=np.float32).reshape(3)
        energy = float(cmd[0] ** 2 + cmd[1] ** 2 + 0.04 * cmd[2] ** 2)
        sigma = float(gait.get("command_gate_sigma", 0.0))
        if sigma <= 0.0:
            return 1.0 if energy > 0.0 else 0.0
        return float(1.0 - np.exp(-energy / sigma))

    def deployment_gait_offset(self, phase: float) -> np.ndarray:
        """Exact NumPy counterpart of ``gym_dog/mujoko/sim2sim.py``."""
        gait = self.deployment_config["gait"]
        result = np.zeros(12, dtype=np.float32)
        stance_ratio = float(gait["stance_ratio"])
        offsets = gait["phase_offsets"]
        for i, name in enumerate(self.deployment_config["joint_names"]):
            leg = name[:2]
            leg_phase = (float(phase) + float(offsets[leg])) % 1.0
            swing = np.clip(
                (leg_phase - stance_ratio) / (1.0 - stance_ratio),
                0.0,
                1.0,
            )
            smooth = swing * swing * (3.0 - 2.0 * swing)
            if "thigh" in name:
                if leg_phase < stance_ratio:
                    profile = -1.0 + 2.0 * np.clip(
                        leg_phase / stance_ratio,
                        0.0,
                        1.0,
                    )
                else:
                    profile = 1.0 - 2.0 * smooth
                result[i] = float(gait["thigh_amplitude"]) * profile
            elif "calf" in name:
                result[i] = (
                    float(gait["calf_amplitude"])
                    * np.sin(np.pi * smooth)
                    * float(leg_phase >= stance_ratio)
                )

        self._contract_gait_gate = self.deployment_gait_gate()
        return (result * self._contract_gait_gate).astype(np.float32)

    @staticmethod
    def _disabled_pre_limit_info(q_raw, q_current, torque_budget):
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
        q_current = np.asarray(q_current, dtype=np.float32).reshape(12)
        delta = q_raw - q_current
        zeros = np.zeros(12, dtype=np.float32)
        false_mask = np.zeros(12, dtype=bool)
        infs = np.full(12, np.inf, dtype=np.float32)
        return {
            "enabled": False,
            "dt": 0.0,
            "torque_budget": float(torque_budget),
            "base_err_limit": float("inf"),
            "err_limit_min": float("inf"),
            "err_limit_max": float("inf"),
            "err_limit": infs,
            "max_rate": infs,
            "max_accel": infs,
            "q_raw_error_abs_max": float(np.max(np.abs(delta))),
            "q_cmd_error_abs_max": float(np.max(np.abs(delta))),
            "qdot_cmd_abs_max": 0.0,
            "qddot_cmd_abs_max": 0.0,
            "pre_limited_count": 0,
            "rate_limited_count": 0,
            "accel_limited_count": 0,
            "post_limited_count": 0,
            "pre_limited_mask": false_mask,
            "rate_limited_mask": false_mask,
            "accel_limited_mask": false_mask,
            "post_limited_mask": false_mask,
            "raw_delta": delta.astype(np.float32),
            "safe_delta": delta.astype(np.float32),
            "qdot_cmd": zeros,
            "qddot_cmd": zeros,
        }

    def _active_torque_limits_real(self) -> np.ndarray:
        ros_budget = self.compute_torque_safety_budget_nm()
        if ros_budget <= 0.0:
            raise RuntimeError("active torque safety budget must be positive")
        return np.minimum(
            self.contract_torque_limits_real,
            np.full(12, ros_budget, dtype=np.float32),
        ).astype(np.float32)

    def _apply_pd_equivalent_limit(self, q_raw, q_current, dq_current):
        limits = self._active_torque_limits_real()
        qd_target = np.zeros(12, dtype=np.float32)
        torque_ff = np.zeros(12, dtype=np.float32)
        q_safe, info = self.contract_torque_limiter.limit_with_torque_feedforward(
            q_raw=q_raw,
            q_current=q_current,
            dq_current=dq_current,
            kp=self.send_kp_real,
            kd=self.send_kd_real,
            torque_limits=limits,
            qd_target=qd_target,
            torque_ff=torque_ff,
            torque_ff_limit=self.motion_torque_ff_limit_nm,
        )

        # Preserve the existing mechanical joint limits after the equivalent
        # torque conversion. Normally this is a no-op because q_raw was already
        # clamped in policy coordinates.
        mapper = self.obs_builder.mapper
        q_safe_policy, _ = mapper.real_to_policy_abs_q_dq(
            q_safe,
            np.zeros(12, dtype=np.float32),
        )
        q_safe_clamped = mapper.policy_target_to_real_target(q_safe_policy, clamp=True)
        joint_limit_mask = np.abs(q_safe_clamped - q_safe) > 1.0e-6
        if np.any(joint_limit_mask):
            velocity_error = -np.asarray(dq_current, dtype=np.float32).reshape(12)
            torque_ff_cmd = np.asarray(
                info.get("torque_ff_cmd", np.zeros(12, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(12)
            tau_after_clamp = (
                self.send_kp_real * (q_safe_clamped - q_current)
                + self.send_kd_real * velocity_error
                + torque_ff_cmd
            )
            if np.any(np.abs(tau_after_clamp) > limits + 1.0e-3):
                raise RuntimeError(
                    "mechanical joint clamp conflicts with the active torque limit; skip send"
                )
            q_safe = q_safe_clamped.astype(np.float32)
            info["tau_reconstructed_signed"] = tau_after_clamp.astype(np.float32)
            info["tau_final_signed"] = tau_after_clamp.astype(np.float32)
        info["joint_limit_adjusted_mask"] = joint_limit_mask
        info["rear_torque_boost_active"] = bool(
            getattr(self, "_rear_torque_boost_active", lambda: False)()
        )
        info["rear_torque_boost_elapsed_s"] = float(
            getattr(self, "_rear_torque_boost_elapsed", lambda: 0.0)()
        )
        info["rear_torque_limit_nm"] = float(np.max(limits[6:12]))
        info["protection_mode"] = "pd_torque_saturation_with_ff"
        return q_safe.astype(np.float32), info

    def control_loop(self):
        self._record_control_loop_rate()
        self._last_onnx_infer_dt_ms = 0.0
        try:
            measured_dt = self.get_control_dt()
            self.update_smoothed_command(measured_dt)
            obs, info = self.obs_builder.build_obs()

            max_age = float(np.max(info["age_ms"]))
            self._last_state_age_ms = max_age
            self._last_motor_state_poll_dt_ms = float(
                info.get("motor_state_poll_dt_ms", 0.0)
            )

            if max_age > self.max_motor_age_ms:
                stale = np.where(np.asarray(info["age_ms"]) > self.max_motor_age_ms)[0]
                if self.recheck_stale_motor_once:
                    obs, info = self.recheck_stale_motor_feedback(obs, info, stale)
                    max_age = float(np.max(info["age_ms"]))
                if max_age > self.max_motor_age_ms:
                    self.get_logger().warn(
                        f"[PARITY][SAFE] motor feedback too old: {max_age:.1f} ms; skip policy"
                    )
                    return

            if self.require_online and not np.all(info["online"]):
                self.get_logger().warn("[PARITY][SAFE] some motors offline; skip policy")
                return
            if self._startup_stand_state != "complete":
                self.handle_startup_stand(obs, info, measured_dt, max_age)
                return
            if self.stand_only:
                self.handle_stand_only(obs, info, max_age)
                return
            if self.joint_probe_enable:
                self.handle_joint_probe(obs, info, max_age)
                return
            if not self.policy_enable or self.policy is None:
                self.handle_stand_only(obs, info, max_age, mode="policy_disabled")
                return

            obs = self.ensure_policy_obs_dim(obs)
            zero_cmd_stand = self.zero_cmd_stand_active()
            self._last_zero_cmd_stand_active = zero_cmd_stand
            self._last_action_cmd_gate_scale = 1.0
            nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)

            if zero_cmd_stand:
                self._reset_contract_state(reset_phase=False)
                action_raw = np.zeros(12, dtype=np.float32)
                filtered_action = np.zeros(12, dtype=np.float32)
                target_policy_abs = self.obs_builder.mapper.default_joint_angle.copy()
                self._last_gait_offset_policy = np.zeros(12, dtype=np.float32)
                self._contract_gait_gate = 0.0
                cpg_action_info = {
                    "action_mode": "zero_cmd_stand",
                    "command_gate_scale": 0.0,
                    "zero_cmd_stand": True,
                    "frequency": 0.0,
                    "phase": float(self.obs_builder.last_gait_phase),
                    "base_phase": float(self._contract_phase),
                    "phase_lead_sec": float(self.gait_phase_lead_sec),
                    "gait_gate": 0.0,
                    "model_gait_period_s": self.model_gait_phase_period,
                    "gait_period_scale": self.contract_gait_period_scale,
                    "active_gait_period_s": self.gait_phase_period,
                    "protection_mode": "pd_equivalent",
                }
            else:
                infer_t0 = time.perf_counter()
                action_raw = self.policy.infer(obs)
                self._last_onnx_infer_dt_ms = (
                    time.perf_counter() - infer_t0
                ) * 1000.0
                filtered_action = self.contract_action_filter.step(
                    action_raw,
                    dt=nominal_dt,
                )
                target_policy_abs, cpg_action_info = self.action_to_policy_target_abs_by_mode(
                    filtered_action,
                    dt=nominal_dt,
                )
                cpg_action_info["command_gate_scale"] = 1.0
                cpg_action_info["zero_cmd_stand"] = False
                cpg_action_info["phase"] = float(self.obs_builder.last_gait_phase)
                cpg_action_info["base_phase"] = float(self._contract_phase)
                cpg_action_info["phase_lead_sec"] = float(self.gait_phase_lead_sec)
                cpg_action_info["frequency"] = 1.0 / self.gait_phase_period
                cpg_action_info["gait_gate"] = float(self._contract_gait_gate)
                cpg_action_info["model_gait_period_s"] = self.model_gait_phase_period
                cpg_action_info["gait_period_scale"] = self.contract_gait_period_scale
                cpg_action_info["active_gait_period_s"] = self.gait_phase_period
                cpg_action_info["protection_mode"] = "pd_equivalent"

            target_real_raw = self.obs_builder.mapper.policy_target_to_real_target(
                target_policy_abs,
                clamp=True,
            )
            current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
            current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
            torque_budget_nm = self.compute_torque_safety_budget_nm()
            pre_limit_info = self._disabled_pre_limit_info(
                target_real_raw,
                current_q,
                torque_budget_nm,
            )
            motor_vel_cmd = np.zeros(12, dtype=np.float32)

            if self.enable_send:
                # Normal deployment path: retain the complete PD-equivalent
                # limiter and mechanical-joint-limit consistency check.
                target_real, torque_limit_info = self._apply_pd_equivalent_limit(
                    target_real_raw,
                    current_q,
                    current_dq,
                )
            else:
                # Target-only diagnostic path:
                # calculate the same signed PD torque clipping for CSV preview,
                # but do not let the post-conversion mechanical clamp abort the
                # policy cycle. Nothing from this branch is sent to the motors.
                active_limits = self._active_torque_limits_real()
                target_real, torque_limit_info = (
                    self.contract_torque_limiter.limit(
                        q_raw=target_real_raw,
                        q_current=current_q,
                        dq_current=current_dq,
                        kp=self.send_kp_real,
                        kd=self.send_kd_real,
                        torque_limits=active_limits,
                        qd_target=np.zeros(12, dtype=np.float32),
                        torque_ff=np.zeros(12, dtype=np.float32),
                    )
                )
                torque_limit_info["joint_limit_adjusted_mask"] = np.zeros(
                    12, dtype=bool
                )
                torque_limit_info["protection_mode"] = (
                    "pd_equivalent_preview_no_send"
                )

            error = target_real - current_q

            self.publish_array(self.pub_obs, obs)
            self.publish_array(self.pub_action_raw, action_raw)
            self.publish_array(self.pub_action, filtered_action)
            self.publish_array(self.pub_target, target_real)
            if self._control_summary_log_due():
                tau_raw = np.asarray(
                    torque_limit_info["tau_raw_signed"], dtype=np.float32
                )
                tau_safe = np.asarray(
                    torque_limit_info["tau_safe_signed"], dtype=np.float32
                )
                self.get_logger().info(
                    f"[PARITY] cmd={self.cmd.tolist()} "
                    f"base_phase={self._contract_phase:.3f} "
                    f"phase={self.obs_builder.last_gait_phase:.3f} "
                    f"phase_lead={self.gait_phase_lead_sec:.3f}s "
                    f"gait_period={self.gait_phase_period:.3f}s "
                    f"gait_scale={self.contract_gait_period_scale:.2f} "
                    f"gait_gate={self._contract_gait_gate:.3f} "
                    f"raw_action_max={float(np.max(np.abs(action_raw))):.3f} "
                    f"filtered_action_max={float(np.max(np.abs(filtered_action))):.3f} "
                    f"tau_raw_max={float(np.max(np.abs(tau_raw))):.2f}Nm "
                    f"tau_safe_max={float(np.max(np.abs(tau_safe))):.2f}Nm "
                    f"torque_clipped={torque_limit_info['limited_count']}/12 "
                    f"max_age={max_age:.1f}ms send={self.enable_send}"
                )

            self.maybe_print_policy_debug(
                action_raw=action_raw,
                action=filtered_action,
                q_des=target_real,
                current_q=current_q,
                error=error,
                max_age=max_age,
                pre_limit_info=pre_limit_info,
                torque_limit_info=torque_limit_info,
                rear_bias_info={"enabled": False, "bias_vec_policy": np.zeros(12)},
            )

            if self.enable_send and self.command_is_fresh():
                self.send_motion_batch(target_real, info, motor_vel_cmd=motor_vel_cmd)
            elif self.enable_send:
                now = time.time()
                if now - self._last_cmd_timeout_log_time >= 1.0:
                    self._last_cmd_timeout_log_time = now
                    self.get_logger().warn(
                        "[PARITY][SAFE] /cmd_vel missing or stale; target not sent"
                    )

            # Write after the send attempt so communication timestamps and
            # command_sequence belong to this exact policy cycle.
            self.maybe_write_policy_csv(
                obs=obs,
                action_raw=action_raw,
                action_policy_obs=filtered_action,
                action=filtered_action,
                q_des=target_real,
                current_q=current_q,
                current_dq=current_dq,
                error=error,
                max_age=max_age,
                measured_torque=info.get("torque_real"),
                motor_temp=info.get("temp_real"),
                motor_online=info.get("online"),
                motor_error_code=info.get("error_code"),
                motor_age_ms=info.get("age_ms"),
                q_raw_des=target_real_raw,
                q_smooth_des=target_real_raw,
                motor_vel_cmd=motor_vel_cmd,
                smoothing_info=pre_limit_info,
                torque_limit_info=torque_limit_info,
                cpg_action_info=cpg_action_info,
                mode="pure_rl_pd_equivalent",
            )

            # Training observation uses the action that actually passed through
            # the policy-action filter, not the actor output and not q_safe.
            self.obs_builder.set_last_action(filtered_action)
            self._walking_active_prev = not zero_cmd_stand

        except Exception as exc:
            self.get_logger().error(f"parity policy loop error: {exc}")
        finally:
            self._advance_contract_phase()


def main(args=None):
    rclpy.init(args=args)
    node = MydogPolicyParityNode()
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
