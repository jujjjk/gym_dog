#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


class DeployJointCPG:
    """Deployment-side joint-space CPG matching FanfanRlCpg training."""

    LEG_ORDER = ("FR", "FL", "RR", "RL")

    PHASE_OFFSETS = {
        "trot": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5},
        "pace": {"FR": 0.0, "RR": 0.0, "FL": 0.5, "RL": 0.5},
        "bound": {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
        "walk": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75},
    }

    def __init__(
        self,
        default_joint_angle,
        lower_limit,
        upper_limit,
        policy_hz,
        gait="trot",
        freq_min=0.8,
        freq_max=1.8,
        k_freq=3.0,
        standing_cmd_threshold=0.03,
        duty_factor=0.60,
        hip_amp=0.025,
        thigh_amp=0.18,
        calf_lift_amp=0.60,
        stance_calf_amp=0.08,
        stride_sign=-1.0,
        enable_hip_balance=True,
        hip_stance_widen_amp=0.020,
        hip_swing_relax_amp=0.008,
        hip_balance_signs=(-1.0, 1.0, -1.0, 1.0),
        hip_balance_use_stance_mask=True,
        hip_balance_smooth_shape="sin",
        hip_balance_max_abs=0.06,
        residual_limit_hip=0.03,
        residual_limit_thigh=0.06,
        residual_limit_calf=0.06,
        enable_phase_aware_hip_gate=True,
        hip_gate_stance_min_outward=0.008,
        hip_gate_swing_max_outward=0.035,
        hip_gate_side_signs=(-1.0, 1.0, -1.0, 1.0),
    ):
        self.default_joint_angle = np.asarray(default_joint_angle, dtype=np.float32).reshape(12)
        self.lower_limit = np.asarray(lower_limit, dtype=np.float32).reshape(12)
        self.upper_limit = np.asarray(upper_limit, dtype=np.float32).reshape(12)
        self.policy_hz = max(float(policy_hz), 1.0e-3)
        self.gait = str(gait).strip().lower()
        if self.gait not in self.PHASE_OFFSETS:
            self.gait = "trot"

        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.k_freq = float(k_freq)
        self.standing_cmd_threshold = float(standing_cmd_threshold)
        self.duty_factor = float(duty_factor)
        self.hip_amp = float(hip_amp)
        self.thigh_amp = float(thigh_amp)
        self.calf_lift_amp = float(calf_lift_amp)
        self.stance_calf_amp = float(stance_calf_amp)
        self.stride_sign = float(stride_sign)
        self.enable_hip_balance = bool(enable_hip_balance)
        self.hip_stance_widen_amp = float(hip_stance_widen_amp)
        self.hip_swing_relax_amp = float(hip_swing_relax_amp)
        self.hip_balance_signs = np.asarray(hip_balance_signs, dtype=np.float32).reshape(4)
        self.hip_balance_use_stance_mask = bool(hip_balance_use_stance_mask)
        self.hip_balance_smooth_shape = str(hip_balance_smooth_shape).strip().lower()
        self.hip_balance_max_abs = float(hip_balance_max_abs)
        self.enable_phase_aware_hip_gate = bool(enable_phase_aware_hip_gate)
        self.hip_gate_stance_min_outward = float(hip_gate_stance_min_outward)
        self.hip_gate_swing_max_outward = float(hip_gate_swing_max_outward)
        self.hip_gate_side_signs = np.asarray(hip_gate_side_signs, dtype=np.float32).reshape(4)
        self.phase = 0.0
        self.last_frequency = 0.0
        self.last_leg_phase = np.zeros(4, dtype=np.float32)
        self.last_hip_stride_delta = np.zeros(4, dtype=np.float32)
        self.last_hip_balance_delta = np.zeros(4, dtype=np.float32)
        self.last_hip_gate_clamp_count = 0
        self.last_hip_gate_clamp_ratio = 0.0
        self.last_hip_outward_before_gate = np.zeros(4, dtype=np.float32)
        self.last_hip_outward_after_gate = np.zeros(4, dtype=np.float32)

        self.residual_limits = np.asarray(
            [
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
            ],
            dtype=np.float32,
        )

    def update(self, command, dt=None):
        command = np.asarray(command, dtype=np.float32).reshape(3)
        dt = (1.0 / self.policy_hz) if dt is None else max(float(dt), 1.0e-4)
        freq = self.compute_frequency(command)
        self.phase = (self.phase + freq * dt) % 1.0

        offsets = self.PHASE_OFFSETS.get(self.gait, self.PHASE_OFFSETS["trot"])
        q = self.default_joint_angle.copy()
        duty = min(max(float(self.duty_factor), 0.50), 0.90)
        swing_fraction = max(1.0 - duty, 0.05)
        moving = 1.0 if freq > 0.0 else 0.0

        leg_phases = []
        hip_stride_delta = np.zeros(4, dtype=np.float32)
        hip_balance_delta = np.zeros(4, dtype=np.float32)
        for leg_i, leg in enumerate(self.LEG_ORDER):
            p = (self.phase + float(offsets[leg])) % 1.0
            leg_phases.append(p)
            if p < swing_fraction:
                s = p / swing_fraction
                swing = math.sin(math.pi * s)
                stance = 0.0
                stride = -1.0 + 2.0 * s
            else:
                s = (p - swing_fraction) / max(1.0 - swing_fraction, 1.0e-5)
                swing = 0.0
                stance = math.sin(math.pi * s)
                stride = 1.0 - 2.0 * s

            base = leg_i * 3
            hip_stride = self.hip_amp * stride
            hip_balance = 0.0
            if self.enable_hip_balance:
                if self.hip_balance_smooth_shape == "sin":
                    stance_weight = stance
                    swing_weight = swing
                else:
                    stance_weight = 0.0 if p < swing_fraction else 1.0
                    swing_weight = 1.0 if p < swing_fraction else 0.0
                if not self.hip_balance_use_stance_mask:
                    stance_weight = 1.0
                hip_side_sign = float(self.hip_balance_signs[leg_i])
                hip_balance = (
                    hip_side_sign * self.hip_stance_widen_amp * stance_weight
                    - hip_side_sign * self.hip_swing_relax_amp * swing_weight
                )
                hip_balance = float(
                    np.clip(
                        hip_balance,
                        -self.hip_balance_max_abs,
                        self.hip_balance_max_abs,
                    )
                )
            hip_stride_delta[leg_i] = moving * hip_stride
            hip_balance_delta[leg_i] = moving * hip_balance
            q[base + 0] += moving * (hip_stride + hip_balance)
            q[base + 1] += moving * self.stride_sign * self.thigh_amp * stride
            q[base + 2] += moving * (
                -self.calf_lift_amp * swing
                + self.stance_calf_amp * stance
            )

        self.last_leg_phase[:] = np.asarray(leg_phases, dtype=np.float32)
        self.last_hip_stride_delta[:] = hip_stride_delta
        self.last_hip_balance_delta[:] = hip_balance_delta
        q = np.clip(q, self.lower_limit, self.upper_limit)
        return q.astype(np.float32)

    def compute_frequency(self, command):
        cmd_x = abs(float(command[0]))
        if cmd_x < self.standing_cmd_threshold:
            self.last_frequency = 0.0
            return 0.0
        freq = self.freq_min + self.k_freq * cmd_x
        freq = min(max(freq, self.freq_min), self.freq_max)
        self.last_frequency = float(freq)
        return float(freq)

    def apply_phase_aware_hip_gate(self, target_policy_abs, q_cpg_policy_abs):
        target = np.asarray(target_policy_abs, dtype=np.float32).reshape(12).copy()
        q_cpg = np.asarray(q_cpg_policy_abs, dtype=np.float32).reshape(12)
        hip_ids = np.asarray([0, 3, 6, 9], dtype=np.int64)

        outward_before = self.hip_gate_side_signs * (target[hip_ids] - self.default_joint_angle[hip_ids])
        outward_after = outward_before.copy()
        if self.enable_phase_aware_hip_gate and self.last_frequency > 1.0e-6:
            duty = min(max(float(self.duty_factor), 0.50), 0.90)
            swing_fraction = max(1.0 - duty, 0.05)
            phase01 = np.remainder(self.last_leg_phase, 1.0)
            swing = phase01 < swing_fraction
            stance = ~swing
            stance_low = stance & (outward_after < self.hip_gate_stance_min_outward)
            swing_high = swing & (outward_after > self.hip_gate_swing_max_outward)
            outward_after[stance_low] = self.hip_gate_stance_min_outward
            outward_after[swing_high] = self.hip_gate_swing_max_outward

        changed = np.abs(outward_after - outward_before) > 1.0e-7
        target[hip_ids] = self.default_joint_angle[hip_ids] + self.hip_gate_side_signs * outward_after
        target = np.clip(target, self.lower_limit, self.upper_limit).astype(np.float32)

        self.last_hip_gate_clamp_count = int(np.count_nonzero(changed))
        self.last_hip_gate_clamp_ratio = float(self.last_hip_gate_clamp_count) / 4.0
        self.last_hip_outward_before_gate[:] = outward_before.astype(np.float32)
        self.last_hip_outward_after_gate[:] = outward_after.astype(np.float32)
        return target, (target - q_cpg).astype(np.float32)

    def info(self):
        return {
            "frequency": float(self.last_frequency),
            "phase": float(self.phase),
            "leg_phase": self.last_leg_phase.astype(np.float32).copy(),
            "hip_stride_delta": self.last_hip_stride_delta.astype(np.float32).copy(),
            "hip_balance_delta": self.last_hip_balance_delta.astype(np.float32).copy(),
            "hip_gate_clamp_count": int(self.last_hip_gate_clamp_count),
            "hip_gate_clamp_ratio": float(self.last_hip_gate_clamp_ratio),
            "hip_outward_before_gate": self.last_hip_outward_before_gate.astype(np.float32).copy(),
            "hip_outward_after_gate": self.last_hip_outward_after_gate.astype(np.float32).copy(),
            "residual_limits": self.residual_limits.astype(np.float32).copy(),
        }
