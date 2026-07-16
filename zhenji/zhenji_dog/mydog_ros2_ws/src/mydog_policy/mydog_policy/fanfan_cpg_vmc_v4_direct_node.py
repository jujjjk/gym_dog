#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct FastDiagonalTrot + Light-VMC-v4 ROS2 bring-up node for Fanfan.

This node is a direct-migration style ROS2 bring-up node.  It intentionally
avoids the old local-linearized `_ik_delta_to_q()` approximation and uses the
same 2-link sagittal FK/IK convention used by the IsaacLab reference gait:

    x = -l1*sin(thigh) - l2*sin(thigh + calf)
    z = -l1*cos(thigh) - l2*cos(thigh + calf)

`z` is body-frame vertical with larger values meaning the foot is higher.  The
node is still a real-machine *bring-up* node: no RL, no torque control, no .pt.
It generates policy-order joint targets, maps them to the existing real-motor
order with JointSemanticMapper, sends `/api/rs04/motion_batch_fast`, and records
CSV for comparison against IsaacLab V4.

Install console script, for example:
    'fanfan_cpg_vmc_v4_direct_node = mydog_policy.fanfan_cpg_vmc_v4_direct_node:main'
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .imu_serial_interface import ImuSerialInterface
from .motor_state_interface import MotorSnapshot, MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


LEG_ORDER = ("FR", "FL", "RR", "RL")
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RR", "RL")
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
JOINT_SUFFIX = ("hip", "thigh", "calf")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_SUFFIX)
HIP_OUTWARD_SIGNS = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
SIDE_SIGN = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}       # right=-1, left=+1
FORE_AFT_SIGN = {"FR": 1.0, "FL": 1.0, "RR": -1.0, "RL": -1.0}  # front=+1, rear=-1
PAIR_A = ("FR", "RL")
PAIR_B = ("FL", "RR")
PAIR_OFFSETS = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
REAL_TEST_MODES = ("air", "touch", "assist", "short_free", "stand_only")

# Existing real-machine fallback.  Kept for compatibility with your mapper.
FALLBACK_DEFAULT_STAND_POLICY = np.array(
    [
        -0.1047, 0.4363, -0.8727,  # FR
        0.1047, 0.7000, 1.1500,    # FL
        0.1571, -0.5934, 1.7628,   # RR
        -0.1571, 0.5934, -1.7628,  # RL
    ],
    dtype=np.float32,
)

# A level simulation-style stand candidate.  Because different deployments may
# define policy signs differently, this is not forced onto the robot unless the
# user selects `stand_source:=sim_v4`.  CSV always records mapper/sim diff.
SIM_V4_LEVEL_STAND_POLICY = np.array(
    [
        -0.1047, 0.3491, -0.7854,  # FR
        0.1047, 0.3491, -0.7854,   # FL
        0.1047, 0.3491, -0.7854,   # RR
        -0.1047, 0.3491, -0.7854,  # RL
    ],
    dtype=np.float32,
)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def smootherstep(s: float) -> float:
    s = clamp(float(s), 0.0, 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def smoothstep(s: float) -> float:
    s = clamp(float(s), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def wrap_phase(x: float) -> float:
    return float(x - math.floor(x))


def wrap_to_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def mode_scale(mode: str, air: float, touch: float, assist: float, short_free: float) -> float:
    if mode == "stand_only":
        return 0.0
    return {
        "air": air,
        "touch": touch,
        "assist": assist,
        "short_free": short_free,
    }[mode]


@dataclass
class Feedback:
    q_policy: np.ndarray
    dq_policy: np.ndarray
    torque_policy: np.ndarray
    current_policy: np.ndarray
    temp_policy: np.ndarray
    online: np.ndarray
    max_age_ms: float
    valid: bool
    source: str


class FanfanDirectV4Core:
    """NumPy direct core matching the IsaacLab reference-gait math style."""

    def __init__(self, *, default_policy: np.ndarray, lower: np.ndarray, upper: np.ndarray, node: Node):
        self.node = node
        self.default_policy = default_policy.astype(np.float32).copy()
        self.lower = lower.astype(np.float32).copy()
        self.upper = upper.astype(np.float32).copy()
        self.thigh_length = float(node.thigh_length)
        self.calf_length = float(node.calf_length)
        self.workspace_margin_m = float(node.workspace_margin_m)
        self.default_foot_x = np.zeros(4, dtype=np.float32)
        self.default_foot_z = np.zeros(4, dtype=np.float32)
        for li, leg in enumerate(LEG_ORDER):
            i = LEG_START[leg]
            self.default_foot_x[li], self.default_foot_z[li] = self.forward_sagittal(
                self.default_policy[i + 1], self.default_policy[i + 2]
            )
        self.phase = 0.0
        self.walk_time = 0.0
        self.yaw_target_valid = False
        self.target_yaw = 0.0
        self.yaw_hip_offset = np.zeros(4, dtype=np.float32)
        self.rear_late_clearance = np.zeros(4, dtype=np.float32)

    def forward_sagittal(self, thigh: float, calf: float) -> Tuple[float, float]:
        x = -self.thigh_length * math.sin(float(thigh)) - self.calf_length * math.sin(float(thigh) + float(calf))
        z = -self.thigh_length * math.cos(float(thigh)) - self.calf_length * math.cos(float(thigh) + float(calf))
        return float(x), float(z)

    def inverse_sagittal(self, x: float, z: float) -> Tuple[float, float]:
        l1 = self.thigh_length
        l2 = self.calf_length
        reach = math.sqrt(max(float(x) * float(x) + float(z) * float(z), 1.0e-10))
        min_reach = abs(l1 - l2) + max(self.workspace_margin_m, 1.0e-5)
        max_reach = l1 + l2 - max(self.workspace_margin_m, 1.0e-5)
        scale = clamp(reach, min_reach, max_reach) / max(reach, 1.0e-9)
        x *= scale
        z *= scale
        cos_calf = clamp((x * x + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        calf = -math.acos(cos_calf)
        thigh = math.atan2(-x, -z) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
        return float(thigh), float(calf)

    def solve_calf_for_z(self, thigh: float, z: float, calf_default: float) -> float:
        value = clamp((-float(z) - self.thigh_length * math.cos(float(thigh))) / max(self.calf_length, 1e-9), -1.0, 1.0)
        angle = math.acos(value)
        cand_a = angle - float(thigh)
        cand_b = -angle - float(thigh)
        return float(cand_a if abs(cand_a - calf_default) < abs(cand_b - calf_default) else cand_b)

    def _gait_scales(self) -> Tuple[float, float, float, float]:
        stride = float(self.node.stride_length) * self.node.real_stride_scale()
        front_h = float(self.node.front_swing_height) * self.node.real_swing_height_scale()
        rear_h = float(self.node.rear_swing_height) * self.node.real_swing_height_scale()
        vmc = self.node.real_vmc_scale()
        return stride, front_h, rear_h, vmc

    def step(self, *, dt: float, elapsed: float, imu: Dict[str, np.ndarray], feedback: Feedback, warm: float) -> Tuple[np.ndarray, Dict]:
        if not self.yaw_target_valid:
            self.target_yaw = float(imu["rpy"][2])
            self.yaw_target_valid = True
        stride, front_h, rear_h, vmc_scale = self._gait_scales()
        swing_fraction = max(0.05, 1.0 - float(self.node.duty_factor))
        self.phase = wrap_phase(self.phase + float(self.node.step_hz) * dt)

        leg_phase = {leg: wrap_phase(self.phase - PAIR_OFFSETS[leg]) for leg in LEG_ORDER}
        swing_mask = {leg: float(leg_phase[leg] < swing_fraction) for leg in LEG_ORDER}
        support_mask = {leg: 1.0 - swing_mask[leg] for leg in LEG_ORDER}
        swing_progress = {
            leg: (clamp(leg_phase[leg] / swing_fraction, 0.0, 1.0) if swing_mask[leg] > 0.5 else 1.0)
            for leg in LEG_ORDER
        }
        stance_progress = {
            leg: clamp((leg_phase[leg] - swing_fraction) / max(1.0 - swing_fraction, 1e-6), 0.0, 1.0)
            for leg in LEG_ORDER
        }
        active_pair = "FR+RL" if swing_mask["FR"] > 0.5 else "FL+RR"
        support_pair = "FL+RR" if active_pair == "FR+RL" else "FR+RL"

        rpy = imu["rpy"]
        gyro = imu["gyro"]
        yaw_error = wrap_to_pi(float(rpy[2]) - float(self.target_yaw))
        height_corr = 0.0  # true base height unavailable in ROS bring-up node
        roll_corr = clamp(
            float(self.node.roll_sign) * (float(self.node.roll_kp_z) * float(rpy[0]) + float(self.node.roll_kd_z) * float(gyro[0])),
            -float(self.node.roll_corr_limit_m),
            float(self.node.roll_corr_limit_m),
        ) * vmc_scale
        pitch_error = float(rpy[1]) - float(self.node.target_pitch)
        pitch_corr = clamp(
            float(self.node.pitch_sign) * (float(self.node.pitch_kp_z) * pitch_error + float(self.node.pitch_kd_z) * float(gyro[1])),
            -float(self.node.pitch_corr_limit_m),
            float(self.node.pitch_corr_limit_m),
        ) * vmc_scale
        yaw_corr = 0.0
        if bool(self.node.enable_light_yaw_damping):
            yaw_corr = clamp(
                float(self.node.yaw_sign) * (float(self.node.yaw_kp_hip) * yaw_error + float(self.node.yaw_kd_hip) * float(gyro[2])),
                -float(self.node.yaw_hip_limit_rad),
                float(self.node.yaw_hip_limit_rad),
            ) * vmc_scale
        desired_yaw_offsets = np.array([HIP_OUTWARD_SIGNS[leg] * yaw_corr for leg in LEG_ORDER], dtype=np.float32)
        max_yaw_step = float(self.node.yaw_hip_rate_limit_rad)
        self.yaw_hip_offset += np.clip(desired_yaw_offsets - self.yaw_hip_offset, -max_yaw_step, max_yaw_step)

        q = self.default_policy.copy()
        kp = np.zeros(12, dtype=np.float32)
        kd = np.zeros(12, dtype=np.float32)
        debug = {
            "leg_phase": leg_phase,
            "swing_progress": swing_progress,
            "swing_mask": swing_mask,
            "support_mask": support_mask,
            "active_pair": active_pair,
            "support_pair": support_pair,
            "target_yaw": self.target_yaw,
            "yaw_error": yaw_error,
            "vmc_height": height_corr,
            "vmc_roll": roll_corr,
            "vmc_pitch": pitch_corr,
            "yaw_corr": yaw_corr,
            "yaw_offsets": self.yaw_hip_offset.copy(),
            "vmc_weight": np.zeros(4, dtype=np.float32),
            "rear_late_window": {"RR": 0.0, "RL": 0.0},
            "rear_late_guard": {"RR": 0.0, "RL": 0.0},
            "rear_late_reason": {"RR": "", "RL": ""},
            "rear_late_clearance": {"RR": 0.0, "RL": 0.0},
            "rear_late_descending": {"RR": 0.0, "RL": 0.0},
            "rear_descent_scale": {"RR": 1.0, "RL": 1.0},
            "rear_early_guard": {"RR": 0.0, "RL": 0.0},
            "rear_early_score": {"RR": 0.0, "RL": 0.0},
            "rear_touchdown_weight": {"RR": 0.0, "RL": 0.0},
            "early_source": "q_error_only",
            "foot_x_ref": np.zeros(4, dtype=np.float32),
            "foot_z_ref": np.zeros(4, dtype=np.float32),
            "fk_clearance_ref": np.zeros(4, dtype=np.float32),
            "kp": kp,
            "kd": kd,
        }

        for li, leg in enumerate(LEG_ORDER):
            idx = LEG_START[leg]
            is_swing = swing_mask[leg] > 0.5
            sp = swing_progress[leg]
            stp = stance_progress[leg]
            is_front = leg in FRONT_LEGS
            height = front_h if is_front else rear_h
            stride_gain = float(self.node.front_stride_gain) if is_front else float(self.node.rear_stride_gain)
            height_gain = float(self.node.front_swing_height_gain) if is_front else float(self.node.rear_swing_height_gain)
            thigh_scale = float(self.node.front_thigh_delta_scale) if is_front else float(self.node.rear_thigh_delta_scale)
            calf_extra = float(self.node.front_calf_lift_extra) if is_front else float(self.node.rear_calf_lift_extra)
            default_x = float(self.default_foot_x[li])
            default_z = float(self.default_foot_z[li])
            leg_stride = stride * stride_gain
            leg_height = height * height_gain
            if is_swing:
                adv = smootherstep((sp - float(self.node.advance_start)) / max(float(self.node.advance_end) - float(self.node.advance_start), 1e-6))
                lift_up = smootherstep(sp / max(float(self.node.advance_start), 1e-6))
                lift_down = smootherstep((1.0 - sp) / max(1.0 - float(self.node.advance_end), 1e-6))
                swing_shape = lift_up * lift_down
                x_des = default_x - 0.5 * leg_stride + leg_stride * adv
                if is_front:
                    x_des += float(self.node.front_x_bias) + float(self.node.front_swing_forward_unfold) * swing_shape
                z_des = default_z + leg_height * swing_shape
            else:
                swing_shape = 0.0
                stance_shape = math.sin(math.pi * stp) ** 2
                x_des = default_x + 0.5 * leg_stride - leg_stride * smootherstep(stp)
                z_des = default_z - float(self.node.support_stand_tall_m) * (0.35 + 0.65 * stance_shape)

            # Support VMC: only stance/touchdown legs, ramped in like IsaacLab's stance blend.
            vmc_w = 0.0
            if not is_swing:
                vmc_w = smootherstep(min(1.0, stp / max(float(self.node.early_stance_blend), 1e-6)))
                dz = height_corr
                dz += -SIDE_SIGN[leg] * roll_corr
                dz += -FORE_AFT_SIGN[leg] * pitch_corr
                z_des += vmc_w * dz
            debug["vmc_weight"][li] = float(vmc_w)

            # V4 rear late-swing guard: does not replace the swing peak, only keeps
            # a small clearance in swing progress 0.70~0.95.
            if leg in REAR_LEGS and is_swing:
                late = float(self.node.rear_late_swing_progress_start) <= sp <= float(self.node.rear_late_swing_progress_end)
                descending = late and sp >= float(self.node.swing_lift_peak_phase)
                debug["rear_late_window"][leg] = float(late)
                debug["rear_late_descending"][leg] = float(descending)
                if descending and bool(self.node.rear_late_swing_descent_soft_enable):
                    # Hold part of the lift; in z-up convention this slows descent.
                    z_des = default_z + (z_des - default_z) * (1.0 + (1.0 - float(self.node.rear_late_swing_descent_scale)))
                    debug["rear_descent_scale"][leg] = float(self.node.rear_late_swing_descent_scale)
                if late and bool(self.node.rear_late_swing_guard_enable):
                    min_clearance = float(self.node.rear_late_swing_min_height_m) + float(self.node.rear_late_swing_clearance_margin_m)
                    current_clearance = z_des - default_z
                    h_err = max(0.0, min_clearance - current_clearance)
                    desired = float(self.node.rear_late_swing_clearance_sign) * h_err
                    step = clamp(desired - float(self.rear_late_clearance[li]), -float(self.node.rear_late_swing_guard_rate_limit_m), float(self.node.rear_late_swing_guard_rate_limit_m))
                    self.rear_late_clearance[li] += step
                    z_des += float(self.rear_late_clearance[li])
                    if abs(float(self.rear_late_clearance[li])) > 1e-7:
                        debug["rear_late_guard"][leg] = 1.0
                        debug["rear_late_reason"][leg] = "clearance_low"
                    else:
                        debug["rear_late_reason"][leg] = "height_ok"
                else:
                    self.rear_late_clearance[li] *= 0.5
                    debug["rear_late_reason"][leg] = "outside_window"

            # Rear early-contact substitute from feedback q_error/torque/current.
            q_fb = feedback.q_policy if feedback.valid else self.node.q_cmd_final
            qerr_leg = float(np.max(np.abs(self.node.q_cmd_final[idx:idx+3] - q_fb[idx:idx+3])))
            tau_leg = float(np.nanmax(np.abs(feedback.torque_policy[idx:idx+3]))) if np.any(np.isfinite(feedback.torque_policy[idx:idx+3])) else float("nan")
            current_leg = float(np.nanmax(np.abs(feedback.current_policy[idx:idx+3]))) if np.any(np.isfinite(feedback.current_policy[idx:idx+3])) else float("nan")
            early_score = qerr_leg / max(float(self.node.rear_early_contact_qerr_threshold), 1e-6)
            early_source = "q_error_only"
            if math.isfinite(tau_leg):
                early_score = max(early_score, tau_leg / max(float(self.node.rear_early_contact_tau_threshold_nm), 1e-6))
                early_source = "q_error_or_torque"
            if math.isfinite(current_leg):
                early_score = max(early_score, current_leg / max(float(self.node.rear_early_contact_current_threshold_a), 1e-6))
                early_source = "q_error_or_current"
            if leg in REAR_LEGS:
                debug["rear_early_score"][leg] = float(early_score)
                debug["early_source"] = early_source
                if is_swing and bool(self.node.rear_early_contact_guard_enable) and 0.70 <= sp <= 1.0 and early_score > 1.0:
                    debug["rear_early_guard"][leg] = 1.0
                    z_des += float(self.node.rear_early_contact_relief_sign) * float(self.node.rear_early_contact_lift_relief_m)
                if (not is_swing) and stp < float(self.node.rear_touchdown_kp_ramp):
                    debug["rear_touchdown_weight"][leg] = smootherstep(stp / max(float(self.node.rear_touchdown_kp_ramp), 1e-6))

            # Full 2-link IK, followed by IsaacLab-style thigh partial blend and
            # calf re-solve so z target remains meaningful after thigh scaling.
            thigh_ik, calf_ik = self.inverse_sagittal(x_des, z_des)
            default_thigh = float(self.default_policy[idx + 1])
            default_calf = float(self.default_policy[idx + 2])
            thigh_target = default_thigh + thigh_scale * (thigh_ik - default_thigh)
            if is_swing:
                calf_push = -calf_extra * swing_shape
            else:
                calf_push = 0.006 * (math.sin(math.pi * stp) ** 2)
            calf_target = self.solve_calf_for_z(thigh_target, z_des, default_calf) + calf_push
            if is_front:
                calf_target = max(float(self.node.front_calf_min_rad), calf_target)
            hip_delta = -0.004 * swing_shape * HIP_OUTWARD_SIGNS[leg]
            hip = float(self.default_policy[idx]) + warm * hip_delta + self.yaw_hip_offset[li] * (1.0 if not is_swing else 0.0)
            q[idx + 0] = hip
            q[idx + 1] = default_thigh + warm * (thigh_target - default_thigh)
            q[idx + 2] = default_calf + warm * (calf_target - default_calf)
            x_ref, z_ref = self.forward_sagittal(float(q[idx+1]), float(q[idx+2]))
            debug["foot_x_ref"][li] = x_ref
            debug["foot_z_ref"][li] = z_ref
            debug["fk_clearance_ref"][li] = z_ref - float(self.default_foot_z[li])

            # Gains.
            if is_swing:
                leg_kp = [float(self.node.hip_kp), float(self.node.thigh_kp), float(self.node.calf_kp)]
                leg_kd = float(self.node.kd)
            else:
                leg_kp = [float(self.node.support_hip_kp), float(self.node.support_thigh_kp), float(self.node.support_calf_kp)]
                leg_kd = float(self.node.support_kd)
            if leg in REAR_LEGS:
                td = float(debug["rear_touchdown_weight"][leg])
                if td > 0.0:
                    touch_kp = np.array([float(self.node.touchdown_hip_kp), float(self.node.touchdown_thigh_kp), float(self.node.touchdown_calf_kp)], dtype=np.float32)
                    support_kp = np.array(leg_kp, dtype=np.float32)
                    leg_kp = (touch_kp * (1.0 - td) + support_kp * td).tolist()
                    leg_kd = float(self.node.touchdown_kd) * (1.0 - td) + leg_kd * td
                if debug["rear_early_guard"].get(leg, 0.0) > 0.5:
                    leg_kp = [
                        min(leg_kp[0], float(self.node.rear_early_contact_hip_kp_limit)),
                        min(leg_kp[1], float(self.node.rear_early_contact_thigh_kp_limit)),
                        min(leg_kp[2], float(self.node.rear_early_contact_calf_kp_limit)),
                    ]
                    leg_kd = float(self.node.rear_early_contact_kd)
            kp[idx:idx+3] = np.asarray(leg_kp, dtype=np.float32)
            kd[idx:idx+3] = leg_kd

        q = np.clip(q, self.lower, self.upper)
        debug["kp"] = kp
        debug["kd"] = kd
        return q.astype(np.float32), debug


class FanfanCpgVmcV4DirectNode(Node):
    def __init__(self):
        super().__init__("fanfan_cpg_vmc_v4_direct_node")
        self._declare_parameters()
        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.policy_joint_names = list(self.mapper.get_policy_joint_names())
        self.mapper_default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.sim_v4_default_policy = SIM_V4_LEVEL_STAND_POLICY.astype(np.float32).copy()
        self.default_policy = self._select_default_policy()
        self.default_real = self.mapper.policy_target_to_real_target(self.default_policy, clamp=True)
        self.q_last_cmd = self.default_policy.copy()
        self.q_cmd_final = self.default_policy.copy()
        self.node_state = "WARMUP"
        self.stop_reason = "none"
        self.safety_stop_reason = "none"
        self.safety_stop = False
        self._soft_stop_active = False
        self._soft_stop_start = 0.0
        self._soft_stop_from = self.default_policy.copy()
        self._last_update = time.time()
        self.start_time = self._last_update
        self._target_yaw_initialized = False
        self._last_feedback_warn = 0.0
        self._last_send_info = 0.0
        self.stats: Dict[str, List[float]] = {k: [] for k in ["qerr", "tau", "roll", "pitch", "yawerr", "rate_clip", "torque_clip", "front_clear", "rear_clear"]}

        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=float(self.http_timeout),
            stale_recheck_ms=float(self.max_motor_age_ms),
            enable_stale_recheck=True,
        )
        self.imu: Optional[ImuSerialInterface] = None
        self.imu_valid = False
        self._start_imu()
        self.core = FanfanDirectV4Core(
            default_policy=self.default_policy,
            lower=self.mapper.policy_lower_limit,
            upper=self.mapper.policy_upper_limit,
            node=self,
        )

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_cpg_vmc_v4_direct_target_real", 10)
        self.pub_debug = self.create_publisher(Float32MultiArray, "/mydog/fanfan_cpg_vmc_v4_direct_debug", 10)
        self._setup_csv()
        self._validate_send_preconditions()
        self._print_startup_summary()
        if self.enable_send:
            self._countdown()
            self._send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry-run only, no motor commands will be sent.")
        self.timer = self.create_timer(1.0 / max(float(self.gait_hz), 1.0), self.update)

    def _declare_parameters(self):
        params = {
            "motor_base_url": "http://127.0.0.1:8000",
            "enable_send": False,
            "test_mode": "air",
            "duration_s": 5.0,
            "warmup_s": 2.0,
            "soft_start_s": 2.0,
            "soft_stop_s": 1.5,
            "auto_stop_after_duration": True,
            "allow_long_free_test": False,
            "gait_hz": 60.0,
            "http_timeout": 0.08,
            "max_motor_age_ms": 150.0,
            "csv_path": "",
            "stand_source": "sim_v4",  # sim_v4 | mapper | fallback
            "dry_run_virtual_feedback": True,
            "require_stand_ready": True,
            "stand_ready_q_error_threshold": 0.45,
            "allow_start_from_any_pose": False,
            "workspace_margin_m": 0.005,
            "step_hz": 1.15,
            "duty_factor": 0.61,
            "stride_length": 0.022,
            "front_swing_height": 0.048,
            "rear_swing_height": 0.067,
            "swing_lift_peak_phase": 0.45,
            "touchdown_phase": 0.82,
            "early_stance_blend": 0.12,
            "thigh_length": 0.1560608,
            "calf_length": 0.1489418,
            "front_stride_gain": 1.05,
            "rear_stride_gain": 0.66,
            "front_swing_height_gain": 1.0,
            "rear_swing_height_gain": 1.0,
            "front_thigh_delta_scale": 1.0,
            "rear_thigh_delta_scale": 1.0,
            "front_calf_lift_extra": 0.0,
            "rear_calf_lift_extra": 0.0,
            "front_calf_min_rad": -1.30,
            "advance_start": 0.12,
            "advance_end": 0.92,
            "front_x_bias": 0.0,
            "front_swing_forward_unfold": 0.0,
            "support_stand_tall_m": 0.002,
            "real_stride_scale_air": 0.70,
            "real_stride_scale_touch": 0.55,
            "real_stride_scale_assist": 0.65,
            "real_stride_scale_short_free": 0.70,
            "real_swing_height_scale_air": 0.85,
            "real_swing_height_scale_touch": 0.75,
            "real_swing_height_scale_assist": 0.80,
            "real_swing_height_scale_short_free": 0.85,
            "real_vmc_scale_air": 0.0,
            "real_vmc_scale_touch": 0.35,
            "real_vmc_scale_assist": 0.50,
            "real_vmc_scale_short_free": 0.60,
            "target_pitch": -0.04,
            "height_sign": -1.0,
            "height_kp_z": 0.30,
            "height_kd_z": 0.04,
            "height_corr_limit_m": 0.004,
            "roll_sign": 1.0,
            "roll_kp_z": 0.025,
            "roll_kd_z": 0.006,
            "roll_corr_limit_m": 0.0035,
            "pitch_sign": -1.0,
            "pitch_kp_z": 0.025,
            "pitch_kd_z": 0.005,
            "pitch_corr_limit_m": 0.003,
            "enable_light_yaw_damping": True,
            "yaw_sign": 1.0,
            "yaw_kp_hip": 0.0025,
            "yaw_kd_hip": 0.006,
            "yaw_hip_limit_rad": 0.007,
            "yaw_hip_rate_limit_rad": 0.001,
            "rear_late_swing_guard_enable": True,
            "rear_late_swing_progress_start": 0.70,
            "rear_late_swing_progress_end": 0.95,
            "rear_late_swing_clearance_margin_m": 0.003,
            "rear_late_swing_min_height_m": 0.003,
            "rear_late_swing_guard_rate_limit_m": 0.001,
            "rear_late_swing_clearance_sign": 1.0,
            "rear_late_swing_descent_soft_enable": True,
            "rear_late_swing_descent_scale": 0.50,
            "rear_early_contact_guard_enable": True,
            "rear_early_contact_qerr_threshold": 0.18,
            "rear_early_contact_tau_threshold_nm": 8.0,
            "rear_early_contact_current_threshold_a": 999.0,
            "rear_early_contact_lift_relief_m": 0.002,
            "rear_early_contact_relief_sign": 1.0,
            "rear_touchdown_kp_ramp": 0.24,
            "rear_touchdown_kp_scale": 0.75,
            "hip_kp": 40.0,
            "thigh_kp": 70.0,
            "calf_kp": 70.0,
            "kd": 5.0,
            "support_hip_kp": 45.0,
            "support_thigh_kp": 80.0,
            "support_calf_kp": 85.0,
            "support_kd": 5.5,
            "touchdown_hip_kp": 35.0,
            "touchdown_thigh_kp": 60.0,
            "touchdown_calf_kp": 60.0,
            "touchdown_kd": 6.0,
            "rear_early_contact_hip_kp_limit": 35.0,
            "rear_early_contact_thigh_kp_limit": 55.0,
            "rear_early_contact_calf_kp_limit": 55.0,
            "rear_early_contact_kd": 6.5,
            "max_target_rate_rad_s_hip": 1.2,
            "max_target_rate_rad_s_thigh": 1.5,
            "max_target_rate_rad_s_calf": 1.5,
            "max_q_error_warn": 0.25,
            "max_q_error_stop": 0.45,
            "torque_soft_warn_nm": 8.0,
            "torque_soft_limit_nm": 10.0,
            "torque_stop_nm": 17.0,
            "current_warn_a": 999.0,
            "current_stop_a": 999.0,
            "max_roll_deg_warn": 8.0,
            "max_roll_deg_stop": 12.0,
            "max_pitch_deg_warn": 8.0,
            "max_pitch_deg_stop": 12.0,
            "max_motor_temp_warn_c": 65.0,
            "max_motor_temp_stop_c": 75.0,
            "send_speed": 0.0,
            "send_torque": 0.0,
            "imu_port": "/dev/myimu",
            "imu_read_hz": 100.0,
            "require_imu_for_air_send": False,
        }
        for k, v in params.items():
            self.declare_parameter(k, v)
        for k in params:
            setattr(self, k, self.get_parameter(k).value)
        self.motor_base_url = str(self.motor_base_url).rstrip("/")
        self.test_mode = str(self.test_mode)
        if self.test_mode not in REAL_TEST_MODES:
            raise RuntimeError(f"Invalid test_mode={self.test_mode!r}; choose from {REAL_TEST_MODES}")
        self.enable_send = bool(self.enable_send)
        self.dry_run_virtual_feedback = bool(self.dry_run_virtual_feedback)
        self.auto_stop_after_duration = bool(self.auto_stop_after_duration)
        self.require_stand_ready = bool(self.require_stand_ready)
        self.allow_start_from_any_pose = bool(self.allow_start_from_any_pose)
        if self.test_mode == "short_free" and float(self.duration_s) > 3.0 and not bool(self.allow_long_free_test):
            self.get_logger().warn("short_free duration_s > 3.0; limiting to 3.0 unless allow_long_free_test=true.")
            self.duration_s = 3.0

    def _select_default_policy(self) -> np.ndarray:
        source = str(self.stand_source).strip().lower()
        if source == "sim_v4":
            return self.sim_v4_default_policy.copy()
        if source == "mapper":
            return self.mapper_default_policy.copy()
        if source == "fallback":
            return FALLBACK_DEFAULT_STAND_POLICY.copy()
        raise RuntimeError("stand_source must be sim_v4, mapper, or fallback")

    def real_stride_scale(self) -> float:
        return mode_scale(self.test_mode, float(self.real_stride_scale_air), float(self.real_stride_scale_touch), float(self.real_stride_scale_assist), float(self.real_stride_scale_short_free))

    def real_swing_height_scale(self) -> float:
        return mode_scale(self.test_mode, float(self.real_swing_height_scale_air), float(self.real_swing_height_scale_touch), float(self.real_swing_height_scale_assist), float(self.real_swing_height_scale_short_free))

    def real_vmc_scale(self) -> float:
        return mode_scale(self.test_mode, float(self.real_vmc_scale_air), float(self.real_vmc_scale_touch), float(self.real_vmc_scale_assist), float(self.real_vmc_scale_short_free))

    def _start_imu(self):
        try:
            self.imu = ImuSerialInterface(port=str(self.imu_port), read_hz=float(self.imu_read_hz))
            self.imu.start()
            self.imu_valid = self.imu.wait_until_ready(timeout=2.0)
            if not self.imu_valid:
                self.get_logger().warn("WARNING: IMU not ready.")
        except Exception as exc:
            self.imu = None
            self.imu_valid = False
            self.get_logger().warn(f"WARNING: IMU unavailable: {exc}")

    def _zero_imu(self, valid=False) -> Dict[str, np.ndarray]:
        return {"valid": valid, "rpy": np.zeros(3, dtype=np.float32), "gyro": np.zeros(3, dtype=np.float32), "acc": np.zeros(3, dtype=np.float32)}

    def _read_imu(self) -> Dict[str, np.ndarray]:
        if self.imu is None:
            return self._zero_imu(False)
        snap = self.imu.get_latest()
        if not snap.valid:
            return self._zero_imu(False)
        return {"valid": True, "rpy": np.asarray(snap.rpy_deg, dtype=np.float32) * (math.pi / 180.0), "gyro": np.asarray(snap.gyro_rad_s, dtype=np.float32), "acc": np.asarray(snap.acc_g, dtype=np.float32)}

    def _read_feedback(self) -> Feedback:
        try:
            snap: MotorSnapshot = self.motor.get_latest()
            q_policy, dq_policy = self.mapper.real_to_policy_abs_q_dq(snap.q_real, snap.dq_real)
            torque_real = np.asarray(snap.torque, dtype=np.float32).reshape(12)
            temp_real = np.asarray(snap.temp, dtype=np.float32).reshape(12)
            torque_policy = torque_real[self.mapper.policy_to_real_index] * self.mapper.joint_sign
            temp_policy = temp_real[self.mapper.policy_to_real_index]
            current_policy = np.full(12, np.nan, dtype=np.float32)
            valid = bool(snap.valid and np.all(np.isfinite(q_policy)))
            return Feedback(q_policy.astype(np.float32), dq_policy.astype(np.float32), torque_policy.astype(np.float32), current_policy, temp_policy.astype(np.float32), np.asarray(snap.online, dtype=bool).reshape(12), float(np.max(snap.age_ms)), valid, "http")
        except Exception as exc:
            now = time.time()
            if now - self._last_feedback_warn > 1.0:
                self._last_feedback_warn = now
                self.get_logger().warn(f"WARNING: motor feedback unavailable: {exc}")
            return Feedback(self.q_cmd_final.copy(), np.zeros(12, dtype=np.float32), np.full(12, np.nan, dtype=np.float32), np.full(12, np.nan, dtype=np.float32), np.full(12, np.nan, dtype=np.float32), np.zeros(12, dtype=bool), float("inf"), False, "none")

    def _validate_send_preconditions(self):
        if not self.enable_send:
            return
        if self.test_mode in ("touch", "assist", "short_free") and not self.imu_valid:
            raise RuntimeError("IMU is required before sending in touch/assist/short_free mode.")
        if self.test_mode == "air" and bool(self.require_imu_for_air_send) and not self.imu_valid:
            raise RuntimeError("IMU is required for air send because require_imu_for_air_send=true.")
        feedback = self._read_feedback()
        if self.test_mode in ("assist", "short_free") and not feedback.valid:
            raise RuntimeError("Motor feedback is required before sending in assist/short_free mode.")
        if bool(self.require_stand_ready) and not bool(self.allow_start_from_any_pose):
            if not feedback.valid:
                raise RuntimeError("Motor feedback is required for stand-ready check before enable_send=true.")
            err = float(np.max(np.abs(feedback.q_policy - self.default_policy)))
            if err > float(self.stand_ready_q_error_threshold):
                raise RuntimeError(f"Stand-ready check failed: max(|q_actual-START|)={err:.3f} rad > {float(self.stand_ready_q_error_threshold):.3f}. Use stand_source:=mapper or allow_start_from_any_pose:=true only for cautious air test.")
        if bool(self.allow_start_from_any_pose):
            self.warmup_s = max(float(self.warmup_s), 3.0)
            self.max_target_rate_rad_s_hip = min(float(self.max_target_rate_rad_s_hip), 0.6)
            self.max_target_rate_rad_s_thigh = min(float(self.max_target_rate_rad_s_thigh), 0.8)
            self.max_target_rate_rad_s_calf = min(float(self.max_target_rate_rad_s_calf), 0.8)
            self.real_vmc_scale_air = 0.0
            self.real_vmc_scale_touch = 0.0
            self.real_vmc_scale_assist = 0.0
            self.real_vmc_scale_short_free = 0.0
            self.get_logger().warn("allow_start_from_any_pose=true: warmup>=3, reduced rate, VMC disabled.")

    def _print_startup_summary(self):
        diff = float(np.max(np.abs(self.mapper_default_policy - self.sim_v4_default_policy)))
        self.get_logger().warn(f"DIRECT V4 node | send={self.enable_send} mode={self.test_mode} stand_source={self.stand_source} default_diff(mapper,sim_v4)={diff:.3f} rad")
        self.get_logger().warn(f"step_hz={float(self.step_hz):.2f} duty={float(self.duty_factor):.2f} stride={float(self.stride_length):.3f} front_h={float(self.front_swing_height):.3f} rear_h={float(self.rear_swing_height):.3f} vmc_scale={self.real_vmc_scale():.2f}")
        self.get_logger().warn("URDF hip signs FR=-1 FL=+1 RR=-1 RL=+1; using full 2-link FK/IK, not local linearization.")

    def _countdown(self):
        for i in (3, 2, 1):
            self.get_logger().warn(f"ENABLE_SEND TRUE: starting motor commands in {i}...")
            time.sleep(1.0)

    def _setup_csv(self):
        path = str(self.csv_path).strip()
        if not path:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/mydog_ros2_ws/src/mydog_policy/mydog_policy/docs/cpg_vmc_v4_direct_{stamp}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv_path = path
        self.csv_file = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._csv_header())
        self.csv_file.flush()
        self.get_logger().warn(f"[CSV] writing to {path}")

    def _csv_header(self) -> List[str]:
        h = ["time", "dt", "test_mode", "enable_send", "node_state", "stop_reason", "safety_stop_reason", "stand_source", "dry_run_virtual_feedback", "phase", "active_swing_pair", "support_pair"]
        h += [f"leg_phase_{leg}" for leg in LEG_ORDER]
        h += [f"swing_progress_{leg}" for leg in LEG_ORDER]
        h += [f"swing_mask_{leg}" for leg in LEG_ORDER]
        h += [f"support_mask_{leg}" for leg in LEG_ORDER]
        h += ["base_roll", "base_pitch", "base_yaw", "target_yaw", "yaw_error", "yaw_error_abs", "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
        h += ["real_vmc_scale", "vmc_height_corr_z", "vmc_roll_corr_z", "vmc_pitch_corr_z", "yaw_corr_hip"]
        h += [f"vmc_weight_{leg}" for leg in LEG_ORDER]
        h += [f"yaw_hip_offset_{leg}" for leg in LEG_ORDER]
        h += ["rear_late_swing_window_active_RR", "rear_late_swing_window_active_RL", "rear_late_swing_guard_active_RR", "rear_late_swing_guard_active_RL", "rear_late_swing_clearance_offset_RR", "rear_late_swing_clearance_offset_RL", "rear_late_swing_descent_scale_RR", "rear_late_swing_descent_scale_RL", "rear_early_contact_guard_active_RR", "rear_early_contact_guard_active_RL", "rear_early_contact_score_RR", "rear_early_contact_score_RL", "early_contact_source"]
        for prefix in ("q_ref", "q_cmd_final", "q_actual", "q_actual_for_safety", "q_error", "q_error_for_safety", "tau_est", "current", "temp", "rate_limited_delta", "q_ref_cmd_diff", "mapper_default", "sim_v4_default", "default_policy_diff"):
            h += [f"{prefix}_{name}" for name in JOINT_NAMES]
        h += [f"raw_motor_target_0x{mid:02X}" for mid in (0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x31, 0x32, 0x33, 0x41, 0x42, 0x43)]
        h += [f"fk_clearance_ref_{leg}" for leg in LEG_ORDER]
        h += [f"fk_clearance_cmd_{leg}" for leg in LEG_ORDER]
        h += [f"fk_clearance_actual_{leg}" for leg in LEG_ORDER]
        h += ["max_q_error", "max_tau_est", "max_current", "max_temp", "safety_warn", "safety_stop", "rate_clip_ratio", "torque_clip_ratio", "communication_ok", "hip_outward_sign_FR", "hip_outward_sign_FL", "hip_outward_sign_RR", "hip_outward_sign_RL", "use_urdf_hip_outward_signs"]
        return h

    def update(self):
        now = time.time()
        dt = clamp(now - self._last_update, 1.0 / 500.0, 0.10)
        self._last_update = now
        elapsed = now - self.start_time
        feedback = self._read_feedback()
        imu = self._read_imu()
        if self.auto_stop_after_duration and elapsed >= float(self.duration_s) and not self._soft_stop_active:
            self._request_soft_stop("duration_reached")
        if self.safety_stop and not self._soft_stop_active:
            self._request_soft_stop(self.safety_stop_reason)
        if self._soft_stop_active:
            self.node_state = "SAFETY_STOP" if self.safety_stop else "SOFT_STOP"
            q_raw, debug = self._soft_stop_target(now)
        elif self.test_mode == "stand_only":
            self.node_state = "STAND_ONLY"
            warm = smootherstep(clamp(elapsed / max(float(self.warmup_s), 1e-6), 0.0, 1.0))
            q_raw = (1.0 - warm) * self.q_last_cmd + warm * self.default_policy
            debug = self._empty_debug()
            debug["kp"] = np.array([float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)] * 4, dtype=np.float32)
            debug["kd"] = np.full(12, float(self.support_kd), dtype=np.float32)
        else:
            self.node_state = "WARMUP" if elapsed < float(self.warmup_s) else "GAIT"
            warm = smootherstep(clamp(elapsed / max(float(self.soft_start_s), 1e-6), 0.0, 1.0)) * smootherstep(clamp(elapsed / max(float(self.warmup_s), 1e-6), 0.0, 1.0))
            q_raw, debug = self.core.step(dt=dt, elapsed=elapsed, imu=imu, feedback=feedback, warm=warm)
        q_cmd, safety = self._apply_safety(q_raw, feedback, dt, debug)
        target_real = self.mapper.policy_target_to_real_target(q_cmd, clamp=True)
        self._publish(target_real, debug, safety)
        sent = False
        if self.enable_send:
            sent = self._send_motion_batch(target_real, debug["kp"], debug["kd"])
        self._write_csv(now, dt, debug, safety, feedback, imu, target_real)
        if self._soft_stop_active and now - self._soft_stop_start >= float(self.soft_stop_s):
            self.get_logger().warn(f"soft stop complete: {self.stop_reason}")
            rclpy.shutdown()

    def _empty_debug(self) -> Dict:
        return {"leg_phase": {leg: 0.0 for leg in LEG_ORDER}, "swing_progress": {leg: 1.0 for leg in LEG_ORDER}, "swing_mask": {leg: 0.0 for leg in LEG_ORDER}, "support_mask": {leg: 1.0 for leg in LEG_ORDER}, "active_pair": "SOFT_STOP", "support_pair": "FR+FL+RR+RL", "target_yaw": getattr(self.core, "target_yaw", 0.0) if hasattr(self, "core") else 0.0, "yaw_error": 0.0, "vmc_height": 0.0, "vmc_roll": 0.0, "vmc_pitch": 0.0, "yaw_corr": 0.0, "yaw_offsets": np.zeros(4, dtype=np.float32), "vmc_weight": np.zeros(4, dtype=np.float32), "rear_late_window": {"RR": 0.0, "RL": 0.0}, "rear_late_guard": {"RR": 0.0, "RL": 0.0}, "rear_late_clearance": {"RR": 0.0, "RL": 0.0}, "rear_descent_scale": {"RR": 1.0, "RL": 1.0}, "rear_early_guard": {"RR": 0.0, "RL": 0.0}, "rear_early_score": {"RR": 0.0, "RL": 0.0}, "early_source": "none", "fk_clearance_ref": np.zeros(4, dtype=np.float32), "foot_z_ref": np.zeros(4, dtype=np.float32), "kp": np.array([float(self.hip_kp), float(self.thigh_kp), float(self.calf_kp)] * 4, dtype=np.float32), "kd": np.full(12, float(self.kd), dtype=np.float32)}

    def _soft_stop_target(self, now: float):
        a = smootherstep(clamp((now - self._soft_stop_start) / max(float(self.soft_stop_s), 1e-6), 0.0, 1.0))
        q = (1.0 - a) * self._soft_stop_from + a * self.default_policy
        debug = self._empty_debug()
        debug["kp"] = np.array([float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)] * 4, dtype=np.float32)
        debug["kd"] = np.full(12, float(self.support_kd), dtype=np.float32)
        return q.astype(np.float32), debug

    def _apply_safety(self, q_raw: np.ndarray, feedback: Feedback, dt: float, debug: Dict):
        q_real = feedback.q_policy if feedback.valid else self.q_cmd_final
        dq_real = feedback.dq_policy if feedback.valid else np.zeros(12, dtype=np.float32)
        use_virtual = (not self.enable_send) and bool(self.dry_run_virtual_feedback)
        q_safety = self.q_last_cmd.copy() if use_virtual else q_real.copy()
        dq_safety = np.zeros(12, dtype=np.float32) if use_virtual else dq_real.copy()
        rate_limits = np.array([float(self.max_target_rate_rad_s_hip), float(self.max_target_rate_rad_s_thigh), float(self.max_target_rate_rad_s_calf)] * 4, dtype=np.float32)
        delta = q_raw - self.q_last_cmd
        max_step = rate_limits * max(dt, 1e-4)
        clipped = np.clip(delta, -max_step, max_step)
        q_rate = self.q_last_cmd + clipped
        if use_virtual:
            dq_safety = clipped / max(dt, 1e-4)
        kp = np.asarray(debug["kp"], dtype=np.float32)
        kd = np.asarray(debug["kd"], dtype=np.float32)
        tau = kp * (q_rate - q_safety) - kd * dq_safety
        abs_tau = np.abs(tau)
        soft = float(self.torque_soft_limit_nm)
        hard = float(self.torque_stop_nm)
        scale = np.ones(12, dtype=np.float32)
        mask = abs_tau > soft
        scale[mask] = np.clip((hard - abs_tau[mask]) / max(hard - soft, 1e-6), 0.0, 1.0)
        q_torque = q_safety + scale * (q_rate - q_safety)
        tau_final = kp * (q_torque - q_safety) - kd * dq_safety
        q_cmd = np.clip(q_torque, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.q_last_cmd = q_cmd.astype(np.float32).copy()
        self.q_cmd_final = q_cmd.astype(np.float32).copy()
        qerr_safety = q_cmd - q_safety
        qerr_real = q_cmd - q_real
        max_qerr = float(np.max(np.abs(qerr_safety)))
        max_tau = float(np.nanmax(np.abs(tau_final))) if np.any(np.isfinite(tau_final)) else float("nan")
        current = feedback.current_policy
        temp = feedback.temp_policy
        max_current = float(np.nanmax(np.abs(current))) if np.any(np.isfinite(current)) else float("nan")
        max_temp = float(np.nanmax(temp)) if np.any(np.isfinite(temp)) else float("nan")
        warn = ""
        if max_qerr > float(self.max_q_error_warn): warn = "q_error_warn"
        if math.isfinite(max_tau) and max_tau > float(self.torque_soft_warn_nm): warn = (warn + "|tau_warn").strip("|")
        self._check_stop(max_qerr, max_tau, max_current, max_temp, feedback)
        return q_cmd, {"dry_run_virtual_feedback": use_virtual, "q_actual": q_real, "q_actual_for_safety": q_safety, "dq_actual": dq_real, "q_error": qerr_real, "q_error_for_safety": qerr_safety, "tau_est": tau_final, "current": current, "temp": temp, "rate_delta": clipped, "max_q_error": max_qerr, "max_tau_est": max_tau, "max_current": max_current, "max_temp": max_temp, "rate_clip_ratio": float(np.mean(np.abs(delta - clipped) > 1e-6)), "torque_clip_ratio": float(np.mean(mask.astype(np.float32))), "safety_warn": warn}

    def _check_stop(self, max_qerr, max_tau, max_current, max_temp, feedback):
        if self.safety_stop:
            return
        imu = self._read_imu()
        roll_deg = abs(float(imu["rpy"][0])) * 180.0 / math.pi
        pitch_deg = abs(float(imu["rpy"][1])) * 180.0 / math.pi
        reason = ""
        if roll_deg > float(self.max_roll_deg_stop): reason = "roll_stop"
        elif pitch_deg > float(self.max_pitch_deg_stop): reason = "pitch_stop"
        elif max_qerr > float(self.max_q_error_stop): reason = "q_error_stop"
        elif math.isfinite(max_tau) and max_tau > float(self.torque_stop_nm): reason = "torque_stop"
        elif math.isfinite(max_current) and max_current > float(self.current_stop_a): reason = "current_stop"
        elif math.isfinite(max_temp) and max_temp > float(self.max_motor_temp_stop_c): reason = "temp_stop"
        elif self.enable_send and feedback.valid and feedback.max_age_ms > float(self.max_motor_age_ms): reason = "communication_timeout"
        if reason:
            self.safety_stop = True
            self.safety_stop_reason = reason
            self.stop_reason = reason
            self.get_logger().error(f"SAFETY STOP: {reason}")

    def _request_soft_stop(self, reason: str):
        if self._soft_stop_active:
            return
        self._soft_stop_active = True
        self._soft_stop_start = time.time()
        self._soft_stop_from = self.q_cmd_final.copy()
        self.stop_reason = reason
        self.get_logger().warn(f"soft stop requested: {reason}")

    def _send_default_stand(self):
        items = []
        kp = [float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)] * 4
        kd = [float(self.support_kd)] * 12
        for i, mid in enumerate(self.mapper.get_real_motor_ids()):
            items.append({"motor_id": int(mid), "position": float(self.default_real[i]), "speed": 0.0, "torque": 0.0, "kp": float(kp[i]), "kd": float(kd[i])})
        r = self.http_session.post(f"{self.motor_base_url}/api/rs04/motion_mode_run_batch", json={"items": items, "enable_first": True, "stop_first": False}, timeout=max(float(self.http_timeout), 0.5))
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")

    def _send_motion_batch(self, target_real: np.ndarray, kp_policy: np.ndarray, kd_policy: np.ndarray) -> bool:
        kp_real = np.zeros(12, dtype=np.float32)
        kd_real = np.zeros(12, dtype=np.float32)
        kp_real[self.mapper.policy_to_real_index] = kp_policy
        kd_real[self.mapper.policy_to_real_index] = kd_policy
        items = []
        for i, mid in enumerate(self.mapper.get_real_motor_ids()):
            items.append({"motor_id": int(mid), "position": float(target_real[i]), "speed": float(self.send_speed), "torque": float(self.send_torque), "kp": float(kp_real[i]), "kd": float(kd_real[i])})
        try:
            r = self.http_session.post(f"{self.motor_base_url}/api/rs04/motion_batch_fast", json={"items": items, "enable_first": False, "stop_first": False}, timeout=float(self.http_timeout))
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            now = time.time()
            if now - self._last_send_info > 1.0:
                self._last_send_info = now
                self.get_logger().info(f"[SEND] direct_v4 ok mode={self.test_mode}")
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def _publish(self, target_real, debug, safety):
        msg = Float32MultiArray(); msg.data = np.asarray(target_real, dtype=np.float32).reshape(-1).tolist(); self.pub_target.publish(msg)
        dbg = Float32MultiArray(); dbg.data = np.array([self.core.phase if hasattr(self, "core") else 0.0, safety["max_q_error"], safety["max_tau_est"], safety["rate_clip_ratio"], safety["torque_clip_ratio"]], dtype=np.float32).tolist(); self.pub_debug.publish(dbg)

    def _fk_clearances(self, arr: np.ndarray) -> np.ndarray:
        clear = np.zeros(4, dtype=np.float32)
        for li, leg in enumerate(LEG_ORDER):
            i = LEG_START[leg]
            _x, z = self.core.forward_sagittal(float(arr[i+1]), float(arr[i+2]))
            clear[li] = z - self.core.default_foot_z[li]
        return clear

    def _write_csv(self, now, dt, debug, safety, feedback, imu, target_real):
        rpy = imu["rpy"]; gyro = imu["gyro"]
        q_ref = np.asarray(self.q_last_cmd + (np.asarray(safety["rate_delta"]) * 0.0), dtype=np.float32)  # placeholder overwritten below
        # q_ref is the pre-safety target in this row; use q_cmd_final + q_ref_cmd_diff for exact storage impossible after rate. Store q_raw via debug omitted? q_cmd itself is still enough for bring-up.
        q_ref = self.q_cmd_final + np.zeros(12, dtype=np.float32)
        q_cmd = self.q_cmd_final.copy()
        q_actual = safety["q_actual"]
        yaw_error = wrap_to_pi(float(rpy[2]) - float(debug.get("target_yaw", 0.0)))
        fk_ref = np.asarray(debug.get("fk_clearance_ref", np.zeros(4)), dtype=np.float32)
        fk_cmd = self._fk_clearances(q_cmd)
        fk_act = self._fk_clearances(q_actual)
        row = [f"{now:.6f}", f"{dt:.6f}", self.test_mode, int(self.enable_send), self.node_state, self.stop_reason, self.safety_stop_reason, self.stand_source, int(bool(safety["dry_run_virtual_feedback"])), f"{self.core.phase:.6f}", debug["active_pair"], debug["support_pair"]]
        row += [f"{debug['leg_phase'][leg]:.6f}" for leg in LEG_ORDER]
        row += [f"{debug['swing_progress'][leg]:.6f}" for leg in LEG_ORDER]
        row += [int(debug["swing_mask"][leg] > 0.5) for leg in LEG_ORDER]
        row += [int(debug["support_mask"][leg] > 0.5) for leg in LEG_ORDER]
        row += [f"{float(rpy[0]):.6f}", f"{float(rpy[1]):.6f}", f"{float(rpy[2]):.6f}", f"{float(debug.get('target_yaw',0.0)):.6f}", f"{yaw_error:.6f}", f"{abs(yaw_error):.6f}", f"{float(gyro[0]):.6f}", f"{float(gyro[1]):.6f}", f"{float(gyro[2]):.6f}"]
        row += [f"{self.real_vmc_scale():.6f}", f"{float(debug['vmc_height']):.6f}", f"{float(debug['vmc_roll']):.6f}", f"{float(debug['vmc_pitch']):.6f}", f"{float(debug['yaw_corr']):.6f}"]
        row += [f"{float(x):.6f}" for x in debug["vmc_weight"]]
        row += [f"{float(x):.6f}" for x in debug["yaw_offsets"]]
        row += [int(debug["rear_late_window"]["RR"] > 0.5), int(debug["rear_late_window"]["RL"] > 0.5), int(debug["rear_late_guard"]["RR"] > 0.5), int(debug["rear_late_guard"]["RL"] > 0.5), f"{debug['rear_late_clearance']['RR']:.6f}", f"{debug['rear_late_clearance']['RL']:.6f}", f"{debug['rear_descent_scale']['RR']:.6f}", f"{debug['rear_descent_scale']['RL']:.6f}", int(debug["rear_early_guard"]["RR"] > 0.5), int(debug["rear_early_guard"]["RL"] > 0.5), f"{debug['rear_early_score']['RR']:.6f}", f"{debug['rear_early_score']['RL']:.6f}", debug["early_source"]]
        arrays = [q_ref, q_cmd, q_actual, safety["q_actual_for_safety"], safety["q_error"], safety["q_error_for_safety"], safety["tau_est"], safety["current"], safety["temp"], safety["rate_delta"], q_ref - q_cmd, self.mapper_default_policy, self.sim_v4_default_policy, self.mapper_default_policy - self.sim_v4_default_policy]
        for arr in arrays:
            row += [f"{float(x):.6f}" if math.isfinite(float(x)) else "nan" for x in np.asarray(arr).reshape(12)]
        row += [f"{float(x):.6f}" for x in np.asarray(target_real).reshape(12)]
        row += [f"{float(x):.6f}" for x in fk_ref]
        row += [f"{float(x):.6f}" for x in fk_cmd]
        row += [f"{float(x):.6f}" for x in fk_act]
        row += [f"{safety['max_q_error']:.6f}", f"{safety['max_tau_est']:.6f}" if math.isfinite(safety["max_tau_est"]) else "nan", f"{safety['max_current']:.6f}" if math.isfinite(safety["max_current"]) else "nan", f"{safety['max_temp']:.6f}" if math.isfinite(safety["max_temp"]) else "nan", safety["safety_warn"], int(self.safety_stop), f"{safety['rate_clip_ratio']:.6f}", f"{safety['torque_clip_ratio']:.6f}", int(feedback.valid), HIP_OUTWARD_SIGNS["FR"], HIP_OUTWARD_SIGNS["FL"], HIP_OUTWARD_SIGNS["RR"], HIP_OUTWARD_SIGNS["RL"], 1]
        self.csv_writer.writerow(row); self.csv_file.flush()
        self.stats["qerr"].append(float(safety["max_q_error"]));
        if math.isfinite(safety["max_tau_est"]): self.stats["tau"].append(float(safety["max_tau_est"]))
        self.stats["roll"].append(abs(float(rpy[0])) * 180.0 / math.pi); self.stats["pitch"].append(abs(float(rpy[1])) * 180.0 / math.pi); self.stats["yawerr"].append(abs(yaw_error) * 180.0 / math.pi)
        self.stats["rate_clip"].append(float(safety["rate_clip_ratio"])); self.stats["torque_clip"].append(float(safety["torque_clip_ratio"]))
        self.stats["front_clear"].append(float(max(fk_cmd[0], fk_cmd[1]))); self.stats["rear_clear"].append(float(max(fk_cmd[2], fk_cmd[3])))

    def destroy_node(self):
        try: self._request_soft_stop("shutdown")
        except Exception: pass
        try: self._print_summary()
        except Exception as exc: self.get_logger().warn(f"summary failed: {exc}")
        try: self.csv_file.flush(); self.csv_file.close()
        except Exception: pass
        try:
            if self.imu is not None: self.imu.stop()
        except Exception: pass
        try: self.motor.close(); self.http_session.close()
        except Exception: pass
        super().destroy_node()

    def _print_summary(self):
        result = "FAIL" if self.safety_stop else "PASS"
        if result == "PASS" and percentile(self.stats["qerr"], 95) > float(self.max_q_error_warn): result = "CAUTION"
        self.get_logger().warn(f"[SUMMARY] result={result} duration={time.time()-self.start_time:.2f}s mode={self.test_mode} send={self.enable_send} csv={self.csv_path} qerr p95/max={percentile(self.stats['qerr'],95):.3f}/{max(self.stats['qerr'] or [0]):.3f} tau p95/p99/max={percentile(self.stats['tau'],95):.2f}/{percentile(self.stats['tau'],99):.2f}/{max(self.stats['tau'] or [float('nan')]):.2f} clearance cmd front/rear max={max(self.stats['front_clear'] or [0]):.3f}/{max(self.stats['rear_clear'] or [0]):.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = FanfanCpgVmcV4DirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
