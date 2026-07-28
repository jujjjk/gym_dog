#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import copy
import hashlib
import json
import os
import queue
import threading
import time
import numpy as np
import onnxruntime as ort
import requests

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from .deploy_cpg import DeployJointCPG
from .obs_builder import ObsBuilder36
from .semantic_mapper import JointSemanticMapper


# ============================================================
# 默认站立姿态：真机电机顺序
# 0x11,0x12,0x13, 0x21,0x22,0x23, 0x31,0x32,0x33, 0x41,0x42,0x43
#
# 注意：这是“真机电机目标角”，不是 policy_joint_names 顺序。
# ============================================================
DEFAULT_STAND_POSE_REAL_ORDER = [
    (0x11,  0.157100),   # FR_hip
    (0x12,  0.349066),   # FR_thigh
    (0x13, -0.785400),   # FR_calf

    (0x21, -0.157100),   # FL_hip
    (0x22, -0.349067),   # FL_thigh
    (0x23,  0.785400),   # FL_calf

    (0x31,  0.157100),   # RL_hip
    (0x32, -0.226900),   # RL_thigh
    (0x33,  0.349065),   # RL_calf

    (0x41, -0.157100),   # RR_hip
    (0x42,  0.226900),   # RR_thigh
    (0x43, -0.349066),   # RR_calf
]


class OnnxPolicyRunner:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = self._get_obs_dim(input_shape)
        metadata = self.session.get_modelmeta().custom_metadata_map
        raw_config = metadata.get("fanfan_deployment_config")
        self.deployment_config = json.loads(raw_config) if raw_config else None
        self.output_transform = "identity"
        if self.deployment_config is not None:
            if self.deployment_config.get("schema_version") != 1:
                raise RuntimeError(
                    "Unsupported fanfan_deployment_config schema: "
                    f"{self.deployment_config.get('schema_version')}"
                )
            dimensions = self.deployment_config.get("dimensions", {})
            if int(dimensions.get("observations", -1)) != self.obs_dim:
                raise RuntimeError("ONNX graph and deployment metadata dimensions disagree")
            if int(dimensions.get("actions", -1)) != 12:
                raise RuntimeError("Fanfan deployment contract must contain 12 actions")
            self.output_transform = str(
                self.deployment_config["control"].get("output_transform", "identity")
            ).lower()

        print("[ONNX] input :", self.input_name, self.session.get_inputs()[0].shape)
        print("[ONNX] output:", self.output_name, self.session.get_outputs()[0].shape)
        print("[ONNX] expected obs_dim:", self.obs_dim)

    @staticmethod
    def _get_obs_dim(input_shape) -> int:
        if len(input_shape) < 2:
            raise RuntimeError(f"Unsupported ONNX input shape: {input_shape}")

        dim = input_shape[1]
        if isinstance(dim, str) or dim is None:
            raise RuntimeError(
                f"ONNX input obs dimension is dynamic/unknown: {input_shape}. "
                "Please export with a fixed obs size."
            )

        dim = int(dim)
        if dim not in (36, 48, 50, 52):
            raise RuntimeError(
                f"Unsupported ONNX obs dimension: {dim}, expected 36, 48, 50, or 52"
            )
        return dim

    def infer(self, obs_in: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_in, dtype=np.float32).reshape(1, self.obs_dim)

        out = self.session.run(
            [self.output_name],
            {self.input_name: obs},
        )[0]

        action = np.asarray(out, dtype=np.float32).reshape(-1)

        if action.shape[0] < 12:
            raise RuntimeError(f"ONNX output size < 12, got {action.shape}")

        action = action[:12]
        if self.output_transform == "tanh":
            action = np.tanh(action)
        elif self.output_transform not in ("identity", "none"):
            raise RuntimeError(f"Unsupported ONNX output transform: {self.output_transform}")
        return action.astype(np.float32)


class SafeTargetLimiter:
    def __init__(self):
        self.q_last_cmd = np.zeros(12, dtype=np.float32)
        self.qdot_last_cmd = np.zeros(12, dtype=np.float32)
        self.initialized = False

    def reset(self, q_current):
        self.q_last_cmd = np.asarray(q_current, dtype=np.float32).reshape(12).copy()
        self.qdot_last_cmd = np.zeros(12, dtype=np.float32)
        self.initialized = True

    def limit(
        self,
        q_raw,
        q_current,
        dt,
        kp,
        torque_budget_nm,
        err_limit_safety_factor,
        max_target_rate_rad_s,
        max_target_accel_rad_s2,
        err_limit_mul,
        target_rate_mul,
        target_accel_mul,
        absolute_error_limit_rad=None,
    ):
        q_raw = np.asarray(q_raw, dtype=np.float32).reshape(12)
        q_current = np.asarray(q_current, dtype=np.float32).reshape(12)
        err_limit_mul = np.asarray(err_limit_mul, dtype=np.float32).reshape(12)
        target_rate_mul = np.asarray(target_rate_mul, dtype=np.float32).reshape(12)
        target_accel_mul = np.asarray(target_accel_mul, dtype=np.float32).reshape(12)
        dt = max(float(dt), 1e-4)
        kp = np.asarray(kp, dtype=np.float32)
        if kp.shape == ():
            kp = np.full(12, abs(float(kp)), dtype=np.float32)
        else:
            kp = np.abs(kp.reshape(12))
        kp = np.maximum(kp, 1e-6)
        torque_budget = max(0.0, float(torque_budget_nm))
        base_err_limit = torque_budget / kp
        err_limit_vec = (
            max(0.0, float(err_limit_safety_factor))
            * base_err_limit
            * np.maximum(err_limit_mul, 0.0)
        ).astype(np.float32)
        if absolute_error_limit_rad is not None:
            absolute_limit = np.asarray(
                absolute_error_limit_rad, dtype=np.float32
            ).reshape(12)
            # Models exported before the PD8 contract have no absolute error
            # limit. A legacy caller may represent that as all +inf; treat it
            # exactly like None instead of turning the 50 Hz loop into errors.
            if np.all(np.isposinf(absolute_limit)):
                absolute_limit = None
            elif not np.all(np.isfinite(absolute_limit)) or np.any(absolute_limit <= 0.0):
                raise ValueError("absolute position error limits must be positive finite values")
            if absolute_limit is not None:
                err_limit_vec = np.minimum(err_limit_vec, absolute_limit)

        if not self.initialized:
            self.reset(q_current)

        q_last = self.q_last_cmd.copy()
        qdot_prev = self.qdot_last_cmd.copy()

        raw_delta = q_raw - q_current
        safe_delta = np.clip(raw_delta, -err_limit_vec, err_limit_vec)
        q_safe = q_current + safe_delta
        pre_limited_mask = np.abs(raw_delta - safe_delta) > 1e-6

        qdot_raw_unclipped = (q_safe - q_last) / dt
        max_rate_vec = (
            abs(float(max_target_rate_rad_s)) * np.maximum(target_rate_mul, 0.0)
        ).astype(np.float32)
        if np.max(max_rate_vec) > 0.0:
            qdot_raw = np.clip(qdot_raw_unclipped, -max_rate_vec, max_rate_vec)
        else:
            qdot_raw = qdot_raw_unclipped
        rate_limited_mask = np.abs(qdot_raw_unclipped - qdot_raw) > 1e-6

        qddot_raw = (qdot_raw - qdot_prev) / dt
        max_accel_vec = (
            abs(float(max_target_accel_rad_s2)) * np.maximum(target_accel_mul, 0.0)
        ).astype(np.float32)
        if np.max(max_accel_vec) > 0.0:
            qdot_step = np.clip(
                qdot_raw - qdot_prev,
                -max_accel_vec * dt,
                max_accel_vec * dt,
            )
        else:
            qdot_step = qdot_raw - qdot_prev
        qdot_cmd_pre_clamp = qdot_prev + qdot_step
        accel_limited_mask = np.abs((qdot_raw - qdot_prev) - qdot_step) > 1e-6

        q_cmd_pre_clamp = q_last + qdot_cmd_pre_clamp * dt
        q_cmd_delta = np.clip(
            q_cmd_pre_clamp - q_current,
            -err_limit_vec,
            err_limit_vec,
        )
        q_cmd = q_current + q_cmd_delta
        post_limited_mask = np.abs((q_cmd_pre_clamp - q_current) - q_cmd_delta) > 1e-6

        qdot_cmd = (q_cmd - q_last) / dt
        qddot_cmd = (qdot_cmd - qdot_prev) / dt

        self.q_last_cmd = q_cmd.astype(np.float32).copy()
        self.qdot_last_cmd = qdot_cmd.astype(np.float32).copy()

        return q_cmd.astype(np.float32), {
            "enabled": True,
            "dt": dt,
            "torque_budget": torque_budget,
            "base_err_limit": base_err_limit,
            "err_limit_min": float(np.min(err_limit_vec)),
            "err_limit_max": float(np.max(err_limit_vec)),
            "err_limit": err_limit_vec,
            "max_rate": max_rate_vec,
            "max_accel": max_accel_vec,
            "q_raw_error_abs_max": float(np.max(np.abs(q_raw - q_current))),
            "q_cmd_error_abs_max": float(np.max(np.abs(q_cmd - q_current))),
            "qdot_cmd_abs_max": float(np.max(np.abs(qdot_cmd))),
            "qddot_cmd_abs_max": float(np.max(np.abs(qddot_cmd))),
            "pre_limited_count": int(np.count_nonzero(pre_limited_mask)),
            "rate_limited_count": int(np.count_nonzero(rate_limited_mask)),
            "accel_limited_count": int(np.count_nonzero(accel_limited_mask)),
            "post_limited_count": int(np.count_nonzero(post_limited_mask)),
            "qdot_cmd": qdot_cmd.astype(np.float32),
            "qddot_cmd": qddot_cmd.astype(np.float32),
            "qddot_raw": qddot_raw.astype(np.float32),
            "pre_limited_mask": pre_limited_mask,
            "rate_limited_mask": rate_limited_mask,
            "accel_limited_mask": accel_limited_mask,
            "post_limited_mask": post_limited_mask,
            "raw_delta": raw_delta.astype(np.float32),
            "safe_delta": safe_delta.astype(np.float32),
        }


class MydogPolicyNode(Node):
    def __init__(self):
        super().__init__("mydog_policy_node")

        # ============================================================
        # 基础参数
        # ============================================================
        self.declare_parameter(
            "onnx_path",
            "/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/policy.onnx",
        )
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("policy_hz", 50.0)
        self.declare_parameter("action_scale", 0.10)
        self.declare_parameter("front_action_scale_mul", 1.0)
        self.declare_parameter("rear_action_scale_mul", 1.0)
        self.declare_parameter("hip_action_scale_mul", 1.0)
        self.declare_parameter("thigh_action_scale_mul", 1.0)
        self.declare_parameter("calf_action_scale_mul", 1.0)
        self.declare_parameter("rear_thigh_action_scale_mul", 1.0)
        self.declare_parameter("rear_calf_action_scale_mul", 1.0)
        self.declare_parameter("thigh_action_sign", 1.0)
        self.declare_parameter("action_leg_yaw_180", False)
        self.declare_parameter("semantic_yaw_180", False)
        self.declare_parameter("action_mode", "cpg_residual")
        self.declare_parameter("cpg_gait", "trot")
        self.declare_parameter("cpg_freq_min", 0.8)
        self.declare_parameter("cpg_freq_max", 1.8)
        self.declare_parameter("cpg_k_freq", 3.0)
        self.declare_parameter("cpg_standing_cmd_threshold", 0.03)
        self.declare_parameter("cpg_duty_factor", 0.60)
        self.declare_parameter("cpg_hip_amp", 0.025)
        self.declare_parameter("cpg_thigh_amp", 0.18)
        self.declare_parameter("cpg_calf_lift_amp", 0.60)
        self.declare_parameter("cpg_stance_calf_amp", 0.08)
        self.declare_parameter("cpg_stride_sign", -1.0)
        self.declare_parameter("cpg_enable_hip_balance", True)
        self.declare_parameter("cpg_hip_stance_widen_amp", 0.020)
        self.declare_parameter("cpg_hip_swing_relax_amp", 0.008)
        self.declare_parameter("cpg_hip_balance_signs", "-1.0,1.0,-1.0,1.0")
        self.declare_parameter("cpg_hip_balance_use_stance_mask", True)
        self.declare_parameter("cpg_hip_balance_smooth_shape", "sin")
        self.declare_parameter("cpg_hip_balance_max_abs", 0.06)
        self.declare_parameter("cpg_zero_residual_when_standing", True)
        self.declare_parameter("enable_phase_aware_hip_gate", True)
        self.declare_parameter("hip_gate_stance_min_outward", 0.008)
        self.declare_parameter("hip_gate_swing_max_outward", 0.035)
        self.declare_parameter("hip_gate_side_signs", "-1.0,1.0,-1.0,1.0")
        self.declare_parameter("residual_limit_hip", 0.03)
        self.declare_parameter("residual_limit_thigh", 0.06)
        self.declare_parameter("residual_limit_calf", 0.06)
        self.declare_parameter("clip_action", True)
        self.declare_parameter("action_clip_limit", 1.0)
        self.declare_parameter("enable_target_smoothing", True)
        self.declare_parameter("max_target_rate_rad_s", 2.0)
        self.declare_parameter("max_target_accel_rad_s2", 60.0)
        self.declare_parameter("err_limit_safety_factor", 1.0)
        self.declare_parameter("hip_err_limit_mul", 1.0)
        self.declare_parameter("thigh_err_limit_mul", 1.2)
        self.declare_parameter("calf_err_limit_mul", 1.4)
        self.declare_parameter("hip_target_rate_mul", 1.0)
        self.declare_parameter("thigh_target_rate_mul", 1.3)
        self.declare_parameter("calf_target_rate_mul", 1.6)
        self.declare_parameter("hip_target_accel_mul", 1.0)
        self.declare_parameter("thigh_target_accel_mul", 1.3)
        self.declare_parameter("calf_target_accel_mul", 1.6)
        self.declare_parameter("target_smoothing_fixed_dt", True)
        self.declare_parameter("target_smoothing_follow_sent_target", True)
        self.declare_parameter("gait_phase_period", 0.55)
        self.declare_parameter("deployment_gait_phase_period_scale", 1.0)
        self.declare_parameter("deployment_command_scale_x_mul", 1.0)
        self.declare_parameter("deployment_command_scale_y_mul", 1.0)
        self.declare_parameter("deployment_command_scale_yaw_mul", 1.0)
        self.declare_parameter("enable_cmd_smoothing", True)
        self.declare_parameter("max_cmd_x_rate_mps2", 0.05)
        self.declare_parameter("max_cmd_y_rate_mps2", 0.05)
        self.declare_parameter("max_cmd_yaw_rate_rad_s2", 0.3)
        self.declare_parameter("enable_cmd_limits", False)
        self.declare_parameter("cmd_min_x", -1.0)
        self.declare_parameter("cmd_max_x", 1.0)
        self.declare_parameter("cmd_min_y", -1.0)
        self.declare_parameter("cmd_max_y", 1.0)
        self.declare_parameter("cmd_min_yaw", -1.0)
        self.declare_parameter("cmd_max_yaw", 1.0)
        self.declare_parameter("require_cmd_vel", True)
        self.declare_parameter("cmd_vel_timeout_sec", 0.5)
        self.declare_parameter("zero_cmd_inhibits_policy", True)
        self.declare_parameter("enable_zero_cmd_stand_protection", False)
        self.declare_parameter("zero_cmd_stand_x_threshold", 0.01)
        self.declare_parameter("zero_cmd_stand_y_threshold", 0.01)
        self.declare_parameter("zero_cmd_stand_yaw_threshold", 0.03)
        self.declare_parameter("enable_policy_action_cmd_gate", False)
        self.declare_parameter("policy_action_cmd_gate_start_ratio", 0.05)
        self.declare_parameter("policy_action_cmd_gate_full_ratio", 1.0)
        self.declare_parameter("policy_action_cmd_gate_max_scale", 1.0)
        self.declare_parameter("reset_gait_phase_on_command_start", False)
        self.declare_parameter("base_lin_vel_source", "zero")
        self.declare_parameter("state_estimator_timeout_sec", 0.25)
        self.declare_parameter("max_motor_age_ms", 100.0)
        self.declare_parameter("recheck_stale_motor_once", True)
        self.declare_parameter("motor_state_async", True)
        self.declare_parameter("motor_state_poll_hz", 50.0)
        self.declare_parameter("enable_send", False)
        self.declare_parameter("print_only", False)
        self.declare_parameter("debug_print_arrays", False)
        self.declare_parameter("debug_print_period_sec", 0.5)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.0)
        self.declare_parameter("debug_csv_async", True)
        self.declare_parameter("debug_csv_queue_size", 128)
        self.declare_parameter("debug_csv_flush_every_n", 20)
        self.declare_parameter("debug_warn_action_abs", 0.95)
        self.declare_parameter("debug_warn_error_rad", 0.5)
        self.declare_parameter("debug_critical_error_rad", 1.0)
        self.declare_parameter("stand_only", False)
        self.declare_parameter("stand_only_duration_sec", 5.0)
        self.declare_parameter("policy_enable", True)
        self.declare_parameter("expected_policy_task", "")
        self.declare_parameter("expected_policy_sha256", "")
        self.declare_parameter("joint_probe_enable", False)
        self.declare_parameter("joint_probe_name", "")
        self.declare_parameter("joint_probe_delta_rad", 0.05)
        self.declare_parameter("joint_probe_period_sec", 2.0)

        # online 有时不稳定，所以默认不强制检查
        self.declare_parameter("require_online", False)

        # ============================================================
        # 启动前默认站立参数
        # ============================================================
        self.declare_parameter("startup_stand_first", False)
        self.declare_parameter("startup_stand_kp", 12.0)
        self.declare_parameter("startup_stand_kd", 2.0)
        self.declare_parameter("startup_stand_speed", 0.0)
        self.declare_parameter("startup_stand_torque", 0.0)
        self.declare_parameter("startup_stand_enable_first", True)
        self.declare_parameter("startup_stand_stop_first", False)
        self.declare_parameter("startup_stand_settle_sec", 1.5)
        self.declare_parameter("startup_stand_hold_current_sec", 0.5)
        self.declare_parameter("startup_stand_ramp_sec", 12.0)
        self.declare_parameter("startup_stand_timeout_sec", 25.0)
        self.declare_parameter("startup_stand_max_rate_hip", 0.12)
        self.declare_parameter("startup_stand_max_rate_thigh", 0.15)
        self.declare_parameter("startup_stand_max_rate_calf", 0.15)
        self.declare_parameter("startup_stand_max_step_rad", 0.003)
        self.declare_parameter("startup_stand_ready_error_rad", 0.08)
        self.declare_parameter("startup_stand_stop_error_rad", 0.30)
        self.declare_parameter("startup_stand_stop_torque_nm", 8.0)
        self.declare_parameter("enable_default_pose_check", True)
        self.declare_parameter("default_pose_check_sec", 2.0)
        self.declare_parameter("use_policy_default_as_stand_pose", True)
        self.declare_parameter("stand_pose_source", "policy_default")

        # ============================================================
        # 真机发送安全参数
        # ============================================================
        self.declare_parameter("max_target_delta", 0.80)
        self.declare_parameter("send_kp", 40.0)
        self.declare_parameter("send_kd", 1.2)
        self.declare_parameter("use_model_pd_gains", True)
        self.declare_parameter("model_kp_scale", 1.0)
        self.declare_parameter("model_kd_scale", 1.0)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("enable_velocity_ff", False)
        self.declare_parameter("velocity_ff_scale", 0.3)
        self.declare_parameter("max_motor_vel_cmd_rad_s", 8.0)
        self.declare_parameter("log_motor_vel_cmd", True)
        self.declare_parameter("enable_torque_error_limit", True)
        self.declare_parameter("torque_limit_nm", -1.0)
        self.declare_parameter("motor_torque_limit_nm", 6.0)
        self.declare_parameter("torque_safety_ratio", 1.0)
        self.declare_parameter("torque_safety_budget_nm", -1.0)
        self.declare_parameter("expected_active_torque_budget_nm", -1.0)
        self.declare_parameter("send_enable_first", False)
        self.declare_parameter("send_stop_first", False)
        self.declare_parameter("http_timeout", 0.05)
        self.declare_parameter("enable_rear_leg_posture_bias", False)
        self.declare_parameter("rear_calf_extend_bias_policy_rad", 0.0)
        self.declare_parameter("rear_thigh_bias_policy_rad", 0.0)

        # ============================================================
        # 读取参数
        # ============================================================
        self.onnx_path = self.get_parameter("onnx_path").value
        self.motor_base_url = self.get_parameter("motor_base_url").value.rstrip("/")
        self.policy_hz = float(self.get_parameter("policy_hz").value)
        self.base_lin_vel_source = str(self.get_parameter("base_lin_vel_source").value).lower()
        self.state_estimator_timeout_sec = float(
            self.get_parameter("state_estimator_timeout_sec").value
        )

        self.action_scale = float(self.get_parameter("action_scale").value)
        self.front_action_scale_mul = float(
            self.get_parameter("front_action_scale_mul").value
        )
        self.rear_action_scale_mul = float(
            self.get_parameter("rear_action_scale_mul").value
        )
        self.hip_action_scale_mul = float(
            self.get_parameter("hip_action_scale_mul").value
        )
        self.thigh_action_scale_mul = float(
            self.get_parameter("thigh_action_scale_mul").value
        )
        self.calf_action_scale_mul = float(
            self.get_parameter("calf_action_scale_mul").value
        )
        self.rear_thigh_action_scale_mul = float(
            self.get_parameter("rear_thigh_action_scale_mul").value
        )
        self.rear_calf_action_scale_mul = float(
            self.get_parameter("rear_calf_action_scale_mul").value
        )
        self.thigh_action_sign = float(self.get_parameter("thigh_action_sign").value)
        self.action_leg_yaw_180 = bool(self.get_parameter("action_leg_yaw_180").value)
        self.semantic_yaw_180 = bool(self.get_parameter("semantic_yaw_180").value)
        self.action_mode = str(self.get_parameter("action_mode").value).strip().lower()
        if self.action_mode not in ("pure_rl", "cpg_residual", "cpg_only"):
            self.get_logger().warn(
                f"Unknown action_mode={self.action_mode!r}; using pure_rl."
            )
            self.action_mode = "pure_rl"
        self.cpg_gait = str(self.get_parameter("cpg_gait").value).strip().lower()
        self.cpg_freq_min = float(self.get_parameter("cpg_freq_min").value)
        self.cpg_freq_max = float(self.get_parameter("cpg_freq_max").value)
        self.cpg_k_freq = float(self.get_parameter("cpg_k_freq").value)
        self.cpg_standing_cmd_threshold = float(
            self.get_parameter("cpg_standing_cmd_threshold").value
        )
        self.cpg_duty_factor = float(self.get_parameter("cpg_duty_factor").value)
        self.cpg_hip_amp = float(self.get_parameter("cpg_hip_amp").value)
        self.cpg_thigh_amp = float(self.get_parameter("cpg_thigh_amp").value)
        self.cpg_calf_lift_amp = float(self.get_parameter("cpg_calf_lift_amp").value)
        self.cpg_stance_calf_amp = float(self.get_parameter("cpg_stance_calf_amp").value)
        self.cpg_stride_sign = float(self.get_parameter("cpg_stride_sign").value)
        self.cpg_enable_hip_balance = bool(
            self.get_parameter("cpg_enable_hip_balance").value
        )
        self.cpg_hip_stance_widen_amp = float(
            self.get_parameter("cpg_hip_stance_widen_amp").value
        )
        self.cpg_hip_swing_relax_amp = float(
            self.get_parameter("cpg_hip_swing_relax_amp").value
        )
        self.cpg_hip_balance_signs = self.parse_hip_balance_signs(
            self.get_parameter("cpg_hip_balance_signs").value
        )
        self.cpg_hip_balance_use_stance_mask = bool(
            self.get_parameter("cpg_hip_balance_use_stance_mask").value
        )
        self.cpg_hip_balance_smooth_shape = str(
            self.get_parameter("cpg_hip_balance_smooth_shape").value
        ).strip().lower()
        self.cpg_hip_balance_max_abs = float(
            self.get_parameter("cpg_hip_balance_max_abs").value
        )
        self.cpg_zero_residual_when_standing = bool(
            self.get_parameter("cpg_zero_residual_when_standing").value
        )
        self.enable_phase_aware_hip_gate = bool(
            self.get_parameter("enable_phase_aware_hip_gate").value
        )
        self.hip_gate_stance_min_outward = float(
            self.get_parameter("hip_gate_stance_min_outward").value
        )
        self.hip_gate_swing_max_outward = float(
            self.get_parameter("hip_gate_swing_max_outward").value
        )
        self.hip_gate_side_signs = self.parse_hip_balance_signs(
            self.get_parameter("hip_gate_side_signs").value
        )
        self.residual_limit_hip = float(self.get_parameter("residual_limit_hip").value)
        self.residual_limit_thigh = float(self.get_parameter("residual_limit_thigh").value)
        self.residual_limit_calf = float(self.get_parameter("residual_limit_calf").value)
        self.clip_action = bool(self.get_parameter("clip_action").value)
        self.action_clip_limit = float(self.get_parameter("action_clip_limit").value)
        self.enable_target_smoothing = bool(
            self.get_parameter("enable_target_smoothing").value
        )
        self.max_target_rate_rad_s = float(
            self.get_parameter("max_target_rate_rad_s").value
        )
        self.max_target_accel_rad_s2 = float(
            self.get_parameter("max_target_accel_rad_s2").value
        )
        self.err_limit_safety_factor = float(
            self.get_parameter("err_limit_safety_factor").value
        )
        self.hip_err_limit_mul = float(self.get_parameter("hip_err_limit_mul").value)
        self.thigh_err_limit_mul = float(self.get_parameter("thigh_err_limit_mul").value)
        self.calf_err_limit_mul = float(self.get_parameter("calf_err_limit_mul").value)
        self.hip_target_rate_mul = float(self.get_parameter("hip_target_rate_mul").value)
        self.thigh_target_rate_mul = float(
            self.get_parameter("thigh_target_rate_mul").value
        )
        self.calf_target_rate_mul = float(self.get_parameter("calf_target_rate_mul").value)
        self.hip_target_accel_mul = float(
            self.get_parameter("hip_target_accel_mul").value
        )
        self.thigh_target_accel_mul = float(
            self.get_parameter("thigh_target_accel_mul").value
        )
        self.calf_target_accel_mul = float(
            self.get_parameter("calf_target_accel_mul").value
        )
        self.target_smoothing_fixed_dt = bool(
            self.get_parameter("target_smoothing_fixed_dt").value
        )
        self.target_smoothing_follow_sent_target = bool(
            self.get_parameter("target_smoothing_follow_sent_target").value
        )
        self.gait_phase_period = float(self.get_parameter("gait_phase_period").value)
        self.deployment_gait_phase_period_scale = max(
            0.1, float(self.get_parameter("deployment_gait_phase_period_scale").value)
        )
        self.deployment_command_scale_mul = np.array([
            float(self.get_parameter("deployment_command_scale_x_mul").value),
            float(self.get_parameter("deployment_command_scale_y_mul").value),
            float(self.get_parameter("deployment_command_scale_yaw_mul").value),
        ], dtype=np.float32)
        if (
            not np.all(np.isfinite(self.deployment_command_scale_mul))
            or np.any(self.deployment_command_scale_mul < 0.0)
        ):
            raise ValueError("deployment command scale multipliers must be finite and non-negative")
        self.enable_cmd_smoothing = bool(self.get_parameter("enable_cmd_smoothing").value)
        self.max_cmd_x_rate_mps2 = float(self.get_parameter("max_cmd_x_rate_mps2").value)
        self.max_cmd_y_rate_mps2 = float(self.get_parameter("max_cmd_y_rate_mps2").value)
        self.max_cmd_yaw_rate_rad_s2 = float(
            self.get_parameter("max_cmd_yaw_rate_rad_s2").value
        )
        self.enable_cmd_limits = bool(self.get_parameter("enable_cmd_limits").value)
        self.cmd_min = np.array([
            float(self.get_parameter("cmd_min_x").value),
            float(self.get_parameter("cmd_min_y").value),
            float(self.get_parameter("cmd_min_yaw").value),
        ], dtype=np.float32)
        self.cmd_max = np.array([
            float(self.get_parameter("cmd_max_x").value),
            float(self.get_parameter("cmd_max_y").value),
            float(self.get_parameter("cmd_max_yaw").value),
        ], dtype=np.float32)
        if np.any(self.cmd_min > self.cmd_max):
            raise ValueError("command minimums must not exceed command maximums")
        self.require_cmd_vel = bool(self.get_parameter("require_cmd_vel").value)
        self.cmd_vel_timeout_sec = float(
            self.get_parameter("cmd_vel_timeout_sec").value
        )
        self.zero_cmd_inhibits_policy = bool(
            self.get_parameter("zero_cmd_inhibits_policy").value
        )
        self.enable_zero_cmd_stand_protection = bool(
            self.get_parameter("enable_zero_cmd_stand_protection").value
        )
        self.zero_cmd_stand_threshold = np.array([
            abs(float(self.get_parameter("zero_cmd_stand_x_threshold").value)),
            abs(float(self.get_parameter("zero_cmd_stand_y_threshold").value)),
            abs(float(self.get_parameter("zero_cmd_stand_yaw_threshold").value)),
        ], dtype=np.float32)
        self.enable_policy_action_cmd_gate = bool(
            self.get_parameter("enable_policy_action_cmd_gate").value
        )
        self.policy_action_cmd_gate_start_ratio = max(
            0.0, float(self.get_parameter("policy_action_cmd_gate_start_ratio").value)
        )
        self.policy_action_cmd_gate_full_ratio = max(
            self.policy_action_cmd_gate_start_ratio + 1.0e-6,
            float(self.get_parameter("policy_action_cmd_gate_full_ratio").value),
        )
        self.policy_action_cmd_gate_max_scale = float(
            np.clip(
                float(self.get_parameter("policy_action_cmd_gate_max_scale").value),
                0.0,
                1.0,
            )
        )
        self.reset_gait_phase_on_command_start = bool(
            self.get_parameter("reset_gait_phase_on_command_start").value
        )
        self.max_motor_age_ms = float(self.get_parameter("max_motor_age_ms").value)
        self.recheck_stale_motor_once = bool(
            self.get_parameter("recheck_stale_motor_once").value
        )
        self.motor_state_async = bool(self.get_parameter("motor_state_async").value)
        self.motor_state_poll_hz = float(self.get_parameter("motor_state_poll_hz").value)
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.print_only = bool(self.get_parameter("print_only").value)
        self.debug_print_arrays = bool(self.get_parameter("debug_print_arrays").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_csv_async = bool(self.get_parameter("debug_csv_async").value)
        self.debug_csv_queue_size = max(
            8, int(self.get_parameter("debug_csv_queue_size").value)
        )
        self.debug_csv_flush_every_n = max(
            1, int(self.get_parameter("debug_csv_flush_every_n").value)
        )
        self.debug_warn_action_abs = float(self.get_parameter("debug_warn_action_abs").value)
        self.debug_warn_error_rad = float(self.get_parameter("debug_warn_error_rad").value)
        self.debug_critical_error_rad = float(self.get_parameter("debug_critical_error_rad").value)
        self.stand_only = bool(self.get_parameter("stand_only").value)
        self.stand_only_duration_sec = float(
            self.get_parameter("stand_only_duration_sec").value
        )
        self.policy_enable = bool(self.get_parameter("policy_enable").value)
        self.expected_policy_task = str(
            self.get_parameter("expected_policy_task").value
        ).strip()
        self.expected_policy_sha256 = str(
            self.get_parameter("expected_policy_sha256").value
        ).strip().lower()
        self.joint_probe_enable = bool(self.get_parameter("joint_probe_enable").value)
        self.joint_probe_name = str(self.get_parameter("joint_probe_name").value)
        self.joint_probe_delta_rad = float(
            self.get_parameter("joint_probe_delta_rad").value
        )
        self.joint_probe_period_sec = float(
            self.get_parameter("joint_probe_period_sec").value
        )
        self.require_online = bool(self.get_parameter("require_online").value)

        self.startup_stand_first = bool(self.get_parameter("startup_stand_first").value)
        self.startup_stand_kp = float(self.get_parameter("startup_stand_kp").value)
        self.startup_stand_kd = float(self.get_parameter("startup_stand_kd").value)
        self.startup_stand_speed = float(self.get_parameter("startup_stand_speed").value)
        self.startup_stand_torque = float(self.get_parameter("startup_stand_torque").value)
        self.startup_stand_enable_first = bool(self.get_parameter("startup_stand_enable_first").value)
        self.startup_stand_stop_first = bool(self.get_parameter("startup_stand_stop_first").value)
        self.startup_stand_settle_sec = float(self.get_parameter("startup_stand_settle_sec").value)
        self.startup_stand_hold_current_sec = float(
            self.get_parameter("startup_stand_hold_current_sec").value
        )
        self.startup_stand_ramp_sec = float(
            self.get_parameter("startup_stand_ramp_sec").value
        )
        self.startup_stand_timeout_sec = float(
            self.get_parameter("startup_stand_timeout_sec").value
        )
        self.startup_stand_max_rate = self.make_joint_type_vector(
            float(self.get_parameter("startup_stand_max_rate_hip").value),
            float(self.get_parameter("startup_stand_max_rate_thigh").value),
            float(self.get_parameter("startup_stand_max_rate_calf").value),
        )
        self.startup_stand_max_step_rad = float(
            self.get_parameter("startup_stand_max_step_rad").value
        )
        self.startup_stand_ready_error_rad = float(
            self.get_parameter("startup_stand_ready_error_rad").value
        )
        self.startup_stand_stop_error_rad = float(
            self.get_parameter("startup_stand_stop_error_rad").value
        )
        self.startup_stand_stop_torque_nm = float(
            self.get_parameter("startup_stand_stop_torque_nm").value
        )
        self.enable_default_pose_check = bool(
            self.get_parameter("enable_default_pose_check").value
        )
        self.default_pose_check_sec = float(
            self.get_parameter("default_pose_check_sec").value
        )
        self.use_policy_default_as_stand_pose = bool(
            self.get_parameter("use_policy_default_as_stand_pose").value
        )
        self.stand_pose_source = str(self.get_parameter("stand_pose_source").value).strip().lower()
        if self.stand_pose_source not in ("policy_default", "legacy"):
            self.get_logger().warn(
                f"Unknown stand_pose_source={self.stand_pose_source!r}; using policy_default."
            )
            self.stand_pose_source = "policy_default"

        self.max_target_delta = float(self.get_parameter("max_target_delta").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.use_model_pd_gains = bool(
            self.get_parameter("use_model_pd_gains").value
        )
        self.model_kp_scale = max(
            0.0, float(self.get_parameter("model_kp_scale").value)
        )
        self.model_kd_scale = max(
            0.0, float(self.get_parameter("model_kd_scale").value)
        )
        self.send_kp_real = np.full(12, abs(self.send_kp), dtype=np.float32)
        self.send_kd_real = np.full(12, abs(self.send_kd), dtype=np.float32)
        self.model_position_error_limits_real = None
        self.pd_gain_source = "scalar_fallback"
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.enable_velocity_ff = bool(self.get_parameter("enable_velocity_ff").value)
        self.velocity_ff_scale = float(self.get_parameter("velocity_ff_scale").value)
        self.max_motor_vel_cmd_rad_s = float(
            self.get_parameter("max_motor_vel_cmd_rad_s").value
        )
        self.log_motor_vel_cmd = bool(self.get_parameter("log_motor_vel_cmd").value)
        self.enable_torque_error_limit = bool(
            self.get_parameter("enable_torque_error_limit").value
        )
        self.motor_torque_limit_nm = float(
            self.get_parameter("motor_torque_limit_nm").value
        )
        torque_limit_alias_nm = float(self.get_parameter("torque_limit_nm").value)
        if torque_limit_alias_nm >= 0.0:
            self.motor_torque_limit_nm = torque_limit_alias_nm
        self.torque_safety_ratio = float(
            self.get_parameter("torque_safety_ratio").value
        )
        self.torque_safety_budget_nm = float(
            self.get_parameter("torque_safety_budget_nm").value
        )
        self.expected_active_torque_budget_nm = float(
            self.get_parameter("expected_active_torque_budget_nm").value
        )
        active_torque_budget_nm = self.compute_torque_safety_budget_nm()
        if (
            self.expected_active_torque_budget_nm >= 0.0
            and abs(active_torque_budget_nm - self.expected_active_torque_budget_nm) > 1.0e-6
        ):
            raise RuntimeError(
                "active torque budget mismatch: "
                f"expected {self.expected_active_torque_budget_nm:.3f} Nm, "
                f"resolved {active_torque_budget_nm:.3f} Nm "
                f"(motor_torque_limit_nm={self.motor_torque_limit_nm:.3f}, "
                f"torque_safety_ratio={self.torque_safety_ratio:.3f}, "
                f"torque_safety_budget_nm={self.torque_safety_budget_nm:.3f})"
            )
        self.send_enable_first = bool(self.get_parameter("send_enable_first").value)
        self.send_stop_first = bool(self.get_parameter("send_stop_first").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.enable_rear_leg_posture_bias = bool(
            self.get_parameter("enable_rear_leg_posture_bias").value
        )
        self.rear_calf_extend_bias_policy_rad = float(
            self.get_parameter("rear_calf_extend_bias_policy_rad").value
        )
        self.rear_thigh_bias_policy_rad = float(
            self.get_parameter("rear_thigh_bias_policy_rad").value
        )

        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cmd_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.err_limit_mul_real = self.make_joint_type_vector(
            self.hip_err_limit_mul,
            self.thigh_err_limit_mul,
            self.calf_err_limit_mul,
        )
        self.target_rate_mul_real = self.make_joint_type_vector(
            self.hip_target_rate_mul,
            self.thigh_target_rate_mul,
            self.calf_target_rate_mul,
        )
        self.target_accel_mul_real = self.make_joint_type_vector(
            self.hip_target_accel_mul,
            self.thigh_target_accel_mul,
            self.calf_target_accel_mul,
        )
        self._last_cmd_log_time = 0.0
        self._last_cmd_receive_time = 0.0
        self._last_cmd_timeout_log_time = 0.0
        self._last_send_ok_log_time = 0.0
        self._last_control_summary_log_time = 0.0
        self._last_torque_limit_log_time = 0.0
        self._loop_rate_window_start = time.perf_counter()
        self._loop_rate_last_tick = None
        self._loop_period_samples = []
        self._policy_loop_count = 0
        self._policy_steps_since_last_csv = 0
        self._motor_send_count = 0
        self._motor_sends_since_last_csv = 0
        self._csv_cycle_sequence = 0
        self._last_command_sequence = 0
        self._last_send_start_time = ""
        self._last_send_end_time = ""
        self._last_board_a_send_time = ""
        self._last_board_b_send_time = ""
        self._last_policy_loop_dt_ms = 0.0
        self._last_policy_loop_hz = 0.0
        self._last_loop_overrun_ms = 0.0
        self._last_motor_send_time_perf = None
        self._last_motor_send_dt_ms = 0.0
        self._last_motor_send_hz = 0.0
        self._last_motor_state_poll_dt_ms = 0.0
        self._last_onnx_infer_dt_ms = 0.0
        self._last_state_age_ms = 0.0
        self._last_debug_print_time = 0.0
        self._last_debug_csv_time = 0.0
        self._smooth_target_real = None
        self._last_smoothing_time = None
        self._last_control_time = None
        self._warned_obs_dim_padding = False
        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_csv_queue = queue.Queue(maxsize=self.debug_csv_queue_size)
        self._debug_csv_stop = threading.Event()
        self._debug_csv_thread = None
        self._debug_csv_dropped = 0
        self._debug_csv_rows_since_flush = 0
        self.safe_target_limiter = SafeTargetLimiter()
        self.deploy_cpg = None
        self._last_q_cpg_policy_abs = np.zeros(12, dtype=np.float32)
        self._last_delta_q_rl_policy = np.zeros(12, dtype=np.float32)
        self._last_gait_offset_policy = np.zeros(12, dtype=np.float32)
        self._last_action_cmd_gate_scale = 1.0
        self._last_zero_cmd_stand_active = False
        self._walk_command_was_active = False
        self._last_cpg_info = {}
        self._stand_mapper = JointSemanticMapper()
        self.policy = None
        self.deployment_config = None
        if self.policy_enable and not self.stand_only and not self.joint_probe_enable:
            self.get_logger().info(f"Loading ONNX policy: {self.onnx_path}")
            self.policy = OnnxPolicyRunner(self.onnx_path)
            self.deployment_config = self.policy.deployment_config
            actual_sha256 = self.file_sha256(self.onnx_path)
            if self.expected_policy_sha256 and actual_sha256 != self.expected_policy_sha256:
                raise RuntimeError(
                    "ONNX SHA256 mismatch: "
                    f"expected {self.expected_policy_sha256}, got {actual_sha256}"
                )
            if self.expected_policy_task:
                actual_task = (
                    self.deployment_config.get("task", "")
                    if self.deployment_config is not None else ""
                )
                if actual_task != self.expected_policy_task:
                    raise RuntimeError(
                        "ONNX task mismatch: "
                        f"expected {self.expected_policy_task!r}, got {actual_task!r}"
                    )
            self.get_logger().info(
                f"[ONNX] identity verified task="
                f"{(self.deployment_config or {}).get('task', '<missing>')} "
                f"sha256={actual_sha256}"
            )
            if self.deployment_config is not None:
                self._stand_mapper.configure_policy_contract(
                    self.deployment_config["joint_names"],
                    self.deployment_config["default_joint_angles"],
                )
                control = self.deployment_config["control"]
                self.configure_policy_pd_gains(control, self._stand_mapper)
                exported_hz = 1.0 / (
                    float(control["sim_dt"]) * int(control["decimation"])
                )
                if abs(self.policy_hz - exported_hz) > 1.0e-3:
                    self.get_logger().warn(
                        f"policy_hz={self.policy_hz:.3f} overridden by ONNX contract "
                        f"({exported_hz:.3f} Hz)"
                    )
                self.policy_hz = exported_hz
                self.gait_phase_period = (
                    float(self.deployment_config["gait"]["period"])
                    * self.deployment_gait_phase_period_scale
                )
                if abs(self.deployment_gait_phase_period_scale - 1.0) > 1.0e-6:
                    self.get_logger().warn(
                        "deployment gait phase period scaled: "
                        f"model={float(self.deployment_config['gait']['period']):.3f}s "
                        f"scale={self.deployment_gait_phase_period_scale:.3f} "
                        f"active={self.gait_phase_period:.3f}s"
                    )
                if self.action_mode != "pure_rl":
                    self.get_logger().warn(
                        f"action_mode={self.action_mode!r} overridden with pure_rl; "
                        "the exported model already defines its gait reference"
                    )
                    self.action_mode = "pure_rl"
        self.http_session = requests.Session()
        self._mode_start_time = time.time()
        self._stand_only_done_logged = False
        self._joint_probe_last_state = None
        self._joint_probe_policy_index = None
        self._startup_stand_state = "complete"
        self._startup_stand_start_time = None
        self._startup_stand_start_real = None
        self._startup_stand_cmd_real = None
        self._startup_stand_target_real = None
        self._startup_stand_ready_since = None
        self._startup_stand_last_log_time = 0.0
        self._startup_stand_enable_sent = False

        if self.print_only:
            self.enable_send = False
            self.startup_stand_first = False
            self.send_enable_first = False
            self.send_stop_first = False
            self.startup_stand_enable_first = False
            self.startup_stand_stop_first = False
            self.get_logger().warn(
                "print_only=True: policy will infer and print arrays only. "
                "Motor command sending and startup stand are disabled."
            )

        # ============================================================
        # 可选：启动时先进入默认站立
        # ============================================================
        deferred_startup_stand = bool(
            self.startup_stand_first
            and self.policy_enable
            and not self.stand_only
            and not self.joint_probe_enable
        )
        immediate_diagnostic_stand = bool(self.stand_only or self.joint_probe_enable)
        if deferred_startup_stand:
            self._startup_stand_state = "waiting_feedback"
            self.print_stand_pose_source_comparison()
            self.get_logger().warn(
                "Startup stand will be ramped from live feedback before ONNX control."
            )
        elif immediate_diagnostic_stand:
            self.print_stand_pose_source_comparison()
            self.get_logger().warn(
                "sending DEFAULT_STAND_POSE before control starts."
            )
            ok = self.send_default_stand()
            if not ok:
                self.get_logger().error("Default stand failed. Continue with caution.")
            else:
                self.get_logger().info(
                    f"Default stand sent. Settling {self.startup_stand_settle_sec:.2f}s..."
                )
                time.sleep(self.startup_stand_settle_sec)

        # ============================================================
        # ObsBuilder：IMU + 电机反馈 + 36维 obs
        # ============================================================
        if self.policy is not None:
            obs_dim = self.policy.obs_dim
        else:
            obs_dim = 50
            self.get_logger().warn(
                f"policy disabled for diagnostic mode: "
                f"stand_only={self.stand_only}, "
                f"joint_probe_enable={self.joint_probe_enable}, "
                f"policy_enable={self.policy_enable}. ONNX will not run."
            )

        self.get_logger().info(
            f"Starting ObsBuilder36 with obs_dim={obs_dim}..."
        )
        self.obs_builder = ObsBuilder36(
            motor_base_url=self.motor_base_url,
            base_lin_vel_source=self.base_lin_vel_source,
            state_estimator_timeout_sec=self.state_estimator_timeout_sec,
            max_motor_age_ms=self.max_motor_age_ms,
            obs_dim=obs_dim,
            semantic_yaw_180=self.semantic_yaw_180,
            gait_phase_period=self.gait_phase_period,
            deployment_config=self.deployment_config,
            motor_state_async=self.motor_state_async,
            motor_state_poll_hz=self.motor_state_poll_hz,
        )
        self.obs_builder.gait_phase_period = self.gait_phase_period
        self.obs_builder.gait_phase_start_time = time.time()
        self.obs_builder.last_gait_phase = 0.0
        if np.any(np.abs(self.deployment_command_scale_mul - 1.0) > 1.0e-6):
            old_scale = self.obs_builder.command_scale.copy()
            self.obs_builder.command_scale = (
                self.obs_builder.command_scale * self.deployment_command_scale_mul
            ).astype(np.float32)
            self.get_logger().warn(
                "[OBS] deployment command scale override: "
                f"model={old_scale.tolist()} "
                f"mul={self.deployment_command_scale_mul.tolist()} "
                f"active={self.obs_builder.command_scale.tolist()}"
            )
        self.obs_builder.start()
        self.obs_builder.set_command(0.0, 0.0, 0.0)
        self.deploy_cpg = self.create_deploy_cpg()
        if immediate_diagnostic_stand and self.enable_default_pose_check:
            self.run_default_pose_check()
        self.setup_debug_csv()

        # ============================================================
        # ROS2 输入输出
        # ============================================================
        self.sub_cmd = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10,
        )

        self.sub_state_estimator = None
        if self.base_lin_vel_source == "estimator":
            self.sub_state_estimator = self.create_subscription(
                Float32MultiArray,
                "/mydog/state_estimator",
                self.state_estimator_callback,
                10,
            )
            self.get_logger().info(
                "base_lin_vel_source=estimator: subscribing /mydog/state_estimator "
                "for obs[0:9]."
            )

        self.pub_obs = self.create_publisher(Float32MultiArray, "/mydog/policy_obs", 10)
        self.pub_action_raw = self.create_publisher(Float32MultiArray, "/mydog/policy_action_raw", 10)
        self.pub_action = self.create_publisher(Float32MultiArray, "/mydog/policy_action", 10)
        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/target_real", 10)

        period = 1.0 / self.policy_hz
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f"mydog_policy_node started. "
            f"hz={self.policy_hz}, "
            f"enable_send={self.enable_send}, "
            f"print_only={self.print_only}, "
            f"debug_print_arrays={self.debug_print_arrays}, "
            f"startup_stand_first={self.startup_stand_first}, "
            f"stand_pose_source={self.stand_pose_source}, "
            f"stand_only={self.stand_only}, "
            f"joint_probe_enable={self.joint_probe_enable}, "
            f"joint_probe_name={self.joint_probe_name}, "
            f"policy_enable={self.policy_enable}, "
            f"base_lin_vel_source={self.base_lin_vel_source}, "
            f"state_estimator_timeout_sec={self.state_estimator_timeout_sec:.2f}, "
            f"action_scale={self.action_scale}, "
            f"front_action_scale_mul={self.front_action_scale_mul:.2f}, "
            f"rear_action_scale_mul={self.rear_action_scale_mul:.2f}, "
            f"hip/thigh/calf_mul="
            f"{self.hip_action_scale_mul:.2f}/"
            f"{self.thigh_action_scale_mul:.2f}/"
            f"{self.calf_action_scale_mul:.2f}, "
            f"rear_thigh/calf_mul="
            f"{self.rear_thigh_action_scale_mul:.2f}/"
            f"{self.rear_calf_action_scale_mul:.2f}, "
            f"thigh_action_sign={self.thigh_action_sign:+.1f}, "
            f"action_leg_yaw_180={self.action_leg_yaw_180}, "
            f"semantic_yaw_180={self.semantic_yaw_180}, "
            f"action_mode={self.action_mode}, "
            f"cpg_gait={self.cpg_gait}, "
            f"cpg_freq={self.cpg_freq_min:.2f}-{self.cpg_freq_max:.2f}, "
            f"cpg_k_freq={self.cpg_k_freq:.2f}, "
            f"cpg_standing_cmd_threshold={self.cpg_standing_cmd_threshold:.3f}, "
            f"cpg_amp hip/thigh/calf_lift/stance="
            f"{self.cpg_hip_amp:.3f}/"
            f"{self.cpg_thigh_amp:.3f}/"
            f"{self.cpg_calf_lift_amp:.3f}/"
            f"{self.cpg_stance_calf_amp:.3f}, "
            f"hip_balance enabled={self.cpg_enable_hip_balance}, "
            f"stance/swing/max="
            f"{self.cpg_hip_stance_widen_amp:.3f}/"
            f"{self.cpg_hip_swing_relax_amp:.3f}/"
            f"{self.cpg_hip_balance_max_abs:.3f}, "
            f"signs={self.cpg_hip_balance_signs.tolist()}, "
            f"cpg_zero_residual_when_standing={self.cpg_zero_residual_when_standing}, "
            f"hip_gate={self.enable_phase_aware_hip_gate} "
            f"stance_min={self.hip_gate_stance_min_outward:.3f} "
            f"swing_max={self.hip_gate_swing_max_outward:.3f}, "
            f"residual_limit hip/thigh/calf="
            f"{self.residual_limit_hip:.3f}/"
            f"{self.residual_limit_thigh:.3f}/"
            f"{self.residual_limit_calf:.3f}, "
            f"gait_phase_period={self.gait_phase_period:.3f}, "
            f"clip_action={self.clip_action}, "
            f"action_clip_limit={self.action_clip_limit:.2f}, "
            f"max_target_rate={self.max_target_rate_rad_s:.3f}, "
            f"max_target_accel={self.max_target_accel_rad_s2:.3f}, "
            f"err_limit_safety_factor={self.err_limit_safety_factor:.3f}, "
            f"err_mul={self.hip_err_limit_mul:.2f}/"
            f"{self.thigh_err_limit_mul:.2f}/"
            f"{self.calf_err_limit_mul:.2f}, "
            f"rate_mul={self.hip_target_rate_mul:.2f}/"
            f"{self.thigh_target_rate_mul:.2f}/"
            f"{self.calf_target_rate_mul:.2f}, "
            f"accel_mul={self.hip_target_accel_mul:.2f}/"
            f"{self.thigh_target_accel_mul:.2f}/"
            f"{self.calf_target_accel_mul:.2f}, "
            f"pd_gain_source={self.pd_gain_source}, "
            f"model_kp/kd_scale={self.model_kp_scale:.2f}/"
            f"{self.model_kd_scale:.2f}, "
            f"kp_real={self.send_kp_real.tolist()}, "
            f"kd_real={self.send_kd_real.tolist()}, "
            f"torque_limit_nm={self.motor_torque_limit_nm:.2f}, "
            f"torque_safety_ratio={self.torque_safety_ratio:.2f}, "
            f"torque_safety_budget_nm={self.torque_safety_budget_nm:.2f}, "
            f"active_torque_budget_nm={self.compute_torque_safety_budget_nm():.2f}, "
            f"velocity_ff={self.enable_velocity_ff}, "
            f"velocity_ff_scale={self.velocity_ff_scale:.3f}, "
            f"max_motor_vel_cmd={self.max_motor_vel_cmd_rad_s:.3f}, "
            f"rear_leg_bias={self.enable_rear_leg_posture_bias}, "
            f"rear_thigh_bias={self.rear_thigh_bias_policy_rad:+.3f}, "
            f"rear_calf_extend_bias={self.rear_calf_extend_bias_policy_rad:+.3f}, "
            f"cmd_smoothing={self.enable_cmd_smoothing}"
        )

    def cmd_callback(self, msg: Twist):
        now = time.time()
        was_stale = not self.command_is_fresh(now)
        requested = np.array([
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
        ], dtype=np.float32)
        if not np.all(np.isfinite(requested)):
            self.get_logger().error("[SAFE] rejected non-finite /cmd_vel")
            return
        if self.zero_cmd_inhibits_policy and np.all(np.abs(requested) <= 1.0e-6):
            self._last_cmd_receive_time = 0.0
            self.cmd_target.fill(0.0)
            self.cmd.fill(0.0)
            self.obs_builder.set_command(0.0, 0.0, 0.0)
            self.get_logger().warn(
                "[SAFE] zero /cmd_vel received; policy motor targets are inhibited"
            )
            return
        if self.enable_cmd_limits:
            requested = np.clip(requested, self.cmd_min, self.cmd_max)
        self._last_cmd_receive_time = now
        self.cmd_target[:] = requested

        if not self.enable_cmd_smoothing:
            self.cmd[:] = self.cmd_target
            self.obs_builder.set_command(self.cmd[0], self.cmd[1], self.cmd[2])

        if was_stale and hasattr(self, "obs_builder"):
            self.obs_builder.gait_phase_start_time = now
            self.obs_builder.last_gait_phase = 0.0
            self.obs_builder.set_last_action(np.zeros(12, dtype=np.float32))
        if now - self._last_cmd_log_time > 1.0:
            self._last_cmd_log_time = now
            self.get_logger().info(
                f"[CMD] received /cmd_vel target={self.cmd_target.tolist()} "
                f"smoothed={self.cmd.tolist()}"
            )

    def command_is_fresh(self, now=None):
        if not self.require_cmd_vel:
            return True
        now = time.time() if now is None else float(now)
        return (
            self._last_cmd_receive_time > 0.0
            and now - self._last_cmd_receive_time <= self.cmd_vel_timeout_sec
        )

    @staticmethod
    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def make_joint_type_vector(hip_value, thigh_value, calf_value):
        per_leg = np.array(
            [hip_value, thigh_value, calf_value],
            dtype=np.float32,
        )
        return np.tile(per_leg, 4).astype(np.float32)

    @staticmethod
    def parse_hip_balance_signs(value):
        if isinstance(value, str):
            parts = [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
            signs = np.asarray([float(x) for x in parts], dtype=np.float32)
        else:
            signs = np.asarray(value, dtype=np.float32).reshape(-1)
        if signs.shape[0] != 4:
            raise ValueError(
                "cpg_hip_balance_signs must contain four values in FR,FL,RR,RL order, "
                f"got {value!r}"
            )
        return signs.astype(np.float32)

    def configure_policy_pd_gains(self, control: dict, mapper: JointSemanticMapper):
        """Load per-joint PD gains from the ONNX contract in motor order."""
        if not self.use_model_pd_gains:
            self.pd_gain_source = "scalar_override"
            return

        try:
            kp_policy = np.asarray(control["stiffness"], dtype=np.float32).reshape(-1)
            kd_policy = np.asarray(control["damping"], dtype=np.float32).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "ONNX deployment contract must provide 12-element stiffness/damping"
            ) from exc

        if (
            kp_policy.shape[0] != 12
            or kd_policy.shape[0] != 12
            or not np.all(np.isfinite(kp_policy))
            or not np.all(np.isfinite(kd_policy))
            or np.any(kp_policy <= 0.0)
            or np.any(kd_policy < 0.0)
        ):
            raise RuntimeError(
                "ONNX stiffness must be 12 positive finite values and damping "
                "must be 12 non-negative finite values"
            )

        self.send_kp_real = (
            mapper.policy_values_to_real_order(kp_policy) * self.model_kp_scale
        ).astype(np.float32)
        self.send_kd_real = (
            mapper.policy_values_to_real_order(kd_policy) * self.model_kd_scale
        ).astype(np.float32)
        position_error_limits = control.get("position_error_limits")
        if position_error_limits is not None:
            limits_policy = np.asarray(position_error_limits, dtype=np.float32).reshape(-1)
            if (
                limits_policy.shape[0] != 12
                or not np.all(np.isfinite(limits_policy))
                or np.any(limits_policy <= 0.0)
            ):
                raise RuntimeError(
                    "ONNX position_error_limits must contain 12 positive finite values"
                )
            self.model_position_error_limits_real = mapper.policy_values_to_real_order(
                limits_policy
            ).astype(np.float32)
        if np.any(self.send_kp_real <= 0.0):
            raise RuntimeError("model_kp_scale must keep all effective Kp values positive")
        self.pd_gain_source = "onnx_contract"

    def compute_torque_safety_budget_nm(self):
        if self.torque_safety_budget_nm >= 0.0:
            return max(0.0, float(self.torque_safety_budget_nm))
        return max(
            0.0,
            float(self.motor_torque_limit_nm) * float(self.torque_safety_ratio),
        )

    def compute_motor_velocity_command(self, smoothing_info: dict):
        qdot_cmd = np.asarray(
            smoothing_info.get("qdot_cmd", np.zeros(12, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(12)

        if not self.enable_velocity_ff:
            return np.zeros(12, dtype=np.float32)

        max_vel = abs(float(self.max_motor_vel_cmd_rad_s))
        motor_vel_cmd = float(self.velocity_ff_scale) * qdot_cmd
        if max_vel > 0.0:
            motor_vel_cmd = np.clip(motor_vel_cmd, -max_vel, max_vel)
        return motor_vel_cmd.astype(np.float32)

    def zero_cmd_stand_active(self) -> bool:
        if not self.enable_zero_cmd_stand_protection:
            return False
        return bool(np.all(np.abs(self.cmd) < self.zero_cmd_stand_threshold))

    def command_gate_metric(self) -> float:
        cmd_abs = np.abs(np.asarray(self.cmd, dtype=np.float32).reshape(3))
        envelope = np.maximum(np.abs(self.cmd_min), np.abs(self.cmd_max))
        envelope = np.maximum(envelope, self.zero_cmd_stand_threshold)
        envelope = np.maximum(envelope, 1.0e-6)
        return float(np.max(cmd_abs / envelope))

    def policy_action_command_gate_scale(self) -> float:
        if not self.enable_policy_action_cmd_gate:
            return 1.0
        if self.zero_cmd_stand_active():
            return 0.0
        metric = self.command_gate_metric()
        start = self.policy_action_cmd_gate_start_ratio
        full = self.policy_action_cmd_gate_full_ratio
        if metric <= start:
            return 0.0
        if metric >= full:
            return self.policy_action_cmd_gate_max_scale
        x = (metric - start) / (full - start)
        smooth = float(x * x * (3.0 - 2.0 * x))
        return self.policy_action_cmd_gate_max_scale * smooth

    def walking_command_active(self) -> bool:
        if self.zero_cmd_stand_active():
            return False
        return self.command_gate_metric() > self.policy_action_cmd_gate_start_ratio

    def maybe_reset_gait_phase_on_command_start(self):
        active = self.walking_command_active()
        if self.reset_gait_phase_on_command_start and active and not self._walk_command_was_active:
            self.obs_builder.gait_phase_start_time = time.time()
            self.obs_builder.last_gait_phase = 0.0
            self.obs_builder.set_last_action(np.zeros(12, dtype=np.float32))
            self.get_logger().warn(
                "[GAIT] command start detected: reset gait phase and last_action"
            )
        self._walk_command_was_active = active

    def create_deploy_cpg(self):
        mapper = self.obs_builder.mapper
        return DeployJointCPG(
            default_joint_angle=mapper.default_joint_angle,
            lower_limit=mapper.policy_lower_limit,
            upper_limit=mapper.policy_upper_limit,
            policy_hz=self.policy_hz,
            gait=self.cpg_gait,
            freq_min=self.cpg_freq_min,
            freq_max=self.cpg_freq_max,
            k_freq=self.cpg_k_freq,
            standing_cmd_threshold=self.cpg_standing_cmd_threshold,
            duty_factor=self.cpg_duty_factor,
            hip_amp=self.cpg_hip_amp,
            thigh_amp=self.cpg_thigh_amp,
            calf_lift_amp=self.cpg_calf_lift_amp,
            stance_calf_amp=self.cpg_stance_calf_amp,
            stride_sign=self.cpg_stride_sign,
            enable_hip_balance=self.cpg_enable_hip_balance,
            hip_stance_widen_amp=self.cpg_hip_stance_widen_amp,
            hip_swing_relax_amp=self.cpg_hip_swing_relax_amp,
            hip_balance_signs=self.cpg_hip_balance_signs,
            hip_balance_use_stance_mask=self.cpg_hip_balance_use_stance_mask,
            hip_balance_smooth_shape=self.cpg_hip_balance_smooth_shape,
            hip_balance_max_abs=self.cpg_hip_balance_max_abs,
            residual_limit_hip=self.residual_limit_hip,
            residual_limit_thigh=self.residual_limit_thigh,
            residual_limit_calf=self.residual_limit_calf,
            enable_phase_aware_hip_gate=self.enable_phase_aware_hip_gate,
            hip_gate_stance_min_outward=self.hip_gate_stance_min_outward,
            hip_gate_swing_max_outward=self.hip_gate_swing_max_outward,
            hip_gate_side_signs=self.hip_gate_side_signs,
        )

    def action_to_policy_target_abs(self, action_policy: np.ndarray) -> np.ndarray:
        mapper = self.obs_builder.mapper
        action_policy = np.asarray(action_policy, dtype=np.float32).reshape(12)
        if self.deployment_config is not None:
            action_scale = np.asarray(
                self.deployment_config["control"]["action_scale"], dtype=np.float32
            )
        else:
            action_scale = np.asarray(self.action_scale, dtype=np.float32)
        if action_scale.shape == ():
            action_scale = np.full(12, float(action_scale), dtype=np.float32)
        else:
            action_scale = action_scale.reshape(12)

        target_policy_abs = mapper.default_joint_angle + action_scale * action_policy
        gait_offset = np.zeros(12, dtype=np.float32)
        if self.deployment_config is not None:
            gait_offset = self.deployment_gait_offset(
                self.obs_builder.last_gait_phase
            )
            target_policy_abs += gait_offset
        self._last_gait_offset_policy = np.asarray(gait_offset, dtype=np.float32).reshape(12).copy()
        return np.clip(
            target_policy_abs,
            mapper.policy_lower_limit,
            mapper.policy_upper_limit,
        ).astype(np.float32)

    def deployment_gait_offset(self, phase: float) -> np.ndarray:
        """Reproduce the gait reference used by training and sim2sim."""
        if self.deployment_config is None:
            return np.zeros(12, dtype=np.float32)
        gait = self.deployment_config["gait"]
        result = np.zeros(12, dtype=np.float32)
        gait_scale = 1.0
        if bool(gait.get("gate_with_command", False)):
            gait_scale = float(self._last_action_cmd_gate_scale)
        if gait_scale <= 1.0e-6:
            return result
        stance_ratio = float(gait["stance_ratio"])
        offsets = gait["phase_offsets"]
        for i, name in enumerate(self.deployment_config["joint_names"]):
            leg = name[:2]
            leg_phase = (float(phase) + float(offsets[leg])) % 1.0
            swing = np.clip(
                (leg_phase - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0
            )
            smooth = swing * swing * (3.0 - 2.0 * swing)
            if "thigh" in name:
                if leg_phase < stance_ratio:
                    profile = -1.0 + 2.0 * np.clip(
                        leg_phase / stance_ratio, 0.0, 1.0
                    )
                else:
                    profile = 1.0 - 2.0 * smooth
                result[i] = float(gait["thigh_amplitude"]) * profile
            elif "calf" in name and leg_phase >= stance_ratio:
                result[i] = float(gait["calf_amplitude"]) * np.sin(np.pi * smooth)
        return (result * gait_scale).astype(np.float32)

    def action_to_policy_target_abs_by_mode(
        self,
        action_policy: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, dict]:
        mapper = self.obs_builder.mapper
        action_policy = np.asarray(action_policy, dtype=np.float32).reshape(12)
        q_cpg = mapper.default_joint_angle.astype(np.float32).copy()
        delta_q_rl = np.zeros(12, dtype=np.float32)
        cpg_info = {}

        if self.action_mode == "pure_rl":
            target_policy_abs = self.action_to_policy_target_abs(action_policy)
        else:
            if self.deploy_cpg is None:
                self.deploy_cpg = self.create_deploy_cpg()
            q_cpg = self.deploy_cpg.update(self.cmd, dt=dt)
            cpg_info = self.deploy_cpg.info()
            if self.action_mode == "cpg_only":
                target_policy_abs = q_cpg.copy()
            else:
                residual_limits = np.asarray(
                    cpg_info.get("residual_limits", self.deploy_cpg.residual_limits),
                    dtype=np.float32,
                ).reshape(12)
                if (
                    self.cpg_zero_residual_when_standing
                    and float(cpg_info.get("frequency", 0.0)) <= 1.0e-6
                ):
                    delta_q_rl = np.zeros(12, dtype=np.float32)
                else:
                    delta_q_rl = action_policy * residual_limits
                target_policy_abs = q_cpg + delta_q_rl
                target_policy_abs, delta_q_rl = self.deploy_cpg.apply_phase_aware_hip_gate(
                    target_policy_abs,
                    q_cpg,
                )
                target_policy_abs = np.clip(
                    target_policy_abs,
                    mapper.policy_lower_limit,
                    mapper.policy_upper_limit,
                ).astype(np.float32)
                cpg_info = self.deploy_cpg.info()

        self._last_q_cpg_policy_abs = np.asarray(q_cpg, dtype=np.float32).reshape(12).copy()
        self._last_delta_q_rl_policy = np.asarray(delta_q_rl, dtype=np.float32).reshape(12).copy()
        self._last_cpg_info = cpg_info
        return target_policy_abs.astype(np.float32), {
            "action_mode": self.action_mode,
            "q_cpg_policy_abs": self._last_q_cpg_policy_abs.copy(),
            "delta_q_rl_policy": self._last_delta_q_rl_policy.copy(),
            **cpg_info,
        }

    @staticmethod
    def leg_prefix_from_joint_name(joint_name: str) -> str:
        name = str(joint_name)
        if name.endswith("_joint"):
            name = name[:-6]
        return name.split("_", 1)[0]

    def apply_rear_leg_posture_bias(
        self,
        target_policy_abs: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        target = np.asarray(target_policy_abs, dtype=np.float32).reshape(12).copy()
        bias_vec = np.zeros(12, dtype=np.float32)

        if self.enable_rear_leg_posture_bias:
            thigh_bias = float(self.rear_thigh_bias_policy_rad)
            calf_bias = float(self.rear_calf_extend_bias_policy_rad)
            mapper = self.obs_builder.mapper
            for i, name in enumerate(mapper.policy_joint_names):
                leg = self.leg_prefix_from_joint_name(name)
                if leg not in ("RL", "RR"):
                    continue
                if "thigh" in name:
                    bias_vec[i] = thigh_bias
                elif "calf" in name:
                    bias_vec[i] = calf_bias
            target = target + bias_vec
            mapper = self.obs_builder.mapper
            target = np.clip(
                target,
                mapper.policy_lower_limit,
                mapper.policy_upper_limit,
            ).astype(np.float32)

        return target, {
            "enabled": bool(self.enable_rear_leg_posture_bias),
            "bias_vec_policy": bias_vec,
            "rear_thigh_bias_policy_rad": float(self.rear_thigh_bias_policy_rad),
            "rear_calf_extend_bias_policy_rad": float(
                self.rear_calf_extend_bias_policy_rad
            ),
        }

    def run_default_pose_check(self):
        duration = max(0.0, float(self.default_pose_check_sec))
        mapper = self.obs_builder.mapper
        samples = []
        deadline = time.time() + duration

        self.get_logger().warn(
            f"[DEFAULT_POSE_CHECK] collecting motor feedback for {duration:.2f}s"
        )
        while True:
            try:
                snapshot = self.obs_builder.motor.get_latest()
                if snapshot.valid:
                    samples.append(snapshot.q_real.copy())
            except Exception as e:
                self.get_logger().warn(f"[DEFAULT_POSE_CHECK] read failed: {e}")

            if time.time() >= deadline:
                break
            time.sleep(0.05)

        if not samples:
            self.get_logger().warn("[DEFAULT_POSE_CHECK] no valid motor samples")
            return

        q_real_mean = np.mean(np.asarray(samples, dtype=np.float32), axis=0)
        self.print_default_pose_alignment_report(
            q_real_mean,
            q_target_real=self.default_stand_target_real_order(),
        )

    def get_stand_mapper(self):
        if hasattr(self, "obs_builder"):
            return self.obs_builder.mapper
        return self._stand_mapper

    @staticmethod
    def legacy_default_stand_pose_real_order():
        by_id = {int(mid): float(pos) for mid, pos in DEFAULT_STAND_POSE_REAL_ORDER}
        motor_ids = [0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x31, 0x32, 0x33, 0x41, 0x42, 0x43]
        return np.asarray([by_id[mid] for mid in motor_ids], dtype=np.float32)

    def build_default_stand_pose_real_from_policy_default(self):
        mapper = self.get_stand_mapper()
        return mapper.real_default_pose_for_motor_order().astype(np.float32)

    def default_stand_target_real_order(self):
        if self.stand_pose_source == "legacy":
            return self.legacy_default_stand_pose_real_order()
        return self.build_default_stand_pose_real_from_policy_default()

    def print_stand_pose_source_comparison(self):
        mapper = self.get_stand_mapper()
        motor_ids = mapper.get_real_motor_ids()
        real_names = mapper.real_joint_names
        legacy = self.legacy_default_stand_pose_real_order()
        policy_default = self.build_default_stand_pose_real_from_policy_default()
        diff = legacy - policy_default

        self.get_logger().warn(
            f"[STAND_POSE_SOURCE] active={self.stand_pose_source}; "
            "legacy minus policy_default shown below"
        )
        for i, (mid, name) in enumerate(zip(motor_ids, real_names)):
            message = (
                f"[STAND_POSE_SOURCE] real[{i:02d}] motor_id=0x{int(mid):02X} "
                f"{name:16s} legacy={float(legacy[i]):+.4f} "
                f"policy_default={float(policy_default[i]):+.4f} "
                f"diff={float(diff[i]):+.4f}"
            )
            if abs(float(diff[i])) > 0.10:
                self.get_logger().error(message)
            elif abs(float(diff[i])) > 0.05:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)

    def print_default_pose_alignment_report(self, q_current_real, q_target_real=None):
        mapper = self.get_stand_mapper()
        q_current_real = np.asarray(q_current_real, dtype=np.float32).reshape(12)
        if q_target_real is None:
            q_target_real = self.default_stand_target_real_order()
        q_target_real = np.asarray(q_target_real, dtype=np.float32).reshape(12)
        q_policy_default_real = self.build_default_stand_pose_real_from_policy_default()
        q_current_policy_abs, _ = mapper.real_to_policy_abs_q_dq(
            q_current_real,
            np.zeros(12, dtype=np.float32),
        )
        q_default_policy = mapper.default_joint_angle.astype(np.float32)
        diff = q_current_policy_abs - q_default_policy
        abs_diff = np.abs(diff)
        motor_ids = mapper.get_real_motor_ids()
        real_to_policy_index = np.zeros(12, dtype=np.int64)
        real_to_policy_index[mapper.policy_to_real_index] = np.arange(12, dtype=np.int64)

        self.get_logger().warn(
            "[DEFAULT_POSE_ALIGNMENT] policy order = "
            f"{mapper.get_policy_joint_names()}"
        )
        for policy_i, policy_name in enumerate(mapper.policy_joint_names):
            real_i = int(mapper.policy_to_real_index[policy_i])
            message = (
                f"[DEFAULT_POSE_ALIGNMENT] "
                f"joint_name={mapper.real_joint_names[real_i]:16s} "
                f"policy_joint_name={policy_name:16s} "
                f"q_default_policy={float(q_default_policy[policy_i]):+.4f} "
                f"q_current_policy_abs={float(q_current_policy_abs[policy_i]):+.4f} "
                f"q_current_minus_default_policy={float(diff[policy_i]):+.4f} "
                f"q_target_real={float(q_target_real[real_i]):+.4f} "
                f"q_current_real={float(q_current_real[real_i]):+.4f} "
                f"target_real_minus_policy_default_real="
                f"{float(q_target_real[real_i] - q_policy_default_real[real_i]):+.4f} "
                f"motor_id=0x{int(motor_ids[real_i]):02X}"
            )
            if abs_diff[policy_i] > 0.10:
                self.get_logger().error(message)
            elif abs_diff[policy_i] > 0.05:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)

        joint_type_indices = {
            "hip": [0, 3, 6, 9],
            "thigh": [1, 4, 7, 10],
            "calf": [2, 5, 8, 11],
        }
        for joint_type, indices in joint_type_indices.items():
            mean_abs = float(np.mean(abs_diff[indices]))
            message = (
                f"[DEFAULT_POSE_ALIGNMENT] joint_type={joint_type:5s} "
                f"mean_abs_error={mean_abs:.4f} rad"
            )
            if mean_abs > 0.10:
                self.get_logger().error(message)
            elif mean_abs > 0.05:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)

        leg_indices = {
            "FR": [0, 1, 2],
            "FL": [3, 4, 5],
            "RR": [6, 7, 8],
            "RL": [9, 10, 11],
        }
        for leg, indices in leg_indices.items():
            mean_abs = float(np.mean(abs_diff[indices]))
            message = f"[DEFAULT_POSE_ALIGNMENT] leg={leg} mean_abs_error={mean_abs:.4f} rad"
            if mean_abs > 0.10:
                self.get_logger().error(message)
            elif mean_abs > 0.05:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)

    def policy_index_for_joint_name(self, joint_name: str):
        name = str(joint_name).strip()
        mapper = self.obs_builder.mapper
        if name in mapper.policy_joint_names:
            return int(mapper.policy_joint_names.index(name))
        if name in mapper.real_joint_names:
            real_i = int(mapper.real_joint_names.index(name))
            matches = np.where(mapper.policy_to_real_index == real_i)[0]
            if matches.size:
                return int(matches[0])
        return None

    def joint_probe_target(self, elapsed: float):
        mapper = self.obs_builder.mapper
        policy_i = self._joint_probe_policy_index
        if policy_i is None:
            policy_i = self.policy_index_for_joint_name(self.joint_probe_name)
            self._joint_probe_policy_index = policy_i
        if policy_i is None:
            self.get_logger().error(
                f"[JOINT_PROBE] unknown joint_probe_name={self.joint_probe_name!r}"
            )
            return self.default_stand_target_real_order(), 0.0, "invalid"

        period = max(0.1, float(self.joint_probe_period_sec))
        phase = int(elapsed / period) % 4
        delta_seq = [0.0, float(self.joint_probe_delta_rad), 0.0, -float(self.joint_probe_delta_rad)]
        delta = delta_seq[phase]
        state_name = ["base", "plus", "base", "minus"][phase]

        q_base_real = self.default_stand_target_real_order()
        q_base_policy_abs, _ = mapper.real_to_policy_abs_q_dq(
            q_base_real,
            np.zeros(12, dtype=np.float32),
        )
        q_target_policy_abs = q_base_policy_abs.copy()
        q_target_policy_abs[policy_i] += delta
        q_target_real = mapper.policy_target_to_real_target(
            q_target_policy_abs,
            clamp=True,
        )

        if state_name != self._joint_probe_last_state:
            self._joint_probe_last_state = state_name
            self.get_logger().warn(
                f"[JOINT_PROBE] joint={mapper.policy_joint_names[policy_i]} "
                f"state={state_name} delta_policy={delta:+.4f} rad"
            )

        return q_target_real.astype(np.float32), delta, state_name

    def get_control_dt(self):
        now = time.time()
        nominal_dt = 1.0 / max(self.policy_hz, 1e-3)
        if self.target_smoothing_fixed_dt or self._last_control_time is None:
            dt = nominal_dt
        else:
            dt = now - self._last_control_time
            if dt <= 0.0:
                dt = nominal_dt
            else:
                dt = min(dt, nominal_dt * 2.0)
        self._last_control_time = now
        return dt

    def update_smoothed_command(self, dt: float):
        if self.enable_cmd_smoothing:
            rates = np.array(
                [
                    abs(self.max_cmd_x_rate_mps2),
                    abs(self.max_cmd_y_rate_mps2),
                    abs(self.max_cmd_yaw_rate_rad_s2),
                ],
                dtype=np.float32,
            )
            max_step = rates * max(float(dt), 1e-4)
            delta = np.asarray(self.cmd_target - self.cmd, dtype=np.float32)
            self.cmd[:] = self.cmd + np.clip(delta, -max_step, max_step)
        else:
            self.cmd[:] = self.cmd_target

        self.obs_builder.set_command(self.cmd[0], self.cmd[1], self.cmd[2])

    def state_estimator_callback(self, msg: Float32MultiArray):
        try:
            self.obs_builder.set_state_estimator(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Invalid /mydog/state_estimator message: {e}")

    @staticmethod
    def smootherstep01(value):
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    def handle_startup_stand(self, obs, info: dict, dt: float, max_age: float):
        """Move from live motor feedback to the ONNX default before inference."""
        now = time.time()
        current = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        torque = np.asarray(info.get("torque_real", np.zeros(12)), dtype=np.float32)

        if self._startup_stand_state == "fault":
            return

        if self._startup_stand_state == "waiting_feedback":
            self._startup_stand_start_time = now
            self._startup_stand_start_real = current.copy()
            self._startup_stand_cmd_real = current.copy()
            self._startup_stand_target_real = self.default_stand_target_real_order()
            self._startup_stand_ready_since = None
            self._startup_stand_state = "hold_current"
            max_delta = float(
                np.max(np.abs(self._startup_stand_target_real - current))
            )
            self.get_logger().warn(
                "[STARTUP_STAND] captured live pose; ONNX inference is locked | "
                f"max_transition={max_delta:.3f}rad "
                f"hold={self.startup_stand_hold_current_sec:.1f}s "
                f"ramp={self.startup_stand_ramp_sec:.1f}s "
                f"send={self.enable_send}"
            )

        elapsed = now - self._startup_stand_start_time
        if elapsed > self.startup_stand_timeout_sec:
            self._startup_stand_state = "fault"
            self.get_logger().error(
                "[STARTUP_STAND][FAULT] timeout before reaching ONNX default; "
                "motor targets stopped. Keep support engaged."
            )
            return

        hold = max(0.0, self.startup_stand_hold_current_sec)
        ramp = max(0.1, self.startup_stand_ramp_sec)
        if elapsed <= hold:
            desired = self._startup_stand_start_real.copy()
            alpha = 0.0
            stage = "hold_current"
        else:
            alpha = self.smootherstep01((elapsed - hold) / ramp)
            desired = (
                self._startup_stand_start_real
                + alpha
                * (self._startup_stand_target_real - self._startup_stand_start_real)
            ).astype(np.float32)
            stage = "ramp" if alpha < 1.0 else "settle"

        max_step = np.maximum(self.startup_stand_max_rate, 0.0) * max(dt, 1.0e-4)
        if self.startup_stand_max_step_rad > 0.0:
            max_step = np.minimum(max_step, self.startup_stand_max_step_rad)
        step = np.clip(
            desired - self._startup_stand_cmd_real,
            -max_step,
            max_step,
        )
        self._startup_stand_cmd_real = (
            self._startup_stand_cmd_real + step
        ).astype(np.float32)

        # Keep the low-gain transition inside the same real-motor torque
        # budget used by policy control: |Kp*e - Kd*dq| <= budget.
        torque_budget = self.compute_torque_safety_budget_nm()
        kd_load = abs(self.startup_stand_kd) * np.abs(current_dq)
        available = np.maximum(0.0, torque_budget - kd_load)
        position_error_limit = available / max(abs(self.startup_stand_kp), 1.0e-6)
        self._startup_stand_cmd_real = (
            current
            + np.clip(
                self._startup_stand_cmd_real - current,
                -position_error_limit,
                position_error_limit,
            )
        ).astype(np.float32)

        tracking_error = self._startup_stand_cmd_real - current
        tracking_error_max = float(np.max(np.abs(tracking_error)))
        torque_abs_max = float(np.max(np.abs(torque)))
        target_error_max = float(
            np.max(np.abs(self._startup_stand_target_real - current))
        )

        if tracking_error_max > self.startup_stand_stop_error_rad:
            self._startup_stand_state = "fault"
            self.get_logger().error(
                "[STARTUP_STAND][FAULT] feedback stopped following ramp | "
                f"tracking_error={tracking_error_max:.3f}rad > "
                f"{self.startup_stand_stop_error_rad:.3f}rad"
            )
            return
        if torque_abs_max > self.startup_stand_stop_torque_nm:
            self._startup_stand_state = "fault"
            self.get_logger().error(
                "[STARTUP_STAND][FAULT] measured torque too high | "
                f"max={torque_abs_max:.2f}Nm > "
                f"{self.startup_stand_stop_torque_nm:.2f}Nm"
            )
            return

        self.publish_array(self.pub_target, self._startup_stand_cmd_real)
        sent = False
        if self.enable_send:
            sent = self.send_startup_stand_target(self._startup_stand_cmd_real)

        ramp_finished = elapsed >= hold + ramp
        ready = target_error_max <= self.startup_stand_ready_error_rad
        if ramp_finished and ready:
            if self._startup_stand_ready_since is None:
                self._startup_stand_ready_since = now
            elif now - self._startup_stand_ready_since >= self.startup_stand_settle_sec:
                self.finish_startup_stand(current)
                return
        else:
            self._startup_stand_ready_since = None

        if now - self._startup_stand_last_log_time >= 1.0:
            self._startup_stand_last_log_time = now
            self.get_logger().info(
                f"[STARTUP_STAND] stage={stage} alpha={alpha:.3f} "
                f"target_error={target_error_max:.3f}rad "
                f"tracking_error={tracking_error_max:.3f}rad "
                f"torque_max={torque_abs_max:.2f}Nm age={max_age:.1f}ms "
                f"sent={sent}"
            )

        current = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        zeros = np.zeros(12, dtype=np.float32)
        self.maybe_write_policy_csv(
            obs=obs,
            action_raw=zeros,
            action_policy_obs=zeros,
            action=zeros,
            q_des=self._startup_stand_cmd_real,
            current_q=current,
            current_dq=current_dq,
            error=self._startup_stand_cmd_real - current,
            max_age=max_age,
            measured_torque=info.get("torque_real"),
            motor_temp=info.get("temp_real"),
            motor_online=info.get("online"),
            motor_error_code=info.get("error_code"),
            motor_age_ms=info.get("age_ms"),
            q_raw_des=self._startup_stand_cmd_real,
            q_smooth_des=self._startup_stand_cmd_real,
            motor_vel_cmd=zeros,
            mode="startup_stand",
        )

    def finish_startup_stand(self, current_real):
        now = time.time()
        current = np.asarray(current_real, dtype=np.float32).reshape(12)
        self._startup_stand_state = "complete"
        self._last_cmd_receive_time = 0.0
        self.cmd[:] = 0.0
        self.cmd_target[:] = 0.0
        self.obs_builder.set_command(0.0, 0.0, 0.0)
        self.obs_builder.set_last_action(np.zeros(12, dtype=np.float32))
        self.obs_builder.gait_phase_start_time = now
        self.obs_builder.last_gait_phase = 0.0
        self.safe_target_limiter.reset(current)
        self._smooth_target_real = current.copy()
        self._last_control_time = None
        self.get_logger().warn(
            "[STARTUP_STAND][READY] ONNX default pose is stable. "
            "Publish a fresh /cmd_vel to hand control to ONNX."
        )

    def send_startup_stand_target(self, target_real):
        target = np.asarray(target_real, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(target)):
            return False
        items = []
        for mid, position in zip(
            self.obs_builder.mapper.get_real_motor_ids(), target
        ):
            items.append({
                "motor_id": int(mid),
                "position": float(position),
                "speed": float(self.startup_stand_speed),
                "torque": float(self.startup_stand_torque),
                "kp": float(self.startup_stand_kp),
                "kd": float(self.startup_stand_kd),
            })
        payload = {
            "items": items,
            "enable_first": bool(
                self.startup_stand_enable_first
                and not self._startup_stand_enable_sent
            ),
            "stop_first": bool(
                self.startup_stand_stop_first
                and not self._startup_stand_enable_sent
            ),
        }
        if bool(getattr(self, "use_hardware_torque_limits", False)):
            payload["require_hardware_torque_limits"] = True
        if bool(getattr(self, "require_verified_hardware_limits", False)):
            payload["require_verified_hardware_safety_limits"] = True
        enable_requested = bool(payload["enable_first"])
        request_timeout = (
            max(float(self.http_timeout), 0.35)
            if enable_requested or bool(payload["stop_first"])
            else self.http_timeout
        )
        try:
            response = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json=payload,
                timeout=request_timeout,
            )
            if response.status_code != 200:
                self.get_logger().warn(
                    f"[STARTUP_STAND] HTTP {response.status_code}: {response.text}"
                )
                return False
            body = response.json()
            if enable_requested and not (
                bool(body.get("enable_applied", False))
                and bool(body.get("target_primed", False))
            ):
                self.get_logger().error(
                    "[STARTUP_STAND] motor service did not acknowledge the "
                    "one-shot target-prime/enable handshake; refusing to "
                    "advance startup"
                )
                return False
            self._startup_stand_enable_sent = True
            if enable_requested:
                self.get_logger().warn(
                    "[STARTUP_STAND] one-shot live-target prime and all-motor "
                    "ENABLE acknowledged"
                )
            return True
        except Exception as exc:
            self.get_logger().warn(f"[STARTUP_STAND] send failed: {exc}")
            return False

    def control_loop(self):
        self._record_control_loop_rate()
        self._last_onnx_infer_dt_ms = 0.0
        try:
            dt = self.get_control_dt()
            self.update_smoothed_command(dt)
            self.maybe_reset_gait_phase_on_command_start()
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
                    stale = np.where(
                        np.asarray(info["age_ms"]) > self.max_motor_age_ms
                    )[0]
                    if max_age <= self.max_motor_age_ms:
                        self.get_logger().info(
                            f"[SAFE] Stale motor feedback recovered after one recheck. "
                            f"max_age={max_age:.1f} ms."
                        )

                if max_age <= self.max_motor_age_ms:
                    pass
                else:
                    stale_items = []
                    motor_ids = self.obs_builder.mapper.get_real_motor_ids()
                    real_names = self.obs_builder.mapper.real_joint_names
                    for i in stale[:6]:
                        stale_items.append(
                            f"0x{motor_ids[int(i)]:02X}/{real_names[int(i)]}:"
                            f"{float(info['age_ms'][int(i)]):.0f}ms"
                        )
                    suffix = ""
                    if stale.shape[0] > 6:
                        suffix = f", ... +{stale.shape[0] - 6} more"
                    self.get_logger().warn(
                        f"[SAFE] Motor feedback too old: "
                        f"max_age={max_age:.1f} ms > {self.max_motor_age_ms:.1f} ms. "
                        f"stale=[{', '.join(stale_items)}{suffix}]. Skip policy."
                    )
                    return

            if self.require_online and not np.all(info["online"]):
                self.get_logger().warn("[SAFE] Some motors offline. Skip policy.")
                return

            if self._startup_stand_state != "complete":
                self.handle_startup_stand(obs, info, dt, max_age)
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
            action_gate_scale = self.policy_action_command_gate_scale()
            self._last_zero_cmd_stand_active = zero_cmd_stand
            self._last_action_cmd_gate_scale = action_gate_scale

            if zero_cmd_stand:
                action_raw = np.zeros(12, dtype=np.float32)
                action_for_obs = np.zeros(12, dtype=np.float32)
                action_for_target = np.zeros(12, dtype=np.float32)
                target_policy_abs = self.obs_builder.mapper.default_joint_angle.copy()
                self._last_gait_offset_policy = np.zeros(12, dtype=np.float32)
                cpg_action_info = {
                    "action_mode": "zero_cmd_stand",
                    "command_gate_scale": 0.0,
                    "zero_cmd_stand": True,
                    "frequency": 0.0,
                    "phase": float(self.obs_builder.last_gait_phase),
                }
            else:
                infer_t0 = time.perf_counter()
                action_raw = self.policy.infer(obs)
                self._last_onnx_infer_dt_ms = (
                    time.perf_counter() - infer_t0
                ) * 1000.0

                if self.clip_action:
                    clip_limit = abs(float(self.action_clip_limit))
                    action_for_obs = np.clip(
                        action_raw,
                        -clip_limit,
                        clip_limit,
                    ).astype(np.float32)
                else:
                    action_for_obs = action_raw.astype(np.float32)

                action_for_obs = (action_for_obs * action_gate_scale).astype(np.float32)
                action_for_target = self.prepare_action_for_target(action_for_obs)

                target_policy_abs, cpg_action_info = self.action_to_policy_target_abs_by_mode(
                    action_for_target,
                    dt=dt,
                )
                cpg_action_info["command_gate_scale"] = float(action_gate_scale)
                cpg_action_info["zero_cmd_stand"] = False
            target_policy_abs, rear_bias_info = self.apply_rear_leg_posture_bias(
                target_policy_abs
            )
            target_real = self.obs_builder.mapper.policy_target_to_real_target(
                target_policy_abs,
                clamp=True,
            )
            current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
            current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
            torque_budget_nm = self.compute_torque_safety_budget_nm()
            target_raw_real = target_real.copy()
            if self.enable_target_smoothing:
                target_pre_safe_real, pre_limit_info = self.safe_target_limiter.limit(
                    q_raw=target_raw_real,
                    q_current=current_q,
                    dt=dt,
                    kp=self.send_kp_real,
                    torque_budget_nm=torque_budget_nm,
                    err_limit_safety_factor=self.err_limit_safety_factor,
                    max_target_rate_rad_s=self.max_target_rate_rad_s,
                    max_target_accel_rad_s2=self.max_target_accel_rad_s2,
                    err_limit_mul=self.err_limit_mul_real,
                    target_rate_mul=self.target_rate_mul_real,
                    target_accel_mul=self.target_accel_mul_real,
                    absolute_error_limit_rad=self.model_position_error_limits_real,
                )
            else:
                target_pre_safe_real = target_raw_real.copy()
                raw_delta = target_raw_real - current_q
                pre_limit_info = {
                    "enabled": False,
                    "dt": dt,
                    "torque_budget": torque_budget_nm,
                    "base_err_limit": float("inf"),
                    "err_limit_min": float("inf"),
                    "err_limit_max": float("inf"),
                    "err_limit": np.full(12, np.inf, dtype=np.float32),
                    "max_rate": np.full(12, np.inf, dtype=np.float32),
                    "max_accel": np.full(12, np.inf, dtype=np.float32),
                    "q_raw_error_abs_max": float(np.max(np.abs(raw_delta))),
                    "q_cmd_error_abs_max": float(np.max(np.abs(raw_delta))),
                    "qdot_cmd_abs_max": 0.0,
                    "qddot_cmd_abs_max": 0.0,
                    "pre_limited_count": 0,
                    "rate_limited_count": 0,
                    "accel_limited_count": 0,
                    "post_limited_count": 0,
                    "pre_limited_mask": np.zeros(12, dtype=bool),
                    "rate_limited_mask": np.zeros(12, dtype=bool),
                    "accel_limited_mask": np.zeros(12, dtype=bool),
                    "post_limited_mask": np.zeros(12, dtype=bool),
                    "raw_delta": raw_delta.astype(np.float32),
                    "safe_delta": raw_delta.astype(np.float32),
                    "qdot_cmd": np.zeros(12, dtype=np.float32),
                    "qddot_cmd": np.zeros(12, dtype=np.float32),
                }
            motor_vel_cmd = self.compute_motor_velocity_command(pre_limit_info)
            target_real, torque_limit_info = self.apply_torque_error_limit(
                target_pre_safe_real,
                current_q,
                current_dq,
                torque_budget_nm,
                qd_target=motor_vel_cmd,
            )
            error = target_real - current_q

            self.publish_array(self.pub_obs, obs)
            self.publish_array(self.pub_action_raw, action_raw)
            self.publish_array(self.pub_action, action_for_target)
            self.publish_array(self.pub_target, target_real)
            self.maybe_write_policy_csv(
                obs=obs,
                action_raw=action_raw,
                action_policy_obs=action_for_obs,
                action=action_for_target,
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
                q_raw_des=target_raw_real,
                q_smooth_des=target_pre_safe_real,
                motor_vel_cmd=motor_vel_cmd,
                smoothing_info=pre_limit_info,
                torque_limit_info=torque_limit_info,
                rear_bias_info=rear_bias_info,
                cpg_action_info=cpg_action_info,
                mode=self.action_mode,
            )

            self.get_logger().info(
                f"cmd={self.cmd.tolist()} "
                f"cmd_target={self.cmd_target.tolist()} "
                f"mode={self.action_mode} "
                f"zero_stand={int(zero_cmd_stand)} "
                f"action_gate={action_gate_scale:.3f} "
                f"cpg_freq={float(cpg_action_info.get('frequency', 0.0)):.2f} "
                f"cpg_phase={float(cpg_action_info.get('phase', 0.0)):.2f} "
                f"max_age={max_age:.1f}ms "
                f"raw_min={float(np.min(action_raw)):.3f} "
                f"raw_max={float(np.max(action_raw)):.3f} "
                f"action_min={float(np.min(action_for_target)):.3f} "
                f"action_max={float(np.max(action_for_target)):.3f} "
                f"target_min={float(np.min(target_real)):.3f} "
                f"target_max={float(np.max(target_real)):.3f} "
                f"error_abs_max={float(np.max(np.abs(error))):.3f} "
                f"q_raw_error_abs_max={pre_limit_info.get('q_raw_error_abs_max', 0.0):.3f} "
                f"q_cmd_error_abs_max={pre_limit_info.get('q_cmd_error_abs_max', 0.0):.3f} "
                f"qdot_cmd_abs_max={pre_limit_info.get('qdot_cmd_abs_max', 0.0):.3f} "
                f"qddot_cmd_abs_max={pre_limit_info.get('qddot_cmd_abs_max', 0.0):.3f} "
                f"motor_vel_cmd_abs_max={float(np.max(np.abs(motor_vel_cmd))):.3f} "
                f"velocity_ff_scale={self.velocity_ff_scale:.3f} "
                f"rear_bias={int(self.enable_rear_leg_posture_bias)} "
                f"rear_calf_bias={self.rear_calf_extend_bias_policy_rad:+.3f} "
                f"rear_thigh_bias={self.rear_thigh_bias_policy_rad:+.3f} "
                f"err_limit_min={pre_limit_info.get('err_limit_min', 0.0):.3f} "
                f"err_limit_max={pre_limit_info.get('err_limit_max', 0.0):.3f} "
                f"pre_limited={pre_limit_info.get('pre_limited_count', 0)} "
                f"rate_limited={pre_limit_info.get('rate_limited_count', 0)} "
                f"accel_limited={pre_limit_info.get('accel_limited_count', 0)} "
                f"tau_est_max={torque_limit_info.get('tau_est_max', 0.0):.2f} "
                f"final_limited={torque_limit_info.get('limited_count', 0)} "
                f"send={self.enable_send}"
            ) if self._control_summary_log_due() else None

            self.maybe_print_policy_debug(
                action_raw=action_raw,
                action=action_for_target,
                q_des=target_real,
                current_q=current_q,
                error=error,
                max_age=max_age,
                pre_limit_info=pre_limit_info,
                torque_limit_info=torque_limit_info,
                rear_bias_info=rear_bias_info,
            )

            if self.enable_send and self.command_is_fresh():
                self.send_motion_batch(target_real, info, motor_vel_cmd=motor_vel_cmd)
            elif self.enable_send:
                now = time.time()
                if now - self._last_cmd_timeout_log_time >= 1.0:
                    self._last_cmd_timeout_log_time = now
                    self.get_logger().warn(
                        "[SAFE] /cmd_vel missing or stale; policy target is not sent"
                    )

            self.obs_builder.set_last_action(action_for_obs)

        except Exception as e:
            self.get_logger().error(f"policy loop error: {e}")

    def _record_control_loop_rate(self):
        now = time.perf_counter()
        if self._loop_rate_last_tick is not None:
            dt = now - self._loop_rate_last_tick
            self._loop_period_samples.append(dt)
            self._last_policy_loop_dt_ms = dt * 1000.0
            self._last_policy_loop_hz = 1.0 / dt if dt > 0.0 else 0.0
            nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)
            self._last_loop_overrun_ms = max(0.0, (dt - nominal_dt) * 1000.0)
        self._policy_loop_count += 1
        self._policy_steps_since_last_csv += 1
        self._loop_rate_last_tick = now

        elapsed = now - self._loop_rate_window_start
        if elapsed < 2.0 or not self._loop_period_samples:
            return
        periods = np.asarray(self._loop_period_samples, dtype=np.float64)
        self.get_logger().info(
            f"[LOOP_RATE] hz={1.0 / float(np.mean(periods)):.2f} "
            f"period_med={float(np.median(periods)) * 1000.0:.2f}ms "
            f"p95={float(np.percentile(periods, 95)) * 1000.0:.2f}ms "
            f"max={float(np.max(periods)) * 1000.0:.2f}ms "
            f"csv_queue={self._debug_csv_queue.qsize()} "
            f"csv_dropped={self._debug_csv_dropped}"
        )
        self._loop_period_samples.clear()
        self._loop_rate_window_start = now

    def _control_summary_log_due(self) -> bool:
        now = time.monotonic()
        if now - self._last_control_summary_log_time < 1.0:
            return False
        self._last_control_summary_log_time = now
        return True

    def handle_stand_only(self, obs, info, max_age, mode="stand_only"):
        elapsed = time.time() - self._mode_start_time
        duration = float(self.stand_only_duration_sec)
        if duration > 0.0 and elapsed > duration:
            if not self._stand_only_done_logged:
                self._stand_only_done_logged = True
                self.get_logger().warn(
                    f"[STAND_ONLY] recorded {duration:.2f}s; holding DEFAULT_STAND_POSE."
                )
            return

        current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        target_real = self.default_stand_target_real_order()
        error = target_real - current_q
        zeros = np.zeros(12, dtype=np.float32)

        if self.debug_print_arrays and elapsed < 0.1:
            self.print_default_pose_alignment_report(current_q, q_target_real=target_real)

        self.publish_array(self.pub_target, target_real)
        self.maybe_write_policy_csv(
            obs=obs,
            action_raw=zeros,
            action_policy_obs=zeros,
            action=zeros,
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
            q_raw_des=target_real,
            q_smooth_des=target_real,
            motor_vel_cmd=zeros,
            mode=mode,
        )

    def handle_joint_probe(self, obs, info, max_age):
        elapsed = time.time() - self._mode_start_time
        current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        target_real, delta, state_name = self.joint_probe_target(elapsed)
        error = target_real - current_q
        zeros = np.zeros(12, dtype=np.float32)

        if self.enable_send:
            self.send_motion_batch(target_real, info, motor_vel_cmd=zeros)

        policy_i = self._joint_probe_policy_index
        if policy_i is not None:
            real_i = int(self.obs_builder.mapper.policy_to_real_index[policy_i])
            self.get_logger().info(
                f"[JOINT_PROBE] state={state_name} "
                f"joint={self.obs_builder.mapper.policy_joint_names[policy_i]} "
                f"q_des={float(target_real[real_i]):+.4f} "
                f"current={float(current_q[real_i]):+.4f} "
                f"error={float(error[real_i]):+.4f}"
            )

        self.publish_array(self.pub_target, target_real)
        self.maybe_write_policy_csv(
            obs=obs,
            action_raw=zeros,
            action_policy_obs=zeros,
            action=zeros,
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
            q_raw_des=target_real,
            q_smooth_des=target_real,
            motor_vel_cmd=zeros,
            mode="joint_probe",
            joint_probe_delta_rad=delta,
            joint_probe_name=self.joint_probe_name,
        )

    def ensure_policy_obs_dim(self, obs_in: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_in, dtype=np.float32).reshape(-1)
        expected = int(self.policy.obs_dim)

        if obs.shape[0] == expected:
            return obs

        if obs.shape[0] == 36 and expected in (48, 50):
            fixed = np.zeros(expected, dtype=np.float32)
            fixed[0:36] = obs
            fixed[36:48] = getattr(
                self.obs_builder,
                "last_action",
                np.zeros(12, dtype=np.float32),
            )
            if expected == 50:
                fixed[48:50] = self.obs_builder.get_gait_phase_obs()
            if not self._warned_obs_dim_padding:
                self._warned_obs_dim_padding = True
                self.get_logger().warn(
                    f"[ONNX] ObsBuilder returned 36 dims but model expects {expected}. "
                    "Padding missing policy-only terms. Please rebuild/sync "
                    f"obs_builder.py so it starts with obs_dim={expected}."
                )
            return fixed

        if obs.shape[0] > expected:
            if not self._warned_obs_dim_padding:
                self._warned_obs_dim_padding = True
                self.get_logger().warn(
                    f"[ONNX] ObsBuilder returned {obs.shape[0]} dims but model "
                    f"expects {expected}. Cropping extra dims."
                )
            return obs[:expected].astype(np.float32)

        raise RuntimeError(
            f"Obs dimension mismatch: got {obs.shape[0]}, expected {expected}"
        )

    def apply_target_smoothing(self, target_real: np.ndarray, q_real: np.ndarray):
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        q_real = np.asarray(q_real, dtype=np.float32).reshape(12)

        if (
            not self.enable_target_smoothing
            or self.max_target_rate_rad_s <= 0.0
        ):
            raw_step = target_real - q_real
            return target_real, {
                "enabled": False,
                "step_max": float(np.max(np.abs(target_real - q_real))),
                "raw_step_max": float(np.max(np.abs(raw_step))),
                "max_step": float("inf"),
                "limited_count": 0,
                "raw_step": raw_step.astype(np.float32),
                "safe_step": raw_step.astype(np.float32),
                "limited_mask": np.zeros(12, dtype=bool),
            }

        now = time.time()
        nominal_dt = 1.0 / max(self.policy_hz, 1e-3)
        if self._smooth_target_real is None:
            self._smooth_target_real = q_real.copy()
            self._last_smoothing_time = now

        if self.target_smoothing_fixed_dt:
            dt = nominal_dt
        elif self._last_smoothing_time is None:
            dt = nominal_dt
        else:
            dt = now - self._last_smoothing_time
            if dt <= 0.0:
                dt = nominal_dt
            else:
                dt = min(dt, nominal_dt * 2.0)

        max_step = float(self.max_target_rate_rad_s) * dt
        prev_target = np.asarray(self._smooth_target_real, dtype=np.float32).reshape(12)
        raw_step = target_real - prev_target
        safe_step = np.clip(raw_step, -max_step, max_step)
        smoothed_target = prev_target + safe_step
        limited_mask = np.abs(raw_step - safe_step) > 1e-6
        limited_count = int(np.count_nonzero(limited_mask))

        self._smooth_target_real = smoothed_target.astype(np.float32).copy()
        self._last_smoothing_time = now

        return smoothed_target.astype(np.float32), {
            "enabled": True,
            "step_max": float(np.max(np.abs(safe_step))),
            "raw_step_max": float(np.max(np.abs(raw_step))),
            "max_step": max_step,
            "limited_count": limited_count,
            "raw_step": raw_step.astype(np.float32),
            "safe_step": safe_step.astype(np.float32),
            "limited_mask": limited_mask,
        }

    def apply_torque_error_limit(
        self,
        target_real: np.ndarray,
        q_real: np.ndarray,
        dq_real: np.ndarray,
        torque_budget_nm: float | None = None,
        qd_target=None,
    ):
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        q_real = np.asarray(q_real, dtype=np.float32).reshape(12)
        dq_real = np.asarray(dq_real, dtype=np.float32).reshape(12)

        kp = np.abs(np.asarray(self.send_kp_real, dtype=np.float32).reshape(12))
        kd = np.abs(np.asarray(self.send_kd_real, dtype=np.float32).reshape(12))
        if qd_target is None:
            qd_target = np.full(12, float(self.send_speed), dtype=np.float32)
        else:
            qd_target = np.asarray(qd_target, dtype=np.float32).reshape(12)
        feedforward_torque = abs(float(self.send_torque))

        if torque_budget_nm is None:
            torque_budget_nm = self.compute_torque_safety_budget_nm()
        torque_budget = max(0.0, float(torque_budget_nm) - feedforward_torque)

        vel_error = qd_target - dq_real
        kd_torque = kd * np.abs(vel_error)

        if not self.enable_torque_error_limit or np.all(kp <= 1e-6):
            delta = target_real - q_real
            tau_est = kp * np.abs(delta) + kd_torque + feedforward_torque
            return target_real, {
                "enabled": False,
                "limited_count": 0,
                "err_limit_min": float("inf"),
                "err_limit_max": float("inf"),
                "tau_est_max": float(np.max(tau_est)),
                "torque_budget": torque_budget,
                "err_limit": np.full(12, np.inf, dtype=np.float32),
                "tau_est": tau_est.astype(np.float32),
                "raw_delta": delta.astype(np.float32),
                "safe_delta": delta.astype(np.float32),
                "limited_mask": np.zeros(12, dtype=bool),
            }

        available_for_position = np.maximum(0.0, torque_budget - kd_torque)
        err_limit = available_for_position / np.maximum(kp, 1e-6)

        raw_delta = target_real - q_real
        safe_delta = np.clip(raw_delta, -err_limit, err_limit)
        safe_target = q_real + safe_delta

        tau_est = kp * np.abs(safe_delta) + kd_torque + feedforward_torque
        limited_mask = np.abs(raw_delta - safe_delta) > 1e-6
        limited_count = int(np.count_nonzero(limited_mask))

        now = time.monotonic()
        if limited_count > 0 and now - self._last_torque_limit_log_time >= 1.0:
            self._last_torque_limit_log_time = now
            self.get_logger().warn(
                "[SAFE] torque error limit active: "
                f"limited={limited_count}/12 "
                f"raw_err_max={float(np.max(np.abs(raw_delta))):.3f} rad "
                f"safe_err_max={float(np.max(np.abs(safe_delta))):.3f} rad "
                f"err_limit_min={float(np.min(err_limit)):.3f} rad "
                f"tau_est_max={float(np.max(tau_est)):.2f} Nm "
                f"budget={torque_budget:.2f} Nm"
            )

        return safe_target.astype(np.float32), {
            "enabled": True,
            "limited_count": limited_count,
            "err_limit_min": float(np.min(err_limit)),
            "err_limit_max": float(np.max(err_limit)),
            "tau_est_max": float(np.max(tau_est)),
            "torque_budget": torque_budget,
            "err_limit": err_limit.astype(np.float32),
            "tau_est": tau_est.astype(np.float32),
            "raw_delta": raw_delta.astype(np.float32),
            "safe_delta": safe_delta.astype(np.float32),
            "limited_mask": limited_mask,
        }

    def update_smoothing_reference_after_limits(self, sent_target_real: np.ndarray):
        if not self.enable_target_smoothing:
            return
        if not self.target_smoothing_follow_sent_target:
            return
        if self._smooth_target_real is None:
            return

        self._smooth_target_real = np.asarray(
            sent_target_real,
            dtype=np.float32,
        ).reshape(12).copy()

    def send_default_stand(self) -> bool:
        items = []
        target_real = self.default_stand_target_real_order()
        motor_ids = self.get_stand_mapper().get_real_motor_ids()
        for mid, pos in zip(motor_ids, target_real):
            items.append({
                "motor_id": int(mid),
                "position": float(pos),
                "speed": float(self.startup_stand_speed),
                "torque": float(self.startup_stand_torque),
                "kp": float(self.startup_stand_kp),
                "kd": float(self.startup_stand_kd),
            })

        payload = {
            "items": items,
            "enable_first": bool(self.startup_stand_enable_first),
            "stop_first": bool(self.startup_stand_stop_first),
        }

        url = f"{self.motor_base_url}/api/rs04/motion_mode_run_batch"

        try:
            r = self.http_session.post(
                url,
                json=payload,
                timeout=max(self.http_timeout, 0.5),
            )

            if r.status_code != 200:
                self.get_logger().error(
                    f"[DEFAULT_STAND] HTTP {r.status_code}: {r.text}"
                )
                return False

            self.get_logger().info(
                f"[DEFAULT_STAND] sent 12 motors | "
                f"source={self.stand_pose_source} "
                f"kp={self.startup_stand_kp:.2f} kd={self.startup_stand_kd:.2f}"
            )
            return True

        except Exception as e:
            self.get_logger().error(f"[DEFAULT_STAND] request failed: {e}")
            return False

    def send_motion_batch(self, target_real: np.ndarray, info: dict, motor_vel_cmd=None):
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        q_real = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        if motor_vel_cmd is None:
            motor_vel_cmd = np.full(12, float(self.send_speed), dtype=np.float32)
        else:
            motor_vel_cmd = np.asarray(motor_vel_cmd, dtype=np.float32).reshape(12)

        if not np.all(np.isfinite(target_real)):
            self.get_logger().warn("[SAFE] target_real has NaN/Inf. Skip send.")
            return False
        if not np.all(np.isfinite(motor_vel_cmd)):
            self.get_logger().warn("[SAFE] motor_vel_cmd has NaN/Inf. Skip send.")
            return False

        delta = target_real - q_real
        max_delta = float(np.max(np.abs(delta)))

        if max_delta > self.max_target_delta:
            self.get_logger().warn(
                f"[SAFE] target jump too large: "
                f"{max_delta:.3f} rad > {self.max_target_delta:.3f} rad. Skip send."
            )
            return False

        motor_ids = self.obs_builder.mapper.get_real_motor_ids()

        items = []
        for i, mid in enumerate(motor_ids):
            items.append({
                "motor_id": int(mid),
                "position": float(target_real[i]),
                "speed": float(motor_vel_cmd[i]),
                "torque": float(self.send_torque),
                "kp": float(self.send_kp_real[i]),
                "kd": float(self.send_kd_real[i]),
            })

        payload = {
            "items": items,
            "enable_first": bool(self.send_enable_first),
            "stop_first": bool(self.send_stop_first),
        }

        url = f"{self.motor_base_url}/api/rs04/motion_batch_fast"

        send_start_time = time.time()
        self._last_send_start_time = f"{send_start_time:.6f}"
        self._last_command_sequence += 1
        try:
            r = self.http_session.post(
                url,
                json=payload,
                timeout=self.http_timeout,
            )

            if r.status_code != 200:
                self._last_send_end_time = f"{time.time():.6f}"
                self.get_logger().warn(
                    f"[SEND] HTTP {r.status_code}: {r.text}"
                )
                return False

            self._last_send_end_time = f"{time.time():.6f}"
            send_time_perf = time.perf_counter()
            if self._last_motor_send_time_perf is not None:
                send_dt = send_time_perf - self._last_motor_send_time_perf
                self._last_motor_send_dt_ms = send_dt * 1000.0
                self._last_motor_send_hz = 1.0 / send_dt if send_dt > 0.0 else 0.0
            self._last_motor_send_time_perf = send_time_perf
            self._motor_send_count += 1
            self._motor_sends_since_last_csv += 1

            vel_log = (
                f"vel_cmd_abs_max={float(np.max(np.abs(motor_vel_cmd))):.3f} | "
                if self.log_motor_vel_cmd
                else ""
            )
            now = time.monotonic()
            if now - self._last_send_ok_log_time >= 1.0:
                self._last_send_ok_log_time = now
                self.get_logger().info(
                    f"[SEND] motion batch ok | "
                    f"max_delta={max_delta:.3f} | "
                    f"{vel_log}"
                    f"kp={self.send_kp_real.tolist()} "
                    f"kd={self.send_kd_real.tolist()}"
                )
            return True

        except Exception as e:
            self.get_logger().warn(f"[SEND] request failed: {e}")
            return False

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def setup_debug_csv(self):
        path = self.debug_csv_path.strip()
        if not path:
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._debug_csv_file = open(path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        # One row is one policy cycle.  Per-joint values are expanded into
        # columns so cycle_sequence remains the unambiguous join key.
        joint_names = list(self.obs_builder.mapper.real_joint_names)
        cycle_headers = [
            "timestamp", "cycle_sequence", "phase", "cmd_vx", "cmd_vy", "cmd_yaw",
            "policy_loop_hz", "state_age_ms", "send_start_time", "send_end_time",
            "board_A_send_time", "board_B_send_time", "command_sequence",
        ]
        for joint_name in joint_names:
            safe_name = str(joint_name).replace("-", "_").replace(" ", "_")
            cycle_headers.extend([
                f"{safe_name}_q_target_raw",
                f"{safe_name}_q_target_safe",
                f"{safe_name}_q_current",
                f"{safe_name}_dq_current",
                f"{safe_name}_raw_pd_torque",
                f"{safe_name}_limited_torque",
                f"{safe_name}_feedback_torque",
                f"{safe_name}_torque_limited_flag",
                f"{safe_name}_motor_online",
                f"{safe_name}_motor_fault",
            ])
        self._debug_csv_writer.writerow(cycle_headers)
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing 50Hz cycle log to {path}")
        if self.debug_csv_async:
            self._debug_csv_stop.clear()
            self._debug_csv_thread = threading.Thread(
                target=self._debug_csv_worker,
                name="policy-debug-csv",
                daemon=True,
            )
            self._debug_csv_thread.start()
            self.get_logger().info(
                f"[DEBUG_CSV] async writer enabled queue={self.debug_csv_queue_size}"
            )
        # The legacy 12-rows-per-cycle schema below is retained as dead code
        # for easy comparison during review; all active writes use the schema
        # above and return one row per policy cycle.
        return
        self._debug_csv_writer.writerow(
            [
                "time",
                "joint_index",
                "motor_id",
                "joint_name",
                "cmd_x",
                "cmd_y",
                "cmd_wz",
                "action_raw",
                "action_used",
                "q_target_real",
                "q_target_raw_real",
                "q_target_smooth_real",
                "q_current_real",
                "dq_current_real",
                "q_error_real",
                "q_raw_error_real",
                "q_smooth_error_real",
                "torque_measured",
                "motor_temp",
                "motor_online",
                "motor_error_code",
                "motor_age_ms",
                "smooth_step_real",
                "smooth_raw_step_real",
                "smooth_limited",
                "err_limit_real",
                "tau_est_real",
                "torque_limited",
                "kp",
                "kd",
                "max_age_ms",
                "base_lin_vel_source",
                "semantic_yaw_180",
                "action_leg_yaw_180",
                "q_raw_error_abs_max",
                "q_cmd_error_abs_max",
                "qdot_cmd_abs_max",
                "qddot_cmd_abs_max",
                "pre_err_limit_min",
                "pre_err_limit_max",
                "pre_limited_count",
                "rate_limited_count",
                "accel_limited_count",
                "final_limited_count",
                "qdot_cmd_real",
                "qddot_cmd_real",
                "pre_limited",
                "rate_limited",
                "accel_limited",
                "policy_index",
                "policy_joint_name",
                "leg_name",
                "joint_type",
                "obs_base_lin_x",
                "obs_base_lin_y",
                "obs_base_lin_z",
                "obs_base_ang_x",
                "obs_base_ang_y",
                "obs_base_ang_z",
                "obs_gravity_x",
                "obs_gravity_y",
                "obs_gravity_z",
                "obs_cmd_x",
                "obs_cmd_y",
                "obs_cmd_wz",
                "command_scale_effective_x",
                "obs_joint_pos_policy",
                "obs_joint_vel_policy",
                "obs_last_action_policy",
                "action_raw_policy",
                "action_clipped_policy",
                "action_scaled_policy",
                "action_mode",
                "cpg_frequency_hz",
                "cpg_phase",
                "cpg_leg_phase",
                "q_cpg_policy_abs",
                "delta_q_rl_policy",
                "residual_limit_policy",
                "q_default_policy",
                "q_current_policy_abs",
                "q_raw_target_policy_abs",
                "q_smooth_target_policy_abs",
                "q_final_target_policy_abs",
                "q_current_minus_default_policy",
                "q_raw_target_minus_default_policy",
                "q_final_target_minus_default_policy",
                "gait_ref_policy",
                "rl_action_contrib_policy",
                "global_gait_phase",
                "raw_to_smooth_delta_real",
                "smooth_to_final_delta_real",
                "raw_to_final_delta_real",
                "torque_budget_nm",
                "torque_limit_nm",
                "torque_safety_budget_nm",
                "pre_limited_joint_mask",
                "rate_limited_joint_mask",
                "accel_limited_joint_mask",
                "final_limited_joint_mask",
                "pre_err_limit_real",
                "pre_max_rate_real",
                "pre_max_accel_real",
                "motor_vel_cmd",
                "velocity_ff_scale",
                "enable_velocity_ff",
                "max_motor_vel_cmd_rad_s",
                "enable_rear_leg_posture_bias",
                "rear_calf_extend_bias_policy_rad",
                "rear_thigh_bias_policy_rad",
                "mode",
                "default_pose_abs_error",
                "joint_probe_delta_rad",
                "joint_probe_name",
                "target_real_minus_policy_default_real",
                "stand_pose_source",
                "expected_support_pair",
                "support_proxy_winner",
                "fr_torque_abs_sum",
                "fl_torque_abs_sum",
                "rr_torque_abs_sum",
                "rl_torque_abs_sum",
                "diag_FR_RL_torque_abs_sum",
                "diag_FL_RR_torque_abs_sum",
                "hip_gate_clamp_count",
                "hip_outward_before_gate",
                "hip_outward_after_gate",
                "action_cmd_gate_scale",
                "zero_cmd_stand_active",
                "policy_loop_dt_ms",
                "policy_loop_hz",
                "motor_send_dt_ms",
                "motor_send_hz",
                "motor_state_poll_dt_ms",
                "onnx_infer_dt_ms",
                "policy_steps_since_last_csv",
                "motor_sends_since_last_csv",
                "state_age_ms",
                "loop_overrun_ms",
            ]
        )
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing joint debug log to {path}")

        if self.debug_csv_async:
            self._debug_csv_stop.clear()
            self._debug_csv_thread = threading.Thread(
                target=self._debug_csv_worker,
                name="policy-debug-csv",
                daemon=True,
            )
            self._debug_csv_thread.start()
            self.get_logger().info(
                f"[DEBUG_CSV] async writer enabled queue={self.debug_csv_queue_size}"
            )

    def _debug_csv_worker(self):
        while not self._debug_csv_stop.is_set() or not self._debug_csv_queue.empty():
            try:
                args, kwargs = self._debug_csv_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_policy_csv_sync(*args, **kwargs)
            except Exception as exc:
                self.get_logger().error(f"[DEBUG_CSV] async write failed: {exc}")
                self._debug_csv_stop.set()
                break
            finally:
                self._debug_csv_queue.task_done()

    def _stop_debug_csv_worker(self):
        self._debug_csv_stop.set()
        thread = self._debug_csv_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.get_logger().error("[DEBUG_CSV] writer did not stop within 5 seconds")
        self._debug_csv_thread = None

    def make_debug_timing_info(self) -> dict:
        return {
            "policy_loop_dt_ms": float(self._last_policy_loop_dt_ms),
            "policy_loop_hz": float(self._last_policy_loop_hz),
            "motor_send_dt_ms": float(self._last_motor_send_dt_ms),
            "motor_send_hz": float(self._last_motor_send_hz),
            "motor_state_poll_dt_ms": float(self._last_motor_state_poll_dt_ms),
            "onnx_infer_dt_ms": float(self._last_onnx_infer_dt_ms),
            "policy_steps_since_last_csv": int(self._policy_steps_since_last_csv),
            "motor_sends_since_last_csv": int(self._motor_sends_since_last_csv),
            "state_age_ms": float(self._last_state_age_ms),
            "loop_overrun_ms": float(self._last_loop_overrun_ms),
        }

    def reset_debug_csv_timing_counters(self):
        self._policy_steps_since_last_csv = 0
        self._motor_sends_since_last_csv = 0

    def maybe_write_policy_csv(self, *args, **kwargs):
        if self._debug_csv_writer is None:
            return

        now = time.time()
        if self.debug_csv_period_sec > 0.0:
            if now - self._last_debug_csv_time < self.debug_csv_period_sec:
                return
        self._last_debug_csv_time = now
        self._csv_cycle_sequence += 1

        kwargs = dict(kwargs)
        kwargs["timing_info"] = self.make_debug_timing_info()
        if not self.debug_csv_async:
            self._write_policy_csv_sync(*args, record_time=now, **kwargs)
            self.reset_debug_csv_timing_counters()
            return
        if self._debug_csv_stop.is_set():
            return

        # Control-loop values are small (mostly 12-element arrays). Copy them
        # before returning so the writer thread sees a consistent cycle.
        queued_kwargs = dict(kwargs)
        queued_kwargs["record_time"] = now
        record = copy.deepcopy((args, queued_kwargs))
        try:
            self._debug_csv_queue.put_nowait(record)
            self.reset_debug_csv_timing_counters()
        except queue.Full:
            self._debug_csv_dropped += 1

    def _write_policy_csv_sync(
        self,
        obs: np.ndarray,
        action_raw: np.ndarray,
        action_policy_obs: np.ndarray,
        action: np.ndarray,
        q_des: np.ndarray,
        current_q: np.ndarray,
        error: np.ndarray,
        max_age: float,
        current_dq: np.ndarray = None,
        measured_torque: np.ndarray = None,
        motor_temp: np.ndarray = None,
        motor_online: np.ndarray = None,
        motor_error_code: np.ndarray = None,
        motor_age_ms: np.ndarray = None,
        q_raw_des: np.ndarray = None,
        q_smooth_des: np.ndarray = None,
        motor_vel_cmd: np.ndarray = None,
        smoothing_info: dict = None,
        torque_limit_info: dict = None,
        rear_bias_info: dict = None,
        cpg_action_info: dict = None,
        mode: str = "policy",
        joint_probe_delta_rad: float = 0.0,
        joint_probe_name: str = "",
        record_time: float = None,
        timing_info: dict = None,
    ):
        if self._debug_csv_writer is None:
            return

        # Active cycle schema: exactly one CSV row for each policy invocation.
        # Keep all values in the control-loop snapshot, including the send
        # timestamps captured by send_motion_batch().
        now = time.time() if record_time is None else float(record_time)
        q_des = np.asarray(q_des, dtype=np.float32).reshape(12)
        q_raw_des = q_des if q_raw_des is None else np.asarray(q_raw_des, dtype=np.float32).reshape(12)
        current_q = np.asarray(current_q, dtype=np.float32).reshape(12)
        current_dq = np.zeros(12, dtype=np.float32) if current_dq is None else np.asarray(current_dq, dtype=np.float32).reshape(12)
        measured_torque = np.zeros(12, dtype=np.float32) if measured_torque is None else np.asarray(measured_torque, dtype=np.float32).reshape(12)
        motor_online = np.zeros(12, dtype=bool) if motor_online is None else np.asarray(motor_online, dtype=bool).reshape(12)
        motor_error_code = np.zeros(12, dtype=np.int32) if motor_error_code is None else np.asarray(motor_error_code, dtype=np.int32).reshape(12)
        torque_limit_info = {} if torque_limit_info is None else torque_limit_info
        timing_info = self.make_debug_timing_info() if timing_info is None else timing_info

        def vec(info, key, default):
            value = np.asarray(info.get(key, default), dtype=np.float32)
            if value.shape == ():
                value = np.full(12, float(value), dtype=np.float32)
            return value.reshape(12)

        raw_pd_torque = self.send_kp_real * (q_raw_des - current_q) - self.send_kd_real * current_dq
        limited_torque = vec(torque_limit_info, "tau_safe_signed", np.zeros(12, dtype=np.float32))
        torque_limited = np.asarray(
            torque_limit_info.get("limited_mask", np.zeros(12, dtype=bool)), dtype=bool
        ).reshape(12)
        phase = float(getattr(getattr(self, "obs_builder", None), "last_gait_phase", 0.0))
        row = [
            f"{now:.6f}", self._csv_cycle_sequence, f"{phase:.6f}",
            f"{float(self.cmd[0]):.6f}", f"{float(self.cmd[1]):.6f}", f"{float(self.cmd[2]):.6f}",
            f"{float(timing_info.get('policy_loop_hz', 0.0)):.3f}",
            f"{float(timing_info.get('state_age_ms', max_age)):.3f}",
            self._last_send_start_time, self._last_send_end_time,
            self._last_board_a_send_time, self._last_board_b_send_time,
            self._last_command_sequence,
        ]
        for i in range(12):
            row.extend([
                f"{float(q_raw_des[i]):.6f}", f"{float(q_des[i]):.6f}",
                f"{float(current_q[i]):.6f}", f"{float(current_dq[i]):.6f}",
                f"{float(raw_pd_torque[i]):.6f}", f"{float(limited_torque[i]):.6f}",
                f"{float(measured_torque[i]):.6f}", int(torque_limited[i]),
                int(motor_online[i]), int(motor_error_code[i] != 0),
            ])
        self._debug_csv_writer.writerow(row)
        self._debug_csv_rows_since_flush += 1
        if self._debug_csv_rows_since_flush >= self.debug_csv_flush_every_n:
            self._debug_csv_file.flush()
            self._debug_csv_rows_since_flush = 0
        return

        now = time.time() if record_time is None else float(record_time)

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(12)
        action_policy_obs = np.asarray(action_policy_obs, dtype=np.float32).reshape(12)
        action = np.asarray(action, dtype=np.float32).reshape(12)
        q_des = np.asarray(q_des, dtype=np.float32).reshape(12)
        current_q = np.asarray(current_q, dtype=np.float32).reshape(12)
        current_dq = (
            np.zeros(12, dtype=np.float32)
            if current_dq is None
            else np.asarray(current_dq, dtype=np.float32).reshape(12)
        )
        error = np.asarray(error, dtype=np.float32).reshape(12)
        measured_torque = (
            np.zeros(12, dtype=np.float32)
            if measured_torque is None
            else np.asarray(measured_torque, dtype=np.float32).reshape(12)
        )
        motor_temp = (
            np.zeros(12, dtype=np.float32)
            if motor_temp is None
            else np.asarray(motor_temp, dtype=np.float32).reshape(12)
        )
        motor_online = (
            np.zeros(12, dtype=bool)
            if motor_online is None
            else np.asarray(motor_online, dtype=bool).reshape(12)
        )
        motor_error_code = (
            np.zeros(12, dtype=np.int32)
            if motor_error_code is None
            else np.asarray(motor_error_code, dtype=np.int32).reshape(12)
        )
        motor_age_ms = (
            np.full(12, float(max_age), dtype=np.float32)
            if motor_age_ms is None
            else np.asarray(motor_age_ms, dtype=np.float32).reshape(12)
        )
        q_raw_des = q_des if q_raw_des is None else np.asarray(q_raw_des, dtype=np.float32).reshape(12)
        q_smooth_des = q_des if q_smooth_des is None else np.asarray(q_smooth_des, dtype=np.float32).reshape(12)
        motor_vel_cmd = (
            np.zeros(12, dtype=np.float32)
            if motor_vel_cmd is None
            else np.asarray(motor_vel_cmd, dtype=np.float32).reshape(12)
        )
        smoothing_info = {} if smoothing_info is None else smoothing_info
        torque_limit_info = {} if torque_limit_info is None else torque_limit_info
        rear_bias_info = {} if rear_bias_info is None else rear_bias_info
        cpg_action_info = {} if cpg_action_info is None else cpg_action_info
        timing_info = (
            self.make_debug_timing_info() if timing_info is None else timing_info
        )

        def info_vec(info, key, default):
            value = info.get(key, default)
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape == ():
                arr = np.full(12, float(arr), dtype=np.float32)
            return arr.reshape(12)

        def info_bool_vec(info, key):
            value = info.get(key, np.zeros(12, dtype=bool))
            arr = np.asarray(value, dtype=bool)
            if arr.shape == ():
                arr = np.full(12, bool(arr), dtype=bool)
            return arr.reshape(12)

        raw_error = q_raw_des - current_q
        smooth_error = q_smooth_des - current_q
        raw_to_smooth_delta = q_smooth_des - q_raw_des
        smooth_to_final_delta = q_des - q_smooth_des
        raw_to_final_delta = q_des - q_raw_des
        smooth_step = info_vec(smoothing_info, "safe_delta", np.zeros(12, dtype=np.float32))
        smooth_raw_step = info_vec(smoothing_info, "raw_delta", np.zeros(12, dtype=np.float32))
        smooth_limited = info_bool_vec(smoothing_info, "pre_limited_mask")
        qdot_cmd = info_vec(smoothing_info, "qdot_cmd", np.zeros(12, dtype=np.float32))
        qddot_cmd = info_vec(smoothing_info, "qddot_cmd", np.zeros(12, dtype=np.float32))
        rate_limited = info_bool_vec(smoothing_info, "rate_limited_mask")
        accel_limited = info_bool_vec(smoothing_info, "accel_limited_mask")
        pre_err_limit = info_vec(
            smoothing_info,
            "err_limit",
            np.full(12, np.inf, dtype=np.float32),
        )
        pre_max_rate = info_vec(
            smoothing_info,
            "max_rate",
            np.full(12, np.inf, dtype=np.float32),
        )
        pre_max_accel = info_vec(
            smoothing_info,
            "max_accel",
            np.full(12, np.inf, dtype=np.float32),
        )
        err_limit = info_vec(torque_limit_info, "err_limit", np.full(12, np.inf, dtype=np.float32))
        tau_est = info_vec(torque_limit_info, "tau_est", np.zeros(12, dtype=np.float32))
        torque_limited = info_bool_vec(torque_limit_info, "limited_mask")
        torque_budget_nm = float(
            torque_limit_info.get(
                "torque_budget",
                smoothing_info.get("torque_budget", self.compute_torque_safety_budget_nm()),
            )
        )

        motor_ids = self.obs_builder.mapper.get_real_motor_ids()
        mapper = self.obs_builder.mapper
        real_names = mapper.real_joint_names
        policy_names = mapper.policy_joint_names
        policy_to_real_index = mapper.policy_to_real_index
        real_to_policy_index = np.zeros(12, dtype=np.int64)
        real_to_policy_index[policy_to_real_index] = np.arange(12, dtype=np.int64)

        q_current_policy_abs, _ = mapper.real_to_policy_abs_q_dq(current_q, current_dq)
        q_policy_default_real = mapper.real_default_pose_for_motor_order()

        def real_target_to_policy_abs(q_real_target):
            q_real_target = np.asarray(q_real_target, dtype=np.float32).reshape(12)
            q_real_ordered = q_real_target[policy_to_real_index]
            return (
                mapper.joint_sign
                * (q_real_ordered - mapper.real_zero_offset_policy_order)
            ).astype(np.float32)

        q_raw_policy_abs = real_target_to_policy_abs(q_raw_des)
        q_smooth_policy_abs = real_target_to_policy_abs(q_smooth_des)
        q_final_policy_abs = real_target_to_policy_abs(q_des)
        q_cpg_policy_abs = np.asarray(
            cpg_action_info.get(
                "q_cpg_policy_abs",
                self._last_q_cpg_policy_abs,
            ),
            dtype=np.float32,
        ).reshape(12)
        delta_q_rl_policy = np.asarray(
            cpg_action_info.get(
                "delta_q_rl_policy",
                self._last_delta_q_rl_policy,
            ),
            dtype=np.float32,
        ).reshape(12)
        residual_limit_policy = np.asarray(
            cpg_action_info.get("residual_limits", np.zeros(12, dtype=np.float32)),
            dtype=np.float32,
        )
        if residual_limit_policy.shape == ():
            residual_limit_policy = np.full(12, float(residual_limit_policy), dtype=np.float32)
        residual_limit_policy = residual_limit_policy.reshape(12)
        cpg_leg_phase = np.asarray(
            cpg_action_info.get("leg_phase", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)

        obs_base_lin = np.zeros(3, dtype=np.float32)
        obs_base_ang = np.zeros(3, dtype=np.float32)
        obs_gravity = np.zeros(3, dtype=np.float32)
        obs_cmd = np.zeros(3, dtype=np.float32)
        obs_joint_pos = np.zeros(12, dtype=np.float32)
        obs_joint_vel = np.zeros(12, dtype=np.float32)
        obs_last_action = np.zeros(12, dtype=np.float32)
        if obs.shape[0] >= 3:
            obs_base_lin[:] = obs[0:3]
        if obs.shape[0] >= 6:
            obs_base_ang[:] = obs[3:6]
        if obs.shape[0] >= 9:
            obs_gravity[:] = obs[6:9]
        if obs.shape[0] >= 12:
            obs_cmd[:] = obs[9:12]
        command_scale_effective_x = float(
            np.asarray(self.obs_builder.command_scale, dtype=np.float32).reshape(3)[0]
        )
        if obs.shape[0] >= 24:
            obs_joint_pos[:] = obs[12:24]
        if obs.shape[0] >= 36:
            obs_joint_vel[:] = obs[24:36]
        if obs.shape[0] >= 48:
            obs_last_action[:] = obs[36:48]

        action_raw_real_order = np.zeros(12, dtype=np.float32)
        action_policy_obs_real_order = np.zeros(12, dtype=np.float32)
        action_real_order = np.zeros(12, dtype=np.float32)
        action_raw_real_order[policy_to_real_index] = action_raw
        action_policy_obs_real_order[policy_to_real_index] = action_policy_obs
        action_real_order[policy_to_real_index] = action

        measured_torque_policy = np.asarray(measured_torque[policy_to_real_index], dtype=np.float32)
        leg_torque = {"FR": 0.0, "FL": 0.0, "RR": 0.0, "RL": 0.0}
        for policy_i, name in enumerate(policy_names):
            leg = self.leg_prefix_from_joint_name(name)
            if leg in leg_torque:
                leg_torque[leg] += abs(float(measured_torque_policy[policy_i]))
        leg_torque_abs_sum = np.asarray(
            [
                leg_torque["FR"],
                leg_torque["FL"],
                leg_torque["RR"],
                leg_torque["RL"],
            ],
            dtype=np.float32,
        )
        diag_fr_rl_torque = float(leg_torque["FR"] + leg_torque["RL"])
        diag_fl_rr_torque = float(leg_torque["FL"] + leg_torque["RR"])
        support_proxy_winner = "FR_RL" if diag_fr_rl_torque >= diag_fl_rr_torque else "FL_RR"
        expected_support_pair = "unknown"
        if self.deployment_config is not None:
            phase = float(self.obs_builder.last_gait_phase)
            gait = self.deployment_config.get("gait", {})
            offsets = gait.get("phase_offsets", {})
            stance_ratio = float(gait.get("stance_ratio", 0.62))
            fr_rl_score = 0.0
            fl_rr_score = 0.0
            for leg in ("FR", "RL"):
                leg_phase = (phase + float(offsets.get(leg, 0.0))) % 1.0
                fr_rl_score += float(leg_phase < stance_ratio)
            for leg in ("FL", "RR"):
                leg_phase = (phase + float(offsets.get(leg, 0.0))) % 1.0
                fl_rr_score += float(leg_phase < stance_ratio)
            expected_support_pair = "FR_RL" if fr_rl_score >= fl_rr_score else "FL_RR"
        elif cpg_leg_phase.size >= 4:
            phase01 = np.remainder(cpg_leg_phase[:4], 1.0)
            swing_fraction = max(1.0 - float(self.cpg_duty_factor), 0.05)
            stance = phase01 >= swing_fraction
            fr_rl_score = float(stance[0]) + float(stance[3])
            fl_rr_score = float(stance[1]) + float(stance[2])
            expected_support_pair = "FR_RL" if fr_rl_score >= fl_rr_score else "FL_RR"
        hip_outward_before_gate = np.asarray(
            cpg_action_info.get("hip_outward_before_gate", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        hip_outward_after_gate = np.asarray(
            cpg_action_info.get("hip_outward_after_gate", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        hip_gate_clamp_count = int(cpg_action_info.get("hip_gate_clamp_count", 0))

        if self.deployment_config is not None:
            action_scale_arr = np.asarray(
                self.deployment_config["control"]["action_scale"], dtype=np.float32
            ).reshape(12)
        else:
            action_scale_arr = np.full(12, float(self.action_scale), dtype=np.float32)
        if mode == "startup_stand":
            csv_kp_real = np.full(12, abs(self.startup_stand_kp), dtype=np.float32)
            csv_kd_real = np.full(12, abs(self.startup_stand_kd), dtype=np.float32)
        else:
            csv_kp_real = self.send_kp_real
            csv_kd_real = self.send_kd_real
        global_gait_phase = float(self.obs_builder.last_gait_phase)
        policy_loop_dt_ms = float(timing_info.get("policy_loop_dt_ms", 0.0))
        policy_loop_hz = float(timing_info.get("policy_loop_hz", 0.0))
        motor_send_dt_ms = float(timing_info.get("motor_send_dt_ms", 0.0))
        motor_send_hz = float(timing_info.get("motor_send_hz", 0.0))
        motor_state_poll_dt_ms = float(
            timing_info.get("motor_state_poll_dt_ms", 0.0)
        )
        onnx_infer_dt_ms = float(timing_info.get("onnx_infer_dt_ms", 0.0))
        policy_steps_since_last_csv = int(
            timing_info.get("policy_steps_since_last_csv", 0)
        )
        motor_sends_since_last_csv = int(
            timing_info.get("motor_sends_since_last_csv", 0)
        )
        state_age_ms = float(timing_info.get("state_age_ms", max_age))
        loop_overrun_ms = float(timing_info.get("loop_overrun_ms", 0.0))

        for i, (mid, name) in enumerate(zip(motor_ids, real_names)):
            policy_i = int(real_to_policy_index[i])
            policy_name = policy_names[policy_i]
            leg_i = int(policy_i // 3)
            parts = name.split("_")
            leg_name = parts[0] if len(parts) >= 1 else ""
            joint_type = parts[1] if len(parts) >= 2 else ""
            self._debug_csv_writer.writerow(
                [
                    f"{now:.6f}",
                    int(i),
                    f"0x{int(mid):02X}",
                    name,
                    f"{float(self.cmd[0]):.6f}",
                    f"{float(self.cmd[1]):.6f}",
                    f"{float(self.cmd[2]):.6f}",
                    f"{float(action_raw_real_order[i]):.6f}",
                    f"{float(action_real_order[i]):.6f}",
                    f"{float(q_des[i]):.6f}",
                    f"{float(q_raw_des[i]):.6f}",
                    f"{float(q_smooth_des[i]):.6f}",
                    f"{float(current_q[i]):.6f}",
                    f"{float(current_dq[i]):.6f}",
                    f"{float(error[i]):.6f}",
                    f"{float(raw_error[i]):.6f}",
                    f"{float(smooth_error[i]):.6f}",
                    f"{float(measured_torque[i]):.6f}",
                    f"{float(motor_temp[i]):.3f}",
                    int(motor_online[i]),
                    int(motor_error_code[i]),
                    f"{float(motor_age_ms[i]):.3f}",
                    f"{float(smooth_step[i]):.6f}",
                    f"{float(smooth_raw_step[i]):.6f}",
                    int(smooth_limited[i]),
                    f"{float(err_limit[i]):.6f}",
                    f"{float(tau_est[i]):.6f}",
                    int(torque_limited[i]),
                    f"{float(csv_kp_real[i]):.6f}",
                    f"{float(csv_kd_real[i]):.6f}",
                    f"{float(max_age):.3f}",
                    self.base_lin_vel_source,
                    int(self.semantic_yaw_180),
                    int(self.action_leg_yaw_180),
                    f"{float(smoothing_info.get('q_raw_error_abs_max', 0.0)):.6f}",
                    f"{float(smoothing_info.get('q_cmd_error_abs_max', 0.0)):.6f}",
                    f"{float(smoothing_info.get('qdot_cmd_abs_max', 0.0)):.6f}",
                    f"{float(smoothing_info.get('qddot_cmd_abs_max', 0.0)):.6f}",
                    f"{float(smoothing_info.get('err_limit_min', 0.0)):.6f}",
                    f"{float(smoothing_info.get('err_limit_max', 0.0)):.6f}",
                    int(smoothing_info.get("pre_limited_count", 0)),
                    int(smoothing_info.get("rate_limited_count", 0)),
                    int(smoothing_info.get("accel_limited_count", 0)),
                    int(torque_limit_info.get("limited_count", 0)),
                    f"{float(qdot_cmd[i]):.6f}",
                    f"{float(qddot_cmd[i]):.6f}",
                    int(smooth_limited[i]),
                    int(rate_limited[i]),
                    int(accel_limited[i]),
                    policy_i,
                    policy_name,
                    leg_name,
                    joint_type,
                    f"{float(obs_base_lin[0]):.6f}",
                    f"{float(obs_base_lin[1]):.6f}",
                    f"{float(obs_base_lin[2]):.6f}",
                    f"{float(obs_base_ang[0]):.6f}",
                    f"{float(obs_base_ang[1]):.6f}",
                    f"{float(obs_base_ang[2]):.6f}",
                    f"{float(obs_gravity[0]):.6f}",
                    f"{float(obs_gravity[1]):.6f}",
                    f"{float(obs_gravity[2]):.6f}",
                    f"{float(obs_cmd[0]):.6f}",
                    f"{float(obs_cmd[1]):.6f}",
                    f"{float(obs_cmd[2]):.6f}",
                    f"{command_scale_effective_x:.6f}",
                    f"{float(obs_joint_pos[policy_i]):.6f}",
                    f"{float(obs_joint_vel[policy_i]):.6f}",
                    f"{float(obs_last_action[policy_i]):.6f}",
                    f"{float(action_raw[policy_i]):.6f}",
                    f"{float(action_policy_obs[policy_i]):.6f}",
                    f"{float(action[policy_i]):.6f}",
                    str(cpg_action_info.get("action_mode", mode)),
                    f"{float(cpg_action_info.get('frequency', 0.0)):.6f}",
                    f"{float(cpg_action_info.get('phase', 0.0)):.6f}",
                    f"{float(cpg_leg_phase[policy_i // 3]) if cpg_leg_phase.size >= 4 else 0.0:.6f}",
                    f"{float(q_cpg_policy_abs[policy_i]):.6f}",
                    f"{float(delta_q_rl_policy[policy_i]):.6f}",
                    f"{float(residual_limit_policy[policy_i]):.6f}",
                    f"{float(mapper.default_joint_angle[policy_i]):.6f}",
                    f"{float(q_current_policy_abs[policy_i]):.6f}",
                    f"{float(q_raw_policy_abs[policy_i]):.6f}",
                    f"{float(q_smooth_policy_abs[policy_i]):.6f}",
                    f"{float(q_final_policy_abs[policy_i]):.6f}",
                    f"{float(q_current_policy_abs[policy_i] - mapper.default_joint_angle[policy_i]):.6f}",
                    f"{float(q_raw_policy_abs[policy_i] - mapper.default_joint_angle[policy_i]):.6f}",
                    f"{float(q_final_policy_abs[policy_i] - mapper.default_joint_angle[policy_i]):.6f}",
                    f"{float(self._last_gait_offset_policy[policy_i]):.6f}",
                    f"{float(action[policy_i] * action_scale_arr[policy_i]):.6f}",
                    f"{global_gait_phase:.6f}",
                    f"{float(raw_to_smooth_delta[i]):.6f}",
                    f"{float(smooth_to_final_delta[i]):.6f}",
                    f"{float(raw_to_final_delta[i]):.6f}",
                    f"{torque_budget_nm:.6f}",
                    f"{float(self.motor_torque_limit_nm):.6f}",
                    f"{float(self.torque_safety_budget_nm):.6f}",
                    int(smooth_limited[i]),
                    int(rate_limited[i]),
                    int(accel_limited[i]),
                    int(torque_limited[i]),
                    f"{float(pre_err_limit[i]):.6f}",
                    f"{float(pre_max_rate[i]):.6f}",
                    f"{float(pre_max_accel[i]):.6f}",
                    f"{float(motor_vel_cmd[i]):.6f}",
                    f"{float(self.velocity_ff_scale):.6f}",
                    int(self.enable_velocity_ff),
                    f"{float(self.max_motor_vel_cmd_rad_s):.6f}",
                    int(rear_bias_info.get("enabled", self.enable_rear_leg_posture_bias)),
                    f"{float(rear_bias_info.get('rear_calf_extend_bias_policy_rad', self.rear_calf_extend_bias_policy_rad)):.6f}",
                    f"{float(rear_bias_info.get('rear_thigh_bias_policy_rad', self.rear_thigh_bias_policy_rad)):.6f}",
                    str(mode),
                    f"{float(abs(q_current_policy_abs[policy_i] - mapper.default_joint_angle[policy_i])):.6f}",
                    f"{float(joint_probe_delta_rad):.6f}",
                    str(joint_probe_name),
                    f"{float(q_des[i] - q_policy_default_real[i]):.6f}",
                    str(self.stand_pose_source),
                    expected_support_pair,
                    support_proxy_winner,
                    f"{float(leg_torque_abs_sum[0]):.6f}",
                    f"{float(leg_torque_abs_sum[1]):.6f}",
                    f"{float(leg_torque_abs_sum[2]):.6f}",
                    f"{float(leg_torque_abs_sum[3]):.6f}",
                    f"{diag_fr_rl_torque:.6f}",
                    f"{diag_fl_rr_torque:.6f}",
                    hip_gate_clamp_count,
                    f"{float(hip_outward_before_gate[leg_i]) if hip_outward_before_gate.size >= 4 else 0.0:.6f}",
                    f"{float(hip_outward_after_gate[leg_i]) if hip_outward_after_gate.size >= 4 else 0.0:.6f}",
                    f"{float(cpg_action_info.get('command_gate_scale', self._last_action_cmd_gate_scale)):.6f}",
                    int(cpg_action_info.get("zero_cmd_stand", self._last_zero_cmd_stand_active)),
                    f"{policy_loop_dt_ms:.3f}",
                    f"{policy_loop_hz:.3f}",
                    f"{motor_send_dt_ms:.3f}",
                    f"{motor_send_hz:.3f}",
                    f"{motor_state_poll_dt_ms:.3f}",
                    f"{onnx_infer_dt_ms:.3f}",
                    policy_steps_since_last_csv,
                    motor_sends_since_last_csv,
                    f"{state_age_ms:.3f}",
                    f"{loop_overrun_ms:.3f}",
                ]
            )
        self._debug_csv_rows_since_flush += 1
        if self._debug_csv_rows_since_flush >= self.debug_csv_flush_every_n:
            self._debug_csv_file.flush()
            self._debug_csv_rows_since_flush = 0

    def prepare_action_for_target(self, action_policy: np.ndarray) -> np.ndarray:
        action = np.asarray(action_policy, dtype=np.float32).reshape(12).copy()

        if self.semantic_yaw_180 or self.action_leg_yaw_180:
            action = self.swap_action_legs_yaw_180(action)

        return self.apply_action_multipliers(action)

    def swap_action_legs_yaw_180(self, action_policy: np.ndarray) -> np.ndarray:
        action = np.asarray(action_policy, dtype=np.float32).reshape(12)
        swapped = action.copy()
        mapper = self.obs_builder.mapper
        leg_indices = {}
        for i, name in enumerate(mapper.policy_joint_names):
            leg = self.leg_prefix_from_joint_name(name)
            leg_indices.setdefault(leg, []).append(i)

        for leg_a, leg_b in (("FR", "RL"), ("FL", "RR")):
            idx_a = leg_indices.get(leg_a)
            idx_b = leg_indices.get(leg_b)
            if not idx_a or not idx_b or len(idx_a) != len(idx_b):
                continue
            values_a = action[idx_a].copy()
            values_b = action[idx_b].copy()
            for j, (ia, ib) in enumerate(zip(idx_a, idx_b)):
                swapped[ia] = values_b[j]
                swapped[ib] = values_a[j]
        return swapped.astype(np.float32)

    def apply_action_multipliers(self, action_policy: np.ndarray) -> np.ndarray:
        action = np.asarray(action_policy, dtype=np.float32).reshape(12).copy()
        mapper = self.obs_builder.mapper

        for i, name in enumerate(mapper.policy_joint_names):
            leg = self.leg_prefix_from_joint_name(name)
            is_rear = leg in ("RL", "RR")
            leg_mul = (
                self.rear_action_scale_mul if is_rear else self.front_action_scale_mul
            )
            if "hip" in name:
                joint_mul = self.hip_action_scale_mul
            elif "thigh" in name:
                thigh_mul = (
                    self.rear_thigh_action_scale_mul
                    if is_rear
                    else self.thigh_action_scale_mul
                )
                joint_mul = thigh_mul * self.thigh_action_sign
            else:
                joint_mul = (
                    self.rear_calf_action_scale_mul
                    if is_rear
                    else self.calf_action_scale_mul
                )
            action[i] *= leg_mul * joint_mul

        if self.clip_action:
            clip_limit = abs(float(self.action_clip_limit))
            action = np.clip(action, -clip_limit, clip_limit)

        return action.astype(np.float32)

    def recheck_stale_motor_feedback(
        self,
        obs: np.ndarray,
        info: dict,
        stale_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        motor_ids = self.obs_builder.mapper.get_real_motor_ids()
        real_names = self.obs_builder.mapper.real_joint_names
        old_age_ms = np.asarray(info["age_ms"], dtype=np.float32).copy()
        refreshed = []

        for idx in stale_indices:
            i = int(idx)
            mid = int(motor_ids[i])
            try:
                item = self.obs_builder.motor.get_one_motor_state(mid)
            except Exception as e:
                self.get_logger().warn(
                    f"[SAFE] Recheck failed for 0x{mid:02X}/{real_names[i]}: {e}"
                )
                continue

            new_age_ms = float(item.get("age_ms", 999999.0))
            if new_age_ms > old_age_ms[i]:
                continue

            info["q_real"][i] = float(item.get("angle", info["q_real"][i]))
            info["dq_real"][i] = float(item.get("speed", info["dq_real"][i]))
            if "torque_real" in info:
                info["torque_real"][i] = float(item.get("torque", info["torque_real"][i]))
            if "temp_real" in info:
                info["temp_real"][i] = float(item.get("temp", info["temp_real"][i]))
            info["online"][i] = bool(item.get("online", info["online"][i]))
            if "error_code" in info:
                info["error_code"][i] = int(item.get("error_code", info["error_code"][i]))
            info["age_ms"][i] = new_age_ms

            if new_age_ms <= self.max_motor_age_ms:
                refreshed.append(
                    f"0x{mid:02X}/{real_names[i]}:{old_age_ms[i]:.0f}->{new_age_ms:.0f}ms"
                )

        if refreshed:
            q_policy, dq_policy = self.obs_builder.mapper.real_to_policy_q_dq(
                q_real=info["q_real"],
                dq_real=info["dq_real"],
            )
            q_policy = self.obs_builder.transform_policy_array_for_obs(q_policy)
            dq_policy = self.obs_builder.transform_policy_array_for_obs(dq_policy)
            info["q_policy"] = q_policy.copy()
            info["dq_policy"] = dq_policy.copy()
            obs[12:24] = q_policy * self.obs_builder.dof_pos_scale
            obs[24:36] = dq_policy * self.obs_builder.dof_vel_scale
            if obs.shape[0] >= 48:
                obs[36:48] = self.obs_builder.last_action
            if obs.shape[0] >= 50:
                obs[48:50] = self.obs_builder.get_gait_phase_obs()
            self.get_logger().info(
                f"[SAFE] Rechecked stale motor feedback: {', '.join(refreshed)}"
            )

        return obs, info

    def maybe_print_policy_debug(
        self,
        action_raw: np.ndarray,
        action: np.ndarray,
        q_des: np.ndarray,
        current_q: np.ndarray,
        error: np.ndarray,
        max_age: float,
        pre_limit_info: dict = None,
        torque_limit_info: dict = None,
        rear_bias_info: dict = None,
    ):
        if not self.debug_print_arrays:
            return

        now = time.time()
        if self.debug_print_period_sec > 0.0:
            if now - self._last_debug_print_time < self.debug_print_period_sec:
                return
        self._last_debug_print_time = now

        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(12)
        action = np.asarray(action, dtype=np.float32).reshape(12)
        q_des = np.asarray(q_des, dtype=np.float32).reshape(12)
        current_q = np.asarray(current_q, dtype=np.float32).reshape(12)
        error = np.asarray(error, dtype=np.float32).reshape(12)

        action_raw_abs_max = float(np.max(np.abs(action_raw)))
        action_abs_max = float(np.max(np.abs(action)))
        error_abs_max = float(np.max(np.abs(error)))
        pre_limit_info = {} if pre_limit_info is None else pre_limit_info
        torque_limit_info = {} if torque_limit_info is None else torque_limit_info
        rear_bias_info = {} if rear_bias_info is None else rear_bias_info

        flags = []
        if action_raw_abs_max >= self.debug_warn_action_abs:
            flags.append(
                f"raw action close to limit: max_abs={action_raw_abs_max:.3f} "
                f">= {self.debug_warn_action_abs:.3f}"
            )
        if error_abs_max >= self.debug_critical_error_rad:
            flags.append(
                f"ERROR > {self.debug_critical_error_rad:.3f} rad: "
                f"max_abs={error_abs_max:.3f}"
            )
        elif error_abs_max >= self.debug_warn_error_rad:
            flags.append(
                f"error > {self.debug_warn_error_rad:.3f} rad: "
                f"max_abs={error_abs_max:.3f}"
            )

        print("=" * 100, flush=True)
        print(
            "[POLICY_DEBUG] "
            f"print_only={self.print_only} send={self.enable_send} "
            f"max_age={max_age:.1f}ms "
            f"raw_action_abs_max={action_raw_abs_max:.3f} "
            f"action_abs_max={action_abs_max:.3f} "
            f"error_abs_max={error_abs_max:.3f} "
            f"q_raw_error_abs_max={pre_limit_info.get('q_raw_error_abs_max', 0.0):.3f} "
            f"q_cmd_error_abs_max={pre_limit_info.get('q_cmd_error_abs_max', 0.0):.3f} "
            f"qdot_cmd_abs_max={pre_limit_info.get('qdot_cmd_abs_max', 0.0):.3f} "
            f"qddot_cmd_abs_max={pre_limit_info.get('qddot_cmd_abs_max', 0.0):.3f} "
            f"err_limit_min={pre_limit_info.get('err_limit_min', 0.0):.3f} "
            f"err_limit_max={pre_limit_info.get('err_limit_max', 0.0):.3f} "
            f"pre/rate/accel/final_limited="
            f"{pre_limit_info.get('pre_limited_count', 0)}/"
            f"{pre_limit_info.get('rate_limited_count', 0)}/"
            f"{pre_limit_info.get('accel_limited_count', 0)}/"
            f"{torque_limit_info.get('limited_count', 0)} "
            f"rear_bias={int(rear_bias_info.get('enabled', False))} "
            f"rear_calf_bias={float(rear_bias_info.get('rear_calf_extend_bias_policy_rad', 0.0)):+.3f} "
            f"rear_thigh_bias={float(rear_bias_info.get('rear_thigh_bias_policy_rad', 0.0)):+.3f}",
            flush=True,
        )
        if flags:
            print("[POLICY_DEBUG][WARN] " + " | ".join(flags), flush=True)

        policy_names = self.obs_builder.mapper.get_policy_joint_names()
        print(
            f"action_raw[12] policy order = {policy_names}:",
            self.format_array(action_raw),
            flush=True,
        )
        print(
            f"action[12] used for q_des, policy order = {policy_names}:",
            self.format_array(action),
            flush=True,
        )
        print(
            "q_des[12] real motor order = FR, FL, RL, RR:",
            self.format_array(q_des),
            flush=True,
        )
        print(
            "current_q[12] real motor order = FR, FL, RL, RR:",
            self.format_array(current_q),
            flush=True,
        )
        print(
            "error[12] = q_des - current_q:",
            self.format_array(error),
            flush=True,
        )

        motor_ids = self.obs_builder.mapper.get_real_motor_ids()
        real_names = self.obs_builder.mapper.real_joint_names
        for i, (mid, name) in enumerate(zip(motor_ids, real_names)):
            mark = ""
            abs_error = abs(float(error[i]))
            if abs_error >= self.debug_critical_error_rad:
                mark = "  <-- ERROR > 1 rad"
            elif abs_error >= self.debug_warn_error_rad:
                mark = "  <-- error > 0.5 rad"
            print(
                f"  real[{i:02d}] motor_id=0x{mid:02X} {name:16s} "
                f"q_des={float(q_des[i]):+8.4f} "
                f"current_q={float(current_q[i]):+8.4f} "
                f"error={float(error[i]):+8.4f}{mark}",
                flush=True,
            )

    @staticmethod
    def format_array(arr: np.ndarray) -> str:
        return np.array2string(
            np.asarray(arr, dtype=np.float32).reshape(12),
            precision=4,
            suppress_small=False,
            separator=", ",
            max_line_width=200,
        )

    def destroy_node(self):
        self._stop_debug_csv_worker()
        try:
            if self._debug_csv_file is not None:
                self._debug_csv_file.flush()
                self._debug_csv_file.close()
        except Exception:
            pass
        if self._debug_csv_dropped:
            self.get_logger().warn(
                f"[DEBUG_CSV] dropped {self._debug_csv_dropped} records because the queue was full"
            )
        try:
            self.obs_builder.stop()
        except Exception:
            pass
        try:
            self.http_session.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MydogPolicyNode()

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
