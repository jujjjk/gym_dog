#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fanfan_v4_migration_core.py

IsaacLab V4 FastDiagonalTrot + Light VMC + safety_profile
``performance_soft_output_v2_light_vmc_balance_v4`` 的纯 Python 迁移核心。

设计原则
========
1. 这个文件不依赖 ROS2 / rclpy / HTTP / Jetson 电机接口，只做仿真 V4 的核心计算。
2. 它不是“再写一套相似逻辑”，而是把 IsaacLab 中已经验证的下列源码逐项迁移过来：
     scripts/environments/fanfan_reference_debug.py                 (golden CSV 生成 / safety_profile 覆盖)
     .../fanfan_rl_cpg_residual/residual_action.py                  (FastDiagonalTrot + Light VMC + 输出滤波 + 安全链)
     .../fanfan_rl_cpg_residual/reference_gait.py                   (full sagittal FK/IK + base_phase)
     .../fanfan_rl_cpg_residual/flat_env_cfg.py                     (任务默认站姿 / dt / reference_cfg)
     .../fanfan_a1_clean/fanfan_robot_cfg.py                        (FANFAN_TEXT_STAND_JOINT_POS / 站姿覆盖)
     .../fanfan_a1_clean/deploy_actions.py                          (DeployFilteredJointPositionAction)
     .../fanfan_a1_clean/rs01_motor_params.py                       (RS01 KP/KD/torque)

   仿真中所有数学（FK/IK、phase、swing/stance shaping、light VMC、yaw damping、rear guard、
   rate limiter、torque soft backoff）都按原样迁移，不写 local linearization、不写经验系数 IK。

policy joint order (固定)
=========================
    FR_hip, FR_thigh, FR_calf,
    FL_hip, FL_thigh, FL_calf,
    RR_hip, RR_thigh, RR_calf,
    RL_hip, RL_thigh, RL_calf

leg index: 0=FR, 1=FL, 2=RR, 3=RL.

URDF hip outward signs (固定): FR=-1, FL=+1, RR=-1, RL=+1.

真机传感器缺失替代规则（见 §7）：
- base_height 缺失   -> height VMC = 0 或可配置估计值, height_source 记录
- foot_force 缺失    -> rear early-contact 用 q_error/tau_est 替代, early_contact_source 记录
- foot body z 缺失   -> rear late-swing guard 用 reference predicted foot height, predicted_foot_height 记录
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

MIGRATION_CORE_VERSION = "fanfan_v4_migration_core/1.0.0"

# ----------------------------------------------------------------------------
# 固定常量 (policy 语义)
# ----------------------------------------------------------------------------
POLICY_LEG_ORDER = ("FR", "FL", "RR", "RL")
POLICY_JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

# URDF hip outward signs，禁止使用 legacy (1,1,-1,1)。
URDF_HIP_OUTWARD_SIGNS = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
LEGACY_HIP_OUTWARD_SIGNS = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float64)

# light VMC 的体坐标分量 (residual_action.py)
SIDE_SIGN = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)       # roll: 左右
FORE_AFT_SIGN = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)   # pitch: 前后
REAR_MASK = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

# sim_v4 level rear stand pose (policy order)。
# 来自 FANFAN_TEXT_STAND_JOINT_POS + flat_env_cfg._set_rear_stand_pose 覆盖
# (RR/RL thigh 0.2269->0.3491, calf -0.3491->-0.7854)。FastDiagonalTrot 使用
# FanfanSmallHighFreqReferenceGaitCfg(apply_default_pose_offsets=False)，所以不再做 offset。
SIM_V4_DEFAULT_JOINT_POS_POLICY = np.array(
    [
        -0.1571, 0.3491, -0.7854,   # FR
        0.1571, 0.3491, -0.7854,    # FL
        -0.1571, 0.3491, -0.7854,   # RR
        0.1571, 0.3491, -0.7854,    # RL
    ],
    dtype=np.float64,
)
# 显式别名：core 内部统一使用 sim_semantic 空间。
SIM_V4_DEFAULT_JOINT_POS_SIM = SIM_V4_DEFAULT_JOINT_POS_POLICY

# JointSemanticMapper.default_joint_angle 的站姿 (mapper default)，仅用于对比报警。
MAPPER_DEFAULT_JOINT_POS_POLICY = np.array(
    [
        -0.1571, 0.3491, -0.7854,   # FR
        0.1571, 0.3491, -0.7854,    # FL
        -0.1571, 0.2269, -0.3491,   # RR
        0.1571, 0.2269, -0.3491,    # RL
    ],
    dtype=np.float64,
)


def _smoothstep01(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smootherstep01(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (x * (x * 6.0 - 15.0) + 10.0)


# ----------------------------------------------------------------------------
# V4 配置 (balanced preset + mid_soft KP + performance_soft_output_v2_light_vmc_balance_v4)
# 所有数值都来自上面列出的 IsaacLab 源码，不是新调出来的。
# ----------------------------------------------------------------------------
@dataclass
class V4Config:
    # --- 控制率 (velocity_env_cfg: decimation 4, sim.dt 0.005 -> 50Hz) ---
    dt: float = 0.02

    # --- full sagittal FK/IK (reference_gait.py) ---
    thigh_length: float = 0.1560608
    calf_length: float = 0.1489418
    workspace_margin_m: float = 0.005  # FanfanSmallHighFreqReferenceGaitCfg

    # --- balanced preset (fanfan_reference_debug.py trot_preset=balanced) ---
    fast_trot_step_hz: float = 1.15
    fast_trot_duty_factor: float = 0.61
    fast_trot_stride_length_m: float = 0.022
    fast_trot_front_swing_height_m: float = 0.048
    fast_trot_rear_swing_height_m: float = 0.067
    # balanced 给 0.009，但 v4 profile 覆盖为 0.0055
    fast_trot_support_preload_z_m: float = 0.0055
    fast_trot_warmup_sec: float = 2.0

    # --- gait shaping defaults (residual_action.py cfg) ---
    fast_trot_swing_lift_peak_phase: float = 0.45
    fast_trot_touchdown_phase: float = 0.82
    fast_trot_early_stance_blend: float = 0.24       # v4 override
    fast_trot_support_preload_ramp_in_phase: float = 0.16   # v4 override
    fast_trot_support_preload_ramp_out_phase: float = 0.16  # v4 override
    fast_trot_support_preload_gate_max: float = 0.60        # v4 override
    fast_trot_global_support_height_offset_m: float = 0.0   # v4 不设置

    # --- phase switch guard (v3/v4) ---
    fast_trot_phase_switch_guard_window: float = 0.055
    fast_trot_phase_switch_guard_hip_kp: float = 58.0
    fast_trot_phase_switch_guard_thigh_kp: float = 125.0
    fast_trot_phase_switch_guard_calf_kp: float = 135.0
    fast_trot_phase_switch_guard_kd: float = 6.2
    fast_trot_phase_switch_kp_scale: float = 0.75

    # --- mid_soft KP (v2+ override in fanfan_reference_debug.py) ---
    fast_trot_swing_hip_kp: float = 50.0
    fast_trot_swing_thigh_kp: float = 80.0
    fast_trot_swing_calf_kp: float = 80.0
    fast_trot_swing_kd: float = 5.0
    fast_trot_touchdown_hip_kp: float = 55.0
    fast_trot_touchdown_thigh_kp: float = 110.0
    fast_trot_touchdown_calf_kp: float = 120.0
    fast_trot_touchdown_kd: float = 6.0
    fast_trot_early_stance_hip_kp: float = 60.0
    fast_trot_early_stance_thigh_kp: float = 130.0
    fast_trot_early_stance_calf_kp: float = 140.0
    fast_trot_early_stance_kd: float = 6.0
    fast_trot_support_hip_kp: float = 62.0
    fast_trot_support_thigh_kp: float = 140.0
    fast_trot_support_calf_kp: float = 150.0
    fast_trot_support_kd: float = 6.0

    # --- torque soft backoff (residual_action cfg defaults + v3/v4 guard override) ---
    fast_trot_soft_output_start_torque: float = 10.0
    fast_trot_soft_output_full_torque: float = 14.0
    fast_trot_guard_soft_start_torque: float = 9.5
    fast_trot_guard_soft_full_torque: float = 13.5
    sim_hard_torque_budget: float = 17.0

    # --- light VMC (balance v3/v4) ---
    light_vmc_target_base_height: float = 0.288
    light_vmc_target_roll: float = 0.0
    light_vmc_target_pitch: float = -0.04
    light_vmc_height_kp_z: float = 0.30
    light_vmc_height_kd_z: float = 0.04
    light_vmc_height_corr_limit_m: float = 0.004
    light_vmc_roll_kp_z: float = 0.025
    light_vmc_roll_kd_z: float = 0.006
    light_vmc_roll_corr_limit_m: float = 0.0035
    light_vmc_pitch_kp_z: float = 0.025
    light_vmc_pitch_kd_z: float = 0.005
    light_vmc_pitch_corr_limit_m: float = 0.003
    light_vmc_z_sign: float = -1.0
    light_vmc_roll_sign: float = 1.0
    light_vmc_pitch_sign: float = -1.0
    light_vmc_touchdown_ramp: float = 0.12
    light_vmc_preswing_ramp: float = 0.12
    light_vmc_max_weight: float = 1.0
    light_vmc_phase_switch_weight_scale: float = 0.50
    light_vmc_z_offset_rate_limit_m: float = 0.001
    light_vmc_xy_offset_rate_limit_m: float = 0.001
    light_vmc_enable_foot_placement: bool = False  # balance 系列关闭
    light_vmc_vx_foot_k: float = 0.025
    light_vmc_vy_foot_k: float = 0.020
    light_vmc_pitch_rate_foot_x_k: float = 0.005
    light_vmc_roll_rate_foot_y_k: float = 0.005
    light_vmc_foot_x_corr_limit_m: float = 0.006
    light_vmc_foot_y_corr_limit_m: float = 0.004

    # --- yaw damping (v3/v4) ---
    enable_light_yaw_damping: bool = True
    light_yaw_kp_hip: float = 0.0025
    light_yaw_kd_hip: float = 0.006
    light_yaw_hip_limit_rad: float = 0.007
    light_yaw_hip_rate_limit_rad: float = 0.001
    light_yaw_phase_switch_weight_scale: float = 0.40
    light_yaw_sign: float = 1.0

    # --- rear preswing unload (v4: z=0) ---
    rear_preswing_unload_enable: bool = True
    rear_preswing_unload_window: float = 0.12
    rear_preswing_unload_z_m: float = 0.0   # v4 override
    rear_preswing_vmc_fade_window: float = 0.14
    rear_unload_sign: float = 1.0

    # --- rear touchdown Kp/Kd ramp (v4) ---
    rear_touchdown_vmc_ramp: float = 0.20
    rear_touchdown_kp_ramp: float = 0.24
    rear_touchdown_kp_scale: float = 0.75
    rear_touchdown_hip_kp_limit: float = 58.0
    rear_touchdown_thigh_kp_limit: float = 125.0
    rear_touchdown_calf_kp_limit: float = 130.0
    rear_touchdown_kd: float = 6.2

    # --- rear late-swing clearance guard + descent softening (v4) ---
    rear_late_swing_guard_enable: bool = True
    rear_late_swing_phase_start: float = 0.28
    rear_late_swing_phase_end: float = 0.38
    rear_late_swing_clearance_margin_m: float = 0.003
    rear_late_swing_min_height_m: float = 0.003
    rear_late_swing_guard_rate_limit_m: float = 0.001
    rear_late_swing_clearance_sign: float = 1.0
    rear_late_swing_descent_soft_enable: bool = True
    rear_late_swing_descent_scale: float = 0.50
    rear_late_swing_descent_rate_limit_m: float = 0.001

    # --- rear early-contact guard (v4) ---
    rear_early_contact_guard_enable: bool = True
    rear_early_contact_force_threshold: float = 10.0
    rear_early_contact_phase_start: float = 0.28
    rear_early_contact_phase_end: float = 0.40
    rear_early_contact_lift_relief_m: float = 0.002
    rear_early_contact_relief_sign: float = 1.0
    rear_early_contact_kp_scale: float = 0.60
    rear_early_contact_hip_kp_limit: float = 55.0
    rear_early_contact_thigh_kp_limit: float = 115.0
    rear_early_contact_calf_kp_limit: float = 115.0
    rear_early_contact_kd: float = 6.5
    rear_early_contact_torque_soft_start: float = 9.0
    rear_early_contact_torque_soft_full: float = 13.0

    # --- output safety chain (v4 profile in fanfan_reference_debug.py) ---
    enable_deploy_target_filter: bool = True
    enable_target_rate_limit: bool = True
    enable_target_accel_limit: bool = False
    enable_torque_target_limit: bool = True
    enable_action_delay: bool = False
    sim_target_rate_limit: float = 9.0      # sim_target_rate_limit_range = (9,9)
    hip_target_rate_mul: float = 7.5 / 9.0  # v4 override
    thigh_target_rate_mul: float = 1.0
    calf_target_rate_mul: float = 1.0
    sim_target_accel_limit: float = 1000.0
    hip_target_accel_mul: float = 1.0
    thigh_target_accel_mul: float = 1.0
    calf_target_accel_mul: float = 1.0
    # kd_scale=1.0 -> damping_scale = sqrt(clip(1,0.5,2)) = 1.0
    kd_scale: float = 1.0

    # --- 真机替代 / 估计 ---
    base_height_estimate_m: float = 0.288   # base_height 缺失时的可配置估计值（默认=target,使 height VMC=0)
    use_base_height_estimate: bool = False  # False -> height VMC = 0

    def metadata(self) -> dict:
        return {
            "trot_preset": "balanced",
            "support_kp_level": "mid_soft",
            "safety_profile": "performance_soft_output_v2_light_vmc_balance_v4",
            "migration_core_version": MIGRATION_CORE_VERSION,
        }


# ----------------------------------------------------------------------------
# step() 的输入 / 输出
# ----------------------------------------------------------------------------
@dataclass
class CoreInputs:
    """每个控制周期来自 ROS2 外壳的替代量。"""
    # IMU (真机 / dry-run)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gyro: tuple = (0.0, 0.0, 0.0)          # body angular velocity rad/s (roll,pitch,yaw 轴)
    lin_vel: tuple = (0.0, 0.0, 0.0)       # body linear velocity m/s (一般真机不可用)
    imu_valid: bool = False

    # 电机反馈 (sim_semantic 空间, absolute angle)。
    # 由 ROS2 wrapper 用 SimRealSemanticBridge.real_policy_to_sim(...) 转换后传入。
    q_actual_sim: Optional[np.ndarray] = None   # (12,) rad
    dq_actual_sim: Optional[np.ndarray] = None
    tau_sim: Optional[np.ndarray] = None
    feedback_valid: bool = False

    # base height (一般真机不可用 -> None)
    base_height_m: Optional[float] = None

    # foot normal force (一般真机不可用 -> None)，顺序 FR,FL,RR,RL
    foot_force: Optional[np.ndarray] = None

    # 运行模式
    test_mode: str = "air"
    dry_run_virtual_feedback: bool = True
    # VMC 整体比例 (air≈0, touch 小, assist 中, short_free 满)。
    # CPG reference 本体不受影响，只缩放 VMC correction（见 §7 原则）。
    vmc_scale: float = 1.0


class FanfanV4MigrationCore:
    """IsaacLab V4 FastDiagonalTrot + Light VMC + soft_output_v2_light_vmc_balance_v4 的纯计算核心。"""

    SUPPORTED_PRESET = "balanced"
    SUPPORTED_KP_LEVEL = "mid_soft"
    SUPPORTED_PROFILE = "performance_soft_output_v2_light_vmc_balance_v4"

    def __init__(
        self,
        cfg: Optional[V4Config] = None,
        *,
        default_joint_pos_policy: Optional[np.ndarray] = None,
        trot_preset: str = "balanced",
        support_kp_level: str = "mid_soft",
        safety_profile: str = "performance_soft_output_v2_light_vmc_balance_v4",
    ) -> None:
        if trot_preset != self.SUPPORTED_PRESET:
            raise ValueError(
                f"migration core 只迁移了已验证的 trot_preset={self.SUPPORTED_PRESET!r}, "
                f"收到 {trot_preset!r}。不允许临时发明新步态。"
            )
        if support_kp_level != self.SUPPORTED_KP_LEVEL:
            raise ValueError(
                f"migration core 只迁移了 support_kp_level={self.SUPPORTED_KP_LEVEL!r}, 收到 {support_kp_level!r}."
            )
        if safety_profile != self.SUPPORTED_PROFILE:
            raise ValueError(
                f"migration core 只迁移了 safety_profile={self.SUPPORTED_PROFILE!r}, 收到 {safety_profile!r}."
            )
        self.cfg = cfg or V4Config()
        self.trot_preset = trot_preset
        self.support_kp_level = support_kp_level
        self.safety_profile = safety_profile

        if default_joint_pos_policy is None:
            self.default_joint_pos = SIM_V4_DEFAULT_JOINT_POS_POLICY.copy()
        else:
            self.default_joint_pos = np.asarray(default_joint_pos_policy, dtype=np.float64).reshape(12).copy()

        # default foot 位置 (FK)
        self.default_foot_x, self.default_foot_z = self._forward_sagittal(
            self.default_joint_pos[1::3], self.default_joint_pos[2::3]
        )
        self.reset()

    # ------------------------------------------------------------------
    # full sagittal FK / IK (reference_gait.py 原样迁移)
    # ------------------------------------------------------------------
    def _forward_sagittal(self, thigh: np.ndarray, calf: np.ndarray):
        l1, l2 = self.cfg.thigh_length, self.cfg.calf_length
        x = -l1 * np.sin(thigh) - l2 * np.sin(thigh + calf)
        z = -l1 * np.cos(thigh) - l2 * np.cos(thigh + calf)
        return x, z

    def _inverse_sagittal(self, x: np.ndarray, z: np.ndarray):
        l1, l2 = self.cfg.thigh_length, self.cfg.calf_length
        reach = np.sqrt(np.clip(x * x + z * z, 1.0e-8, None))
        margin = max(self.cfg.workspace_margin_m, 1.0e-5)
        min_reach = abs(l1 - l2) + margin
        max_reach = l1 + l2 - margin
        scale = np.clip(reach, min_reach, max_reach) / reach
        x = x * scale
        z = z * scale
        cos_calf = np.clip((x * x + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        calf = -np.arccos(cos_calf)
        thigh = np.arctan2(-x, -z) - np.arctan2(l2 * np.sin(calf), l1 + l2 * np.cos(calf))
        return thigh, calf

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.step_index = 0          # = sim 中的 _rear_lift_step
        self.base_phase = 0.0
        cfg = self.cfg
        # light VMC 滤波状态
        self._vmc_z_offset = np.zeros(4)
        self._vmc_x_offset = np.zeros(4)
        self._vmc_y_offset = np.zeros(4)
        self._yaw_hip_offset = np.zeros(4)
        self._yaw_target = 0.0
        self._yaw_target_valid = False
        self._late_swing_clearance_offset = np.zeros(4)
        # 输出滤波 (rate limiter) 状态
        self._q_last_cmd = self.default_joint_pos.copy()
        self._qdot_last_cmd = np.zeros(12)
        self.q_cmd_final = self.default_joint_pos.copy()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def step(self, inp: CoreInputs) -> dict:
        cfg = self.cfg
        dt = cfg.dt

        # --- warmup 与 base_phase (residual_action._fast_diagonal_trot_target) ---
        warmup = float(np.clip(self.step_index * dt / max(cfg.fast_trot_warmup_sec, 1.0e-6), 0.0, 1.0))
        frequency = float(cfg.fast_trot_step_hz)
        stride = cfg.fast_trot_stride_length_m * warmup
        self.base_phase = (self.base_phase + frequency * dt) % 1.0
        phase_a = self.base_phase
        phase_b = (self.base_phase + 0.5) % 1.0

        phase_to_switch = min(min(self.base_phase, 1.0 - self.base_phase), abs(self.base_phase - 0.5))
        guard_window = max(1.0e-6, cfg.fast_trot_phase_switch_guard_window)
        guard_strength = float(_smootherstep01((guard_window - phase_to_switch) / guard_window))

        leg_phase = np.zeros(4)
        leg_phase[[0, 3]] = phase_a   # FR, RL
        leg_phase[[1, 2]] = phase_b   # FL, RR

        swing_fraction = max(0.05, min(0.49, 1.0 - cfg.fast_trot_duty_factor))
        swing_mask = leg_phase < swing_fraction
        support_mask = ~swing_mask

        pair_a_swing = bool(swing_mask[0] or swing_mask[3])
        pair_b_swing = bool(swing_mask[1] or swing_mask[2])
        active_swing_pair = 1 if pair_a_swing else (2 if pair_b_swing else 0)
        expected_support_pair = 2 if pair_a_swing else (1 if pair_b_swing else 0)

        # --- swing / stance shaping ---
        s_swing = np.clip(leg_phase / swing_fraction, 0.0, 1.0)
        s_stance = np.clip((leg_phase - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0)
        advance = _smootherstep01(s_swing)
        peak_phase = min(0.80, max(0.20, cfg.fast_trot_swing_lift_peak_phase))
        touchdown_phase = min(0.98, max(peak_phase + 0.05, cfg.fast_trot_touchdown_phase))
        lift_up = _smootherstep01(np.clip(s_swing / peak_phase, 0.0, 1.0))
        lift_down = 1.0 - _smootherstep01(
            np.clip((s_swing - peak_phase) / max(touchdown_phase - peak_phase, 1.0e-6), 0.0, 1.0)
        )
        swing_shape = lift_up * lift_down * swing_mask * (s_swing < touchdown_phase)
        stance_progress = _smootherstep01(s_stance)
        early_stance = min(0.30, max(0.0, cfg.fast_trot_early_stance_blend))
        touchdown_progress = np.clip(
            (s_swing - touchdown_phase) / max(1.0 - touchdown_phase, 1.0e-6), 0.0, 1.0
        )
        touchdown_blend = _smootherstep01(touchdown_progress) * swing_mask
        early_stance_gate = np.clip(1.0 - s_stance / max(early_stance, 1.0e-6), 0.0, 1.0)
        early_stance_gate = _smootherstep01(early_stance_gate) * (~swing_mask)

        # leg swing heights (front / rear)
        leg_height = np.array(
            [
                cfg.fast_trot_front_swing_height_m,
                cfg.fast_trot_front_swing_height_m,
                cfg.fast_trot_rear_swing_height_m,
                cfg.fast_trot_rear_swing_height_m,
            ]
        ) * warmup
        swing_height = float(np.max(leg_height))

        x_default = self.default_foot_x
        z_default = self.default_foot_z

        # --- support preload (soft_output_v2 path) ---
        ramp_in = max(1.0e-6, cfg.fast_trot_support_preload_ramp_in_phase)
        ramp_out = max(1.0e-6, cfg.fast_trot_support_preload_ramp_out_phase)
        ramp_in_gate = _smootherstep01(np.clip(s_stance / ramp_in, 0.0, 1.0))
        ramp_out_gate = _smootherstep01(np.clip((1.0 - s_stance) / ramp_out, 0.0, 1.0))
        preload_gate = ramp_in_gate * ramp_out_gate * (~swing_mask)
        preload_gate = np.clip(preload_gate, None, cfg.fast_trot_support_preload_gate_max)
        support_preload_gate = np.maximum(preload_gate, touchdown_blend)
        support_preload_gate = np.clip(support_preload_gate, None, cfg.fast_trot_support_preload_gate_max)

        support_preload = -cfg.fast_trot_support_preload_z_m * support_preload_gate * warmup
        support_height_offset = (
            -cfg.fast_trot_global_support_height_offset_m * (~swing_mask).astype(np.float64) * warmup
        )

        x_swing = x_default - 0.5 * stride + stride * advance
        x_stance = x_default + 0.5 * stride - stride * stance_progress
        z_swing = z_default + leg_height * swing_shape
        z_stance = z_default + support_height_offset + support_preload
        x_cpg = np.where(swing_mask, x_swing, x_stance)
        # soft_output profile: touchdown blend z
        z_touchdown = z_swing * (1.0 - touchdown_blend) + z_stance * touchdown_blend
        z_cpg = np.where(swing_mask, z_touchdown, z_stance)

        # --- pure CPG q (no VMC), 仅用于诊断 / q_cpg_policy 输出 ---
        thigh_cpg, calf_cpg = self._inverse_sagittal(x_cpg, z_cpg)
        q_cpg = self.default_joint_pos.copy()
        q_cpg[1::3] = thigh_cpg
        q_cpg[2::3] = calf_cpg

        # --- light VMC offsets ---
        vmc = self._light_vmc_offsets(
            inp=inp,
            swing_mask=swing_mask,
            leg_phase=leg_phase,
            s_stance=s_stance,
            touchdown_blend=touchdown_blend,
            guard_strength=guard_strength,
            warmup=warmup,
        )
        vmc_x_offset = vmc["x_offset"]
        vmc_y_offset = vmc["y_offset"]
        vmc_z_offset = vmc["z_offset"]

        # --- rear preswing unload (v4 z=0, 但保留 gate 记录) ---
        rear_unload_offset = (
            cfg.rear_unload_sign * cfg.rear_preswing_unload_z_m * vmc["rear_preswing_unload_gate"] * warmup
        )

        x_target = x_cpg + vmc_x_offset
        z_target = z_cpg + vmc_z_offset + rear_unload_offset

        # --- rear late-swing guard / descent softening / early-contact (v4) ---
        rear_guard = self._rear_guards(
            inp=inp,
            leg_phase=leg_phase,
            swing_mask=swing_mask,
            touchdown_blend=touchdown_blend,
            z_target=z_target,
            z_default=z_default,
        )
        z_target = rear_guard["z_target"]

        # --- 全目标 IK -> q_ref ---
        thigh_ref, calf_ref = self._inverse_sagittal(x_target, z_target)
        q_ref = self.default_joint_pos.copy()
        q_ref[1::3] = thigh_ref
        q_ref[2::3] = calf_ref
        leg_length = np.clip(np.abs(z_default), 0.15, None)
        # hip = default + body-y placement + yaw damping
        hip_extra = -vmc_y_offset / leg_length + self._yaw_hip_offset
        q_ref[0::3] = self.default_joint_pos[0::3] + hip_extra

        q_vmc_delta = q_ref - q_cpg

        # --- gains (residual_action._apply_fast_trot_gains) ---
        kp, kd, gains_dbg = self._fast_trot_gains(
            swing_mask=swing_mask,
            touchdown_blend=touchdown_blend,
            early_stance_gate=early_stance_gate,
            preload_gate=preload_gate,
            guard_strength=guard_strength,
            rear_early_contact_guard_active=rear_guard["early_contact_guard_active"],
        )

        # --- 输出滤波 + 安全链 -> q_cmd_final ---
        q_cmd_raw = q_ref.copy()
        out = self._output_safety_chain(
            q_raw=q_cmd_raw,
            kp=kp,
            kd=kd,
            inp=inp,
            guard_strength=guard_strength,
            rear_early_contact_guard_active=rear_guard["early_contact_guard_active"],
        )
        q_cmd_final = out["q_cmd_final"]
        self.q_cmd_final = q_cmd_final.copy()

        # --- FK clearance (ref / cmd / actual) ---
        fk_ref_z = self._forward_sagittal(q_ref[1::3], q_ref[2::3])[1]
        fk_cmd_z = self._forward_sagittal(q_cmd_final[1::3], q_cmd_final[2::3])[1]
        q_actual = out["q_actual"]
        fk_act_z = self._forward_sagittal(q_actual[1::3], q_actual[2::3])[1]
        fk_clearance_ref = fk_ref_z - z_default
        fk_clearance_cmd = fk_cmd_z - z_default
        fk_clearance_actual = fk_act_z - z_default
        predicted_foot_height = fk_clearance_ref  # = sim predicted_foot_lift

        relative_time = self.step_index * dt
        self.step_index += 1

        debug = {
            # 基础
            "relative_time": relative_time,
            "phase": float(self.base_phase),
            "warmup": warmup,
            "frequency": frequency,
            "stride": stride,
            "swing_height": swing_height,
            "duty_factor": cfg.fast_trot_duty_factor,
            "active_swing_pair": active_swing_pair,
            "support_pair": expected_support_pair,
            "leg_phase": leg_phase.copy(),
            "swing_progress": s_swing.copy(),
            "swing_mask": swing_mask.copy(),
            "support_mask": support_mask.copy(),
            "preload_gate": preload_gate.copy(),
            "support_preload_gate": support_preload_gate.copy(),
            "early_stance_gate": early_stance_gate.copy(),
            "support_gate": np.maximum(
                (~swing_mask).astype(np.float64), np.maximum(touchdown_blend, early_stance_gate)
            ),
            "support_preload_delta_z": support_preload.copy(),
            "global_support_height_offset_m": float(cfg.fast_trot_global_support_height_offset_m),
            # phase switch guard
            "phase_to_switch": phase_to_switch,
            "phase_switch_guard_active": guard_strength > 1.0e-6,
            "phase_switch_guard_strength": guard_strength,
            # reference / output (sim_semantic 空间)
            "q_cpg_sim": q_cpg.copy(),
            "q_ref_sim": q_ref.copy(),
            "q_vmc_delta_sim": q_vmc_delta.copy(),
            "q_cmd_raw_sim": q_cmd_raw.copy(),
            "q_cmd_final_sim": q_cmd_final.copy(),
            "q_actual_sim": q_actual.copy(),
            "q_error_sim": (q_cmd_final - q_actual).copy(),
            "q_ref_cmd_diff_sim": (q_ref - q_cmd_final).copy(),
            # gains
            "kp_sim": kp.copy(),
            "kd_sim": kd.copy(),
            # FK / clearance
            "fk_clearance_ref": fk_clearance_ref.copy(),
            "fk_clearance_cmd": fk_clearance_cmd.copy(),
            "fk_clearance_actual": fk_clearance_actual.copy(),
            "predicted_foot_height": predicted_foot_height.copy(),
        }
        debug.update(vmc["debug"])
        debug.update(rear_guard["debug"])
        debug.update(gains_dbg)
        debug.update(out["debug"])
        return {
            "q_cpg_sim": q_cpg,
            "q_ref_sim": q_ref,
            "q_vmc_delta_sim": q_vmc_delta,
            "q_cmd_raw_sim": q_cmd_raw,
            "q_cmd_final_sim": q_cmd_final,
            "kp_sim": kp,
            "kd_sim": kd,
            "debug_info": debug,
        }

    # ------------------------------------------------------------------
    # light VMC (residual_action._fast_trot_light_vmc_offsets 原样迁移)
    # ------------------------------------------------------------------
    def _light_vmc_offsets(self, *, inp, swing_mask, leg_phase, s_stance, touchdown_blend, guard_strength, warmup):
        cfg = self.cfg
        dtype = np.float64

        touchdown_ramp = max(1.0e-6, cfg.light_vmc_touchdown_ramp)
        rear_touchdown_ramp = max(1.0e-6, cfg.rear_touchdown_vmc_ramp)
        preswing_ramp = max(1.0e-6, cfg.light_vmc_preswing_ramp)
        early_weight = _smootherstep01(np.clip(s_stance / touchdown_ramp, 0.0, 1.0))
        rear_early_weight = _smootherstep01(np.clip(s_stance / rear_touchdown_ramp, 0.0, 1.0))
        early_weight = early_weight * (1.0 - REAR_MASK) + rear_early_weight * REAR_MASK
        preswing_weight = _smootherstep01(np.clip((1.0 - s_stance) / preswing_ramp, 0.0, 1.0))
        stance_weight = (0.5 + 0.5 * early_weight) * preswing_weight * (~swing_mask).astype(dtype)
        swing_touchdown_weight = 0.5 * touchdown_blend
        vmc_weight = np.clip(
            (stance_weight + swing_touchdown_weight) * cfg.light_vmc_max_weight * warmup,
            0.0,
            cfg.light_vmc_max_weight,
        )
        # phase switch weight scale
        scale = 1.0 - guard_strength * (1.0 - cfg.light_vmc_phase_switch_weight_scale)
        vmc_weight = vmc_weight * scale
        phase_switch_vmc_weight_scale_applied = scale

        # 真机 test_mode VMC 比例 (CPG reference 不变, 只缩放 VMC correction)
        vmc_weight = vmc_weight * float(getattr(inp, "vmc_scale", 1.0))

        # rear preswing unload gate + vmc fade
        if cfg.rear_preswing_unload_enable:
            preswing_window = max(1.0e-6, cfg.rear_preswing_unload_window)
            fade_window = max(1.0e-6, cfg.rear_preswing_vmc_fade_window)
            rear_preswing_gate = _smootherstep01(
                np.clip((leg_phase - (1.0 - preswing_window)) / preswing_window, 0.0, 1.0)
            )
            rear_preswing_gate = rear_preswing_gate * REAR_MASK * (~swing_mask).astype(dtype)
            rear_fade = 1.0 - _smootherstep01(
                np.clip((leg_phase - (1.0 - fade_window)) / fade_window, 0.0, 1.0)
            )
            rear_fade = rear_fade * REAR_MASK + (1.0 - REAR_MASK)
            rear_fade = np.where(swing_mask, 1.0, rear_fade)
            vmc_weight = vmc_weight * rear_fade
        else:
            rear_preswing_gate = np.zeros(4)
            rear_fade = np.ones(4)

        # --- base 状态 (真机替代规则) ---
        roll = float(inp.roll)
        pitch = float(inp.pitch)
        yaw = float(inp.yaw)
        gyro = np.asarray(inp.gyro, dtype=dtype).reshape(3)
        lin_vel = np.asarray(inp.lin_vel, dtype=dtype).reshape(3)

        # base_height 替代
        if inp.base_height_m is not None:
            base_height = float(inp.base_height_m)
            height_source = "measured"
        elif cfg.use_base_height_estimate:
            base_height = float(cfg.base_height_estimate_m)
            height_source = "estimated"
        else:
            base_height = float(cfg.light_vmc_target_base_height)  # 使 height VMC = 0
            height_source = "unavailable"

        height_corr = cfg.light_vmc_height_kp_z * (cfg.light_vmc_target_base_height - base_height)
        height_corr -= cfg.light_vmc_height_kd_z * lin_vel[2]
        height_corr = float(np.clip(height_corr, -cfg.light_vmc_height_corr_limit_m, cfg.light_vmc_height_corr_limit_m))

        roll_corr = cfg.light_vmc_roll_kp_z * (roll - cfg.light_vmc_target_roll)
        roll_corr += cfg.light_vmc_roll_kd_z * gyro[0]
        roll_corr = float(np.clip(roll_corr, -cfg.light_vmc_roll_corr_limit_m, cfg.light_vmc_roll_corr_limit_m))

        pitch_corr = cfg.light_vmc_pitch_kp_z * (pitch - cfg.light_vmc_target_pitch)
        pitch_corr += cfg.light_vmc_pitch_kd_z * gyro[1]
        pitch_corr = float(np.clip(pitch_corr, -cfg.light_vmc_pitch_corr_limit_m, cfg.light_vmc_pitch_corr_limit_m))

        if not inp.imu_valid:
            # IMU 不可用 -> 姿态修正归零，但 height 仍走上面的替代规则
            roll_corr = 0.0
            pitch_corr = 0.0

        z_raw = (
            cfg.light_vmc_z_sign * height_corr
            + SIDE_SIGN * cfg.light_vmc_roll_sign * roll_corr
            + FORE_AFT_SIGN * cfg.light_vmc_pitch_sign * pitch_corr
        ) * vmc_weight

        if cfg.light_vmc_enable_foot_placement:
            x_corr = cfg.light_vmc_vx_foot_k * lin_vel[0] + cfg.light_vmc_pitch_rate_foot_x_k * gyro[1]
            y_corr = cfg.light_vmc_vy_foot_k * lin_vel[1] + cfg.light_vmc_roll_rate_foot_y_k * gyro[0]
            x_corr = float(np.clip(x_corr, -cfg.light_vmc_foot_x_corr_limit_m, cfg.light_vmc_foot_x_corr_limit_m))
            y_corr = float(np.clip(y_corr, -cfg.light_vmc_foot_y_corr_limit_m, cfg.light_vmc_foot_y_corr_limit_m))
        else:
            x_corr = 0.0
            y_corr = 0.0
        x_raw = x_corr * vmc_weight
        y_raw = y_corr * vmc_weight

        z_limit = cfg.light_vmc_z_offset_rate_limit_m
        xy_limit = cfg.light_vmc_xy_offset_rate_limit_m
        z_offset = self._vmc_z_offset + np.clip(z_raw - self._vmc_z_offset, -z_limit, z_limit)
        x_offset = self._vmc_x_offset + np.clip(x_raw - self._vmc_x_offset, -xy_limit, xy_limit)
        y_offset = self._vmc_y_offset + np.clip(y_raw - self._vmc_y_offset, -xy_limit, xy_limit)
        self._vmc_z_offset = z_offset
        self._vmc_x_offset = x_offset
        self._vmc_y_offset = y_offset

        # --- yaw damping ---
        if cfg.enable_light_yaw_damping:
            if not self._yaw_target_valid:
                self._yaw_target = yaw
                self._yaw_target_valid = True
            yaw_error = math.atan2(math.sin(yaw - self._yaw_target), math.cos(yaw - self._yaw_target))
            yaw_corr = cfg.light_yaw_kp_hip * yaw_error + cfg.light_yaw_kd_hip * gyro[2]
            yaw_corr = float(np.clip(yaw_corr, -cfg.light_yaw_hip_limit_rad, cfg.light_yaw_hip_limit_rad))
            yaw_raw_corr = yaw_corr
            yaw_guard_scale = 1.0 - guard_strength * (1.0 - cfg.light_yaw_phase_switch_weight_scale)
            yaw_corr = yaw_corr * yaw_guard_scale
            phase_switch_yaw_weight_scale_applied = yaw_guard_scale
            if not inp.imu_valid:
                yaw_error = 0.0
                yaw_corr = 0.0
                yaw_raw_corr = 0.0
            yaw_raw = SIDE_SIGN * cfg.light_yaw_sign * yaw_corr * vmc_weight * warmup
            yaw_limit = cfg.light_yaw_hip_rate_limit_rad
            yaw_offset = self._yaw_hip_offset + np.clip(yaw_raw - self._yaw_hip_offset, -yaw_limit, yaw_limit)
            yaw_rate_limited = yaw_offset - self._yaw_hip_offset
            self._yaw_hip_offset = yaw_offset
        else:
            yaw_error = 0.0
            yaw_raw_corr = 0.0
            yaw_corr = 0.0
            phase_switch_yaw_weight_scale_applied = 1.0
            yaw_rate_limited = np.zeros(4)
            self._yaw_hip_offset = np.zeros(4)

        debug = {
            "height_source": height_source,
            "real_vmc_scale": float(np.max(vmc_weight)) if vmc_weight.size else 0.0,
            "vmc_weight": vmc_weight.copy(),
            "vmc_height_corr_z": height_corr,
            "vmc_roll_corr_z": roll_corr,
            "vmc_pitch_corr_z": pitch_corr,
            "vmc_foot_x_corr": x_corr,
            "vmc_foot_y_corr": y_corr,
            "vmc_foot_z_offset": z_offset.copy(),
            "vmc_foot_x_offset": x_offset.copy(),
            "vmc_foot_y_offset": y_offset.copy(),
            "yaw_target": self._yaw_target,
            "yaw_error": yaw_error,
            "yaw_corr_hip_raw": yaw_raw_corr,
            "yaw_corr_hip": yaw_corr,
            "yaw_hip_offset": self._yaw_hip_offset.copy(),
            "yaw_hip_rate_limited": np.asarray(yaw_rate_limited).copy(),
            "rear_preswing_unload_gate": rear_preswing_gate.copy(),
            "rear_preswing_vmc_fade": rear_fade.copy(),
            "rear_touchdown_vmc_ramp_weight": (early_weight * REAR_MASK).copy(),
            "phase_switch_vmc_weight_scale_applied": float(phase_switch_vmc_weight_scale_applied),
            "phase_switch_yaw_weight_scale_applied": float(phase_switch_yaw_weight_scale_applied),
        }
        return {
            "x_offset": x_offset,
            "y_offset": y_offset,
            "z_offset": z_offset,
            "rear_preswing_unload_gate": rear_preswing_gate,
            "debug": debug,
        }

    # ------------------------------------------------------------------
    # rear late-swing guard / descent soften / early-contact (v4)
    # ------------------------------------------------------------------
    def _rear_guards(self, *, inp, leg_phase, swing_mask, touchdown_blend, z_target, z_default):
        cfg = self.cfg
        z_target = z_target.copy()
        late_guard_active = np.zeros(4, dtype=bool)
        late_swing_height = np.zeros(4)
        late_swing_height_error = np.zeros(4)
        descent_scale_applied = np.ones(4)
        early_contact_guard_active = np.zeros(4, dtype=bool)
        early_contact_score = np.zeros(4)
        early_contact_relief = np.zeros(4)
        rear_unload_z_offset = (cfg.rear_unload_sign * cfg.rear_preswing_unload_z_m) * np.zeros(4)

        late_start = cfg.rear_late_swing_phase_start
        late_end = max(late_start + 1.0e-6, cfg.rear_late_swing_phase_end)
        late_gate = _smootherstep01(np.clip((leg_phase - late_start) / (late_end - late_start), 0.0, 1.0))
        late_gate = late_gate * (1.0 - _smootherstep01(np.clip((leg_phase - late_end) / 0.02, 0.0, 1.0)))
        late_gate = late_gate * swing_mask.astype(np.float64) * REAR_MASK

        rear_height = (z_target - z_default) * REAR_MASK  # predicted foot lift (FK z 之差)
        min_height = cfg.rear_late_swing_min_height_m
        height_error = np.clip(min_height + cfg.rear_late_swing_clearance_margin_m - rear_height, 0.0, None) * late_gate

        # descent softening
        if cfg.rear_late_swing_descent_soft_enable:
            descent_scale = 1.0 - late_gate * (1.0 - cfg.rear_late_swing_descent_scale)
            z_target = z_target + np.clip(rear_height, 0.0, None) * (1.0 - descent_scale)
            descent_scale_applied = descent_scale

        # clearance guard (用 reference predicted foot height 替代真机 foot body z)
        if cfg.rear_late_swing_guard_enable:
            desired_clearance = cfg.rear_late_swing_clearance_sign * np.clip(height_error, None, 0.006)
            rate = cfg.rear_late_swing_guard_rate_limit_m
            clearance_offset = self._late_swing_clearance_offset + np.clip(
                desired_clearance - self._late_swing_clearance_offset, -rate, rate
            )
            z_target = z_target + clearance_offset
            self._late_swing_clearance_offset = clearance_offset
            late_guard_active = height_error > 1.0e-6
        else:
            self._late_swing_clearance_offset = np.zeros(4)
        clearance_offset = self._late_swing_clearance_offset.copy()

        # early-contact guard：真机一般没有 contact force -> 用 q_error / tau 估计替代
        early_contact_source = "force"
        if cfg.rear_early_contact_guard_enable:
            contact_start = cfg.rear_early_contact_phase_start
            contact_end = max(contact_start + 1.0e-6, cfg.rear_early_contact_phase_end)
            contact_phase = (leg_phase >= contact_start) & (leg_phase <= contact_end)
            in_window = contact_phase & (swing_mask | (touchdown_blend > 1.0e-6)) & (REAR_MASK > 0.5)
            if inp.foot_force is not None:
                forces = np.asarray(inp.foot_force, dtype=np.float64).reshape(4)
                early_contact_score = forces.copy()
                contact_guard = in_window & (forces > cfg.rear_early_contact_force_threshold)
                early_contact_source = "force"
            elif inp.feedback_valid and inp.q_actual_sim is not None:
                # 替代：摆动腿 q_error 太大说明脚提前踩到东西被顶住
                q_actual = np.asarray(inp.q_actual_sim, dtype=np.float64).reshape(12)
                # 用上一步命令与实际之差近似 (calf 关节)
                q_err = np.abs(self._q_last_cmd - q_actual)
                leg_err = np.maximum.reduce([q_err[0::3], q_err[1::3], q_err[2::3]])
                early_contact_score = leg_err.copy()
                contact_guard = in_window & (leg_err > 0.12)
                early_contact_source = "q_error"
            else:
                contact_guard = np.zeros(4, dtype=bool)
                early_contact_source = "unavailable"
            early_contact_relief = (
                contact_guard.astype(np.float64) * cfg.rear_early_contact_relief_sign * cfg.rear_early_contact_lift_relief_m
            )
            z_target = z_target + early_contact_relief
            early_contact_guard_active = contact_guard
        late_swing_height = rear_height
        late_swing_height_error = height_error

        debug = {
            "early_contact_source": early_contact_source,
            "rear_late_swing_window_active": (late_gate > 1.0e-6),
            "rear_late_swing_guard_active": late_guard_active.copy(),
            "rear_late_swing_clearance_offset": clearance_offset,
            "rear_late_swing_height": late_swing_height.copy(),
            "rear_late_swing_height_error": late_swing_height_error.copy(),
            "rear_late_swing_descent_scale_applied": descent_scale_applied.copy(),
            "rear_early_contact_guard_active": early_contact_guard_active.copy(),
            "rear_early_contact_score": early_contact_score.copy(),
            "rear_early_contact_relief_offset": np.asarray(early_contact_relief).copy(),
        }
        return {
            "z_target": z_target,
            "early_contact_guard_active": early_contact_guard_active,
            "debug": debug,
        }

    # ------------------------------------------------------------------
    # gains (residual_action._apply_fast_trot_gains 原样迁移)
    # ------------------------------------------------------------------
    def _fast_trot_gains(self, *, swing_mask, touchdown_blend, early_stance_gate, preload_gate,
                         guard_strength, rear_early_contact_guard_active):
        cfg = self.cfg
        kp = np.zeros(12)
        kd = np.zeros(12)
        swing_kp = np.array([cfg.fast_trot_swing_hip_kp, cfg.fast_trot_swing_thigh_kp, cfg.fast_trot_swing_calf_kp])
        touchdown_kp = np.array(
            [cfg.fast_trot_touchdown_hip_kp, cfg.fast_trot_touchdown_thigh_kp, cfg.fast_trot_touchdown_calf_kp]
        )
        early_kp = np.array(
            [cfg.fast_trot_early_stance_hip_kp, cfg.fast_trot_early_stance_thigh_kp, cfg.fast_trot_early_stance_calf_kp]
        )
        guard_kp = np.array(
            [cfg.fast_trot_phase_switch_guard_hip_kp, cfg.fast_trot_phase_switch_guard_thigh_kp,
             cfg.fast_trot_phase_switch_guard_calf_kp]
        )
        support_kp = np.array([cfg.fast_trot_support_hip_kp, cfg.fast_trot_support_thigh_kp, cfg.fast_trot_support_calf_kp])

        guard_kp_scale = np.zeros(12)
        rear_touchdown_kp_scale = np.ones(4)
        rear_early_contact_kp_scale = np.ones(4)
        rear_touchdown_kp_ramp_weight = np.zeros(4)
        phase_switch_kp_scale_applied = 1.0

        for leg in range(4):
            cols = slice(leg * 3, leg * 3 + 3)
            leg_swing = bool(swing_mask[leg])
            touchdown = float(touchdown_blend[leg])
            early = float(early_stance_gate[leg])
            stance = 0.0 if leg_swing else 1.0
            leg_kp = swing_kp.copy() if leg_swing else support_kp.copy()
            leg_kp = leg_kp * (1.0 - early) + early_kp * early
            leg_kp = leg_kp * (1.0 - touchdown) + touchdown_kp * touchdown
            preload = float(preload_gate[leg]) * stance
            leg_kp = leg_kp * (1.0 - preload) + support_kp * preload
            # phase switch guard kp (v3/v4 -> stance legs)
            guard = guard_strength * stance
            leg_kp = leg_kp * (1.0 - guard) + guard_kp * guard
            guard_kp_scale[cols] = guard
            phase_switch_kp_scale_applied = 1.0 - guard_strength * (1.0 - cfg.fast_trot_phase_switch_kp_scale)

            # rear touchdown kp ramp (v3/v4, 后腿)
            if leg >= 2:
                rear_touchdown = max(touchdown, early)
                rear_touchdown = rear_touchdown * np.clip((1.0 - (1.0 if leg_swing else 0.0)) + touchdown, 0.0, 1.0)
                rear_limit_kp = np.array(
                    [cfg.rear_touchdown_hip_kp_limit, cfg.rear_touchdown_thigh_kp_limit, cfg.rear_touchdown_calf_kp_limit]
                )
                rear_soft_kp = np.minimum(leg_kp * cfg.rear_touchdown_kp_scale, rear_limit_kp)
                leg_kp = leg_kp * (1.0 - rear_touchdown) + rear_soft_kp * rear_touchdown
                rear_touchdown_kp_scale[leg] = 1.0 - rear_touchdown * (1.0 - cfg.rear_touchdown_kp_scale)
                rear_touchdown_kp_ramp_weight[leg] = rear_touchdown
                # early-contact kp (v4)
                if bool(rear_early_contact_guard_active[leg]):
                    early_limit_kp = np.array(
                        [cfg.rear_early_contact_hip_kp_limit, cfg.rear_early_contact_thigh_kp_limit,
                         cfg.rear_early_contact_calf_kp_limit]
                    )
                    early_soft_kp = np.minimum(leg_kp * cfg.rear_early_contact_kp_scale, early_limit_kp)
                    leg_kp = early_soft_kp
                    rear_early_contact_kp_scale[leg] = cfg.rear_early_contact_kp_scale

            # kd
            leg_kd = np.full(3, cfg.fast_trot_support_kd)
            if leg_swing:
                leg_kd = np.full(3, cfg.fast_trot_swing_kd)
            leg_kd = leg_kd * (1.0 - touchdown) + cfg.fast_trot_touchdown_kd * touchdown
            leg_kd = leg_kd * (1.0 - early) + cfg.fast_trot_early_stance_kd * early
            if leg >= 2:
                rear_touchdown = max(touchdown, early)
                leg_kd = leg_kd * (1.0 - rear_touchdown) + cfg.rear_touchdown_kd * rear_touchdown
                if bool(rear_early_contact_guard_active[leg]):
                    leg_kd = np.full(3, cfg.rear_early_contact_kd)
            # phase switch guard kd 在 v4 不作用于 kd (只在 small_fix / v3)，v4 不加
            kp[cols] = leg_kp
            kd[cols] = leg_kd

        dbg = {
            "guard_kp_scale": guard_kp_scale.copy(),
            "rear_touchdown_kp_scale": rear_touchdown_kp_scale.copy(),
            "rear_early_contact_kp_scale": rear_early_contact_kp_scale.copy(),
            "rear_touchdown_kp_ramp_weight": rear_touchdown_kp_ramp_weight.copy(),
            "phase_switch_kp_scale_applied": float(phase_switch_kp_scale_applied),
        }
        return kp, kd, dbg

    # ------------------------------------------------------------------
    # 输出滤波 + 安全链 (residual_action.process_actions soft_output_v2 path)
    # ------------------------------------------------------------------
    def _output_safety_chain(self, *, q_raw, kp, kd, inp, guard_strength, rear_early_contact_guard_active):
        cfg = self.cfg
        dt = cfg.dt

        # --- 当前关节位置 q_current / 速度 qd_current (sim_semantic) 替代规则 ---
        # sim 中 q_current = robot.data.joint_pos, qd_current = robot.data.joint_vel。
        # torque backoff = kp*(q_target-q_current) - kd*qd_current，两项都要复刻。
        use_virtual = (not inp.feedback_valid) or inp.dry_run_virtual_feedback
        q_last_cmd_prev = self._q_last_cmd.copy()
        if inp.feedback_valid and inp.q_actual_sim is not None and not inp.dry_run_virtual_feedback:
            q_actual = np.asarray(inp.q_actual_sim, dtype=np.float64).reshape(12)
            if inp.dq_actual_sim is not None:
                qd_current = np.asarray(inp.dq_actual_sim, dtype=np.float64).reshape(12)
            else:
                qd_current = np.zeros(12)
        else:
            # dry-run: 假设电机完美跟随上一周期命令, 速度项取 0 (与 sim 静止初值一致)
            q_actual = self._q_last_cmd.copy()
            qd_current = np.zeros(12)
        q_current = q_actual

        kp_eff = np.clip(kp, 1.0e-6, None)
        kd_eff = np.clip(kd, 0.0, None)

        if not cfg.enable_deploy_target_filter:
            self._q_last_cmd = q_raw.copy()
            self._qdot_last_cmd = np.zeros(12)
            tau = self._pd_torque(q_raw, q_current, qd_current, kp_eff, kd_eff)
            return {
                "q_cmd_final": q_raw.copy(),
                "q_actual": q_actual,
                "debug": {
                    "tau_est": tau.copy(),
                    "rate_limited_delta": np.zeros(12),
                    "rate_clip_ratio": 0.0,
                    "torque_clip_ratio": 0.0,
                    "q_last_cmd_sim": q_last_cmd_prev,
                    "q_rate_limited_sim": q_raw.copy(),
                    "q_torque_filtered_sim": q_raw.copy(),
                },
            }

        # rate limit (per-joint mul; kd_scale=1 -> damping_scale=1)
        damping_scale = math.sqrt(min(max(cfg.kd_scale, 0.5), 2.0))
        rate_mul = self._joint_type_vector(cfg.hip_target_rate_mul, cfg.thigh_target_rate_mul, cfg.calf_target_rate_mul)
        rate_limit = (cfg.sim_target_rate_limit / damping_scale) * rate_mul

        q_after_rate = q_raw.copy()
        rate_limited_delta = np.zeros(12)
        if cfg.enable_target_rate_limit:
            max_step = rate_limit * dt
            target_step = q_raw - self._q_last_cmd
            step = np.clip(target_step, -max_step, max_step)
            q_after_rate = self._q_last_cmd + step
            crossed = (target_step * (q_raw - q_after_rate)) < 0.0
            q_after_rate = np.where(crossed, q_raw, q_after_rate)
            rate_limited_delta = q_after_rate - q_raw
        rate_clip_ratio = float(np.mean(np.abs(rate_limited_delta) > 1.0e-6))
        qdot_rate = (q_after_rate - self._q_last_cmd) / dt

        # accel limit (v4 disabled)
        if cfg.enable_target_accel_limit:
            accel_mul = self._joint_type_vector(
                cfg.hip_target_accel_mul, cfg.thigh_target_accel_mul, cfg.calf_target_accel_mul
            )
            accel_limit = (cfg.sim_target_accel_limit / damping_scale) * accel_mul
            qdot_delta = np.clip(qdot_rate - self._qdot_last_cmd, -accel_limit * dt, accel_limit * dt)
            qdot_cmd = self._qdot_last_cmd + qdot_delta
        else:
            qdot_cmd = qdot_rate
        q_after_accel = self._q_last_cmd + qdot_cmd * dt

        # torque soft-output backoff
        if cfg.enable_torque_target_limit:
            early_contact_guard_strength = (
                float(np.max(rear_early_contact_guard_active.astype(np.float64)))
                if np.any(rear_early_contact_guard_active)
                else 0.0
            )
            q_after_torque = self._soft_output_torque_target(
                q_after_accel, q_current, qd_current, kp_eff, kd_eff,
                guard_strength=guard_strength,
                early_contact_guard_strength=early_contact_guard_strength,
            )
        else:
            q_after_torque = q_after_accel
        torque_clip_delta = q_after_torque - q_after_accel
        torque_clip_ratio = float(np.mean(np.abs(torque_clip_delta) > 1.0e-6))

        actual_qdot_cmd = (q_after_torque - self._q_last_cmd) / dt
        self._q_last_cmd = q_after_torque.copy()
        self._qdot_last_cmd = actual_qdot_cmd

        # action delay disabled
        q_final = q_after_torque
        tau = self._pd_torque(q_final, q_current, qd_current, kp_eff, kd_eff)

        return {
            "q_cmd_final": q_final.copy(),
            "q_actual": q_actual,
            "debug": {
                "tau_est": tau.copy(),
                "rate_limited_delta": rate_limited_delta.copy(),
                "rate_clip_ratio": rate_clip_ratio,
                "torque_clip_ratio": torque_clip_ratio,
                "q_last_cmd_sim": q_last_cmd_prev,
                "q_rate_limited_sim": q_after_rate.copy(),
                "q_torque_filtered_sim": q_after_torque.copy(),
            },
        }

    def _soft_output_torque_target(self, q_target, q_current, qd_current, kp_eff, kd_eff, *, guard_strength, early_contact_guard_strength):
        cfg = self.cfg
        # tau = kp*(q_target-q_current) - kd*qd_current (与 sim _pd_torque_for 一致)
        tau = kp_eff * (q_target - q_current) - kd_eff * qd_current
        abs_tau = np.abs(tau)
        soft_start = np.full(12, cfg.fast_trot_soft_output_start_torque)
        soft_full = np.full(12, cfg.fast_trot_soft_output_full_torque)
        if guard_strength is not None and guard_strength > 0.0:
            g = guard_strength
            soft_start = soft_start * (1.0 - g) + cfg.fast_trot_guard_soft_start_torque * g
            soft_full = soft_full * (1.0 - g) + cfg.fast_trot_guard_soft_full_torque * g
        if early_contact_guard_strength is not None and early_contact_guard_strength > 0.0:
            eg = early_contact_guard_strength
            soft_start = soft_start * (1.0 - eg) + cfg.rear_early_contact_torque_soft_start * eg
            soft_full = soft_full * (1.0 - eg) + cfg.rear_early_contact_torque_soft_full * eg
        hard = cfg.sim_hard_torque_budget
        soft_t = np.clip((abs_tau - soft_start) / np.clip(soft_full - soft_start, 1.0e-6, None), 0.0, 1.0)
        soft_t = soft_t * soft_t * (3.0 - 2.0 * soft_t)
        hard_t = np.clip((abs_tau - soft_full) / max(hard - 0.0, 1.0e-6), 0.0, 1.0)
        hard_t = np.clip((abs_tau - soft_full) / np.clip(hard - soft_full, 1.0e-6, None), 0.0, 1.0)
        hard_t = hard_t * hard_t * (3.0 - 2.0 * hard_t)
        scale = 1.0 - 0.18 * soft_t - 0.32 * hard_t
        hard_scale = np.where(abs_tau > hard, hard / np.clip(abs_tau, 1.0e-6, None), 1.0)
        scale = np.minimum(scale, hard_scale)
        return q_current + scale * (q_target - q_current)

    @staticmethod
    def _pd_torque(q_target, q_current, qd_current, kp_eff, kd_eff):
        # tau = kp*(q_target-q_current) - kd*qd_current (与 sim _pd_torque_for 一致)
        return kp_eff * (q_target - q_current) - kd_eff * np.asarray(qd_current, dtype=np.float64).reshape(12)

    @staticmethod
    def _joint_type_vector(hip, thigh, calf):
        v = np.zeros(12)
        v[0::3] = hip
        v[1::3] = thigh
        v[2::3] = calf
        return v

    # ------------------------------------------------------------------
    # 默认站姿对比 (sim_v4 vs mapper)
    # ------------------------------------------------------------------
    @staticmethod
    def default_pose_comparison(mapper_default_policy: np.ndarray) -> dict:
        sim = SIM_V4_DEFAULT_JOINT_POS_POLICY
        mapper = np.asarray(mapper_default_policy, dtype=np.float64).reshape(12)
        diff = mapper - sim
        return {
            "sim_v4_default_policy": sim.copy(),
            "mapper_default_policy": mapper.copy(),
            "default_policy_diff": diff.copy(),
            "default_policy_diff_max": float(np.max(np.abs(diff))),
        }


# ----------------------------------------------------------------------------
# 自检 (无 ROS2 / 无 IsaacLab 也能跑): python fanfan_v4_migration_core.py
# ----------------------------------------------------------------------------
def _self_test():
    core = FanfanV4MigrationCore()
    print(f"[migration_core] {MIGRATION_CORE_VERSION}")
    print(f"[migration_core] default foot z = {core.default_foot_z}")
    cmp = FanfanV4MigrationCore.default_pose_comparison(MAPPER_DEFAULT_JOINT_POS_POLICY)
    print(f"[migration_core] default_policy_diff_max = {cmp['default_policy_diff_max']:.4f} rad")
    n = int(round(5.0 / core.cfg.dt))
    max_clearance = np.zeros(4)
    for _ in range(n):
        out = core.step(CoreInputs(test_mode="air", dry_run_virtual_feedback=True))
        d = out["debug_info"]
        max_clearance = np.maximum(max_clearance, d["fk_clearance_ref"])
    print(f"[migration_core] after {n} steps phase={core.base_phase:.3f}")
    print(f"[migration_core] max fk_clearance_ref per leg (FR,FL,RR,RL) = {max_clearance}")
    print(f"[migration_core] front swing height cfg = {core.cfg.fast_trot_front_swing_height_m}, "
          f"rear = {core.cfg.fast_trot_rear_swing_height_m}")
    print(f"[migration_core] last q_ref_sim = {out['q_ref_sim']}")
    print(f"[migration_core] last kp = {out['kp_sim']}")


if __name__ == "__main__":
    _self_test()
