#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
TROT_OFFSETS = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
HIP_OUTWARD_SIGNS = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
OLD_HIP_OUTWARD_SIGNS = {"FR": 1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
CRAWL_SWING_ORDER = ("RR", "FR", "RL", "FL")
CRAWL_OFFSETS = {"RR": 0.0, "FR": 0.25, "RL": 0.50, "FL": 0.75}
OLD_STABLE_SWING_ORDER = ("RR", "FL", "RL", "FR")
OLD_LEG_SIDE = {"FR": -1.0, "RR": -1.0, "FL": 1.0, "RL": 1.0}
OLD_DIAGONAL_PARTNER = {"FR": "RL", "RL": "FR", "FL": "RR", "RR": "FL"}
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RR", "RL")
RIGHT_LEGS = ("FR", "RR")
LEFT_LEGS = ("FL", "RL")
LEG_FALLBACK_DEBUG = {
    "leg_phase": 0.0,
    "stance": 1.0,
    "swing": 0.0,
    "stance_shape": 0.0,
    "swing_shape": 0.0,
    "x_foot": 0.0,
    "z_foot": 0.0,
    "hip_delta": 0.0,
    "thigh_ik": 0.0,
    "calf_ik": 0.0,
    "calf_ik_raw": 0.0,
    "front_calf_clipped": 0.0,
    "active_swing_leg": "",
    "leg_state": "SUPPORT",
    "support_legs": "",
    "body_x_shift": 0.0,
    "body_y_shift": 0.0,
    "support_triangle_stable": 1.0,
    "stability_margin_est": 0.0,
    "touchdown_counter": 0.0,
    "preload_gate": 0.0,
    "unload_gate": 0.0,
    "lift_gate": 0.0,
    "advance_gate": 0.0,
    "touchdown_gate": 0.0,
    "settle_gate": 0.0,
}


class FanfanIkGaitNode(Node):
    """
    Structure-aware open-loop forward trot for Fanfan.

    The gait is generated in policy/URDF joint space from a sagittal foot
    trajectory and then converted to real motor order through JointSemanticMapper.
    """

    def __init__(self):
        super().__init__("fanfan_ik_gait_node")

        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("motion_mode", "urdf_forward_crawl")
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("step_hz", 0.55)
        self.declare_parameter("stand_sec", 3.0)
        self.declare_parameter("warmup_sec", 5.0)

        self.declare_parameter("stand_kp", 40.0)
        self.declare_parameter("stand_kd", 4.2)
        self.declare_parameter("send_kp", 50.0)
        self.declare_parameter("send_kd", 4.6)
        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("debug_csv_period_sec", 0.0)
        self.declare_parameter("debug_stale_recheck_ms", 100.0)

        self.declare_parameter("stride_length", 0.014)
        self.declare_parameter("walk_direction", 1.0)
        self.declare_parameter("swing_height", 0.070)
        self.declare_parameter("duty_factor", 0.78)
        self.declare_parameter("diagonal_pair_delay_phase", 0.20)
        self.declare_parameter("lock_support_legs", False)
        self.declare_parameter("settle_fraction", 0.06)
        self.declare_parameter("preload_fraction", 0.18)
        self.declare_parameter("unload_fraction", 0.10)
        self.declare_parameter("lift_fraction", 0.22)
        self.declare_parameter("advance_fraction", 0.38)
        self.declare_parameter("touchdown_fraction", 0.14)
        self.declare_parameter("front_stride_gain", 0.86)
        self.declare_parameter("rear_stride_gain", 0.72)
        self.declare_parameter("rear_stride_min_front_ratio", 0.86)
        self.declare_parameter("walk_stride_scale", 1.65)
        self.declare_parameter("stance_drive_scale", 0.0)
        self.declare_parameter("front_stance_drive_scale", 0.12)
        self.declare_parameter("rear_stance_drive_scale", 0.0)
        self.declare_parameter("body_shift_foot_x_scale", 0.0)
        self.declare_parameter("body_shift_hip_scale", 0.45)
        self.declare_parameter("front_swing_height_gain", 1.36)
        self.declare_parameter("rear_swing_height_gain", 1.10)
        self.declare_parameter("front_x_bias", 0.004)
        self.declare_parameter("front_z_extend", -0.003)
        # Front knee/calf bend limiter. The front calf joint was visually too folded;
        # keep only the front calf joints from going more negative than this value.
        self.declare_parameter("front_calf_min_rad", -1.12)
        self.declare_parameter("hip_default_scale", 0.38)
        self.declare_parameter("hip_default_inward_offset", 0.010)
        self.declare_parameter("hip_stance_widen_amp", 0.0)
        self.declare_parameter("hip_swing_relax_amp", 0.0)
        self.declare_parameter("support_hip_outward_amp", 0.0)
        self.declare_parameter("hip_inward_hold_amp", 0.018)
        self.declare_parameter("hip_inward_active_scale", 0.35)
        self.declare_parameter("side_support_hip_amp", 0.0015)
        self.declare_parameter("right_side_support_scale", 0.65)
        self.declare_parameter("left_side_support_scale", 0.85)
        self.declare_parameter("same_side_support_hip_scale", 0.30)
        self.declare_parameter("hip_body_y_sign", 1.0)
        self.declare_parameter("front_hip_swing_scale", 0.03)
        self.declare_parameter("front_support_hip_scale", 0.002)
        self.declare_parameter("rear_hip_swing_scale", 0.22)
        self.declare_parameter("rear_support_hip_scale", 0.04)
        self.declare_parameter("swing_hip_unload_amp", 0.004)
        self.declare_parameter("fr_swing_hip_inward_amp", 0.005)
        self.declare_parameter("front_touchdown_hip_counter_amp", 0.006)
        self.declare_parameter("front_touchdown_start", 0.66)
        self.declare_parameter("rear_touchdown_hip_counter_amp", 0.010)
        self.declare_parameter("rear_touchdown_start", 0.64)
        self.declare_parameter("front_calf_lift_extra", 0.195)
        self.declare_parameter("rear_calf_lift_extra", 0.145)
        self.declare_parameter("front_thigh_delta_scale", 0.65)
        self.declare_parameter("rear_thigh_delta_scale", 0.06)
        self.declare_parameter("front_swing_forward_unfold", 0.016)
        self.declare_parameter("front_calf_stance_push_amp", 0.030)
        self.declare_parameter("rear_hip_default_outward_offset", 0.006)
        self.declare_parameter("rear_thigh_default_back_offset", 0.030)
        self.declare_parameter("rear_calf_stance_push_amp", 0.004)
        self.declare_parameter("diagonal_push_boost", 0.004)
        self.declare_parameter("rear_push_during_front_swing_amp", 0.006)
        self.declare_parameter("front_push_during_rear_swing_amp", 0.003)
        self.declare_parameter("support_calf_hold_amp", 0.032)
        self.declare_parameter("old_diagonal_front_support_boost_amp", 0.030)
        self.declare_parameter("old_same_front_support_scale", 0.42)
        self.declare_parameter("pre_swing_unload_amp", 0.050)
        self.declare_parameter("pre_swing_support_boost_amp", 0.022)
        self.declare_parameter("rear_support_stand_tall_m", 0.008)
        self.declare_parameter("rear_pre_lift_relief_m", 0.010)
        self.declare_parameter("rear_swing_body_x_shift", 0.070)
        self.declare_parameter("front_swing_body_x_shift", -0.040)
        self.declare_parameter("body_y_shift_amp", 0.040)
        self.declare_parameter("rear_swing_front_support_boost_amp", 0.018)
        self.declare_parameter("rear_swing_opposite_rear_support_boost_amp", 0.005)
        self.declare_parameter("rear_support_shift_ramp_fraction", 0.58)
        self.declare_parameter("rear_support_return_fraction", 0.36)
        self.declare_parameter("rear_swing_rear_support_hold_amp", 0.000)
        self.declare_parameter("rear_swing_rear_support_relief_amp", 0.000)
        self.declare_parameter("rear_swing_front_x_shift_scale", 1.00)
        self.declare_parameter("rear_swing_opposite_rear_x_shift_scale", 0.000)
        self.declare_parameter("rear_swing_swing_leg_x_scale", 0.050)
        self.declare_parameter("rear_swing_lateral_hip_amp", 0.003)
        self.declare_parameter("pre_swing_fraction", 0.30)
        self.declare_parameter("advance_start", 0.20)
        self.declare_parameter("advance_end", 0.82)
        self.declare_parameter("front_thigh_swing_scale", 1.0)
        self.declare_parameter("rear_thigh_swing_scale", 1.0)
        self.declare_parameter("front_calf_swing_scale", 1.0)
        # Extra multiplier for left-front calf swing only.
        # 1.0 keeps the original FL calf motion; 0.5 makes FL calf lift/swing
        # amplitude half of the current front_calf_swing_scale result.
        self.declare_parameter("fl_calf_swing_scale", 1.0)
        self.declare_parameter("rear_calf_swing_scale", 1.0)
        self.declare_parameter("opposite_side_boost", 0.0)
        self.declare_parameter("rear_swing_extra_lift_m", 0.030)
        self.declare_parameter("rear_swing_thigh_lift_amp", 0.055)
        self.declare_parameter("forward_body_x_bias_m", 0.026)
        self.declare_parameter("forward_body_x_foot_scale", 0.38)
        self.declare_parameter("forward_lateral_shift_m", 0.040)
        self.declare_parameter("forward_rear_swing_x_shift_m", 0.080)
        self.declare_parameter("forward_front_swing_x_shift_m", 0.014)
        self.declare_parameter("forward_stability_margin_m", 0.008)
        self.declare_parameter("forward_support_stand_tall_m", 0.006)
        self.declare_parameter("forward_front_support_load_m", 0.010)
        self.declare_parameter("forward_active_body_x_scale", 0.15)
        # URDF default-stand contact estimate for
        # fanfan_mass_scaled_only_trunk_plus_800g.urdf after the local stand
        # offsets above: FR/FL foot x ~= 0.1765 m, RR/RL foot x ~= -0.2219 m,
        # lateral half width ~= 0.138 m.
        self.declare_parameter("front_foot_x_default", 0.1765)
        self.declare_parameter("rear_foot_x_default", -0.2219)
        self.declare_parameter("foot_y_half_width", 0.138)
        self.declare_parameter("com_x_default", 0.00)
        self.declare_parameter("com_y_default", 0.00)
        self.declare_parameter("stability_margin_m", 0.020)
        self.declare_parameter("max_target_rate_rad_s", 2.2)
        self.declare_parameter("max_delta", 0.60)
        self.declare_parameter("torque_warn_nm", 6.0)

        # From fanfan_mass_scaled_only_trunk_plus_800g.urdf.
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)

        self.motor_base_url = str(self.get_parameter("motor_base_url").value).rstrip("/")
        self.enable_send = bool(self.get_parameter("enable_send").value)
        self.motion_mode = self.normalize_motion_mode(
            str(self.get_parameter("motion_mode").value)
        )
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.stand_sec = float(self.get_parameter("stand_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)

        self.stand_kp = float(self.get_parameter("stand_kp").value)
        self.stand_kd = float(self.get_parameter("stand_kd").value)
        self.send_kp = float(self.get_parameter("send_kp").value)
        self.send_kd = float(self.get_parameter("send_kd").value)
        self.send_speed = float(self.get_parameter("send_speed").value)
        self.send_torque = float(self.get_parameter("send_torque").value)
        self.http_timeout = float(self.get_parameter("http_timeout").value)
        self.debug_csv_path = str(self.get_parameter("debug_csv_path").value)
        self.debug_csv_period_sec = float(self.get_parameter("debug_csv_period_sec").value)
        self.debug_stale_recheck_ms = float(self.get_parameter("debug_stale_recheck_ms").value)

        self.stride_length = float(self.get_parameter("stride_length").value)
        self.walk_direction = -1.0 if float(self.get_parameter("walk_direction").value) < 0.0 else 1.0
        self.swing_height = float(self.get_parameter("swing_height").value)
        self.duty_factor = float(self.get_parameter("duty_factor").value)
        self.diagonal_pair_delay_phase = float(
            self.get_parameter("diagonal_pair_delay_phase").value
        )
        self.lock_support_legs = bool(self.get_parameter("lock_support_legs").value)
        self.settle_fraction = float(self.get_parameter("settle_fraction").value)
        self.preload_fraction = float(self.get_parameter("preload_fraction").value)
        self.unload_fraction = float(self.get_parameter("unload_fraction").value)
        self.lift_fraction = float(self.get_parameter("lift_fraction").value)
        self.advance_fraction = float(self.get_parameter("advance_fraction").value)
        self.touchdown_fraction = float(self.get_parameter("touchdown_fraction").value)
        self.front_stride_gain = float(self.get_parameter("front_stride_gain").value)
        self.rear_stride_gain = float(self.get_parameter("rear_stride_gain").value)
        self.rear_stride_min_front_ratio = float(
            self.get_parameter("rear_stride_min_front_ratio").value
        )
        self.walk_stride_scale = float(self.get_parameter("walk_stride_scale").value)
        self.stance_drive_scale = float(self.get_parameter("stance_drive_scale").value)
        self.front_stance_drive_scale = float(
            self.get_parameter("front_stance_drive_scale").value
        )
        self.rear_stance_drive_scale = float(
            self.get_parameter("rear_stance_drive_scale").value
        )
        self.body_shift_foot_x_scale = float(
            self.get_parameter("body_shift_foot_x_scale").value
        )
        self.body_shift_hip_scale = float(self.get_parameter("body_shift_hip_scale").value)
        self.front_swing_height_gain = float(
            self.get_parameter("front_swing_height_gain").value
        )
        self.rear_swing_height_gain = float(
            self.get_parameter("rear_swing_height_gain").value
        )
        self.front_x_bias = float(self.get_parameter("front_x_bias").value)
        self.front_z_extend = float(self.get_parameter("front_z_extend").value)
        self.front_calf_min_rad = float(self.get_parameter("front_calf_min_rad").value)
        self.hip_default_scale = float(self.get_parameter("hip_default_scale").value)
        self.hip_default_inward_offset = float(
            self.get_parameter("hip_default_inward_offset").value
        )
        self.hip_stance_widen_amp = float(self.get_parameter("hip_stance_widen_amp").value)
        self.hip_swing_relax_amp = float(self.get_parameter("hip_swing_relax_amp").value)
        self.support_hip_outward_amp = float(
            self.get_parameter("support_hip_outward_amp").value
        )
        self.hip_inward_hold_amp = float(self.get_parameter("hip_inward_hold_amp").value)
        self.hip_inward_active_scale = float(
            self.get_parameter("hip_inward_active_scale").value
        )
        self.side_support_hip_amp = float(self.get_parameter("side_support_hip_amp").value)
        self.right_side_support_scale = float(
            self.get_parameter("right_side_support_scale").value
        )
        self.left_side_support_scale = float(
            self.get_parameter("left_side_support_scale").value
        )
        self.same_side_support_hip_scale = float(
            self.get_parameter("same_side_support_hip_scale").value
        )
        self.hip_body_y_sign = float(self.get_parameter("hip_body_y_sign").value)
        self.front_hip_swing_scale = float(
            self.get_parameter("front_hip_swing_scale").value
        )
        self.front_support_hip_scale = float(
            self.get_parameter("front_support_hip_scale").value
        )
        self.rear_hip_swing_scale = float(self.get_parameter("rear_hip_swing_scale").value)
        self.rear_support_hip_scale = float(
            self.get_parameter("rear_support_hip_scale").value
        )
        self.swing_hip_unload_amp = float(self.get_parameter("swing_hip_unload_amp").value)
        self.fr_swing_hip_inward_amp = float(
            self.get_parameter("fr_swing_hip_inward_amp").value
        )
        self.front_touchdown_hip_counter_amp = float(
            self.get_parameter("front_touchdown_hip_counter_amp").value
        )
        self.front_touchdown_start = float(
            self.get_parameter("front_touchdown_start").value
        )
        self.rear_touchdown_hip_counter_amp = float(
            self.get_parameter("rear_touchdown_hip_counter_amp").value
        )
        self.rear_touchdown_start = float(self.get_parameter("rear_touchdown_start").value)
        self.front_calf_lift_extra = float(
            self.get_parameter("front_calf_lift_extra").value
        )
        self.rear_calf_lift_extra = float(
            self.get_parameter("rear_calf_lift_extra").value
        )
        self.front_thigh_delta_scale = float(
            self.get_parameter("front_thigh_delta_scale").value
        )
        self.rear_thigh_delta_scale = float(
            self.get_parameter("rear_thigh_delta_scale").value
        )
        self.front_swing_forward_unfold = float(
            self.get_parameter("front_swing_forward_unfold").value
        )
        self.front_calf_stance_push_amp = float(
            self.get_parameter("front_calf_stance_push_amp").value
        )
        self.rear_hip_default_outward_offset = float(
            self.get_parameter("rear_hip_default_outward_offset").value
        )
        self.rear_thigh_default_back_offset = float(
            self.get_parameter("rear_thigh_default_back_offset").value
        )
        self.rear_calf_stance_push_amp = float(
            self.get_parameter("rear_calf_stance_push_amp").value
        )
        self.diagonal_push_boost = float(self.get_parameter("diagonal_push_boost").value)
        self.rear_push_during_front_swing_amp = float(
            self.get_parameter("rear_push_during_front_swing_amp").value
        )
        self.front_push_during_rear_swing_amp = float(
            self.get_parameter("front_push_during_rear_swing_amp").value
        )
        self.support_calf_hold_amp = float(
            self.get_parameter("support_calf_hold_amp").value
        )
        self.old_diagonal_front_support_boost_amp = float(
            self.get_parameter("old_diagonal_front_support_boost_amp").value
        )
        self.old_same_front_support_scale = float(
            self.get_parameter("old_same_front_support_scale").value
        )
        self.pre_swing_unload_amp = float(
            self.get_parameter("pre_swing_unload_amp").value
        )
        self.pre_swing_support_boost_amp = float(
            self.get_parameter("pre_swing_support_boost_amp").value
        )
        self.rear_support_stand_tall_m = float(
            self.get_parameter("rear_support_stand_tall_m").value
        )
        self.rear_pre_lift_relief_m = float(
            self.get_parameter("rear_pre_lift_relief_m").value
        )
        self.rear_swing_body_x_shift = float(
            self.get_parameter("rear_swing_body_x_shift").value
        )
        self.front_swing_body_x_shift = float(
            self.get_parameter("front_swing_body_x_shift").value
        )
        self.body_y_shift_amp = float(self.get_parameter("body_y_shift_amp").value)
        self.rear_swing_front_support_boost_amp = float(
            self.get_parameter("rear_swing_front_support_boost_amp").value
        )
        self.rear_swing_opposite_rear_support_boost_amp = float(
            self.get_parameter("rear_swing_opposite_rear_support_boost_amp").value
        )
        self.rear_support_shift_ramp_fraction = float(
            self.get_parameter("rear_support_shift_ramp_fraction").value
        )
        self.rear_support_return_fraction = float(
            self.get_parameter("rear_support_return_fraction").value
        )
        self.rear_swing_rear_support_hold_amp = float(
            self.get_parameter("rear_swing_rear_support_hold_amp").value
        )
        self.rear_swing_rear_support_relief_amp = float(
            self.get_parameter("rear_swing_rear_support_relief_amp").value
        )
        self.rear_swing_front_x_shift_scale = float(
            self.get_parameter("rear_swing_front_x_shift_scale").value
        )
        self.rear_swing_opposite_rear_x_shift_scale = float(
            self.get_parameter("rear_swing_opposite_rear_x_shift_scale").value
        )
        self.rear_swing_swing_leg_x_scale = float(
            self.get_parameter("rear_swing_swing_leg_x_scale").value
        )
        self.rear_swing_lateral_hip_amp = float(
            self.get_parameter("rear_swing_lateral_hip_amp").value
        )
        self.pre_swing_fraction = float(self.get_parameter("pre_swing_fraction").value)
        self.advance_start = float(self.get_parameter("advance_start").value)
        self.advance_end = float(self.get_parameter("advance_end").value)
        self.front_thigh_swing_scale = float(
            self.get_parameter("front_thigh_swing_scale").value
        )
        self.rear_thigh_swing_scale = float(
            self.get_parameter("rear_thigh_swing_scale").value
        )
        self.front_calf_swing_scale = float(
            self.get_parameter("front_calf_swing_scale").value
        )
        self.fl_calf_swing_scale = float(
            self.get_parameter("fl_calf_swing_scale").value
        )
        self.rear_calf_swing_scale = float(
            self.get_parameter("rear_calf_swing_scale").value
        )
        self.opposite_side_boost = float(self.get_parameter("opposite_side_boost").value)
        self.rear_swing_extra_lift_m = float(
            self.get_parameter("rear_swing_extra_lift_m").value
        )
        self.rear_swing_thigh_lift_amp = float(
            self.get_parameter("rear_swing_thigh_lift_amp").value
        )
        self.forward_body_x_bias_m = float(
            self.get_parameter("forward_body_x_bias_m").value
        )
        self.forward_body_x_foot_scale = float(
            self.get_parameter("forward_body_x_foot_scale").value
        )
        self.forward_lateral_shift_m = float(
            self.get_parameter("forward_lateral_shift_m").value
        )
        self.forward_rear_swing_x_shift_m = float(
            self.get_parameter("forward_rear_swing_x_shift_m").value
        )
        self.forward_front_swing_x_shift_m = float(
            self.get_parameter("forward_front_swing_x_shift_m").value
        )
        self.forward_stability_margin_m = float(
            self.get_parameter("forward_stability_margin_m").value
        )
        self.forward_support_stand_tall_m = float(
            self.get_parameter("forward_support_stand_tall_m").value
        )
        self.forward_front_support_load_m = float(
            self.get_parameter("forward_front_support_load_m").value
        )
        self.forward_active_body_x_scale = float(
            self.get_parameter("forward_active_body_x_scale").value
        )
        self.front_foot_x_default = float(
            self.get_parameter("front_foot_x_default").value
        )
        self.rear_foot_x_default = float(self.get_parameter("rear_foot_x_default").value)
        self.foot_y_half_width = float(self.get_parameter("foot_y_half_width").value)
        self.com_x_default = float(self.get_parameter("com_x_default").value)
        self.com_y_default = float(self.get_parameter("com_y_default").value)
        self.stability_margin_m = float(self.get_parameter("stability_margin_m").value)
        self.max_target_rate_rad_s = float(self.get_parameter("max_target_rate_rad_s").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.torque_warn_nm = float(self.get_parameter("torque_warn_nm").value)
        self.thigh_length = float(self.get_parameter("thigh_length").value)
        self.calf_length = float(self.get_parameter("calf_length").value)

        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.policy_joint_names = self.mapper.get_policy_joint_names()
        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.apply_hip_default_scale()
        self.apply_hip_default_inward_offset()
        self.apply_rear_default_offsets()
        self.apply_front_calf_limit()
        self.default_real = self.mapper.policy_target_to_real_target(
            self.default_policy,
            clamp=True,
        ).astype(np.float32)

        self.default_foot_xz = self.compute_default_foot_xz()
        self.last_target_policy = self.default_policy.copy()
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
        self._last_send_info_time = 0.0
        self._last_stability_warn_time = 0.0
        self._last_torque_warn_time = 0.0

        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=self.debug_stale_recheck_ms,
        )

        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self._last_feedback_warn_time = 0.0
        self.setup_debug_csv()
        self.start_debug_collector()

        self.pub_target = self.create_publisher(
            Float32MultiArray,
            "/mydog/fanfan_ik_target_real",
            10,
        )
        self.pub_phase = self.create_publisher(
            Float32MultiArray,
            "/mydog/fanfan_ik_phase",
            10,
        )

        self.get_logger().warn(
            "Fanfan IK gait is open-loop. First run supported or hand-held, "
            "and be ready to cut power."
        )

        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry run only, no motor commands sent.")

        self.get_logger().info(
            f"Fanfan IK gait: mode={self.motion_mode}, step_hz={self.step_hz:.2f}, "
            f"gait_hz={self.gait_hz:.1f}, "
            f"stride={self.stride_length:.3f}m, direction={self.walk_direction:+.0f}, "
            f"swing={self.swing_height:.3f}m, "
            f"duty={self.duty_factor:.2f}, warmup={self.warmup_sec:.2f}s, "
            f"forward_bias={self.forward_body_x_bias_m:.3f}m, "
            f"front_stride_gain={self.front_stride_gain:.2f}, "
            f"rear_stride_gain={self.rear_stride_gain:.2f}, "
            f"rear_min_front_ratio={self.rear_stride_min_front_ratio:.2f}, "
            f"front_stance_drive={self.front_stance_drive_scale:.2f}, "
            f"rear_stance_drive={self.rear_stance_drive_scale:.2f}, "
            f"rear_stand_tall={self.rear_support_stand_tall_m:.3f}m, "
            f"front_x_bias={self.front_x_bias:.3f}m, "
            f"front_calf_min={self.front_calf_min_rad:.3f}rad, "
            f"hip_scale={self.hip_default_scale:.2f}, "
            f"hip_inward_default={self.hip_default_inward_offset:.3f}, "
            f"hip_inward_hold={self.hip_inward_hold_amp:.3f}, "
            f"hip_widen={self.hip_stance_widen_amp:.3f}, send={self.enable_send}"
        )

        self.timer = self.create_timer(1.0 / max(self.gait_hz, 1e-3), self.update)

    @staticmethod
    def normalize_motion_mode(mode: str) -> str:
        mode = str(mode).strip().lower()
        if mode in ("urdf_forward_crawl", "forward_crawl", "forward", "crawl_forward"):
            return "urdf_forward_crawl"
        if mode in ("stable_calf_walk", "old_forward_calf_walk", "calf_dominant_loop", "loop_calf", "loop_crawl"):
            return "old_forward_calf_walk"
        if mode in ("triangle_stable_walk", ""):
            return "triangle_stable_walk"
        return mode

    def apply_hip_default_scale(self):
        scale = min(1.0, max(0.0, self.hip_default_scale))
        if abs(scale - self.hip_default_scale) > 1e-6:
            self.get_logger().warn(
                f"hip_default_scale={self.hip_default_scale:.3f} clipped to {scale:.3f}"
            )
        self.hip_default_scale = scale
        for leg in LEG_ORDER:
            self.default_policy[LEG_START[leg]] *= scale

    def apply_hip_default_inward_offset(self):
        for leg in LEG_ORDER:
            self.default_policy[LEG_START[leg]] -= (
                HIP_OUTWARD_SIGNS[leg] * self.hip_default_inward_offset
            )

    def apply_rear_default_offsets(self):
        for leg in REAR_LEGS:
            i = LEG_START[leg]
            self.default_policy[i + 0] += (
                HIP_OUTWARD_SIGNS[leg] * self.rear_hip_default_outward_offset
            )
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset

    def apply_front_calf_limit(self):
        # Only touch the front thigh-calf joints. Negative calf is knee flexion in
        # this IK convention; clamping it upward makes the front knee less folded.
        limit = float(self.front_calf_min_rad)
        for leg in ("FR", "FL"):
            idx = LEG_START[leg] + 2
            old = float(self.default_policy[idx])
            if old < limit:
                self.default_policy[idx] = limit
                self.get_logger().warn(
                    f"{leg}_calf default limited: {old:.3f} -> {limit:.3f} rad "
                    f"to reduce front knee folding"
                )

    def compute_default_foot_xz(self) -> dict[str, tuple[float, float]]:
        xz = {}
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            thigh = float(self.default_policy[i + 1])
            calf = float(self.default_policy[i + 2])
            xz[leg] = self.forward_sagittal(thigh, calf)
        return xz

    def forward_sagittal(self, thigh: float, calf: float) -> tuple[float, float]:
        x = -self.thigh_length * math.sin(thigh) - self.calf_length * math.sin(thigh + calf)
        z = -self.thigh_length * math.cos(thigh) - self.calf_length * math.cos(thigh + calf)
        return float(x), float(z)

    def inverse_sagittal(self, x: float, z: float) -> tuple[float, float]:
        x, z = self.clamp_reachable_xz(float(x), float(z))
        l1 = self.thigh_length
        l2 = self.calf_length
        cos_calf = (x * x + z * z - l1 * l1 - l2 * l2) / max(2.0 * l1 * l2, 1e-9)
        cos_calf = min(1.0, max(-1.0, cos_calf))
        calf = -math.acos(cos_calf)
        thigh = math.atan2(-x, -z) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
        return float(thigh), float(calf)

    def clamp_reachable_xz(self, x: float, z: float) -> tuple[float, float]:
        r = math.hypot(x, z)
        max_r = self.thigh_length + self.calf_length - 1e-5
        min_r = abs(self.thigh_length - self.calf_length) + 1e-5
        if r < 1e-9:
            return 0.0, -min_r
        if r > max_r:
            scale = max_r / r
            return x * scale, z * scale
        if r < min_r:
            scale = min_r / r
            return x * scale, z * scale
        return x, z

    @staticmethod
    def smoothstep_half_cos(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return 0.5 - 0.5 * math.cos(math.pi * s)

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    def smooth_window(self, s: float, edge: float = 0.18) -> float:
        edge = min(0.49, max(0.001, float(edge)))
        return self.smootherstep(s / edge) * self.smootherstep((1.0 - s) / edge)

    def send_default_stand(self):
        items = []
        for mid, pos in zip(self.motor_ids, self.default_real):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(pos),
                    "speed": 0.0,
                    "torque": 0.0,
                    "kp": self.stand_kp,
                    "kd": self.stand_kd,
                }
            )

        payload = {"items": items, "enable_first": True, "stop_first": False}
        url = f"{self.motor_base_url}/api/rs04/motion_mode_run_batch"
        r = self.http_session.post(url, json=payload, timeout=max(self.http_timeout, 0.5))
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")

        self.get_logger().info(
            f"Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}"
        )

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time

        if elapsed < self.stand_sec:
            target_policy = self.default_policy.copy()
            phase = 0.0
            warm = 0.0
            leg_debug = self.default_leg_debug(phase, warm)
            self._phase_acc = 0.0
        else:
            gait_time = elapsed - self.stand_sec
            self._phase_acc = (self._phase_acc + self.step_hz * dt) % 1.0
            phase = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 1e-3))
            target_policy, leg_debug = self.build_target_policy(phase, warm)

        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True)

        self.publish_array(self.pub_target, target_real)
        self.publish_array(
            self.pub_phase,
            np.array([phase, warm, self.step_hz], dtype=np.float32),
        )

        sent = False
        if not self.enable_send:
            self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)
            return

        delta = target_real - self.last_target_real
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self.max_delta:
            self.get_logger().warn(
                f"[SAFE] IK target jump too large: {max_delta:.3f} rad > {self.max_delta:.3f} rad. Skip send."
            )
            self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)
            return

        sent = self.send_motion_batch(target_real)
        if sent:
            self.last_target_real = target_real.copy()
        self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)

    def default_leg_debug(self, phase: float, warm: float) -> dict[str, dict[str, float]]:
        data = {}
        for leg in LEG_ORDER:
            x, z = self.default_foot_xz[leg]
            row = dict(LEG_FALLBACK_DEBUG)
            row.update(
                {
                "leg_phase": (phase + TROT_OFFSETS[leg]) % 1.0,
                "stance": 1.0,
                "swing": 0.0,
                "x_foot": x,
                "z_foot": z,
                "warm": warm,
                    "motion_mode": self.motion_mode,
                }
            )
            data[leg] = row
        return data

    def build_target_policy(self, phase: float, warm: float):
        mode = self.normalize_motion_mode(self.motion_mode)
        if mode == "urdf_forward_crawl":
            return self.build_urdf_forward_crawl_target_policy(phase, warm)
        if mode == "old_forward_calf_walk":
            return self.build_old_forward_calf_walk_target_policy(phase, warm)
        if mode == "triangle_stable_walk":
            return self.build_triangle_stable_walk_target_policy(phase, warm)
        return self.build_legacy_trot_target_policy(phase, warm)

    def build_legacy_trot_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        duty = min(max(self.duty_factor, 0.50), 0.90)
        swing_fraction = max(0.05, 1.0 - duty)

        for leg in LEG_ORDER:
            leg_phase = (phase + TROT_OFFSETS[leg]) % 1.0
            x0, z0 = self.default_foot_xz[leg]
            is_front = leg in ("FR", "FL")
            stride = self.stride_length * (
                self.front_stride_gain if is_front else self.rear_stride_gain
            )
            stride_dir = self.walk_direction
            swing_height = self.swing_height * (
                self.front_swing_height_gain if is_front else self.rear_swing_height_gain
            )
            x_center = x0 + (self.front_x_bias if is_front else 0.0)
            stance = 0.0
            swing = 0.0

            if leg_phase < swing_fraction:
                s = leg_phase / swing_fraction
                u = self.smoothstep_half_cos(s)
                x = x_center + stride_dir * (-0.5 * stride + stride * u)
                z = z0 + swing_height * math.sin(math.pi * s)
                swing = math.sin(math.pi * s)
            else:
                s = (leg_phase - swing_fraction) / max(duty, 1e-3)
                x = x_center + stride_dir * (0.5 * stride - stride * s)
                z = z0
                stance = math.sin(math.pi * s)

            thigh, calf = self.inverse_sagittal(x, z)
            calf_raw = calf
            front_calf_clipped = 0.0
            if is_front and calf < self.front_calf_min_rad:
                calf = self.front_calf_min_rad
                front_calf_clipped = 1.0

            i = LEG_START[leg]
            outward = HIP_OUTWARD_SIGNS[leg]
            hip_delta = outward * (
                self.hip_stance_widen_amp * stance
                - self.hip_swing_relax_amp * swing
            )

            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta
            q[i + 1] = self.default_policy[i + 1] + warm * (thigh - self.default_policy[i + 1])
            q[i + 2] = self.default_policy[i + 2] + warm * (calf - self.default_policy[i + 2])

            leg_debug[leg] = {
                "leg_phase": float(leg_phase),
                "stance": float(1.0 if leg_phase >= swing_fraction else 0.0),
                "swing": float(1.0 if leg_phase < swing_fraction else 0.0),
                "stance_shape": float(stance),
                "swing_shape": float(swing),
                "x_foot": float(x),
                "z_foot": float(z),
                "hip_delta": float(hip_delta),
                "thigh_ik": float(thigh),
                "calf_ik": float(calf),
                "calf_ik_raw": float(calf_raw),
                "front_calf_clipped": float(front_calf_clipped),
            }

        return q.astype(np.float32), leg_debug

    def forward_crawl_shift_gate(self, active_swing_leg: Optional[str], phase: float) -> float:
        if active_swing_leg is None:
            return 0.0
        state_info = self.get_leg_state(active_swing_leg, phase)
        event_s = float(state_info.get("event_s", 1.0))
        ramp_in = self.smoothstep_half_cos(event_s / 0.28)
        ramp_out = self.smoothstep_half_cos((1.0 - event_s) / 0.22)
        return float(ramp_in * ramp_out)

    def support_margin_for_shift(
        self,
        swing_leg: Optional[str],
        body_x_shift: float,
        body_y_shift: float,
    ) -> tuple[bool, float]:
        if swing_leg is None:
            return True, float(self.stability_margin_m)
        support_legs = self.support_legs_for_swing(swing_leg)
        if len(support_legs) < 3:
            return False, 0.0
        points = [self.support_foot_xy(leg) for leg in support_legs[:3]]
        com = (
            self.com_x_default + float(body_x_shift),
            self.com_y_default + float(body_y_shift),
        )
        return self.point_in_triangle_with_margin(
            com,
            points[0],
            points[1],
            points[2],
            self.stability_margin_m,
        )

    def clamp_forward_body_shift_to_support(
        self,
        swing_leg: Optional[str],
        target_x: float,
        target_y: float,
    ) -> tuple[float, float, bool, float]:
        stable, margin = self.support_margin_for_shift(swing_leg, target_x, target_y)
        if stable:
            return float(target_x), float(target_y), True, float(margin)

        support_legs = self.support_legs_for_swing(swing_leg)
        if len(support_legs) < 3:
            return 0.0, 0.0, False, 0.0
        centroid = np.mean(np.array([self.support_foot_xy(leg) for leg in support_legs]), axis=0)
        centroid_x = float(centroid[0] - self.com_x_default)
        centroid_y = float(centroid[1] - self.com_y_default)

        best_x, best_y, best_margin = float(target_x), float(target_y), float(margin)
        for blend in (0.25, 0.50, 0.75, 1.0):
            x = (1.0 - blend) * target_x + blend * centroid_x
            y = (1.0 - blend) * target_y + blend * centroid_y
            stable, margin = self.support_margin_for_shift(swing_leg, x, y)
            if margin > best_margin:
                best_x, best_y, best_margin = float(x), float(y), float(margin)
            if stable:
                return float(x), float(y), True, float(margin)
        return best_x, best_y, False, best_margin

    def desired_urdf_forward_body_shift(
        self,
        active_swing_leg: Optional[str],
        phase: float,
    ) -> tuple[float, float, bool, float]:
        bias_x = self.forward_body_x_bias_m
        if active_swing_leg is None:
            return float(bias_x), 0.0, True, float(self.stability_margin_m)

        gate = self.forward_crawl_shift_gate(active_swing_leg, phase)
        target_x = (
            self.forward_rear_swing_x_shift_m
            if active_swing_leg in REAR_LEGS
            else self.forward_front_swing_x_shift_m
        )
        target_y = (
            self.forward_lateral_shift_m
            if active_swing_leg in RIGHT_LEGS
            else -self.forward_lateral_shift_m
        )
        body_x = bias_x + gate * (target_x - bias_x)
        body_y = gate * target_y
        return self.clamp_forward_body_shift_to_support(active_swing_leg, body_x, body_y)

    def build_urdf_forward_crawl_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        active_swing_leg = self.get_active_swing_leg(phase)
        body_x_shift, body_y_shift, stable, margin_est = self.desired_urdf_forward_body_shift(
            active_swing_leg,
            phase,
        )
        support_legs = self.support_legs_for_swing(active_swing_leg)
        event_fraction = self.triangle_event_fraction()
        stride_dir = self.walk_direction

        for leg in LEG_ORDER:
            state_info = self.get_leg_state(leg, phase)
            state = str(state_info.get("leg_state", "SUPPORT"))
            state_s = float(state_info.get("state_s", 0.0))
            rel = float(state_info.get("leg_phase", 0.0))
            is_front = leg in FRONT_LEGS
            is_rear = leg in REAR_LEGS
            is_active = leg == active_swing_leg

            x0, z0 = self.default_foot_xz[leg]
            x_center = x0 + (self.front_x_bias if is_front else 0.0)
            stride_gain = self.front_stride_gain if is_front else max(
                self.rear_stride_gain,
                self.front_stride_gain * max(0.0, self.rear_stride_min_front_ratio),
            )
            stride = self.stride_length * self.walk_stride_scale * stride_gain
            swing_height = self.swing_height * (
                self.front_swing_height_gain if is_front else self.rear_swing_height_gain
            )
            lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra

            body_scale = self.forward_active_body_x_scale if is_active else self.forward_body_x_foot_scale
            x_body = -warm * body_x_shift * body_scale
            z = z0
            swing_shape = 0.0
            stance_shape = 0.0
            calf_lift_gate = 0.0

            if is_active:
                if state in ("PRELOAD", "UNLOAD"):
                    x = x_center + stride_dir * (-0.5 * stride) + x_body
                    unload = self.smoothstep_half_cos(state_s) if state == "UNLOAD" else 0.0
                    z = z0 + warm * 0.15 * swing_height * unload
                elif state == "LIFT":
                    u = self.smoothstep_half_cos(state_s)
                    x = x_center + stride_dir * (-0.5 * stride) + x_body
                    z = z0 + warm * (0.15 + 0.85 * u) * swing_height
                    swing_shape = math.sin(math.pi * min(1.0, max(0.0, float(state_info.get("event_s", 0.0)))))
                    calf_lift_gate = u
                elif state == "ADVANCE":
                    u = self.smoothstep_half_cos(state_s)
                    x = x_center + stride_dir * (-0.5 * stride + stride * u) + x_body
                    if is_front:
                        x += stride_dir * self.front_swing_forward_unfold * u
                    z = z0 + warm * swing_height
                    swing_shape = 1.0
                    calf_lift_gate = 1.0
                elif state == "TOUCHDOWN":
                    u = self.smoothstep_half_cos(state_s)
                    x = x_center + stride_dir * 0.5 * stride + x_body
                    if is_front:
                        x += stride_dir * self.front_swing_forward_unfold
                    z = z0 + warm * swing_height * (1.0 - u)
                    swing_shape = 1.0 - u
                    calf_lift_gate = 1.0 - u
                else:
                    x = x_center + stride_dir * 0.5 * stride + x_body
            else:
                stance_s = (rel - event_fraction) / max(1.0 - event_fraction, 1e-6)
                stance_s = min(1.0, max(0.0, stance_s))
                drive = self.smoothstep_half_cos(stance_s)
                x = x_center + stride_dir * (0.5 * stride - stride * drive) + x_body
                stance_shape = math.sin(math.pi * stance_s)
                z -= warm * self.forward_support_stand_tall_m * (0.35 + 0.65 * stance_shape)
                if active_swing_leg in REAR_LEGS and is_front:
                    load_gate = self.forward_crawl_shift_gate(active_swing_leg, phase)
                    z -= warm * self.forward_front_support_load_m * load_gate

            thigh, calf = self.inverse_sagittal(x, z)
            calf_raw = calf
            calf -= warm * lift_extra * calf_lift_gate
            front_calf_clipped = 0.0
            if is_front and calf < self.front_calf_min_rad:
                calf = self.front_calf_min_rad
                front_calf_clipped = 1.0
            if is_front:
                thigh = self.default_policy[LEG_START[leg] + 1] + (
                    thigh - self.default_policy[LEG_START[leg] + 1]
                ) * self.front_thigh_delta_scale
            elif is_active:
                thigh += warm * 0.50 * self.rear_swing_thigh_lift_amp * calf_lift_gate

            touchdown_counter = self.compute_touchdown_counter(leg, state_info, active_swing_leg)
            hip_delta = self.compute_hip_balance_delta(
                leg,
                active_swing_leg,
                state_info,
                body_y_shift,
                touchdown_counter,
            )

            i = LEG_START[leg]
            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta
            q[i + 1] = self.default_policy[i + 1] + warm * (
                thigh - self.default_policy[i + 1]
            )
            q[i + 2] = self.default_policy[i + 2] + warm * (
                calf - self.default_policy[i + 2]
            )

            true_swing = 1.0 if is_active and state in ("LIFT", "ADVANCE", "TOUCHDOWN") else 0.0
            leg_debug[leg] = {
                "leg_phase": float(rel),
                "stance": float(0.0 if true_swing > 0.5 else 1.0),
                "swing": float(true_swing),
                "stance_shape": float(stance_shape),
                "swing_shape": float(swing_shape),
                "x_foot": float(x),
                "z_foot": float(z),
                "hip_delta": float(hip_delta),
                "thigh_ik": float(thigh),
                "calf_ik": float(calf),
                "calf_ik_raw": float(calf_raw),
                "front_calf_clipped": float(front_calf_clipped),
                "touchdown_counter": float(touchdown_counter),
                "thigh_target": float(thigh),
                "calf_target": float(calf),
                "motion_mode": self.motion_mode,
                "active_swing_leg": active_swing_leg or "",
                "leg_state": state,
                "support_legs": ",".join(support_legs),
                "body_x_shift": float(body_x_shift),
                "body_y_shift": float(body_y_shift),
                "support_triangle_stable": float(1.0 if stable else 0.0),
                "stability_margin_est": float(margin_est),
                **{
                    key: float(state_info.get(key, 0.0))
                    for key in (
                        "preload_gate",
                        "unload_gate",
                        "lift_gate",
                        "advance_gate",
                        "touchdown_gate",
                        "settle_gate",
                    )
                },
            }

        return q.astype(np.float32), leg_debug

    def old_get_loop_offsets(self, duty: Optional[float] = None) -> dict[str, float]:
        duty_value = self.duty_factor if duty is None else duty
        duty_value = min(max(float(duty_value), 0.74), 0.90)
        swing_fraction = max(0.08, 1.0 - duty_value)
        delay = max(float(self.diagonal_pair_delay_phase), swing_fraction + 0.02)
        delay = min(max(delay, 0.12), 0.32)
        starts = {
            "RR": 0.00,
            "FL": delay,
            "RL": 0.50,
            "FR": (0.50 + delay) % 1.0,
        }
        return {leg: ((1.0 - start) % 1.0) for leg, start in starts.items()}

    def old_get_leg_phase_value(
        self,
        leg: str,
        phase: float,
        duty: Optional[float] = None,
    ) -> float:
        offsets = self.old_get_loop_offsets(duty=duty)
        return float((phase + offsets[leg]) % 1.0)

    def old_get_active_swing_leg(
        self,
        phase: float,
        duty: Optional[float] = None,
    ) -> Optional[str]:
        duty_value = self.duty_factor if duty is None else duty
        duty_value = min(max(float(duty_value), 0.74), 0.90)
        swing_fraction = max(0.08, 1.0 - duty_value)
        for leg in OLD_STABLE_SWING_ORDER:
            if self.old_get_leg_phase_value(leg, phase, duty=duty_value) < swing_fraction:
                return leg
        return None

    def old_get_pre_swing_leg(
        self,
        phase: float,
        duty: Optional[float] = None,
    ) -> Optional[str]:
        duty_value = self.duty_factor if duty is None else duty
        duty_value = min(max(float(duty_value), 0.74), 0.90)
        swing_fraction = max(0.08, 1.0 - duty_value)
        pre_window = max(0.02, min(0.50, self.pre_swing_fraction))
        start = max(swing_fraction, 1.0 - pre_window)
        for leg in OLD_STABLE_SWING_ORDER:
            p = self.old_get_leg_phase_value(leg, phase, duty=duty_value)
            if p >= start:
                return leg
        return None

    def old_compute_touchdown_gate(
        self,
        leg: str,
        leg_phase: float,
        swing_fraction: float,
    ) -> float:
        if leg_phase >= swing_fraction:
            return 0.0
        s = leg_phase / max(swing_fraction, 1e-6)
        if leg in FRONT_LEGS:
            start = min(0.95, max(0.05, self.front_touchdown_start))
        elif leg in REAR_LEGS:
            start = min(0.95, max(0.05, self.rear_touchdown_start))
        else:
            return 0.0
        return self.smootherstep((s - start) / max(1.0 - start, 1e-6))

    def old_solve_calf_for_z(self, thigh: float, z_des: float, calf_default: float) -> float:
        value = (-float(z_des) - self.thigh_length * math.cos(float(thigh))) / max(
            self.calf_length,
            1e-9,
        )
        value = min(1.0, max(-1.0, value))
        angle = math.acos(value)
        candidates = (angle - thigh, -angle - thigh)
        return float(min(candidates, key=lambda calf: abs(calf - calf_default)))

    def old_compute_hip_balance_delta(
        self,
        leg: str,
        active_swing_leg: Optional[str],
        swing_shape: float,
        stance_shape: float,
        touchdown_gate: float = 0.0,
    ) -> float:
        outward = OLD_HIP_OUTWARD_SIGNS[leg]
        is_front = leg in FRONT_LEGS
        support_scale = self.front_support_hip_scale if is_front else self.rear_support_hip_scale
        if active_swing_leg is None:
            return float(outward * self.support_hip_outward_amp * support_scale * 0.2)
        if leg == active_swing_leg:
            if is_front:
                counter = -outward * self.front_touchdown_hip_counter_amp * touchdown_gate
                if leg == "FR":
                    counter += -abs(self.fr_swing_hip_inward_amp) * swing_shape
                return float(counter)
            return float(-outward * self.rear_touchdown_hip_counter_amp * touchdown_gate)

        base = outward * support_scale * self.support_hip_outward_amp * (
            0.65 + 0.35 * stance_shape
        )
        swing_side = OLD_LEG_SIDE[active_swing_leg]
        preferred_support_side = -swing_side if self.hip_body_y_sign >= 0.0 else swing_side
        leg_side_scale = (
            self.left_side_support_scale
            if OLD_LEG_SIDE[leg] > 0.0
            else self.right_side_support_scale
        )
        if OLD_LEG_SIDE[leg] == preferred_support_side:
            side_boost = self.side_support_hip_amp * leg_side_scale
        else:
            side_boost = (
                self.side_support_hip_amp
                * self.same_side_support_hip_scale
                * leg_side_scale
            )
        delta = base + outward * side_boost * (0.75 + 0.25 * stance_shape)
        if active_swing_leg in REAR_LEGS and OLD_LEG_SIDE[leg] == preferred_support_side:
            delta += outward * self.rear_swing_lateral_hip_amp * leg_side_scale
        return float(delta)

    def old_compute_calf_dominant_target(
        self,
        leg: str,
        leg_phase: float,
        duty: float,
        swing_fraction: float,
        thigh_default: float,
        calf_default: float,
        active_swing_leg: Optional[str] = None,
        pre_swing_leg: Optional[str] = None,
        active_swing_phase: Optional[float] = None,
    ) -> dict[str, float]:
        x0, z0 = self.default_foot_xz[leg]
        is_front = leg in FRONT_LEGS
        stride_gain = self.front_stride_gain if is_front else self.rear_stride_gain
        height_gain = self.front_swing_height_gain if is_front else self.rear_swing_height_gain
        thigh_scale = self.front_thigh_delta_scale if is_front else self.rear_thigh_delta_scale
        calf_lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
        calf_stance_push = (
            self.front_calf_stance_push_amp if is_front else self.rear_calf_stance_push_amp
        )
        stride = self.stride_length * stride_gain
        swing_h = self.swing_height * height_gain
        x_center = x0 + (self.front_x_bias if is_front else 0.0)
        z_center = z0 + (self.front_z_extend if is_front else 0.0)
        front_unfold = 0.0
        rear_swing_gate = 0.0
        if active_swing_leg in REAR_LEGS and active_swing_phase is not None:
            s_active = active_swing_phase / max(swing_fraction, 1e-6)
            rear_swing_gate = self.smootherstep(s_active / 0.28) * self.smootherstep(
                (1.0 - s_active) / 0.28
            )

        if leg_phase < swing_fraction:
            s = leg_phase / max(swing_fraction, 1e-6)
            denom = max(1e-6, self.advance_end - self.advance_start)
            u = self.smootherstep((s - self.advance_start) / denom)
            lift_up = self.smootherstep(s / max(self.advance_start, 0.001))
            lift_down = self.smootherstep((1.0 - s) / max(1.0 - self.advance_end, 0.001))
            swing_shape = lift_up * lift_down
            stance_shape = 0.0
            stance_gate = 0.0
            front_unfold = self.front_swing_forward_unfold * swing_shape if is_front else 0.0
            x_des = x_center - 0.5 * stride + stride * u + front_unfold
            z_des = z_center + swing_h * swing_shape
        else:
            s = (leg_phase - swing_fraction) / max(duty, 1e-6)
            u = self.smootherstep(s)
            swing_shape = 0.0
            stance_shape = math.sin(math.pi * min(1.0, max(0.0, s))) ** 2
            stance_gate = self.smooth_window(s, edge=0.18)
            x_des = x_center + 0.5 * stride - stride * u
            z_des = z_center

        body_x_shift_applied = 0.0
        if active_swing_leg in REAR_LEGS and rear_swing_gate > 0.0:
            if leg == active_swing_leg:
                scale = self.rear_swing_swing_leg_x_scale
            elif leg in FRONT_LEGS:
                scale = self.rear_swing_front_x_shift_scale
            else:
                scale = self.rear_swing_opposite_rear_x_shift_scale
            body_x_shift_applied = self.rear_swing_body_x_shift * rear_swing_gate * scale
            x_des -= body_x_shift_applied

        thigh_ik, calf_ik = self.inverse_sagittal(x_des, z_des)
        thigh_target = thigh_default + thigh_scale * (thigh_ik - thigh_default)
        calf_target = self.old_solve_calf_for_z(thigh_target, z_des, calf_default)

        coordinated_push = 0.0
        if active_swing_leg is not None and leg_phase >= swing_fraction:
            if leg == OLD_DIAGONAL_PARTNER.get(active_swing_leg):
                coordinated_push += self.diagonal_push_boost
            if active_swing_leg in FRONT_LEGS and leg in REAR_LEGS:
                coordinated_push += self.rear_push_during_front_swing_amp
            if active_swing_leg in REAR_LEGS and leg in FRONT_LEGS:
                coordinated_push += self.front_push_during_rear_swing_amp

        support_hold = 0.0
        pre_unload = 0.0
        pre_support_boost = 0.0
        if leg_phase >= swing_fraction:
            if active_swing_leg is not None:
                support_hold = self.support_calf_hold_amp * stance_gate
                if active_swing_leg in REAR_LEGS and rear_swing_gate > 0.0:
                    if is_front:
                        front_boost = self.rear_swing_front_support_boost_amp
                        if leg == OLD_DIAGONAL_PARTNER.get(active_swing_leg):
                            front_boost += self.old_diagonal_front_support_boost_amp
                        else:
                            front_boost *= max(0.0, self.old_same_front_support_scale)
                        support_hold += front_boost * rear_swing_gate * stance_gate
                    elif leg != active_swing_leg:
                        support_hold += (
                            self.rear_swing_rear_support_hold_amp
                            - self.rear_swing_rear_support_relief_amp
                        ) * rear_swing_gate * stance_gate
            pre_window = max(0.02, min(0.50, self.pre_swing_fraction))
            pre_gate = self.smootherstep((s - (1.0 - pre_window)) / pre_window)
            if leg == pre_swing_leg:
                pre_unload = -self.pre_swing_unload_amp * pre_gate
            elif pre_swing_leg is not None:
                pre_support_boost = self.pre_swing_support_boost_amp * stance_gate

        calf_target += (
            -calf_lift_extra * swing_shape
            + support_hold
            + pre_unload
            + pre_support_boost
            + (calf_stance_push + coordinated_push) * stance_shape * stance_gate
        )
        x_actual, z_actual = self.forward_sagittal(thigh_target, calf_target)
        return {
            "x_des": float(x_des),
            "z_des": float(z_des),
            "x_actual": float(x_actual),
            "z_actual": float(z_actual),
            "swing_shape": float(swing_shape),
            "stance_shape": float(stance_shape),
            "front_unfold": float(front_unfold),
            "coordinated_push": float(coordinated_push),
            "support_hold": float(support_hold),
            "pre_unload": float(pre_unload),
            "pre_support_boost": float(pre_support_boost),
            "rear_swing_gate": float(rear_swing_gate),
            "body_x_shift": float(body_x_shift_applied),
            "thigh_ik": float(thigh_ik),
            "calf_ik": float(calf_ik),
            "thigh_target": float(thigh_target),
            "calf_target": float(calf_target),
        }

    def build_old_forward_calf_walk_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        duty = min(max(self.duty_factor, 0.74), 0.90)
        swing_fraction = max(0.08, 1.0 - duty)
        active_swing_leg = self.old_get_active_swing_leg(phase, duty=duty)
        pre_swing_leg = self.old_get_pre_swing_leg(phase, duty=duty)
        active_swing_phase = None
        if active_swing_leg is not None:
            active_swing_phase = self.old_get_leg_phase_value(
                active_swing_leg,
                phase,
                duty=duty,
            )

        for leg in LEG_ORDER:
            leg_phase = self.old_get_leg_phase_value(leg, phase, duty=duty)
            is_front = leg in FRONT_LEGS
            is_swing = leg_phase < swing_fraction
            i = LEG_START[leg]
            thigh_default = float(self.default_policy[i + 1])
            calf_default = float(self.default_policy[i + 2])
            target = self.old_compute_calf_dominant_target(
                leg=leg,
                leg_phase=leg_phase,
                duty=duty,
                swing_fraction=swing_fraction,
                thigh_default=thigh_default,
                calf_default=calf_default,
                active_swing_leg=active_swing_leg,
                pre_swing_leg=pre_swing_leg,
                active_swing_phase=active_swing_phase,
            )
            thigh_target = target["thigh_target"]
            calf_target = target["calf_target"]
            swing_shape = target["swing_shape"]
            stance_shape = target["stance_shape"]
            touchdown_gate = self.old_compute_touchdown_gate(leg, leg_phase, swing_fraction)
            hip_delta = self.old_compute_hip_balance_delta(
                leg=leg,
                active_swing_leg=active_swing_leg,
                swing_shape=swing_shape,
                stance_shape=stance_shape,
                touchdown_gate=touchdown_gate,
            )
            hip_scale = self.front_hip_swing_scale if is_front else self.rear_hip_swing_scale
            thigh_swing_scale = (
                self.front_thigh_swing_scale if is_front else self.rear_thigh_swing_scale
            )
            if leg == "FL":
                calf_swing_scale = self.front_calf_swing_scale * self.fl_calf_swing_scale
            elif is_front:
                calf_swing_scale = self.front_calf_swing_scale
            else:
                calf_swing_scale = self.rear_calf_swing_scale
            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta * hip_scale
            q[i + 1] = self.default_policy[i + 1] + warm * (
                thigh_target - self.default_policy[i + 1]
            ) * thigh_swing_scale
            q[i + 2] = self.default_policy[i + 2] + warm * (
                calf_target - self.default_policy[i + 2]
            ) * calf_swing_scale
            leg_debug[leg] = {
                "leg_phase": float(leg_phase),
                "stance": float(0.0 if is_swing else 1.0),
                "swing": float(1.0 if is_swing else 0.0),
                "stance_shape": float(stance_shape),
                "swing_shape": float(swing_shape),
                "x_foot": float(target["x_actual"]),
                "z_foot": float(target["z_actual"]),
                "hip_delta": float(hip_delta),
                "touchdown_counter": float(touchdown_gate),
                "touchdown_gate": float(touchdown_gate),
                "thigh_ik": float(target["thigh_ik"]),
                "calf_ik": float(target["calf_ik"]),
                "calf_ik_raw": float(target["calf_ik"]),
                "front_calf_clipped": 0.0,
                "front_unfold": float(target["front_unfold"]),
                "coordinated_push": float(target["coordinated_push"]),
                "pre_unload": float(target["pre_unload"]),
                "pre_support_boost": float(target["pre_support_boost"]),
                "rear_swing_gate": float(target["rear_swing_gate"]),
                "body_x_shift": float(target["body_x_shift"]),
                "body_y_shift": 0.0,
                "active_swing_leg": active_swing_leg or "",
                "swing_leg": active_swing_leg or "none",
                "pre_swing_leg": pre_swing_leg or "none",
                "leg_state": "SWING" if is_swing else "SUPPORT",
                "support_legs": ",".join(self.support_legs_for_swing(active_swing_leg)),
                "support_triangle_stable": 1.0,
                "stability_margin_est": 0.0,
                "thigh_target": float(thigh_target),
                "calf_target": float(calf_target),
                "hip_swing_scale": float(hip_scale),
                "thigh_swing_scale": float(thigh_swing_scale),
                "calf_swing_scale": float(calf_swing_scale),
                "motion_mode": self.motion_mode,
            }
        return q.astype(np.float32), leg_debug

    def triangle_event_fraction(self) -> float:
        duty = min(max(self.duty_factor, 0.50), 0.95)
        # Four legs are spaced by 0.25 phase. Cap the event window so states
        # cannot overlap even if a less conservative duty factor is requested.
        return min(0.24, max(0.05, 1.0 - duty))

    def normalized_state_fractions(self) -> list[tuple[str, float]]:
        pieces = [
            ("PRELOAD", self.preload_fraction),
            ("UNLOAD", self.unload_fraction),
            ("LIFT", self.lift_fraction),
            ("ADVANCE", self.advance_fraction),
            ("TOUCHDOWN", self.touchdown_fraction),
            ("SETTLE", self.settle_fraction),
        ]
        total = sum(max(0.0, float(value)) for _, value in pieces)
        if total <= 1e-9:
            pieces = [
                ("PRELOAD", 0.10),
                ("UNLOAD", 0.10),
                ("LIFT", 0.22),
                ("ADVANCE", 0.38),
                ("TOUCHDOWN", 0.14),
                ("SETTLE", 0.06),
            ]
            total = 1.0

        acc = 0.0
        out = []
        for name, value in pieces:
            acc += max(0.0, float(value)) / total
            out.append((name, min(1.0, acc)))
        out[-1] = (out[-1][0], 1.0)
        return out

    def get_active_swing_leg(self, phase: float) -> Optional[str]:
        event_fraction = self.triangle_event_fraction()
        for leg in CRAWL_SWING_ORDER:
            rel = (float(phase) - CRAWL_OFFSETS[leg]) % 1.0
            if rel < event_fraction:
                return leg
        return None

    def get_leg_state(self, leg: str, phase: float) -> dict[str, Any]:
        event_fraction = self.triangle_event_fraction()
        rel = (float(phase) - CRAWL_OFFSETS[leg]) % 1.0
        gates = {
            "preload_gate": 0.0,
            "unload_gate": 0.0,
            "lift_gate": 0.0,
            "advance_gate": 0.0,
            "touchdown_gate": 0.0,
            "settle_gate": 0.0,
        }

        if rel >= event_fraction:
            return {
                "leg_phase": float(rel),
                "event_s": 1.0,
                "state_s": 0.0,
                "leg_state": "SUPPORT",
                **gates,
            }

        event_s = rel / max(event_fraction, 1e-6)
        prev_end = 0.0
        state = "SETTLE"
        state_s = 1.0
        for name, end in self.normalized_state_fractions():
            if event_s <= end + 1e-9:
                state = name
                state_s = (event_s - prev_end) / max(end - prev_end, 1e-6)
                break
            prev_end = end

        gate_key = f"{state.lower()}_gate"
        if gate_key in gates:
            gates[gate_key] = self.smoothstep_half_cos(state_s)

        return {
            "leg_phase": float(rel),
            "event_s": float(event_s),
            "state_s": float(min(1.0, max(0.0, state_s))),
            "leg_state": state,
            **gates,
        }

    def desired_body_shift_for_swing(self, swing_leg: Optional[str]) -> tuple[float, float]:
        if swing_leg is None:
            return 0.0, 0.0

        body_x = (
            self.rear_swing_body_x_shift
            if swing_leg in REAR_LEGS
            else self.front_swing_body_x_shift
        )
        body_y = self.body_y_shift_amp if swing_leg in RIGHT_LEGS else -self.body_y_shift_amp
        return float(body_x), float(body_y)

    @staticmethod
    def support_legs_for_swing(swing_leg: Optional[str]) -> list[str]:
        if swing_leg not in LEG_ORDER:
            return list(LEG_ORDER)
        return [leg for leg in LEG_ORDER if leg != swing_leg]

    def support_foot_xy(self, leg: str) -> tuple[float, float]:
        x = self.front_foot_x_default if leg in FRONT_LEGS else self.rear_foot_x_default
        y = -self.foot_y_half_width if leg in RIGHT_LEGS else self.foot_y_half_width
        return float(x), float(y)

    @staticmethod
    def point_in_triangle_with_margin(
        com_xy: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        margin: float,
    ) -> tuple[bool, float]:
        px, py = com_xy
        pts = (p1, p2, p3)
        area = 0.5 * (
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        if abs(area) < 1e-9:
            return False, 0.0

        signs = []
        distances = []
        for a, b in ((p1, p2), (p2, p3), (p3, p1)):
            edge_x = b[0] - a[0]
            edge_y = b[1] - a[1]
            cross = edge_x * (py - a[1]) - edge_y * (px - a[0])
            signs.append(cross)
            distances.append(abs(cross) / max(math.hypot(edge_x, edge_y), 1e-9))

        same_sign = all(v >= -1e-9 for v in signs) or all(v <= 1e-9 for v in signs)
        margin_est = min(distances) if distances else 0.0
        return bool(same_sign and margin_est >= margin), float(margin_est)

    def is_support_triangle_stable(
        self,
        swing_leg: Optional[str],
        body_x_shift: float,
        body_y_shift: float,
    ) -> tuple[bool, float]:
        if swing_leg is None:
            return True, float(self.stability_margin_m)

        support_legs = self.support_legs_for_swing(swing_leg)
        if len(support_legs) < 3:
            return False, 0.0

        points = [self.support_foot_xy(leg) for leg in support_legs[:3]]
        com = (
            self.com_x_default + float(body_x_shift),
            self.com_y_default + float(body_y_shift),
        )
        stable, margin_est = self.point_in_triangle_with_margin(
            com,
            points[0],
            points[1],
            points[2],
            self.forward_stability_margin_m,
        )

        direction_ok = True
        if swing_leg in REAR_LEGS and body_x_shift <= 0.0:
            direction_ok = False
        if swing_leg in FRONT_LEGS and body_x_shift >= 0.0:
            direction_ok = False
        if swing_leg in RIGHT_LEGS and body_y_shift <= 0.0:
            direction_ok = False
        if swing_leg in LEFT_LEGS and body_y_shift >= 0.0:
            direction_ok = False

        # Keep both checks: the support polygon must contain the shifted COM
        # with margin, and the body shift must be toward the support tripod.
        return bool(stable and direction_ok), float(margin_est)

    def compute_touchdown_counter(
        self,
        leg: str,
        state_info: dict[str, Any],
        active_swing_leg: Optional[str],
    ) -> float:
        if leg != active_swing_leg:
            return 0.0
        event_s = float(state_info.get("event_s", 1.0))
        start = self.front_touchdown_start if leg in FRONT_LEGS else self.rear_touchdown_start
        if event_s < start:
            return 0.0
        return self.smoothstep_half_cos((event_s - start) / max(1.0 - start, 1e-6))

    def compute_hip_balance_delta(
        self,
        leg: str,
        active_swing_leg: Optional[str],
        state_info: dict[str, Any],
        body_y_shift: float,
        touchdown_counter: float,
    ) -> float:
        outward = HIP_OUTWARD_SIGNS[leg]
        state = str(state_info.get("leg_state", "SUPPORT"))
        is_front = leg in FRONT_LEGS
        is_active = leg == active_swing_leg

        if is_active:
            swing_scale = self.front_hip_swing_scale if is_front else self.rear_hip_swing_scale
            unload_gate = float(state_info.get("unload_gate", 0.0))
            lift_gate = float(state_info.get("lift_gate", 0.0))
            advance_gate = float(state_info.get("advance_gate", 0.0))
            recenter_gate = max(unload_gate, lift_gate, advance_gate)
            inward_amp = (
                self.fr_swing_hip_inward_amp
                if leg == "FR"
                else self.swing_hip_unload_amp
            )
            counter_amp = (
                self.front_touchdown_hip_counter_amp
                if is_front
                else self.rear_touchdown_hip_counter_amp
            )
            active_inward = (
                -outward
                * self.hip_inward_hold_amp
                * max(0.0, self.hip_inward_active_scale)
            )
            return float(
                -outward * swing_scale * inward_amp * recenter_gate
                -outward * counter_amp * touchdown_counter
                + active_inward
            )

        support_scale = self.front_support_hip_scale if is_front else self.rear_support_hip_scale
        side_norm = 0.0
        if abs(self.body_y_shift_amp) > 1e-9:
            side_norm = (body_y_shift / self.body_y_shift_amp) * self.body_shift_hip_scale
        side_scale = self.right_side_support_scale if leg in RIGHT_LEGS else self.left_side_support_scale
        if active_swing_leg is not None:
            same_side = (leg in RIGHT_LEGS and active_swing_leg in RIGHT_LEGS) or (
                leg in LEFT_LEGS and active_swing_leg in LEFT_LEGS
            )
            if same_side:
                side_scale *= self.same_side_support_hip_scale

        state_gate = 0.0 if state == "SETTLE" else 1.0
        side_delta = (
            self.hip_body_y_sign
            * side_norm
            * self.side_support_hip_amp
            * side_scale
            * state_gate
        )
        support_delta = outward * self.support_hip_outward_amp * support_scale
        inward_hold = -outward * self.hip_inward_hold_amp * state_gate
        return float(side_delta + support_delta + inward_hold)

    def solve_calf_for_z(self, x: float, z: float) -> float:
        _, calf = self.inverse_sagittal(x, z)
        return float(calf)

    def compute_leg_loop_target(
        self,
        leg: str,
        active_swing_leg: Optional[str],
        state_info: dict[str, Any],
        body_x_shift: float,
        body_y_shift: float,
        warm: float,
    ) -> tuple[float, float, float, float, float, dict[str, Any]]:
        is_front = leg in FRONT_LEGS
        is_rear = leg in REAR_LEGS
        is_active = leg == active_swing_leg
        x0, z0 = self.default_foot_xz[leg]
        x_center = x0 + (self.front_x_bias if is_front else 0.0)

        if self.lock_support_legs and not is_active:
            i = LEG_START[leg]
            thigh = float(self.default_policy[i + 1])
            calf = float(self.default_policy[i + 2])
            debug = {
                "leg_phase": float(state_info.get("leg_phase", 0.0)),
                "stance": 1.0,
                "swing": 0.0,
                "stance_shape": 0.0,
                "swing_shape": 0.0,
                "x_foot": float(x0),
                "z_foot": float(z0),
                "hip_delta": 0.0,
                "thigh_ik": thigh,
                "calf_ik": calf,
                "calf_ik_raw": calf,
                "front_calf_clipped": 0.0,
                "touchdown_counter": 0.0,
                "thigh_target": thigh,
                "calf_target": calf,
                **{
                    key: 0.0
                    for key in (
                        "preload_gate",
                        "unload_gate",
                        "lift_gate",
                        "advance_gate",
                        "touchdown_gate",
                        "settle_gate",
                    )
                },
            }
            return float(x0), float(z0), thigh, calf, 0.0, debug

        stride_gain = self.front_stride_gain if is_front else max(
            self.rear_stride_gain,
            self.front_stride_gain * max(0.0, self.rear_stride_min_front_ratio),
        )
        stride = (
            self.stride_length
            * self.walk_stride_scale
            * stride_gain
        )
        stride_dir = self.walk_direction
        height_gain = self.front_swing_height_gain if is_front else self.rear_swing_height_gain
        lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
        swing_height = self.swing_height * height_gain

        x_shift_scale = 1.0
        z_support_delta = 0.0
        if active_swing_leg in REAR_LEGS:
            opposite_rear = "RL" if active_swing_leg == "RR" else "RR"
            if leg in FRONT_LEGS:
                x_shift_scale = self.rear_swing_front_x_shift_scale
                z_support_delta -= self.rear_swing_front_support_boost_amp
            elif leg == opposite_rear:
                x_shift_scale = self.rear_swing_opposite_rear_x_shift_scale
                z_support_delta -= self.rear_swing_opposite_rear_support_boost_amp
                z_support_delta += self.rear_swing_rear_support_relief_amp
            elif leg == active_swing_leg:
                x_shift_scale = self.rear_swing_swing_leg_x_scale
        elif active_swing_leg in FRONT_LEGS and leg in REAR_LEGS and not is_active:
            z_support_delta -= 0.35 * self.rear_calf_stance_push_amp

        state = str(state_info.get("leg_state", "SUPPORT"))
        state_s = float(state_info.get("state_s", 0.0))
        event_s = float(state_info.get("event_s", 1.0))
        support_shift_gate = 1.0
        if active_swing_leg in REAR_LEGS and not is_active:
            ramp_in = self.smoothstep_half_cos(
                event_s / max(self.rear_support_shift_ramp_fraction, 1e-6)
            )
            ramp_out = self.smoothstep_half_cos(
                (1.0 - event_s) / max(self.rear_support_return_fraction, 1e-6)
            )
            support_shift_gate = ramp_in * ramp_out

        x = x_center - warm * body_x_shift * x_shift_scale * self.body_shift_foot_x_scale * support_shift_gate
        z = z0 + warm * z_support_delta * support_shift_gate
        swing_shape = 0.0
        stance_shape = 0.0
        calf_lift_gate = 0.0

        if is_active:
            if state in ("PRELOAD", "UNLOAD"):
                x = x_center
            if is_rear and state == "PRELOAD":
                z = z0 + warm * self.rear_pre_lift_relief_m * self.smoothstep_half_cos(state_s)
            if state == "UNLOAD":
                relief = self.rear_pre_lift_relief_m if is_rear else 0.0
                z = z0 + warm * (
                    relief + 0.12 * swing_height * self.smoothstep_half_cos(state_s)
                )
            elif state == "LIFT":
                u = self.smoothstep_half_cos(state_s)
                x = x_center + stride_dir * (-0.5 * stride * warm * u)
                extra_lift = self.rear_swing_extra_lift_m if is_rear else 0.0
                z = z0 + warm * (0.12 + 0.88 * u) * swing_height + warm * extra_lift * u
                swing_shape = math.sin(math.pi * min(1.0, max(0.0, event_s)))
                calf_lift_gate = u
            elif state == "ADVANCE":
                u = self.smoothstep_half_cos(state_s)
                x = x_center + stride_dir * (-0.5 * stride + stride * u)
                if is_front:
                    x += stride_dir * self.front_swing_forward_unfold * u
                extra_lift = self.rear_swing_extra_lift_m if is_rear else 0.0
                z = z0 + warm * (swing_height + extra_lift)
                swing_shape = 1.0
                calf_lift_gate = 1.0
            elif state == "TOUCHDOWN":
                u = self.smoothstep_half_cos(state_s)
                x = x_center + stride_dir * 0.5 * stride
                if is_front:
                    x += stride_dir * self.front_swing_forward_unfold
                extra_lift = self.rear_swing_extra_lift_m if is_rear else 0.0
                z = z0 + warm * (swing_height + extra_lift) * (1.0 - u)
                swing_shape = 1.0 - u
                calf_lift_gate = 1.0 - u
            elif state == "SETTLE":
                x = x_center + stride_dir * 0.5 * stride * (
                    1.0 - self.smoothstep_half_cos(state_s)
                )
                z = z0
        else:
            support_s = ((float(state_info.get("leg_phase", 0.0)) - self.triangle_event_fraction())
                         / max(1.0 - self.triangle_event_fraction(), 1e-6))
            support_s = min(1.0, max(0.0, support_s))
            stance_shape = math.sin(math.pi * support_s)
            drive = self.smoothstep_half_cos(support_s)
            drive_scale = self.front_stance_drive_scale if is_front else self.rear_stance_drive_scale
            drive_scale += self.stance_drive_scale
            x += warm * stride_dir * drive_scale * 0.5 * stride * (1.0 - 2.0 * drive)
            if is_rear:
                stand_tall_gate = 0.35 + 0.65 * stance_shape
                z -= warm * self.rear_support_stand_tall_m * stand_tall_gate

        thigh, calf = self.inverse_sagittal(x, z)
        calf_raw = calf
        calf -= warm * lift_extra * calf_lift_gate
        front_calf_clipped = 0.0
        if is_front and calf < self.front_calf_min_rad:
            calf = self.front_calf_min_rad
            front_calf_clipped = 1.0

        if is_front:
            thigh = self.default_policy[LEG_START[leg] + 1] + (
                thigh - self.default_policy[LEG_START[leg] + 1]
            ) * self.front_thigh_delta_scale
        elif is_active:
            thigh += warm * self.rear_swing_thigh_lift_amp * calf_lift_gate

        if is_rear and not is_active:
            calf += warm * self.rear_calf_stance_push_amp * stance_shape

        touchdown_counter = self.compute_touchdown_counter(leg, state_info, active_swing_leg)
        hip_delta = self.compute_hip_balance_delta(
            leg,
            active_swing_leg,
            state_info,
            body_y_shift,
            touchdown_counter,
        )

        true_swing = 1.0 if is_active and state in ("LIFT", "ADVANCE", "TOUCHDOWN") else 0.0
        debug = {
            "leg_phase": float(state_info.get("leg_phase", 0.0)),
            "stance": float(0.0 if true_swing > 0.5 else 1.0),
            "swing": float(true_swing),
            "stance_shape": float(stance_shape),
            "swing_shape": float(swing_shape),
            "x_foot": float(x),
            "z_foot": float(z),
            "hip_delta": float(hip_delta),
            "thigh_ik": float(thigh),
            "calf_ik": float(calf),
            "calf_ik_raw": float(calf_raw),
            "front_calf_clipped": float(front_calf_clipped),
            "touchdown_counter": float(touchdown_counter),
            "thigh_target": float(thigh),
            "calf_target": float(calf),
            **{
                key: float(state_info.get(key, 0.0))
                for key in (
                    "preload_gate",
                    "unload_gate",
                    "lift_gate",
                    "advance_gate",
                    "touchdown_gate",
                    "settle_gate",
                )
            },
        }
        return float(x), float(z), float(thigh), float(calf), float(hip_delta), debug

    def build_triangle_stable_walk_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        active_swing_leg = self.get_active_swing_leg(phase)
        body_x_shift, body_y_shift = self.desired_body_shift_for_swing(active_swing_leg)
        stable, margin_est = self.is_support_triangle_stable(
            active_swing_leg,
            body_x_shift,
            body_y_shift,
        )
        support_legs = self.support_legs_for_swing(active_swing_leg)

        if active_swing_leg is not None and not stable:
            now = time.time()
            if now - self._last_stability_warn_time > 1.0:
                self._last_stability_warn_time = now
                self.get_logger().warn(
                    "[STABILITY] support triangle estimate unsafe: "
                    f"swing={active_swing_leg} support={','.join(support_legs)} "
                    f"body_shift=({body_x_shift:.3f},{body_y_shift:.3f}) "
                    f"margin={margin_est:.3f}m. Continue cautiously."
                )

        for leg in LEG_ORDER:
            state_info = self.get_leg_state(leg, phase)
            _, _, thigh, calf, hip_delta, debug = self.compute_leg_loop_target(
                leg,
                active_swing_leg,
                state_info,
                body_x_shift,
                body_y_shift,
                warm,
            )

            i = LEG_START[leg]
            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta
            q[i + 1] = self.default_policy[i + 1] + warm * (
                thigh - self.default_policy[i + 1]
            )
            q[i + 2] = self.default_policy[i + 2] + warm * (
                calf - self.default_policy[i + 2]
            )

            debug.update(
                {
                    "motion_mode": self.motion_mode,
                    "active_swing_leg": active_swing_leg or "",
                    "leg_state": str(state_info.get("leg_state", "SUPPORT")),
                    "support_legs": ",".join(support_legs),
                    "body_x_shift": float(body_x_shift),
                    "body_y_shift": float(body_y_shift),
                    "support_triangle_stable": float(1.0 if stable else 0.0),
                    "stability_margin_est": float(margin_est),
                }
            )
            leg_debug[leg] = debug

        return q.astype(np.float32), leg_debug

    def apply_target_rate_limit(self, target_policy: np.ndarray, dt: float) -> np.ndarray:
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        if self.max_target_rate_rad_s <= 0.0 or dt <= 0.0:
            self.last_target_policy = target_policy.copy()
            return target_policy

        max_step = self.max_target_rate_rad_s * dt
        step = np.clip(target_policy - self.last_target_policy, -max_step, max_step)
        limited = self.last_target_policy + step
        limited = np.clip(limited, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.last_target_policy = limited.astype(np.float32).copy()
        return self.last_target_policy.copy()

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(target_real[i]),
                    "speed": self.send_speed,
                    "torque": self.send_torque,
                    "kp": self.send_kp,
                    "kd": self.send_kd,
                }
            )

        payload = {"items": items, "enable_first": False, "stop_first": False}

        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json=payload,
                timeout=self.http_timeout,
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            now = time.time()
            if now - self._last_send_info_time > 1.0:
                self._last_send_info_time = now
                self.get_logger().info(
                    f"[SEND] fanfan IK ok | target_min={float(np.min(target_real)):.3f} "
                    f"target_max={float(np.max(target_real)):.3f} kp={self.send_kp:.1f} kd={self.send_kd:.1f}"
                )
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def setup_debug_csv(self):
        path = self.debug_csv_path.strip()
        if not path:
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._debug_csv_file = open(path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        self._debug_csv_writer.writerow(
            [
                "time",
                "elapsed",
                "motion_mode",
                "phase",
                "warm",
                "active_swing_leg",
                "leg_phase",
                "leg_state",
                "support_legs",
                "stance",
                "swing",
                "leg_name",
                "joint_index",
                "motor_id",
                "joint_name",
                "policy_joint_name",
                "x_foot",
                "z_foot",
                "body_x_shift",
                "body_y_shift",
                "support_triangle_stable",
                "stability_margin_est",
                "hip_delta",
                "touchdown_counter",
                "preload_gate",
                "unload_gate",
                "lift_gate",
                "advance_gate",
                "touchdown_gate",
                "settle_gate",
                "thigh_target",
                "calf_target",
                "q_target_policy",
                "q_target_real",
                "q_current_real",
                "q_error_real",
                "torque_measured",
                "temp",
                "online",
                "error_code",
                "age_ms",
                "sent",
                "step_hz",
                "stride_length",
                "swing_height",
                "duty_factor",
                "kp",
                "kd",
            ]
        )
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[DEBUG_CSV] writing fanfan IK gait data to {path}")

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(
            target=self.debug_collect_loop,
            name="fanfan_ik_gait_logger",
            daemon=True,
        )
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec
        if period <= 0.0:
            period = 1.0 / max(self.gait_hz, 1e-3)

        while not self._debug_stop_event.wait(period):
            with self._debug_sample_lock:
                sample = self._latest_debug_sample
                if sample is not None:
                    sample = {
                        key: (value.copy() if isinstance(value, np.ndarray) else value)
                        for key, value in sample.items()
                    }
            if sample is None:
                continue
            self.write_debug_csv_sample(**sample)

    def update_debug_sample(
        self,
        target_real: np.ndarray,
        target_policy: np.ndarray,
        phase: float,
        warm: float,
        leg_debug: dict[str, dict[str, float]],
        sent: bool,
    ):
        if self._debug_csv_writer is None:
            return

        with self._debug_sample_lock:
            self._latest_debug_sample = {
                "target_real": np.asarray(target_real, dtype=np.float32).reshape(12).copy(),
                "target_policy": np.asarray(target_policy, dtype=np.float32).reshape(12).copy(),
                "phase": float(phase),
                "warm": float(warm),
                "leg_debug": {leg: dict(data) for leg, data in leg_debug.items()},
                "sent": bool(sent),
                "stamp": time.time(),
            }

    def write_debug_csv_sample(
        self,
        target_real: np.ndarray,
        target_policy: np.ndarray,
        phase: float,
        warm: float,
        leg_debug: dict[str, dict[str, float]],
        sent: bool,
        stamp: float,
    ):
        if self._debug_csv_writer is None:
            return

        now = time.time()
        try:
            snapshot = self.motor.get_latest()
        except Exception as exc:
            if now - self._last_feedback_warn_time > 1.0:
                self._last_feedback_warn_time = now
                self.get_logger().warn(f"[DEBUG_CSV] motor feedback read failed: {exc}")
            return

        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        q_real = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
        torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
        temp = np.asarray(snapshot.temp, dtype=np.float32).reshape(12)
        online = np.asarray(snapshot.online, dtype=bool).reshape(12)
        error_code = np.asarray(snapshot.error_code, dtype=np.int32).reshape(12)
        age_ms = np.asarray(snapshot.age_ms, dtype=np.float32).reshape(12)

        if self.torque_warn_nm > 0.0:
            torque_abs_max = float(np.max(np.abs(torque)))
            if torque_abs_max > self.torque_warn_nm and now - self._last_torque_warn_time > 1.0:
                self._last_torque_warn_time = now
                self.get_logger().warn(
                    f"[TORQUE] measured max |torque|={torque_abs_max:.2f}Nm > "
                    f"{self.torque_warn_nm:.2f}Nm. Reduce step_hz, stride_length, "
                    "or swing_height, or increase front-leg support. Do not raise Kp automatically."
                )

        target_policy_real_order = np.zeros(12, dtype=np.float32)
        target_policy_real_order[self.mapper.policy_to_real_index] = target_policy

        elapsed = float(stamp) - self.start_time
        for real_i, (mid, real_name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
            policy_name = self.policy_joint_names[policy_i]
            leg = policy_name.split("_", 1)[0]
            leg_info = dict(LEG_FALLBACK_DEBUG)
            leg_info.update(leg_debug.get(leg, {}))
            self._debug_csv_writer.writerow(
                [
                    f"{now:.6f}",
                    f"{elapsed:.6f}",
                    str(leg_info.get("motion_mode", self.motion_mode)),
                    f"{phase:.6f}",
                    f"{warm:.6f}",
                    str(leg_info.get("active_swing_leg", "")),
                    f"{float(leg_info.get('leg_phase', 0.0)):.6f}",
                    str(leg_info.get("leg_state", "SUPPORT")),
                    str(leg_info.get("support_legs", "")),
                    int(float(leg_info.get("stance", 0.0)) > 0.5),
                    int(float(leg_info.get("swing", 0.0)) > 0.5),
                    leg,
                    int(real_i),
                    f"0x{int(mid):02X}",
                    real_name,
                    policy_name,
                    f"{float(leg_info.get('x_foot', 0.0)):.6f}",
                    f"{float(leg_info.get('z_foot', 0.0)):.6f}",
                    f"{float(leg_info.get('body_x_shift', 0.0)):.6f}",
                    f"{float(leg_info.get('body_y_shift', 0.0)):.6f}",
                    int(float(leg_info.get("support_triangle_stable", 1.0)) > 0.5),
                    f"{float(leg_info.get('stability_margin_est', 0.0)):.6f}",
                    f"{float(leg_info.get('hip_delta', 0.0)):.6f}",
                    f"{float(leg_info.get('touchdown_counter', 0.0)):.6f}",
                    f"{float(leg_info.get('preload_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('unload_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('lift_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('advance_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('touchdown_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('settle_gate', 0.0)):.6f}",
                    f"{float(leg_info.get('thigh_target', leg_info.get('thigh_ik', 0.0))):.6f}",
                    f"{float(leg_info.get('calf_target', leg_info.get('calf_ik', 0.0))):.6f}",
                    f"{float(target_policy_real_order[real_i]):.6f}",
                    f"{float(target_real[real_i]):.6f}",
                    f"{float(q_real[real_i]):.6f}",
                    f"{float(target_real[real_i] - q_real[real_i]):.6f}",
                    f"{float(torque[real_i]):.6f}",
                    f"{float(temp[real_i]):.3f}",
                    int(online[real_i]),
                    int(error_code[real_i]),
                    f"{float(age_ms[real_i]):.3f}",
                    int(bool(sent)),
                    f"{self.step_hz:.6f}",
                    f"{self.stride_length:.6f}",
                    f"{self.swing_height:.6f}",
                    f"{self.duty_factor:.6f}",
                    f"{self.send_kp:.6f}",
                    f"{self.send_kd:.6f}",
                ]
            )
        self._debug_csv_file.flush()

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def destroy_node(self):
        try:
            self._debug_stop_event.set()
            if self._debug_thread is not None:
                self._debug_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._debug_csv_file is not None:
                self._debug_csv_file.flush()
                self._debug_csv_file.close()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass
        try:
            self.http_session.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FanfanIkGaitNode()

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
