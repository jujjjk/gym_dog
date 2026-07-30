"""Run an RS01 Go2 policy in an independent MuJoCo model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
ROS_POLICY_SOURCE = (
    ROOT / "zhenji/mydog_ros2_ws/src/mydog_policy"
)
if str(ROS_POLICY_SOURCE) not in sys.path:
    sys.path.insert(0, str(ROS_POLICY_SOURCE))

from mydog_policy.rs01_model1850_core import (  # noqa: E402
    Rs01StraightPathEstimator,
)
from mydog_policy.rs01_model930_core import (  # noqa: E402
    Rs01NewMachineLegOdometry,
)


METADATA_KEY = "rs01_go2_deployment_config"
LEGS = ("FR", "FL", "RR", "RL")


def load_contract(session):
    metadata = session.get_modelmeta().custom_metadata_map
    if METADATA_KEY not in metadata:
        raise RuntimeError(f"ONNX is missing {METADATA_KEY}")
    contract = json.loads(metadata[METADATA_KEY])
    if contract.get("schema_version") != 2:
        raise RuntimeError(
            f"Unsupported schema version {contract.get('schema_version')}"
        )
    return contract


def load_scene_contract(scene):
    root = ET.parse(scene).getroot()
    custom = root.find("custom")
    if custom is None:
        return {}
    result = {}
    for entry in custom:
        name = entry.get("name")
        if name is not None and name.startswith("rs01_"):
            result[name] = entry.get("data")
    return result


def quaternion_rotation_matrix(quaternion_wxyz):
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, quaternion_wxyz)
    return result.reshape(3, 3)


def roll_pitch_yaw(quaternion_wxyz):
    rotation = quaternion_rotation_matrix(quaternion_wxyz)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def limited_target_step(
    desired,
    previous,
    previous_rate,
    rate_limit,
    acceleration_limit,
    dt,
):
    desired_rate = np.clip(
        (desired - previous) / dt,
        -rate_limit,
        rate_limit,
    )
    rate_step = np.clip(
        desired_rate - previous_rate,
        -acceleration_limit * dt,
        acceleration_limit * dt,
    )
    limited_rate = np.clip(
        previous_rate + rate_step,
        -rate_limit,
        rate_limit,
    )
    limited_target = previous + limited_rate * dt
    crossed = (desired - previous) * (desired - limited_target) <= 0.0
    limited_target = np.where(crossed, desired, limited_target)
    limited_rate = np.where(crossed, 0.0, limited_rate)
    return limited_target, limited_rate


def compute_rs01_torques(
    response_target,
    position,
    velocity,
    kp,
    kd,
    peak_limit,
    coulomb_friction,
    friction_smoothing,
):
    raw = kp * (response_target - position) - kd * velocity
    motor = np.clip(raw, -peak_limit, peak_limit)
    friction = coulomb_friction * np.tanh(
        velocity / max(float(friction_smoothing), 1.0e-4)
    )
    applied = motor - friction
    return raw.copy(), motor.copy(), applied.copy()


class Rs01Go2Sim:
    def __init__(self, scene, policy, command):
        self.session = ort.InferenceSession(
            str(policy),
            providers=["CPUExecutionProvider"],
        )
        self.cfg = load_contract(self.session)
        scene_contract = load_scene_contract(scene)
        expected_urdf_sha256 = self.cfg["simulator"].get("urdf_sha256")
        if expected_urdf_sha256 is not None:
            scene_urdf_sha256 = scene_contract.get(
                "rs01_source_urdf_sha256"
            )
            if scene_urdf_sha256 != expected_urdf_sha256:
                raise RuntimeError(
                    "Policy and MuJoCo scene use different RS01 URDFs: "
                    f"policy={expected_urdf_sha256}, "
                    f"scene={scene_urdf_sha256}"
                )
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.names = list(self.cfg["joint_names"])
        control = self.cfg["control"]
        observations = self.cfg["observations"]
        mujoco_cfg = self.cfg["simulator"].get("mujoco", {})

        self.default = np.asarray(
            self.cfg["default_joint_angles_rad"],
            dtype=np.float64,
        )
        self.command = np.asarray(command, dtype=np.float64)
        self.kp = np.asarray(control["kp_nm_per_rad"], dtype=np.float64)
        self.kd = np.asarray(control["kd_nm_per_rad_s"], dtype=np.float64)
        self.rate_limit = np.asarray(
            control["target_rate_limit_rad_s"],
            dtype=np.float64,
        )
        self.acceleration_limit = np.asarray(
            control["target_acceleration_limit_rad_s2"],
            dtype=np.float64,
        )
        self.response_gain = np.asarray(
            control["response_gain"],
            dtype=np.float64,
        )
        self.time_constant = np.asarray(
            control["time_constant_s"],
            dtype=np.float64,
        )
        delay_s = np.asarray(
            control["observed_closed_loop_delay_s"],
            dtype=np.float64,
        )
        self.friction = np.asarray(
            control["coulomb_friction_nm"],
            dtype=np.float64,
        )
        self.friction_smoothing = float(
            control["friction_smoothing_rad_s"]
        )
        self.peak_limit = float(control["peak_torque_limit_nm"])
        self.action_scale = np.asarray(
            control["action_scale_rad"],
            dtype=np.float64,
        )
        if self.action_scale.ndim == 0:
            self.action_scale = np.full(
                len(self.names),
                float(self.action_scale),
                dtype=np.float64,
            )
        self.action_clip = float(control["action_clip"])
        self.physics_dt = float(control["physics_dt_s"])
        self.policy_dt = float(control["policy_dt_s"])
        self.decimation = int(control["decimation"])
        self.integration_dt = float(
            mujoco_cfg.get("integration_timestep_s", self.physics_dt)
        )
        self.integration_substeps = int(
            mujoco_cfg.get("integration_substeps_per_motor_step", 1)
        )
        self.delay_steps = np.rint(delay_s / self.physics_dt).astype(int)
        self.max_delay_steps = int(self.delay_steps.max())
        self.response_alpha = 1.0 - np.exp(
            -self.physics_dt / self.time_constant
        )

        if self.integration_substeps < 1:
            raise RuntimeError("MuJoCo integration substeps must be positive")
        if abs(self.model.opt.timestep - self.integration_dt) > 1.0e-12:
            raise RuntimeError(
                f"Scene dt {self.model.opt.timestep} != MuJoCo contract "
                f"{self.integration_dt}"
            )
        if abs(
            self.integration_dt * self.integration_substeps
            - self.physics_dt
        ) > 1.0e-12:
            raise RuntimeError(
                "MuJoCo contact substeps do not equal one RS01 motor step"
            )
        if abs(
            self.decimation * self.physics_dt - self.policy_dt
        ) > 1.0e-12:
            raise RuntimeError("Policy and physics timing contract is inconsistent")

        self.obs_cfg = observations
        self.use_estimated_observations = (
            observations.get("base_linear_velocity_source")
            == "rs01_leg_odometry"
            and observations.get("straight_path_state_source")
            == "rs01_leg_odometry_integral"
        )
        self.leg_odometry = None
        self.walk_guard_odometry = None
        self.estimated_path = None
        self.last_estimated_linear_velocity = np.zeros(
            3, dtype=np.float64
        )
        self.last_estimated_path_state = (0.0, 0.0)
        self.strict_diagonal_odometry = False
        self.actor_uses_strict_diagonal_odometry = False
        self.path_update_min_confidence = 0.0
        if self.use_estimated_observations:
            odometry = observations.get("rs01_leg_odometry") or {}
            self.actor_uses_strict_diagonal_odometry = bool(
                odometry.get("strict_diagonal_pairs", False)
            )
            self.strict_diagonal_odometry = bool(
                self.actor_uses_strict_diagonal_odometry
                or self.cfg.get("task") == "rs01_go2_estimator_parity"
            )
            self.path_update_min_confidence = float(
                odometry.get(
                    "path_update_min_confidence",
                    0.5 if self.strict_diagonal_odometry else 0.0,
                )
            )
            self.leg_odometry = Rs01NewMachineLegOdometry(
                nominal_base_height=float(
                    odometry["nominal_base_height_m"]
                ),
                foot_radius=float(odometry["foot_radius_m"]),
                height_margin=float(odometry["height_margin_m"]),
                vertical_speed_threshold=float(
                    odometry["vertical_speed_threshold_m_s"]
                ),
                velocity_residual_threshold=float(
                    odometry["velocity_residual_threshold_m_s"]
                ),
                filter_alpha=float(odometry["filter_alpha"]),
                no_contact_decay=float(odometry["no_contact_decay"]),
                previous_stance_score_bonus=float(
                    odometry["previous_stance_score_bonus"]
                ),
                strict_diagonal_pairs=(
                    self.actor_uses_strict_diagonal_odometry
                ),
            )
            self.walk_guard_odometry = (
                Rs01NewMachineLegOdometry(
                    nominal_base_height=float(
                        odometry["nominal_base_height_m"]
                    ),
                    foot_radius=float(odometry["foot_radius_m"]),
                    height_margin=float(odometry["height_margin_m"]),
                    vertical_speed_threshold=float(
                        odometry["vertical_speed_threshold_m_s"]
                    ),
                    velocity_residual_threshold=float(
                        odometry["velocity_residual_threshold_m_s"]
                    ),
                    filter_alpha=float(odometry["filter_alpha"]),
                    no_contact_decay=float(odometry["no_contact_decay"]),
                    previous_stance_score_bonus=float(
                        odometry["previous_stance_score_bonus"]
                    ),
                    strict_diagonal_pairs=True,
                )
                if (
                    self.strict_diagonal_odometry
                    and not self.actor_uses_strict_diagonal_odometry
                )
                else self.leg_odometry
            )
            self.estimated_path = Rs01StraightPathEstimator(
                max_update_gap_s=max(0.10, 2.0 * self.policy_dt)
            )
        self.gait_cfg = self.cfg["gait"]
        self.qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                ]
                for name in self.names
            ],
            dtype=int,
        )
        self.qvel_indices = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                ]
                for name in self.names
            ],
            dtype=int,
        )
        self.actuator_indices = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{name}_motor",
                )
                for name in self.names
            ],
            dtype=int,
        )
        self.trunk_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Trunk",
        )
        self.ground_geom = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground",
        )
        self.foot_geoms = self._find_foot_geoms()
        self.reset()

    def _find_foot_geoms(self):
        result = {}
        for leg in LEGS:
            body = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{leg}_calf_joint",
            )
            candidates = [
                geom
                for geom in range(self.model.ngeom)
                if self.model.geom_bodyid[geom] == body
                and self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                and abs(float(self.model.geom_size[geom, 0]) - 0.016)
                < 1.0e-5
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one 16 mm {leg} foot sphere, got {candidates}"
                )
            result[leg] = candidates[0]
        return result

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        position = self.cfg["initial_state"]["base_position_m"]
        xyzw = self.cfg["initial_state"]["base_quaternion_xyzw"]
        self.data.qpos[:7] = [
            *position,
            xyzw[3],
            xyzw[0],
            xyzw[1],
            xyzw[2],
        ]
        self.data.qpos[self.qpos_indices] = self.default
        self.action = np.zeros(len(self.names), dtype=np.float64)
        self.desired_target = self.default.copy()
        self.limited_target = self.default.copy()
        self.target_rate = np.zeros(len(self.names), dtype=np.float64)
        self.response_target = self.default.copy()
        self.last_delayed_target = self.default.copy()
        self.target_history = np.repeat(
            self.default[None, :],
            self.max_delay_steps + 1,
            axis=0,
        )
        self.policy_steps = 0
        self.heading_target = 0.0
        # Match Isaac Gym's straight_path_origin_xy snapshot. This is a
        # navigation state, not a hidden stabilizing controller.
        self.path_origin_xy = self.data.qpos[:2].copy()
        self.last_raw_torque = np.zeros(len(self.names), dtype=np.float64)
        self.last_motor_torque = np.zeros(len(self.names), dtype=np.float64)
        self.last_applied_torque = np.zeros(len(self.names), dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        if self.use_estimated_observations:
            self.leg_odometry.reset()
            if self.walk_guard_odometry is not self.leg_odometry:
                self.walk_guard_odometry.reset()
            self.estimated_path.reset(now=0.0, yaw=0.0)
            self.last_estimated_linear_velocity.fill(0.0)
            self.last_estimated_path_state = (0.0, 0.0)

    @property
    def phase(self):
        return (
            self.policy_steps
            * self.policy_dt
            / float(self.gait_cfg["period_s"])
        ) % 1.0

    def base_velocity_world(self):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.trunk_body,
            velocity,
            0,
        )
        # World-frame object velocity is angular, then linear.
        return velocity[3:].copy(), velocity[:3].copy()

    def base_velocity_body(self):
        # World-frame mj_objectVelocity is ordered angular, then linear.
        # Convert explicitly with the floating-base quaternion, matching
        # Isaac Gym's quat_rotate_inverse. MuJoCo's flg_local=1 convention for
        # this imported free body has a fixed axis permutation and must not be
        # used as the policy body frame.
        linear_world, angular_world = self.base_velocity_world()
        rotation = quaternion_rotation_matrix(self.data.qpos[3:7])
        return (
            rotation.T @ linear_world,
            rotation.T @ angular_world,
        )

    def straight_path_state(self):
        """Match the 54-D task's world/path-frame lateral state."""
        lateral_axis = np.asarray(
            [
                -math.sin(self.heading_target),
                math.cos(self.heading_target),
            ],
            dtype=np.float64,
        )
        displacement = self.data.qpos[:2] - self.path_origin_xy
        linear_world, _ = self.base_velocity_world()
        return (
            float(np.dot(displacement, lateral_axis)),
            float(np.dot(linear_world[:2], lateral_axis)),
        )

    def observation(self):
        quaternion = self.data.qpos[3:7]
        rotation = quaternion_rotation_matrix(quaternion)
        linear, angular = self.base_velocity_body()
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        _, _, yaw = roll_pitch_yaw(quaternion)
        heading_error = np.clip(
            wrapped_angle(self.heading_target - yaw)
            * float(self.obs_cfg["straight_heading_error_scale"]),
            -1.0,
            1.0,
        )
        angle = 2.0 * math.pi * self.phase
        path_observation = None
        if self.use_estimated_observations:
            odometry = self.leg_odometry.estimate(
                self.data.qpos[self.qpos_indices],
                self.data.qvel[self.qvel_indices],
                angular,
            )
            guard_odometry = (
                self.walk_guard_odometry.estimate(
                    self.data.qpos[self.qpos_indices],
                    self.data.qvel[self.qvel_indices],
                    angular,
                )
                if self.walk_guard_odometry is not self.leg_odometry
                else odometry
            )
            linear = np.asarray(
                odometry["base_linear_velocity"], dtype=np.float64
            )
            path_observation = self.estimated_path.update(
                now=self.policy_steps * self.policy_dt,
                yaw=yaw,
                base_linear_velocity_body=linear,
                update_enabled=bool(
                    float(guard_odometry["confidence"])
                    >= self.path_update_min_confidence
                    and (
                        not self.strict_diagonal_odometry
                        or guard_odometry.get(
                            "legal_diagonal_support", False
                        )
                    )
                ),
            )
            self.last_estimated_linear_velocity = linear.copy()
            self.last_estimated_path_state = tuple(path_observation)
        observation_parts = [
            linear * float(self.obs_cfg["lin_vel_scale"]),
            angular * float(self.obs_cfg["ang_vel_scale"]),
            gravity,
            self.command
            * np.asarray(self.obs_cfg["command_scale"], dtype=np.float64),
            (self.data.qpos[self.qpos_indices] - self.default)
            * float(self.obs_cfg["dof_pos_scale"]),
            self.data.qvel[self.qvel_indices]
            * float(self.obs_cfg["dof_vel_scale"]),
            self.action,
            [math.sin(angle), math.cos(angle)],
        ]
        if self.obs_cfg.get(
            "heading_representation",
            "scaled_wrapped_scalar",
        ) == "sin_cos":
            raw_heading_error = wrapped_angle(self.heading_target - yaw)
            observation_parts.append(
                [
                    math.sin(raw_heading_error),
                    math.cos(raw_heading_error),
                ]
            )
        else:
            observation_parts.append([heading_error])
        if self.obs_cfg.get("straight_path_state_enabled", False):
            if path_observation is None:
                lateral_error, lateral_velocity = (
                    self.straight_path_state()
                )
            else:
                lateral_error, lateral_velocity = path_observation
            observation_parts.append(
                [
                    np.clip(
                        lateral_error
                        * float(
                            self.obs_cfg[
                                "straight_path_lateral_position_scale"
                            ]
                        ),
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        lateral_velocity
                        * float(
                            self.obs_cfg[
                                "straight_path_lateral_velocity_scale"
                            ]
                        ),
                        -1.0,
                        1.0,
                    ),
                ]
            )
        observation = np.concatenate(observation_parts).astype(np.float32)
        expected = int(self.cfg["dimensions"]["observations"])
        if observation.size != expected:
            raise RuntimeError(
                f"Observation size {observation.size}, expected {expected}"
            )
        return np.clip(
            observation,
            -float(self.obs_cfg["clip"]),
            float(self.obs_cfg["clip"]),
        )

    def update_policy(self):
        observation = self.observation()
        action = self.session.run(
            ["actions"],
            {"observations": observation[None, :]},
        )[0][0]
        self.action = np.clip(
            np.asarray(action, dtype=np.float64),
            -self.action_clip,
            self.action_clip,
        )
        desired = self.default + self.action_scale * self.action
        self.desired_target = desired.copy()
        self.limited_target, self.target_rate = limited_target_step(
            desired,
            self.limited_target,
            self.target_rate,
            self.rate_limit,
            self.acceleration_limit,
            self.policy_dt,
        )

    def physics_step(self):
        self.target_history = np.roll(self.target_history, 1, axis=0)
        self.target_history[0] = self.limited_target
        delayed = np.asarray(
            [
                self.target_history[self.delay_steps[index], index]
                for index in range(len(self.names))
            ]
        )
        self.last_delayed_target = delayed.copy()
        equilibrium = self.default + self.response_gain * (
            delayed - self.default
        )
        self.response_target += self.response_alpha * (
            equilibrium - self.response_target
        )

        position = self.data.qpos[self.qpos_indices]
        velocity = self.data.qvel[self.qvel_indices]
        raw, motor, applied = compute_rs01_torques(
            self.response_target,
            position,
            velocity,
            self.kp,
            self.kd,
            self.peak_limit,
            self.friction,
            self.friction_smoothing,
        )
        self.data.ctrl[self.actuator_indices] = applied
        # PhysX holds one motor command over two internal 2.5 ms contact
        # substeps. Mirror that timing while keeping the identified delay and
        # first-order response update at its measured 5 ms rate.
        for _ in range(self.integration_substeps):
            mujoco.mj_step(self.model, self.data)
        self.last_raw_torque = raw
        self.last_motor_torque = motor
        self.last_applied_torque = applied

    def control_step(self):
        self.update_policy()
        for _ in range(self.decimation):
            self.physics_step()
        self.policy_steps += 1

    def contact_diagnostics(self):
        force_by_geom = {geom: 0.0 for geom in self.foot_geoms.values()}
        contact_force = np.zeros(6, dtype=np.float64)
        illegal_contact_count = 0
        illegal_geom_names = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.ground_geom not in (contact.geom1, contact.geom2):
                continue
            foot_geom = (
                contact.geom2
                if contact.geom1 == self.ground_geom
                else contact.geom1
            )
            if foot_geom not in force_by_geom:
                illegal_contact_count += 1
                illegal_geom_names.add(
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        foot_geom,
                    ) or f"geom_{foot_geom}"
                )
                continue
            mujoco.mj_contactForce(
                self.model,
                self.data,
                index,
                contact_force,
            )
            force_by_geom[foot_geom] += abs(float(contact_force[0]))
        return (
            np.asarray(
                [force_by_geom[self.foot_geoms[leg]] for leg in LEGS]
            ),
            illegal_contact_count,
            sorted(illegal_geom_names),
        )

    def contact_forces(self):
        return self.contact_diagnostics()[0]

    def foot_slip_speeds(self):
        result = []
        velocity = np.zeros(6, dtype=np.float64)
        for leg in LEGS:
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_GEOM,
                self.foot_geoms[leg],
                velocity,
                0,
            )
            # World-frame object velocity is angular, then linear.
            result.append(float(np.linalg.norm(velocity[3:5])))
        return np.asarray(result)

    def desired_contact(self):
        stance = float(self.gait_cfg["stance_ratio"])
        offsets = self.gait_cfg["phase_offsets"]
        return np.asarray(
            [((self.phase + float(offsets[leg])) % 1.0) < stance for leg in LEGS]
        )

    def foot_heights(self):
        return np.asarray(
            [self.data.geom_xpos[self.foot_geoms[leg], 2] for leg in LEGS]
        )


def percentile(values, quantile):
    return float(np.quantile(np.asarray(values), quantile))


def run(args):
    sim = Rs01Go2Sim(
        args.scene.resolve(),
        args.policy.resolve(),
        args.command,
    )
    csv_handle = None
    writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = args.csv.open("w", newline="")
        headers = [
            "time_s",
            "command_vx_m_s",
            "base_x_m",
            "base_y_m",
            "base_z_m",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "base_vx_body_m_s",
            "base_vy_body_m_s",
            "path_lateral_error_m",
            "path_lateral_velocity_m_s",
            "yaw_rate_body_rad_s",
            "unwrapped_yaw_rad",
            "gait_phase",
            "exact_desired_contact",
            "flight",
            "all_four_contact",
            "illegal_ground_contact_count",
        ]
        for group in (
            "contact",
            "desired_contact",
            "foot_force_n",
            "foot_center_z_m",
            "foot_slip_speed_m_s",
            "action",
            "desired_position_target_rad",
            "limited_position_target_rad",
            "delayed_position_target_rad",
            "response_position_target_rad",
            "raw_pd_torque_nm",
            "motor_electromagnetic_torque_nm",
            "applied_joint_torque_nm",
        ):
            labels = LEGS if group not in (
                "action",
                "desired_position_target_rad",
                "limited_position_target_rad",
                "delayed_position_target_rad",
                "response_position_target_rad",
                "raw_pd_torque_nm",
                "motor_electromagnetic_torque_nm",
                "applied_joint_torque_nm",
            ) else sim.names
            headers.extend(f"{group}_{label}" for label in labels)
        writer = csv.writer(csv_handle)
        writer.writerow(headers)

    start_xy = sim.data.qpos[:2].copy()
    path_y = []
    path_velocities = []
    velocities = []
    yaw_rates = []
    rolls = []
    pitches = []
    raw_torques = []
    motor_torques = []
    applied_torques = []
    contacts = []
    desired_contacts = []
    foot_heights = []
    foot_slip_speeds = []
    illegal_contact_counts = []
    illegal_geom_names = set()
    wrapped_yaws = []
    min_height = float("inf")
    fall_time = None
    fall_reason = None

    viewer_context = None
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(sim.model, sim.data)
        viewer = viewer_context.__enter__()
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

    try:
        steps = int(round(args.duration / sim.policy_dt))
        for step in range(steps):
            wall_start = time.time()
            sim.control_step()
            linear, angular = sim.base_velocity_body()
            path_error, path_velocity = sim.straight_path_state()
            roll, pitch, yaw = roll_pitch_yaw(sim.data.qpos[3:7])
            force, illegal_count, illegal_names = sim.contact_diagnostics()
            contact = force >= float(sim.gait_cfg["contact_threshold_n"])
            desired = sim.desired_contact()
            exact = bool(np.array_equal(contact, desired))
            flight = bool(not contact.any())
            all_four = bool(contact.all())
            height = sim.foot_heights()
            slip_speed = sim.foot_slip_speeds()
            wrapped_yaws.append(yaw)
            unwrapped_yaw = float(np.unwrap(wrapped_yaws)[-1])

            path_y.append(path_error)
            path_velocities.append(path_velocity)
            velocities.append(linear.copy())
            yaw_rates.append(float(angular[2]))
            rolls.append(roll)
            pitches.append(pitch)
            raw_torques.append(sim.last_raw_torque.copy())
            motor_torques.append(sim.last_motor_torque.copy())
            applied_torques.append(sim.last_applied_torque.copy())
            contacts.append(contact.copy())
            desired_contacts.append(desired.copy())
            foot_heights.append(height.copy())
            foot_slip_speeds.append(slip_speed.copy())
            illegal_contact_counts.append(illegal_count)
            illegal_geom_names.update(illegal_names)
            min_height = min(min_height, float(sim.data.qpos[2]))
            current_fall_reason = None
            if float(sim.data.qpos[2]) < 0.18:
                current_fall_reason = "base_height_below_0.18m"
            elif abs(roll) > 0.8:
                current_fall_reason = "absolute_roll_above_0.8rad"
            elif abs(pitch) > 0.8:
                current_fall_reason = "absolute_pitch_above_0.8rad"
            if current_fall_reason is not None and fall_time is None:
                fall_time = (step + 1) * sim.policy_dt
                fall_reason = current_fall_reason

            if writer is not None:
                writer.writerow(
                    [
                        (step + 1) * sim.policy_dt,
                        sim.command[0],
                        *sim.data.qpos[:3],
                        roll,
                        pitch,
                        yaw,
                        linear[0],
                        linear[1],
                        path_error,
                        path_velocity,
                        angular[2],
                        unwrapped_yaw,
                        sim.phase,
                        int(exact),
                        int(flight),
                        int(all_four),
                        illegal_count,
                        *contact.astype(int),
                        *desired.astype(int),
                        *force,
                        *height,
                        *slip_speed,
                        *sim.action,
                        *sim.desired_target,
                        *sim.limited_target,
                        *sim.last_delayed_target,
                        *sim.response_target,
                        *sim.last_raw_torque,
                        *sim.last_motor_torque,
                        *sim.last_applied_torque,
                    ]
                )

            if viewer is not None:
                viewer.cam.lookat[:] = sim.data.qpos[:3]
                viewer.sync()
                remaining = sim.policy_dt - (time.time() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    break
            if fall_time is not None and not args.continue_after_fall:
                break
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

    velocity = np.asarray(velocities)
    raw = np.abs(np.asarray(raw_torques))
    motor = np.abs(np.asarray(motor_torques))
    applied = np.abs(np.asarray(applied_torques))
    contact = np.asarray(contacts)
    desired = np.asarray(desired_contacts)
    height = np.asarray(foot_heights)
    slip_speed = np.asarray(foot_slip_speeds)
    illegal_count = np.asarray(illegal_contact_counts)
    unwrapped_yaw = np.unwrap(np.asarray(wrapped_yaws))
    exact = np.all(contact == desired, axis=1)
    diagonal_a = (
        contact[:, LEGS.index("FL")]
        & contact[:, LEGS.index("RR")]
        & ~contact[:, LEGS.index("FR")]
        & ~contact[:, LEGS.index("RL")]
    )
    diagonal_b = (
        contact[:, LEGS.index("FR")]
        & contact[:, LEGS.index("RL")]
        & ~contact[:, LEGS.index("FL")]
        & ~contact[:, LEGS.index("RR")]
    )
    displacement = sim.data.qpos[:2] - start_xy
    _, _, final_yaw = roll_pitch_yaw(sim.data.qpos[3:7])
    summary = {
        "requested_duration_s": float(args.duration),
        "evaluated_duration_s": len(velocity) * sim.policy_dt,
        "command_vx_m_s": float(sim.command[0]),
        "motor_step_s": sim.physics_dt,
        "integration_timestep_s": sim.integration_dt,
        "integration_substeps_per_motor_step": sim.integration_substeps,
        "mean_vx_body_m_s": float(velocity[:, 0].mean()),
        "speed_error_m_s": float(velocity[:, 0].mean() - sim.command[0]),
        "forward_displacement_m": float(displacement[0]),
        "lateral_path_rms_m": float(np.sqrt(np.mean(np.square(path_y)))),
        "lateral_path_velocity_rms_m_s": float(
            np.sqrt(np.mean(np.square(path_velocities)))
        ),
        "final_lateral_displacement_m": float(displacement[1]),
        "final_yaw_rad": float(final_yaw),
        "final_unwrapped_yaw_rad": float(unwrapped_yaw[-1]),
        "unwrapped_yaw_drift_rad": float(
            unwrapped_yaw[-1] - unwrapped_yaw[0]
        ),
        "yaw_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(yaw_rates)))),
        "roll_rms_rad": float(np.sqrt(np.mean(np.square(rolls)))),
        "pitch_rms_rad": float(np.sqrt(np.mean(np.square(pitches)))),
        "final_base_height_m": float(sim.data.qpos[2]),
        "minimum_base_height_m": float(min_height),
        "fall_flag": fall_time is not None,
        "fall_time_s": fall_time,
        "fall_reason": fall_reason,
        "exact_desired_contact_ratio": float(exact.mean()),
        "diagonal_a_only_ratio": float(diagonal_a.mean()),
        "diagonal_b_only_ratio": float(diagonal_b.mean()),
        "flight_ratio": float((~contact.any(axis=1)).mean()),
        "all_four_contact_ratio": float(contact.all(axis=1).mean()),
        "illegal_ground_contact_frame_ratio": float(
            (illegal_count > 0).mean()
        ),
        "illegal_ground_contact_count": int(illegal_count.sum()),
        "illegal_ground_contact_geoms": sorted(illegal_geom_names),
        "contact_duty_by_leg": {
            leg: float(contact[:, index].mean())
            for index, leg in enumerate(LEGS)
        },
        "foot_center_height_p95_m": {
            leg: percentile(height[:, index], 0.95)
            for index, leg in enumerate(LEGS)
        },
        "foot_contact_slip_speed_p95_m_s": {
            leg: (
                percentile(
                    slip_speed[contact[:, index], index],
                    0.95,
                )
                if np.any(contact[:, index])
                else 0.0
            )
            for index, leg in enumerate(LEGS)
        },
        "raw_pd_torque_p95_nm": percentile(raw, 0.95),
        "raw_pd_torque_max_nm": float(raw.max()),
        "raw_over_17_ratio": float((raw > 17.0).mean()),
        "motor_torque_p95_nm": percentile(motor, 0.95),
        "motor_over_6_ratio": float((motor > 6.0).mean()),
        "motor_over_12_ratio": float((motor > 12.0).mean()),
        "motor_over_15_ratio": float((motor > 15.0).mean()),
        "peak_saturation_ratio": float(
            (motor >= sim.peak_limit - 1.0e-6).mean()
        ),
        "applied_joint_torque_p95_nm": percentile(applied, 0.95),
        "raw_pd_torque_p95_by_joint_nm": {
            name: percentile(raw[:, index], 0.95)
            for index, name in enumerate(sim.names)
        },
    }
    print(json.dumps(summary, indent=2))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Scene generated from the same policy by prepare_model.py.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="ONNX policy carrying the exact RS01 URDF/control contract.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        default=[0.23, 0.0, 0.0],
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--continue-after-fall",
        action="store_true",
        help="Keep simulating after the first fall; default metrics stop at it.",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
