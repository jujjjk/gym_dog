"""Vectorized RS01 leg odometry matching the real-machine estimator.

This module deliberately contains no Isaac Gym dependency.  The training
environment uses it to build actor observations from joint/IMU-like signals,
while simulator root state remains available only to rewards and diagnostics.
"""

from __future__ import annotations

import torch


RS01_LEG_ORDER = ("FL", "FR", "RL", "RR")
RS01_ODOMETRY_DEFAULTS = {
    "nominal_base_height_m": 0.307,
    "foot_radius_m": 0.016,
    "height_margin_m": 0.030,
    "vertical_speed_threshold_m_s": 0.25,
    "velocity_residual_threshold_m_s": 0.35,
    "filter_alpha": 0.35,
    "no_contact_decay": 0.90,
    "previous_stance_score_bonus": 0.08,
}


class Rs01TorchLegOdometry:
    """Batched Torch form of ``Rs01NewMachineLegOdometry``.

    Joint order is FL, FR, RL, RR with hip, thigh, calf for every leg.
    Inputs and outputs are expressed in the robot body frame.
    """

    ORIGINS = (
        (
            (0.216, 0.060, 0.0),
            (0.0, 0.08725, 0.0),
            (-0.155697793236241, 0.0, -0.0903227390049967),
            (0.201984878571, 0.0, 0.00837310150402),
        ),
        (
            (0.216, -0.060, 0.0),
            (0.0, -0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984878568, 0.0, 0.00837310156582),
        ),
        (
            (-0.216, 0.060, 0.0),
            (0.0, 0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984876549, 0.0, 0.00837315026116),
        ),
        (
            (-0.216, -0.060, 0.0),
            (0.0, -0.08725, 0.0),
            (-0.1557, 0.0, -0.090323),
            (0.201984876547, 0.0, 0.0083731503174),
        ),
    )

    def __init__(
        self,
        num_envs,
        device,
        nominal_base_height=RS01_ODOMETRY_DEFAULTS[
            "nominal_base_height_m"
        ],
        foot_radius=RS01_ODOMETRY_DEFAULTS["foot_radius_m"],
        height_margin=RS01_ODOMETRY_DEFAULTS["height_margin_m"],
        vertical_speed_threshold=RS01_ODOMETRY_DEFAULTS[
            "vertical_speed_threshold_m_s"
        ],
        velocity_residual_threshold=RS01_ODOMETRY_DEFAULTS[
            "velocity_residual_threshold_m_s"
        ],
        filter_alpha=RS01_ODOMETRY_DEFAULTS["filter_alpha"],
        no_contact_decay=RS01_ODOMETRY_DEFAULTS["no_contact_decay"],
        previous_stance_score_bonus=RS01_ODOMETRY_DEFAULTS[
            "previous_stance_score_bonus"
        ],
        strict_diagonal_pairs=False,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = torch.float
        self.nominal_base_height = float(nominal_base_height)
        self.foot_radius = float(foot_radius)
        self.height_margin = float(height_margin)
        self.vertical_speed_threshold = float(
            vertical_speed_threshold
        )
        self.velocity_residual_threshold = float(
            velocity_residual_threshold
        )
        self.filter_alpha = float(filter_alpha)
        self.no_contact_decay = float(no_contact_decay)
        self.previous_stance_score_bonus = float(
            previous_stance_score_bonus
        )
        self.strict_diagonal_pairs = bool(strict_diagonal_pairs)
        self.origins = torch.tensor(
            self.ORIGINS,
            device=self.device,
            dtype=self.dtype,
        )
        self.filtered = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=self.dtype
        )
        self.last_stance = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids=None):
        """Clear all persistent odometry state for selected environments."""
        if env_ids is None:
            self.filtered.zero_()
            self.last_stance.zero_()
            return
        self.filtered[env_ids] = 0.0
        self.last_stance[env_ids] = False

    @staticmethod
    def _rotation_x(angle):
        result = torch.zeros(
            angle.shape[0],
            3,
            3,
            device=angle.device,
            dtype=angle.dtype,
        )
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        result[:, 0, 0] = 1.0
        result[:, 1, 1] = cosine
        result[:, 1, 2] = -sine
        result[:, 2, 1] = sine
        result[:, 2, 2] = cosine
        return result

    @staticmethod
    def _rotation_y(angle):
        result = torch.zeros(
            angle.shape[0],
            3,
            3,
            device=angle.device,
            dtype=angle.dtype,
        )
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        result[:, 0, 0] = cosine
        result[:, 0, 2] = sine
        result[:, 1, 1] = 1.0
        result[:, 2, 0] = -sine
        result[:, 2, 2] = cosine
        return result

    def foot_position_and_jacobian(self, q_policy):
        """Return batched body-frame foot positions and Jacobians."""
        q_policy = torch.as_tensor(
            q_policy, device=self.device, dtype=self.dtype
        ).reshape(self.num_envs, 4, 3)
        batch = self.num_envs * 4
        q_flat = q_policy.reshape(batch, 3)
        origins = self.origins.unsqueeze(0).expand(
            self.num_envs, -1, -1, -1
        ).reshape(batch, 4, 3)
        rotation = torch.eye(
            3, device=self.device, dtype=self.dtype
        ).unsqueeze(0).repeat(batch, 1, 1)
        position = torch.zeros(
            batch, 3, device=self.device, dtype=self.dtype
        )
        joint_positions = []
        joint_axes = []
        local_axes = (
            torch.tensor(
                [1.0, 0.0, 0.0],
                device=self.device,
                dtype=self.dtype,
            ),
            torch.tensor(
                [0.0, 1.0, 0.0],
                device=self.device,
                dtype=self.dtype,
            ),
            torch.tensor(
                [0.0, 1.0, 0.0],
                device=self.device,
                dtype=self.dtype,
            ),
        )
        for joint in range(3):
            position = position + torch.bmm(
                rotation, origins[:, joint].unsqueeze(2)
            ).squeeze(2)
            joint_positions.append(position.clone())
            axis = torch.matmul(rotation, local_axes[joint])
            joint_axes.append(axis)
            local_rotation = (
                self._rotation_x(q_flat[:, joint])
                if joint == 0
                else self._rotation_y(q_flat[:, joint])
            )
            rotation = torch.bmm(rotation, local_rotation)
        foot = position + torch.bmm(
            rotation, origins[:, 3].unsqueeze(2)
        ).squeeze(2)
        jacobian = torch.stack(
            [
                torch.linalg.cross(
                    joint_axes[joint],
                    foot - joint_positions[joint],
                    dim=1,
                )
                for joint in range(3)
            ],
            dim=2,
        )
        return (
            foot.reshape(self.num_envs, 4, 3),
            jacobian.reshape(self.num_envs, 4, 3, 3),
        )

    @staticmethod
    def _gather_legs(values, indices):
        tail = values.shape[2:]
        gather_shape = (*indices.shape, *tail)
        index = indices.reshape(*indices.shape, *([1] * len(tail)))
        index = index.expand(gather_shape)
        return torch.gather(values, 1, index)

    def estimate(self, q_policy, dq_policy, omega_body):
        """Estimate body velocity and expose stance-selection diagnostics."""
        q_policy = torch.as_tensor(
            q_policy, device=self.device, dtype=self.dtype
        ).reshape(self.num_envs, 12)
        dq_policy = torch.as_tensor(
            dq_policy, device=self.device, dtype=self.dtype
        ).reshape(self.num_envs, 4, 3)
        omega_body = torch.as_tensor(
            omega_body, device=self.device, dtype=self.dtype
        ).reshape(self.num_envs, 3)
        foot_position, jacobian = self.foot_position_and_jacobian(
            q_policy
        )
        foot_velocity = torch.einsum(
            "nlij,nlj->nli", jacobian, dq_policy
        )
        velocity_by_foot = -(
            foot_velocity
            + torch.linalg.cross(
                omega_body.unsqueeze(1).expand(-1, 4, -1),
                foot_position,
                dim=2,
            )
        )
        base_height_proxy = -foot_position[:, :, 2] + self.foot_radius
        lowest = torch.max(base_height_proxy, dim=1).values
        height_gap = lowest.unsqueeze(1) - base_height_proxy
        vertical_speed = torch.abs(foot_velocity[:, :, 2])
        absolute_height_error = torch.abs(
            base_height_proxy - self.nominal_base_height
        )
        valid = (
            (height_gap <= self.height_margin)
            & (vertical_speed <= self.vertical_speed_threshold)
            & (absolute_height_error <= 0.10)
        )
        score = (
            height_gap / max(self.height_margin, 1.0e-6)
            + vertical_speed
            / max(self.vertical_speed_threshold, 1.0e-6)
            - self.previous_stance_score_bonus
            * self.last_stance.to(dtype=self.dtype)
        )
        if self.strict_diagonal_pairs:
            pair_indices = torch.tensor(
                [[0, 3], [1, 2]],
                device=self.device,
                dtype=torch.long,
            )
            pair_valid = (
                valid[:, pair_indices[:, 0]]
                & valid[:, pair_indices[:, 1]]
            )
            pair_velocity_a = velocity_by_foot[
                :, pair_indices[:, 0], :
            ]
            pair_velocity_b = velocity_by_foot[
                :, pair_indices[:, 1], :
            ]
            pair_residual = torch.linalg.vector_norm(
                (pair_velocity_a - pair_velocity_b)[:, :, :2],
                dim=2,
            )
            pair_valid &= (
                pair_residual <= self.velocity_residual_threshold
            )
            pair_score = (
                score[:, pair_indices[:, 0]]
                + score[:, pair_indices[:, 1]]
                + pair_residual
                / max(self.velocity_residual_threshold, 1.0e-6)
            )
            pair_score = torch.where(
                pair_valid,
                pair_score,
                torch.full_like(pair_score, 1.0e6),
            )
            selected_pair = torch.argmin(pair_score, dim=1)
            selected_valid = pair_valid.gather(
                1, selected_pair.unsqueeze(1)
            ).squeeze(1)
            selected_indices = pair_indices[selected_pair]
            selected_velocity_a = velocity_by_foot[
                torch.arange(self.num_envs, device=self.device),
                selected_indices[:, 0],
            ]
            selected_velocity_b = velocity_by_foot[
                torch.arange(self.num_envs, device=self.device),
                selected_indices[:, 1],
            ]
            raw = 0.5 * (
                selected_velocity_a + selected_velocity_b
            )
            raw[:, :2] = torch.clamp(raw[:, :2], -1.0, 1.0)
            raw[:, 2] = 0.0
            selected_residual = pair_residual.gather(
                1, selected_pair.unsqueeze(1)
            ).squeeze(1)
            confidence = torch.where(
                selected_valid,
                torch.clamp(
                    1.0
                    - selected_residual
                    / max(self.velocity_residual_threshold, 1.0e-6),
                    min=0.0,
                    max=1.0,
                ),
                torch.zeros_like(selected_residual),
            )
            alpha = self.filter_alpha * confidence
            updated = (
                (1.0 - alpha.unsqueeze(1)) * self.filtered
                + alpha.unsqueeze(1) * raw
            )
            decayed = self.filtered * self.no_contact_decay
            self.filtered = torch.where(
                selected_valid.unsqueeze(1), updated, decayed
            )
            self.filtered[:, 2] = 0.0
            stance = torch.zeros(
                self.num_envs,
                4,
                device=self.device,
                dtype=torch.bool,
            )
            rows = torch.arange(self.num_envs, device=self.device)
            stance[rows, selected_indices[:, 0]] = selected_valid
            stance[rows, selected_indices[:, 1]] = selected_valid
            self.last_stance = stance
            return {
                "base_linear_velocity": self.filtered.clone(),
                "raw_base_linear_velocity": raw,
                "confidence": confidence,
                "stance_mask": stance.clone(),
                "foot_position": foot_position,
                "foot_velocity": foot_velocity,
                "velocity_by_foot": velocity_by_foot,
                "base_height_proxy": base_height_proxy,
                "velocity_residual": pair_residual,
                "pair_residual_m_s": selected_residual,
                "selected_pair_index": torch.where(
                    selected_valid,
                    selected_pair,
                    torch.full_like(selected_pair, -1),
                ),
                "legal_diagonal_support": selected_valid,
            }
        invalid_score = torch.full_like(score, 1.0e6)
        ranked_score, ranked_index = torch.topk(
            torch.where(valid, score, invalid_score),
            k=3,
            dim=1,
            largest=False,
            sorted=True,
        )
        ranked_valid = ranked_score < 1.0e5
        ranked_velocity = self._gather_legs(
            velocity_by_foot, ranked_index
        )
        candidate_count = ranked_valid.sum(dim=1)
        center_one = ranked_velocity[:, 0]
        center_two = 0.5 * (
            ranked_velocity[:, 0] + ranked_velocity[:, 1]
        )
        center_three = torch.median(
            ranked_velocity, dim=1
        ).values
        center = torch.where(
            (candidate_count >= 3).unsqueeze(1),
            center_three,
            torch.where(
                (candidate_count == 2).unsqueeze(1),
                center_two,
                center_one,
            ),
        )
        residual = torch.linalg.vector_norm(
            (ranked_velocity - center.unsqueeze(1))[:, :, :2],
            dim=2,
        )
        accepted = (
            ranked_valid
            & (residual <= self.velocity_residual_threshold)
        )
        none_accepted = (~accepted.any(dim=1)) & (candidate_count > 0)
        fallback_residual = torch.where(
            ranked_valid,
            residual,
            torch.full_like(residual, 1.0e6),
        )
        fallback_slot = torch.argmin(fallback_residual, dim=1)
        accepted[
            torch.arange(self.num_envs, device=self.device)[none_accepted],
            fallback_slot[none_accepted],
        ] = True
        accepted_rank = torch.cumsum(
            accepted.to(dtype=torch.int64), dim=1
        )
        selected_ranked = accepted & (accepted_rank <= 2)
        selected_count = selected_ranked.sum(dim=1)
        selected_velocity = (
            ranked_velocity
            * selected_ranked.unsqueeze(2).to(dtype=self.dtype)
        ).sum(dim=1)
        raw = selected_velocity / torch.clamp(
            selected_count.to(dtype=self.dtype).unsqueeze(1),
            min=1.0,
        )
        raw[:, 2] = 0.0
        confidence = selected_count.to(dtype=self.dtype) / 2.0
        alpha = self.filter_alpha * confidence
        updated = (
            (1.0 - alpha.unsqueeze(1)) * self.filtered
            + alpha.unsqueeze(1) * raw
        )
        decayed = self.filtered * self.no_contact_decay
        self.filtered = torch.where(
            (selected_count > 0).unsqueeze(1),
            updated,
            decayed,
        )
        self.filtered[:, 2] = 0.0
        stance = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )
        stance.scatter_(1, ranked_index, selected_ranked)
        self.last_stance = stance
        return {
            "base_linear_velocity": self.filtered.clone(),
            "raw_base_linear_velocity": raw,
            "confidence": confidence,
            "stance_mask": stance.clone(),
            "foot_position": foot_position,
            "foot_velocity": foot_velocity,
            "velocity_by_foot": velocity_by_foot,
            "base_height_proxy": base_height_proxy,
            "velocity_residual": residual,
        }


class Rs01TorchStraightPathEstimator:
    """Policy-rate path integrator matching the model_1850 deployment."""

    def __init__(self, num_envs, device, policy_dt):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.policy_dt = float(policy_dt)
        self.heading_target = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.lateral_displacement = torch.zeros_like(
            self.heading_target
        )
        self.lateral_velocity = torch.zeros_like(self.heading_target)

    def reset(self, env_ids, heading_target):
        heading_target = torch.as_tensor(
            heading_target, device=self.device, dtype=torch.float
        ).reshape(-1)
        self.heading_target[env_ids] = heading_target
        self.lateral_displacement[env_ids] = 0.0
        self.lateral_velocity[env_ids] = 0.0

    def update(
        self,
        yaw,
        base_linear_velocity_body,
        update_mask=None,
    ):
        yaw = torch.as_tensor(
            yaw, device=self.device, dtype=torch.float
        ).reshape(self.num_envs)
        velocity = torch.as_tensor(
            base_linear_velocity_body,
            device=self.device,
            dtype=torch.float,
        ).reshape(self.num_envs, 3)
        relative_yaw = yaw - self.heading_target
        lateral_velocity = (
            torch.sin(relative_yaw) * velocity[:, 0]
            + torch.cos(relative_yaw) * velocity[:, 1]
        )
        if update_mask is None:
            update_mask = torch.ones(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
        else:
            update_mask = torch.as_tensor(
                update_mask,
                device=self.device,
                dtype=torch.bool,
            ).reshape(self.num_envs)
        integrated = (
            self.lateral_displacement
            + 0.5
            * (self.lateral_velocity + lateral_velocity)
            * self.policy_dt
        )
        self.lateral_displacement.copy_(
            torch.where(
                update_mask,
                integrated,
                self.lateral_displacement,
            )
        )
        self.lateral_velocity.copy_(
            torch.where(
                update_mask,
                lateral_velocity,
                torch.zeros_like(lateral_velocity),
            )
        )
        return (
            self.lateral_displacement.clone(),
            self.lateral_velocity.clone(),
        )
