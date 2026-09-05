"""Hardware-constrained omnidirectional diagonal gait for RS01."""

import torch

from .rs01_go2_straight_env import Rs01Go2StraightRobot


class Rs01Go2OmniDiagonalRobot(Rs01Go2StraightRobot):
    """Track body velocity commands without relaxing the RS01 gait contract.

    The inherited actuator path remains unchanged.  This class only replaces
    straight-only command sampling, reference integration, and rewards.
    """

    COMMAND_STAND = 0
    COMMAND_FORWARD = 1
    COMMAND_BACKWARD = 2
    COMMAND_LATERAL = 3
    COMMAND_YAW = 4
    COMMAND_COMBINED = 5
    COMMAND_MODE_COUNT = 6

    def __init__(self, *args, **kwargs):
        self._omni_reference_ready = False
        super().__init__(*args, **kwargs)

        probabilities = torch.tensor(
            list(self.cfg.commands.mode_probabilities),
            device=self.device,
            dtype=torch.float,
        )
        if probabilities.numel() != self.COMMAND_MODE_COUNT:
            raise ValueError(
                "mode_probabilities must contain exactly "
                f"{self.COMMAND_MODE_COUNT} command modes"
            )
        if torch.any(probabilities < 0.0) or not torch.isclose(
            probabilities.sum(),
            torch.tensor(1.0, device=self.device),
            atol=1.0e-6,
        ):
            raise ValueError("mode_probabilities must be nonnegative and sum to 1")
        self.command_mode_probabilities = probabilities
        self.command_mode = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self.omni_desired_position_xy = self.root_states[:, :2].clone()
        self.omni_estimated_position_xy = self.root_states[:, :2].clone()
        self.omni_desired_heading_rad = self.rpy[:, 2].clone()
        self.straight_heading_target_rad.copy_(self.omni_desired_heading_rad)
        self.straight_path_origin_xy.copy_(self.omni_desired_position_xy)
        self._omni_reference_ready = True

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_commands(env_ids)
        if self._rs01_observation_estimator_ready:
            self.rs01_path_estimator.reset(
                env_ids, self.omni_desired_heading_rad
            )

    @staticmethod
    def _uniform_sample(count, value_range, device):
        lower, upper = float(value_range[0]), float(value_range[1])
        return lower + (upper - lower) * torch.rand(count, device=device)

    def _resample_commands(self, env_ids):
        """Stratify commands so rare directions cannot disappear in sampling."""
        count = len(env_ids)
        if count == 0:
            return

        probabilities = getattr(self, "command_mode_probabilities", None)
        if probabilities is None:
            probabilities = torch.tensor(
                list(self.cfg.commands.mode_probabilities),
                device=self.device,
                dtype=torch.float,
            )
            probabilities = probabilities / probabilities.sum()
        modes = torch.multinomial(probabilities, count, replacement=True)
        self.commands[env_ids, :3] = 0.0

        def sample_into(mode, column, value_range):
            selected = env_ids[modes == mode]
            if len(selected) > 0:
                self.commands[selected, column] = self._uniform_sample(
                    len(selected), value_range, self.device
                )

        sample_into(
            self.COMMAND_FORWARD,
            0,
            self.cfg.commands.forward_velocity_range_m_s,
        )
        sample_into(
            self.COMMAND_BACKWARD,
            0,
            self.cfg.commands.backward_velocity_range_m_s,
        )
        sample_into(
            self.COMMAND_LATERAL,
            1,
            self.cfg.commands.lateral_velocity_range_m_s,
        )
        sample_into(
            self.COMMAND_YAW,
            2,
            self.cfg.commands.yaw_velocity_range_rad_s,
        )

        combined = env_ids[modes == self.COMMAND_COMBINED]
        if len(combined) > 0:
            self.commands[combined, 0] = self._uniform_sample(
                len(combined),
                self.cfg.commands.combined_forward_range_m_s,
                self.device,
            )
            self.commands[combined, 1] = self._uniform_sample(
                len(combined),
                self.cfg.commands.combined_lateral_range_m_s,
                self.device,
            )
            self.commands[combined, 2] = self._uniform_sample(
                len(combined),
                self.cfg.commands.combined_yaw_range_rad_s,
                self.device,
            )

        if hasattr(self, "command_mode"):
            self.command_mode[env_ids] = modes

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not self._omni_reference_ready:
            return
        self.omni_desired_position_xy[env_ids] = self.root_states[env_ids, :2]
        self.omni_estimated_position_xy[env_ids] = self.root_states[env_ids, :2]
        self.omni_desired_heading_rad[env_ids] = self.rpy[env_ids, 2]
        self.straight_heading_target_rad[env_ids] = (
            self.omni_desired_heading_rad[env_ids]
        )
        self.straight_path_origin_xy[env_ids] = (
            self.omni_desired_position_xy[env_ids]
        )
        if self._rs01_observation_estimator_ready:
            self.rs01_path_estimator.reset(
                env_ids, self.omni_desired_heading_rad[env_ids]
            )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self._omni_reference_ready:
            self._integrate_omni_reference()

    @staticmethod
    def _body_to_world_xy(body_velocity_xy, heading):
        cosine = torch.cos(heading)
        sine = torch.sin(heading)
        return torch.stack(
            (
                cosine * body_velocity_xy[:, 0]
                - sine * body_velocity_xy[:, 1],
                sine * body_velocity_xy[:, 0]
                + cosine * body_velocity_xy[:, 1],
            ),
            dim=1,
        )

    def _integrate_omni_reference(self):
        desired_velocity_world = self._body_to_world_xy(
            self.commands[:, :2], self.omni_desired_heading_rad
        )
        self.omni_desired_position_xy.add_(self.dt * desired_velocity_world)
        new_heading = (
            self.omni_desired_heading_rad + self.dt * self.commands[:, 2]
        )
        self.omni_desired_heading_rad.copy_(
            torch.atan2(torch.sin(new_heading), torch.cos(new_heading))
        )
        self.straight_heading_target_rad.copy_(self.omni_desired_heading_rad)

        if self._rs01_observation_estimator_ready:
            estimated_velocity_world = self._body_to_world_xy(
                self.estimated_base_lin_vel[:, :2], self.rpy[:, 2]
            )
            update = (
                self.estimated_odom_confidence
                >= float(self.cfg.rs01_odometry.path_update_min_confidence)
            ).unsqueeze(1)
            self.omni_estimated_position_xy.add_(
                self.dt * estimated_velocity_world * update
            )

    def _walking_command_gate(self):
        planar_speed = torch.linalg.vector_norm(self.commands[:, :2], dim=1)
        return (
            (planar_speed > float(self.cfg.commands.walking_speed_threshold_m_s))
            | (
                torch.abs(self.commands[:, 2])
                > float(self.cfg.commands.walking_yaw_threshold_rad_s)
            )
        ).to(dtype=torch.float)

    def _straight_heading_error(self):
        raw_error = self.omni_desired_heading_rad - self.rpy[:, 2]
        return torch.atan2(torch.sin(raw_error), torch.cos(raw_error))

    def _path_frame_lateral_velocity_error(self, body_velocity):
        velocity_world = self._body_to_world_xy(
            body_velocity[:, :2], self.rpy[:, 2]
        )
        heading = self.omni_desired_heading_rad
        lateral_axis = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1)
        actual_lateral_velocity = torch.sum(velocity_world * lateral_axis, dim=1)
        return actual_lateral_velocity - self.commands[:, 1]

    def _trajectory_lateral_state(self, position_xy, body_velocity):
        heading = self.omni_desired_heading_rad
        lateral_axis = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1)
        position_error = position_xy - self.omni_desired_position_xy
        lateral_error = torch.sum(position_error * lateral_axis, dim=1)
        lateral_velocity_error = self._path_frame_lateral_velocity_error(
            body_velocity
        )
        return lateral_error, lateral_velocity_error

    def _straight_path_state(self):
        return self._trajectory_lateral_state(
            self.root_states[:, :2], self.base_lin_vel
        )

    def _straight_path_observation_state(self):
        if self._rs01_observation_estimator_ready:
            return self._trajectory_lateral_state(
                self.omni_estimated_position_xy,
                self.estimated_base_lin_vel,
            )
        return self._straight_path_state()

    def _phase_support_error(self):
        """Enforce diagonal timing without prescribing equal foot loads."""
        return torch.mean(
            (
                self.get_foot_contact_mask() != self._desired_contact_mask()
            ).to(dtype=torch.float),
            dim=1,
        )

    def _reward_tracking_planar_velocity(self):
        error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(
            -error / float(self.cfg.rewards.planar_tracking_sigma)
        )

    def _reward_tracking_yaw_velocity(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / float(self.cfg.rewards.yaw_tracking_sigma))

    def _reward_trajectory_lateral_error(self):
        lateral_error, _ = self._straight_path_state()
        scale = float(self.cfg.rewards.trajectory_lateral_scale_m)
        return (
            torch.square(lateral_error / max(scale, 1.0e-6))
            * self._walking_command_gate()
        )

    def _reward_omni_heading_error(self):
        scale = float(self.cfg.rewards.heading_error_scale_rad)
        return (
            torch.square(self._straight_heading_error() / max(scale, 1.0e-6))
            * self._walking_command_gate()
        )

    def _reward_stance_foot_slip(self):
        contact = self.get_foot_contact_mask().to(dtype=torch.float)
        scale = float(self.cfg.rewards.stance_foot_slip_scale_m_s)
        planar_speed_squared = torch.sum(
            torch.square(self.feet_state[:, :, 7:9] / max(scale, 1.0e-6)),
            dim=2,
        )
        slip = torch.sum(planar_speed_squared * contact, dim=1) / torch.clamp(
            torch.sum(contact, dim=1), min=1.0
        )
        return slip * self._walking_command_gate()

    def _reward_stand_still(self):
        full_command = torch.cat(
            (self.commands[:, :2], self.commands[:, 2:3]), dim=1
        )
        stand = (
            torch.linalg.vector_norm(full_command, dim=1)
            < float(self.cfg.commands.stand_command_threshold)
        ).to(dtype=torch.float)
        return (
            torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
            * stand
        )
