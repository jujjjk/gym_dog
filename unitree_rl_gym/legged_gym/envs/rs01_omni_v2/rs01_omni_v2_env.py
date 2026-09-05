"""RS01 omni v2 with explicit stand/march/motion transitions."""

import torch

from legged_gym.envs.rs01_go2_straight.rs01_go2_omni_env import (
    Rs01Go2OmniDiagonalRobot,
)


class Rs01OmniV2Robot(Rs01Go2OmniDiagonalRobot):
    """Random-start policy task with a continuous diagonal gait clock."""

    COMMAND_STAND = 0
    COMMAND_MARCH = 1
    COMMAND_FORWARD = 2
    COMMAND_BACKWARD = 3
    COMMAND_LATERAL = 4
    COMMAND_YAW = 5
    COMMAND_COMBINED = 6
    COMMAND_MODE_COUNT = 7

    def __init__(self, *args, **kwargs):
        # Base construction invokes the virtual command sampler before the
        # v2-only buffers exist. Keep that initialization command neutral.
        self._v2_command_ready = False
        super().__init__(*args, **kwargs)

        moving_probabilities = torch.tensor(
            list(self.cfg.commands.moving_mode_probabilities),
            device=self.device,
            dtype=torch.float,
        )
        if moving_probabilities.numel() != 5:
            raise ValueError(
                "moving_mode_probabilities must contain forward, backward, "
                "lateral, yaw, and combined probabilities"
            )
        if torch.any(moving_probabilities < 0.0) or not torch.isclose(
            moving_probabilities.sum(),
            torch.tensor(1.0, device=self.device),
            atol=1.0e-6,
        ):
            raise ValueError(
                "moving_mode_probabilities must be nonnegative and sum to 1"
            )
        moving_to_march = float(
            self.cfg.commands.moving_to_march_probability
        )
        moving_to_stand = float(
            self.cfg.commands.moving_to_stand_probability
        )
        if (
            moving_to_march < 0.0
            or moving_to_stand < 0.0
            or moving_to_march + moving_to_stand > 1.0
        ):
            raise ValueError(
                "moving transition probabilities must be nonnegative and "
                "sum to no more than 1"
            )

        self.moving_mode_probabilities = moving_probabilities
        self.gait_enable = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._v2_command_ready = True

        # Every episode begins with in-place diagonal stepping. Four seconds
        # later it changes to a moving command, teaching a real transition.
        env_ids = torch.arange(self.num_envs, device=self.device)
        modes = torch.full_like(env_ids, self.COMMAND_MARCH)
        self._set_command_modes(env_ids, modes)
        self.compute_observations()

    @staticmethod
    def _random_sign(count, device):
        return torch.where(
            torch.rand(count, device=device) < 0.5,
            -torch.ones(count, device=device),
            torch.ones(count, device=device),
        )

    def _sample_signed(self, count, magnitude_range):
        return self._uniform_sample(
            count, magnitude_range, self.device
        ) * self._random_sign(count, self.device)

    def _sample_moving_modes(self, count):
        if count == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        sampled = torch.multinomial(
            self.moving_mode_probabilities, count, replacement=True
        )
        return sampled + self.COMMAND_FORWARD

    def _sample_transition_modes(self, env_ids):
        previous = self.command_mode[env_ids]
        modes = torch.empty_like(previous)

        was_stand = previous == self.COMMAND_STAND
        was_march = previous == self.COMMAND_MARCH
        was_moving = ~(was_stand | was_march)
        modes[was_stand] = self.COMMAND_MARCH
        modes[was_march] = self._sample_moving_modes(
            int(torch.sum(was_march).item())
        )

        moving_count = int(torch.sum(was_moving).item())
        if moving_count > 0:
            decision = torch.rand(moving_count, device=self.device)
            stand_probability = float(
                self.cfg.commands.moving_to_stand_probability
            )
            march_probability = float(
                self.cfg.commands.moving_to_march_probability
            )
            moving_modes = self._sample_moving_modes(moving_count)
            moving_modes = torch.where(
                decision < stand_probability,
                torch.full_like(moving_modes, self.COMMAND_STAND),
                moving_modes,
            )
            moving_modes = torch.where(
                (decision >= stand_probability)
                & (decision < stand_probability + march_probability),
                torch.full_like(moving_modes, self.COMMAND_MARCH),
                moving_modes,
            )
            modes[was_moving] = moving_modes
        return modes

    def _set_command_modes(self, env_ids, modes):
        self.commands[env_ids, :3] = 0.0
        self.command_mode[env_ids] = modes
        self.gait_enable[env_ids] = (
            modes != self.COMMAND_STAND
        ).to(dtype=torch.float)

        def selected(mode):
            return env_ids[modes == mode]

        forward = selected(self.COMMAND_FORWARD)
        if len(forward) > 0:
            self.commands[forward, 0] = self._uniform_sample(
                len(forward),
                self.cfg.commands.forward_velocity_range_m_s,
                self.device,
            )

        backward = selected(self.COMMAND_BACKWARD)
        if len(backward) > 0:
            self.commands[backward, 0] = -self._uniform_sample(
                len(backward),
                self.cfg.commands.backward_speed_range_m_s,
                self.device,
            )

        lateral = selected(self.COMMAND_LATERAL)
        if len(lateral) > 0:
            self.commands[lateral, 1] = self._sample_signed(
                len(lateral), self.cfg.commands.lateral_speed_range_m_s
            )

        yaw = selected(self.COMMAND_YAW)
        if len(yaw) > 0:
            self.commands[yaw, 2] = self._sample_signed(
                len(yaw), self.cfg.commands.yaw_speed_range_rad_s
            )

        combined = selected(self.COMMAND_COMBINED)
        if len(combined) > 0:
            self.commands[combined, 0] = self._sample_signed(
                len(combined),
                self.cfg.commands.combined_forward_speed_range_m_s,
            )
            self.commands[combined, 1] = self._sample_signed(
                len(combined),
                self.cfg.commands.combined_lateral_speed_range_m_s,
            )
            self.commands[combined, 2] = self._sample_signed(
                len(combined),
                self.cfg.commands.combined_yaw_speed_range_rad_s,
            )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if not getattr(self, "_v2_command_ready", False):
            if hasattr(self, "commands"):
                self.commands[env_ids, :3] = 0.0
            return
        modes = self._sample_transition_modes(env_ids)
        self._set_command_modes(env_ids, modes)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not getattr(self, "_v2_command_ready", False):
            return
        modes = torch.full_like(env_ids, self.COMMAND_MARCH)
        self._set_command_modes(env_ids, modes)

    def _walking_command_gate(self):
        if getattr(self, "_v2_command_ready", False):
            return self.gait_enable
        return super()._walking_command_gate()

    def _reward_trajectory_lateral_error(self):
        lateral_error, _ = self._straight_path_state()
        scale = max(
            float(self.cfg.rewards.trajectory_lateral_scale_m), 1.0e-6
        )
        bounded_error = 1.0 - torch.exp(
            -torch.square(lateral_error / scale)
        )
        return bounded_error * self._walking_command_gate()

    def _reward_omni_heading_error(self):
        scale = max(float(self.cfg.rewards.heading_error_scale_rad), 1.0e-6)
        bounded_error = 1.0 - torch.exp(
            -torch.square(self._straight_heading_error() / scale)
        )
        return bounded_error * self._walking_command_gate()

    def _reward_stand_still(self):
        stand = 1.0 - self._walking_command_gate()
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        ) * stand

    def compute_observations(self):
        phase_angle = 2.0 * torch.pi * self._gait_phase()
        phase_observation = torch.stack(
            (torch.sin(phase_angle), torch.cos(phase_angle)), dim=1
        )
        heading_error = self._straight_heading_error()
        lateral_error, lateral_velocity = self._straight_path_observation_state()
        gait_enable = (
            self.gait_enable
            if getattr(self, "_v2_command_ready", False)
            else torch.zeros(self.num_envs, device=self.device)
        )
        self.obs_buf = torch.cat(
            (
                (
                    self.estimated_base_lin_vel
                    if self._rs01_observation_estimator_ready
                    else self.base_lin_vel
                )
                * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                phase_observation,
                torch.stack(
                    (torch.sin(heading_error), torch.cos(heading_error)), dim=1
                ),
                torch.stack(
                    (
                        torch.clamp(
                            lateral_error
                            * float(
                                self.cfg.commands
                                .straight_path_lateral_position_scale
                            ),
                            min=-1.0,
                            max=1.0,
                        ),
                        torch.clamp(
                            lateral_velocity
                            * float(
                                self.cfg.commands
                                .straight_path_lateral_velocity_scale
                            ),
                            min=-1.0,
                            max=1.0,
                        ),
                    ),
                    dim=1,
                ),
                gait_enable.unsqueeze(1),
            ),
            dim=1,
        )
        if self.add_noise:
            self.obs_buf += (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self.noise_scale_vec
