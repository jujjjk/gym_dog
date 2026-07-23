"""Deterministic diagonal CPG and planar IK for the RS01 dog.

The CPG owns only the nominal periodic motion.  A learned policy is added as
a bounded joint-space residual by the environment; motor gain, delay,
friction, backlash, target-rate and torque limits remain downstream.
"""

from __future__ import annotations

import math

import torch


class RS01DiagonalCPG:
    """Two anti-phase, coupled limit-cycle oscillators.

    Oscillator 0 drives FL+RR and oscillator 1 drives FR+RL.  The state is
    stored in Cartesian limit-cycle coordinates so it can be exported with
    the policy controller without a discontinuity at phase wrap.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float,
        radial_rate: float = 30.0,
        coupling_gain: float = 18.0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)
        self.radial_rate = float(radial_rate)
        self.coupling_gain = float(coupling_gain)
        self.state = torch.zeros(
            self.num_envs, 2, 2, dtype=torch.float, device=device
        )
        self.reset(torch.arange(self.num_envs, device=device))

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(
        self, env_ids: torch.Tensor, phase0: torch.Tensor | None = None
    ) -> None:
        if env_ids.numel() == 0:
            return
        if phase0 is None:
            phase0 = torch.zeros(
                env_ids.numel(), dtype=torch.float, device=self.device
            )
        angle0 = 2.0 * torch.pi * phase0
        angles = torch.stack((angle0, angle0 + torch.pi), dim=1)
        self.state[env_ids, :, 0] = torch.cos(angles)
        self.state[env_ids, :, 1] = torch.sin(angles)

    def synchronize(self, phase0: torch.Tensor) -> None:
        """Apply a contact-induced phase hold without breaking anti-phase."""
        current_angle = torch.atan2(
            self.state[:, 0, 1], self.state[:, 0, 0]
        )
        target_angle = 2.0 * torch.pi * phase0
        correction = self._wrap_angle(target_angle - current_angle).unsqueeze(1)
        radius = torch.sqrt(torch.sum(torch.square(self.state), dim=2))
        angle = torch.atan2(self.state[:, :, 1], self.state[:, :, 0])
        angle = angle + correction
        self.state[:, :, 0] = radius * torch.cos(angle)
        self.state[:, :, 1] = radius * torch.sin(angle)

    def step(self, phase_increment: torch.Tensor) -> torch.Tensor:
        """Advance both oscillators and return FL+RR phase in cycles."""
        x = self.state[:, :, 0]
        y = self.state[:, :, 1]
        radius = torch.sqrt(torch.square(x) + torch.square(y)).clip(min=1.0e-6)
        angle = torch.atan2(y, x)

        # Kuramoto-style anti-phase coupling.  Positive error means oscillator
        # 1 is more than pi ahead, so the two angular velocities close it.
        phase_error = self._wrap_angle(angle[:, 1] - angle[:, 0] - torch.pi)
        common_increment = 2.0 * torch.pi * phase_increment
        coupled_increment = 0.5 * self.coupling_gain * self.dt * phase_error
        angle0 = angle[:, 0] + common_increment + coupled_increment
        angle1 = angle[:, 1] + common_increment - coupled_increment

        # Stable unit-radius limit cycle.  The exponential update remains
        # stable at the real 50 Hz controller rate.
        radial_blend = 1.0 - math.exp(-self.radial_rate * self.dt)
        radius = radius + radial_blend * (1.0 - radius)
        new_angle = torch.stack((angle0, angle1), dim=1)
        self.state[:, :, 0] = radius * torch.cos(new_angle)
        self.state[:, :, 1] = radius * torch.sin(new_angle)
        return torch.remainder(new_angle[:, 0] / (2.0 * torch.pi), 1.0)

    @property
    def pair_phases(self) -> torch.Tensor:
        angle = torch.atan2(self.state[:, :, 1], self.state[:, :, 0])
        return torch.remainder(angle / (2.0 * torch.pi), 1.0)


class RS01FootTrajectory:
    """Foot-space CPG trajectory and analytic IK for the supplied URDF."""

    # Vectors from thigh joint to calf joint and calf joint to foot, expressed
    # at q=0 in the URDF sagittal x/z plane.
    THIGH_VECTOR_XZ_M = (-0.1557, -0.090323)
    CALF_VECTOR_XZ_M = (0.201984878568, 0.008373101566)

    def __init__(self) -> None:
        thigh_x, thigh_z = self.THIGH_VECTOR_XZ_M
        calf_x, calf_z = self.CALF_VECTOR_XZ_M
        self.thigh_length = math.hypot(thigh_x, thigh_z)
        self.calf_length = math.hypot(calf_x, calf_z)
        self.thigh_zero_angle = math.atan2(thigh_z, thigh_x)
        self.calf_zero_angle = math.atan2(calf_z, calf_x)

    @staticmethod
    def _smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = value.clip(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def sample(
        self,
        phase: torch.Tensor,
        stance_ratio: torch.Tensor,
        signed_stride_m: torch.Tensor,
        clearance_m: torch.Tensor,
        nominal_x_m: float = 0.0,
        nominal_z_m: float = -0.300,
        lift_fraction: float = 0.18,
        lower_start_fraction: float = 0.62,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x/z targets for a `[num_envs, 4]` leg phase tensor."""
        if stance_ratio.ndim == 1:
            stance_ratio = stance_ratio.unsqueeze(1)
        stride = signed_stride_m.unsqueeze(1)
        clearance = clearance_m.unsqueeze(1)

        stance_progress = (phase / stance_ratio).clip(0.0, 1.0)
        swing_progress = (
            (phase - stance_ratio) / (1.0 - stance_ratio)
        ).clip(0.0, 1.0)
        swing_smooth = self._smoothstep(swing_progress)

        stance_x = nominal_x_m + 0.5 * stride - stride * stance_progress
        swing_x = nominal_x_m - 0.5 * stride + stride * swing_smooth
        foot_x = torch.where(phase < stance_ratio, stance_x, swing_x)

        lift_fraction = min(max(float(lift_fraction), 0.10), 0.45)
        lower_start_fraction = min(
            max(float(lower_start_fraction), lift_fraction + 0.10), 0.90
        )
        rise = self._smoothstep(swing_progress / lift_fraction)
        fall_progress = (
            (swing_progress - lower_start_fraction)
            / (1.0 - lower_start_fraction)
        ).clip(0.0, 1.0)
        fall = 1.0 - self._smoothstep(fall_progress)
        lift = torch.minimum(rise, fall) * (phase >= stance_ratio)
        foot_z = nominal_z_m + clearance * lift
        return foot_x, foot_z

    def inverse_kinematics(
        self, foot_x: torch.Tensor, foot_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert sagittal foot targets to URDF thigh/calf joint angles."""
        l1 = self.thigh_length
        l2 = self.calf_length
        radius_squared = torch.square(foot_x) + torch.square(foot_z)
        cosine_knee = (
            (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        ).clip(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        relative_link_angle = torch.acos(cosine_knee)
        thigh_link_angle = torch.atan2(foot_z, foot_x) - torch.atan2(
            l2 * torch.sin(relative_link_angle),
            l1 + l2 * torch.cos(relative_link_angle),
        )
        thigh_joint = self.thigh_zero_angle - thigh_link_angle
        calf_joint = (
            self.calf_zero_angle
            - self.thigh_zero_angle
            - relative_link_angle
        )
        return thigh_joint, calf_joint
