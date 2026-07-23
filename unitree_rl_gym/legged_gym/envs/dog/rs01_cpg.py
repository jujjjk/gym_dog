"""Deterministic diagonal CPG and URDF-derived planar IK for the RS01 dog."""

from __future__ import annotations

import math
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import torch


class RS01DiagonalCPG:
    """Two anti-phase, coupled limit-cycle oscillators.

    Oscillator 0 drives FL+RR and oscillator 1 drives FR+RL.
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
        """Apply an accepted phase correction without breaking anti-phase."""
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
        """Advance both oscillators and return the FL+RR phase in cycles."""
        x = self.state[:, :, 0]
        y = self.state[:, :, 1]
        radius = torch.sqrt(torch.square(x) + torch.square(y)).clip(min=1.0e-6)
        angle = torch.atan2(y, x)

        phase_error = self._wrap_angle(angle[:, 1] - angle[:, 0] - torch.pi)
        common_increment = 2.0 * torch.pi * phase_increment
        coupled_increment = 0.5 * self.coupling_gain * self.dt * phase_error
        angle0 = angle[:, 0] + common_increment + coupled_increment
        angle1 = angle[:, 1] + common_increment - coupled_increment

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
    """Foot-space trajectory and analytic IK derived from the tracked URDF."""

    LEGS = ("FL", "FR", "RL", "RR")
    FALLBACK_THIGH_VECTOR_XZ_M = (-0.1557, -0.090323)
    FALLBACK_CALF_VECTOR_XZ_M = (0.201984878568, 0.008373101566)

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        path = self._resolve_urdf_path(urdf_path)
        if path is None:
            warnings.warn(
                "RS01 URDF was not found; using the checked fallback link "
                "vectors. Keep dog_urdf beside unitree_rl_gym so geometry is "
                "read directly from the tracked URDF.",
                RuntimeWarning,
                stacklevel=2,
            )
            thigh_vector = self.FALLBACK_THIGH_VECTOR_XZ_M
            calf_vector = self.FALLBACK_CALF_VECTOR_XZ_M
            self.urdf_path = None
        else:
            thigh_vector, calf_vector = self._load_leg_vectors(path)
            self.urdf_path = path

        thigh_x, thigh_z = thigh_vector
        calf_x, calf_z = calf_vector
        self.thigh_vector_xz_m = (float(thigh_x), float(thigh_z))
        self.calf_vector_xz_m = (float(calf_x), float(calf_z))
        self.thigh_length = math.hypot(thigh_x, thigh_z)
        self.calf_length = math.hypot(calf_x, calf_z)
        self.thigh_zero_angle = math.atan2(thigh_z, thigh_x)
        self.calf_zero_angle = math.atan2(calf_z, calf_x)

        if self.thigh_length <= 0.05 or self.calf_length <= 0.05:
            raise ValueError(
                "URDF-derived leg lengths are implausible: "
                f"thigh={self.thigh_length:.6f} m, "
                f"calf={self.calf_length:.6f} m"
            )

    @staticmethod
    def _resolve_urdf_path(
        urdf_path: str | Path | None,
    ) -> Path | None:
        if urdf_path is not None:
            candidate = Path(urdf_path).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"RS01 URDF not found: {candidate}")
            return candidate

        repo_root = Path(__file__).resolve().parents[4]
        candidates = (
            repo_root
            / "dog_urdf"
            / "urdf"
            / "URDFzhuangpei.SLDASM.urdf",
            repo_root / "dog_urdf" / "urdf" / "dog_rs01.urdf",
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    @staticmethod
    def _joint_origin_xyz(root: ET.Element, joint_name: str) -> tuple[float, float, float]:
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"URDF is missing joint {joint_name!r}")
        origin = joint.find("origin")
        if origin is None or "xyz" not in origin.attrib:
            raise ValueError(f"URDF joint {joint_name!r} has no xyz origin")
        values = tuple(float(value) for value in origin.attrib["xyz"].split())
        if len(values) != 3:
            raise ValueError(f"Invalid xyz origin for joint {joint_name!r}")
        return values

    @classmethod
    def _load_leg_vectors(
        cls, urdf_path: Path
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        root = ET.parse(urdf_path).getroot()
        vectors = []
        for leg in cls.LEGS:
            thigh_xyz = cls._joint_origin_xyz(root, f"{leg}_calf_joint")
            calf_xyz = cls._joint_origin_xyz(root, f"{leg}_foot_fixed")
            if abs(thigh_xyz[1]) > 1.0e-5 or abs(calf_xyz[1]) > 1.0e-5:
                raise ValueError(
                    f"{leg} sagittal IK expects near-zero link y offsets, got "
                    f"{thigh_xyz[1]:.6g} and {calf_xyz[1]:.6g}"
                )
            vectors.append(
                ((thigh_xyz[0], thigh_xyz[2]), (calf_xyz[0], calf_xyz[2]))
            )

        reference = vectors[0]
        for leg, vector_pair in zip(cls.LEGS[1:], vectors[1:]):
            for current, expected in zip(vector_pair, reference):
                error = math.hypot(
                    current[0] - expected[0], current[1] - expected[1]
                )
                if error > 1.0e-5:
                    raise ValueError(
                        f"{leg} link geometry differs from FL by {error:.6g} m; "
                        "use per-leg IK instead of silently sharing one model"
                    )
        return reference

    @staticmethod
    def _smootherstep(value: torch.Tensor) -> torch.Tensor:
        """Quintic interpolation with zero velocity and acceleration at ends."""
        value = value.clip(0.0, 1.0)
        return value * value * value * (
            value * (value * 6.0 - 15.0) + 10.0
        )

    def sample(
        self,
        phase: torch.Tensor,
        stance_ratio: torch.Tensor,
        signed_stride_m: torch.Tensor,
        clearance_m: torch.Tensor,
        nominal_x_m: float = 0.0,
        nominal_z_m: float = -0.300,
        lift_fraction: float = 0.40,
        lower_start_fraction: float = 0.60,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smooth x/z targets for a ``[num_envs, 4]`` phase tensor."""
        if stance_ratio.ndim == 1:
            stance_ratio = stance_ratio.unsqueeze(1)
        stride = signed_stride_m.unsqueeze(1)
        clearance = clearance_m.unsqueeze(1)

        stance_progress = (phase / stance_ratio).clip(0.0, 1.0)
        swing_progress = (
            (phase - stance_ratio) / (1.0 - stance_ratio)
        ).clip(0.0, 1.0)

        # Both branches now have zero endpoint velocity and acceleration.
        stance_smooth = self._smootherstep(stance_progress)
        swing_smooth = self._smootherstep(swing_progress)
        stance_x = nominal_x_m + 0.5 * stride - stride * stance_smooth
        swing_x = nominal_x_m - 0.5 * stride + stride * swing_smooth
        foot_x = torch.where(phase < stance_ratio, stance_x, swing_x)

        lift_fraction = min(max(float(lift_fraction), 0.20), 0.45)
        lower_start_fraction = min(
            max(float(lower_start_fraction), lift_fraction + 0.10), 0.85
        )
        rise = self._smootherstep(swing_progress / lift_fraction)
        fall_progress = (
            (swing_progress - lower_start_fraction)
            / (1.0 - lower_start_fraction)
        ).clip(0.0, 1.0)
        fall = 1.0 - self._smootherstep(fall_progress)
        lift = torch.minimum(rise, fall) * (phase >= stance_ratio)
        foot_z = nominal_z_m + clearance * lift
        return foot_x, foot_z

    def inverse_kinematics(
        self, foot_x: torch.Tensor, foot_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert sagittal foot targets to URDF thigh/calf joint angles."""
        l1 = self.thigh_length
        l2 = self.calf_length

        radius = torch.sqrt(torch.square(foot_x) + torch.square(foot_z)).clip(
            min=1.0e-8
        )
        min_radius = abs(l1 - l2) + 1.0e-4
        max_radius = l1 + l2 - 1.0e-4
        feasible_radius = radius.clip(min=min_radius, max=max_radius)
        scale = feasible_radius / radius
        x = foot_x * scale
        z = foot_z * scale

        radius_squared = torch.square(x) + torch.square(z)
        cosine_knee = (
            (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        ).clip(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        relative_link_angle = torch.acos(cosine_knee)
        thigh_link_angle = torch.atan2(z, x) - torch.atan2(
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
