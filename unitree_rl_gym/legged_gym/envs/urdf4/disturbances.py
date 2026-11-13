"""Sensor and external-force disturbance models for the URDF4 task.

Pure tensor code with no simulator dependency so every model can be unit
tested.  Quaternions are (x, y, z, w), matching Isaac Gym.
"""

import math

import torch


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quat_rotate_inverse(q, v):
    """Express world/parent-frame vector ``v`` in the frame whose attitude is ``q``."""
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def pulse_envelope(elapsed_s, duration_s):
    """sin^2 force pulse: zero outside [0, duration), impulse = peak * duration / 2."""
    phase = elapsed_s / duration_s.clamp(min=1.0e-6)
    return torch.where(
        (phase >= 0.0) & (phase < 1.0),
        torch.sin(math.pi * phase.clamp(0.0, 1.0)).square(),
        torch.zeros_like(phase),
    )


class DelayLine:
    """Ring buffer sampled on the physics clock; each env reads its own latency."""

    def __init__(self, num_envs, width, length, device):
        self.buffer = torch.zeros(num_envs, length, width, device=device)
        self.length = int(length)
        self.head = 0
        self._rows = torch.arange(num_envs, device=device)

    def reset(self, env_ids, value):
        # No invented pre-reset history: every slot holds the reset-time sample.
        self.buffer[env_ids] = value.unsqueeze(1)

    def push(self, value):
        self.head = (self.head + 1) % self.length
        self.buffer[:, self.head] = value

    def read(self, delay_steps):
        """``delay_steps`` [num_envs] long, 0 = newest sample."""
        if int(delay_steps.max()) >= self.length:
            raise ValueError("Requested latency exceeds the delay-line length")
        index = torch.remainder(self.head - delay_steps, self.length)
        return self.buffer[self._rows, index]


class ImuDisturbance:
    """What the body IMU reports, given the true attitude and body rate.

    Per-episode (resampled at reset):
      * mounting misalignment (roll/pitch/yaw) applied to both outputs;
      * gyro bias and per-axis scale-factor error.
    Per policy step:
      * gyro bias random walk and white noise;
      * attitude-filter drift: an Ornstein-Uhlenbeck roll/pitch error on the
        gravity estimate, as an AHRS shows under sustained acceleration;
      * white noise on the gravity estimate;
      * impact/vibration spikes on both outputs;
      * stale frames: the previous sample is repeated (dropped packet).
    A configurable fraction of environments keeps an ideal IMU so nominal
    performance is still trained.  Latency is handled by ``DelayLine``.
    """

    def __init__(self, num_envs, device, step_dt_s, cfg):
        self.num_envs, self.device, self.dt, self.cfg = num_envs, device, float(step_dt_s), cfg
        self.clean = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.mount_quat = torch.zeros(num_envs, 4, device=device)
        self.mount_quat[:, 3] = 1.0
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)
        self.gyro_scale = torch.ones(num_envs, 3, device=device)
        self.tilt_error = torch.zeros(num_envs, 2, device=device)
        self.last_gyro = torch.zeros(num_envs, 3, device=device)
        self.last_gravity = torch.zeros(num_envs, 3, device=device)
        self.has_last = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._down = torch.zeros(num_envs, 3, device=device)
        self._down[:, 2] = -1.0

    def _symmetric(self, count, width, bound):
        return (2.0 * torch.rand(count, width, device=self.device) - 1.0) * bound

    def reset(self, env_ids):
        count = len(env_ids)
        if count == 0:
            return
        c = self.cfg
        clean = (
            torch.rand(count, device=self.device) < c.clean_env_fraction
            if c.enabled
            else torch.ones(count, device=self.device, dtype=torch.bool)
        )
        active = (~clean).float().unsqueeze(1)
        angles = self._symmetric(count, 3, 1.0) * active
        tilt = math.radians(c.mount_roll_pitch_deg)
        self.mount_quat[env_ids] = quat_from_rpy(
            angles[:, 0] * tilt, angles[:, 1] * tilt, angles[:, 2] * math.radians(c.mount_yaw_deg)
        )
        self.gyro_bias[env_ids] = self._symmetric(count, 3, c.gyro_bias_rad_s) * active
        self.gyro_scale[env_ids] = 1.0 + self._symmetric(count, 3, c.gyro_scale_error) * active
        # Start inside the stationary drift distribution, not at a perfect zero.
        drift = torch.randn(count, 2, device=self.device) * math.radians(c.attitude_drift_std_deg)
        limit = math.radians(c.attitude_drift_limit_deg)
        self.tilt_error[env_ids] = drift.clamp(-limit, limit) * active
        self.clean[env_ids] = clean
        self.has_last[env_ids] = False

    def step(self):
        """Advance the slow error states by one policy step."""
        c = self.cfg
        active = (~self.clean).float().unsqueeze(1)
        walk = torch.randn_like(self.gyro_bias) * c.gyro_bias_walk_rad_s_sqrt_s * math.sqrt(self.dt)
        self.gyro_bias = (self.gyro_bias + walk * active).clamp(
            -c.gyro_bias_limit_rad_s, c.gyro_bias_limit_rad_s
        )
        tau = max(float(c.attitude_drift_tau_s), self.dt)
        sigma = math.radians(c.attitude_drift_std_deg)
        limit = math.radians(c.attitude_drift_limit_deg)
        diffusion = sigma * math.sqrt(2.0 * self.dt / tau) * torch.randn_like(self.tilt_error)
        self.tilt_error = (
            self.tilt_error * (1.0 - self.dt / tau) + diffusion * active
        ).clamp(-limit, limit)

    def measure(self, base_quat, body_angular_velocity):
        """Return (gyro [rad/s], unit gravity direction) in the sensor frame."""
        c = self.cfg
        active = (~self.clean).float().unsqueeze(1)
        zero = torch.zeros(self.num_envs, device=self.device)
        attitude_error = quat_from_rpy(self.tilt_error[:, 0], self.tilt_error[:, 1], zero)
        true_gravity = quat_rotate_inverse(base_quat, self._down)
        gravity = quat_rotate_inverse(quat_mul(self.mount_quat, attitude_error), true_gravity)
        gyro = self.gyro_scale * quat_rotate_inverse(self.mount_quat, body_angular_velocity) + self.gyro_bias

        spike = (torch.rand(self.num_envs, 1, device=self.device) < c.spike_probability).float() * active
        gyro = gyro + active * torch.randn_like(gyro) * c.gyro_noise_std_rad_s
        gyro = gyro + spike * self._symmetric(self.num_envs, 3, c.gyro_spike_rad_s)
        gravity = gravity + active * torch.randn_like(gravity) * c.gravity_noise_std
        gravity = gravity + spike * self._symmetric(self.num_envs, 3, c.gravity_spike)
        gravity = gravity / gravity.norm(dim=1, keepdim=True).clamp(min=1.0e-6)

        stale = (
            (torch.rand(self.num_envs, device=self.device) < c.stale_frame_probability)
            & self.has_last
            & ~self.clean
        ).unsqueeze(1)
        gyro = torch.where(stale, self.last_gyro, gyro)
        gravity = torch.where(stale, self.last_gravity, gravity)
        self.last_gyro, self.last_gravity = gyro.clone(), gravity.clone()
        self.has_last[:] = True
        return gyro, gravity
