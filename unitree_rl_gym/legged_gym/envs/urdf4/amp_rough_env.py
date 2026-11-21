"""Finite, continuous rough terrain for the blind V4 locomotion policy."""
import math

import numpy as np
import torch
from isaacgym import gymapi, terrain_utils

from .amp_v4_env import Urdf4AMPV4Robot, V4PathController
from .path_tracking import rotate_xy


def rough_height_samples(cfg):
    """The common sampled surface before simulator-specific tessellation."""
    step = float(cfg.grid_m)
    x = np.linspace(-20., 60., round(80 / step) + 1, dtype=np.float32)[:, None]
    y = np.linspace(-20., 20., round(40 / step) + 1, dtype=np.float32)[None, :]
    rng = np.random.default_rng(cfg.terrain_seed)
    height = np.zeros((len(x), y.shape[1]), dtype=np.float32)
    for wavelength in cfg.wavelengths_m:
        angle, phase = rng.uniform(0, 2 * math.pi, 2)
        height += np.sin(2 * math.pi * (np.cos(angle) * x + np.sin(angle) * y) / wavelength + phase)
    height *= cfg.amplitude_m / len(cfg.wavelengths_m)
    height *= np.clip((20. - x) / 1., 0., 1.)
    return height


class RoughPathController(V4PathController):
    def __init__(self, *args, rough_mask, rough_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.rough_mask = rough_mask
        self.rough_cfg = rough_cfg
        self.rough_yaw = torch.zeros_like(self.cross_yaw)

    def reset_ids(self, ids, position, yaw):
        super().reset_ids(ids, position, yaw)
        self.rough_yaw[ids] = 0.

    def update(self, external, position, yaw, velocity_world, yaw_rate):
        output = super().update(external, position, yaw, velocity_world, yaw_rate)
        fast = (external[:, 0] > .35) & (external[:, 1:].abs().sum(1) < .001) & self.rough_mask
        lateral_velocity = rotate_xy(-self.reference_yaw, self.filtered_velocity)[:, 1]
        cfg = self.rough_cfg
        correction = -cfg.fast_cross_yaw_gain * self.cross_error - cfg.fast_cross_yaw_damping * lateral_velocity
        desired = correction.clamp(-cfg.fast_cross_yaw_limit, cfg.fast_cross_yaw_limit) * fast
        step = self.gains.yaw_acceleration * self.dt
        self.rough_yaw += (desired - self.rough_yaw).clamp(-step, step)
        self.rough_yaw[external.abs().sum(1) < .001] = 0.
        limit = self.gains.output_yaw_limit_rad_s
        output[:, 2] = (output[:, 2] + self.rough_yaw).clamp(-limit, limit)
        self.command_output.copy_(output)
        return output


class Urdf4AMPRoughRobot(Urdf4AMPV4Robot):
    def _init_buffers(self):
        super()._init_buffers()
        self.path = RoughPathController(self.num_envs, self.dt, self.device, gains=self.path.gains,
            speed_enabled=self.cfg.direction.speed_feedback_enabled, speed_settings=self.cfg.direction.speed_feedback,
            rough_mask=self.rough_env, rough_cfg=self.cfg.rough)

    def _create_ground_plane(self):
        # One mesh, with adjacent rough and flat spawn regions. Actors are in
        # independent PhysX collision groups and may share terrain locations.
        cfg = self.cfg.rough
        self.terrain_origin_xy = (-20., -20.)
        self.terrain_grid_m = float(cfg.grid_m)
        step = self.terrain_grid_m
        height = rough_height_samples(cfg)
        vertices, faces = terrain_utils.convert_heightfield_to_trimesh(height, step, 1.)
        params = gymapi.TriangleMeshParams()
        params.nb_vertices, params.nb_triangles = len(vertices), len(faces)
        params.transform.p.x, params.transform.p.y = self.terrain_origin_xy
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), faces.flatten(), params)
        self.terrain_heights = torch.as_tensor(height, device=self.device)
        self.terrain_xy_offset = self.terrain_heights.new_tensor(self.terrain_origin_xy)

    def ground_height_at(self, xy):
        """Interpolate the same two triangles used by PhysX, at world XY."""
        limit_x, limit_y = self.terrain_heights.shape
        coordinate = (xy - self.terrain_xy_offset) / self.terrain_grid_m
        x = coordinate[..., 0].clamp(0, limit_x - 1)
        y = coordinate[..., 1].clamp(0, limit_y - 1)
        ix = x.floor().long().clamp(max=limit_x - 2)
        iy = y.floor().long().clamp(max=limit_y - 2)
        u, v = x - ix, y - iy
        z00 = self.terrain_heights[ix, iy]
        z11 = self.terrain_heights[ix + 1, iy + 1]
        z10 = self.terrain_heights[ix + 1, iy]
        z01 = self.terrain_heights[ix, iy + 1]
        return torch.where(u >= v, (1-u)*z00 + (u-v)*z10 + v*z11,
                           (1-v)*z00 + (v-u)*z01 + u*z11)

    def _get_env_origins(self):
        self.custom_origins = False
        ids = torch.arange(self.num_envs, device=self.device)
        self.rough_env = torch.ones_like(ids, dtype=torch.bool)
        if not self.cfg.env.test:
            # Alternate flat slots across CTS teacher/student and nominal/DR
            # partitions instead of assigning all clean robots to flat ground.
            self.rough_env = ((ids + ids // self.cfg.rough.flat_stride) % self.cfg.rough.flat_stride) != 0
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[~self.rough_env, 0] = 40.
        if not self.cfg.env.test:
            jitter = self.cfg.rough.spawn_jitter_m
            self.env_origins[:, :2] += self._uniform((self.num_envs, 2), [-jitter, jitter])
        self.env_origins[:, 2] = self.ground_height_at(self.env_origins[:, :2])

    def check_termination(self):
        trunk_force = self.contact_forces[:, self.termination_contact_indices].norm(dim=-1)
        height = self.root_states[:, 2] - self.ground_height_at(self.root_states[:, :2])
        fallen = (trunk_force > 1.).any(dim=1) | (self.projected_gravity[:, 2] > -.6)
        fallen |= height < self.cfg.rewards.min_base_height_m
        # Keep episodes inside the mesh even when an unstable policy drifts.
        xy = self.root_states[:, :2]
        fallen |= (xy[:, 0] < -19.5) | (xy[:, 0] > 59.5) | (xy[:, 1].abs() > 19.5)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = fallen | self.time_out_buf

    def _reward_gait_quality(self):
        contact = self.get_foot_contact_mask()
        mismatch = contact != self._scheduled_stance()
        phase = self._leg_phase()
        duty = self._stance_ratio()[:, None]
        distance = torch.minimum(torch.minimum(phase, 1-phase), (phase-duty).abs())
        tolerance = self.cfg.rough.handoff_tolerance_s * self.motion_frequency[:, None]
        mismatch &= (~self.rough_env[:, None]) | (distance > tolerance)
        stand = (~contact).float().mean(1) + .05 * self.dof_vel.square().mean(1)
        moving = .20 * mismatch.float().mean(1) * self.recovery_factor()
        return self._reward_illegal_support() + torch.where(self.gait_enabled, moving, stand)

    def _resample_commands(self, ids):
        super()._resample_commands(ids)
        if self.command_override is not None or self.cfg.env.test or not len(ids):
            return
        fast = ids[torch.rand(len(ids), device=self.device) < self.cfg.rough.exact_fast_fraction]
        self.target_commands[fast] = self.target_commands.new_tensor([.50, 0., 0.])
        self.target_gait[fast] = True
        self.command_frontier[fast] = True

    def step(self, actions):
        previous_height = self.ground_height_at(self.root_states[:, :2])
        result = super().step(actions)
        amp = result[4]['amp']
        amp['previous'][:, 45] += self.env_origins[:, 2] - previous_height
        amp['next'][:, 45] += self.env_origins[:, 2] - self.terminal_ground_height
        return result

    def compute_observations(self):
        super().compute_observations()
        public = self.cfg.env.public_obs_size
        self.obs_buf[:, public + 54] -= 5 * self.ground_height_at(self.root_states[:, :2])
        self.obs_buf[:, public + 63:public + 67] -= 10 * self.ground_height_at(self.feet_pos[:, :, :2])

    def _capture_terminal_snapshot(self):
        super()._capture_terminal_snapshot()
        self.terminal_ground_height = self.ground_height_at(self.root_states[:, :2])
