"""Continuous approach, ascent, descent and flat exit on RS01."""
import numpy as np
import torch
from isaacgym import gymapi

from .amp_cts_env import yaw_from_quat
from .amp_transition_training import (
    Urdf4AMPTransitionTrainingCfg, Urdf4AMPTransitionTrainingPPO, Urdf4AMPTransitionTrainingRobot,
)
from .path_tracking import rotate_xy
from .urdf4_omni_env import Urdf4OmniRobot

BASE_PUBLIC = 1029
SCAN_SIZE = 12
SCAN_X = (-.40, -.24, -.08, .08, .24, .44)
SCAN_Y = (-.14, .14)


def stair_scan(position, yaw, ground_height):
    """Body-aligned samples, relative to the level below the trunk."""
    points = position.new_tensor([(x, y) for x in SCAN_X for y in SCAN_Y])
    rotated = rotate_xy(yaw.repeat_interleave(len(points)), points.repeat(len(position),1))
    world = position[:, None, :2] + rotated.reshape(len(position),len(points),2)
    return (ground_height(world)-ground_height(position[:, :2])[:, None])/.14


class Urdf4AMPStairsCfg(Urdf4AMPTransitionTrainingCfg):
    class env(Urdf4AMPTransitionTrainingCfg.env):
        num_observations = 1130 + 2 * SCAN_SIZE
        num_privileged_obs = num_observations
        episode_length_s = 20.

    class amp_cts(Urdf4AMPTransitionTrainingCfg.amp_cts):
        reward_revision = 'urdf4_continuous_stairs_v1'

    class init_state(Urdf4AMPTransitionTrainingCfg.init_state):
        randomize_yaw = False

    class stairs:
        heights_m = [.04, .05, .06]
        lengths_m = [.72, 1.00]
        lane_spacing_m = 3.
        half_width_m = .80
        approach_m = .70
        exit_m = .45
        speed_m_s = .20
        spawn_jitter_m = .08
        scan_noise_m = .004
        edge_clearance_m = .027
        lift_gain = 0.
        lift_phase = .76
        leveling_gain = 0.
        top_start_fraction = 0.

    class rewards(Urdf4AMPTransitionTrainingCfg.rewards):
        class scales(Urdf4AMPTransitionTrainingCfg.rewards.scales):
            stair_edge_contact = -.35


class Urdf4AMPStairsPPO(Urdf4AMPTransitionTrainingPPO):
    class runner(Urdf4AMPTransitionTrainingPPO.runner):
        cts_stairs_enabled = True
        experiment_name = 'urdf4_amp_cts_stairs'
        training_objective = 'continuous_platform_ascent_and_descent'
        seed_weight_index = None


class Urdf4AMPStairsRobot(Urdf4AMPTransitionTrainingRobot):
    def _create_ground_plane(self):
        Urdf4OmniRobot._create_ground_plane(self)
        cfg = self.cfg.stairs
        # The flat lane shares the physical plane with the platform lanes.
        lanes = [(0., 0.)] + [(h, length) for h in cfg.heights_m for length in cfg.lengths_m]
        self.stair_lanes = torch.tensor(lanes, device=self.device)
        vertices, triangles = [], []
        faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                          [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=np.uint32)
        for i, (height, length) in enumerate(lanes[1:], 1):
            left = i*cfg.lane_spacing_m - cfg.half_width_m
            right = i*cfg.lane_spacing_m + cfg.half_width_m
            vertices.extend([(x, y, z) for z in (0., height) for x, y in
                             ((0.,left), (length,left), (length,right), (0.,right))])
            triangles.extend(faces+8*(i-1))
        params = gymapi.TriangleMeshParams()
        params.nb_vertices, params.nb_triangles = len(vertices), len(triangles)
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, np.asarray(vertices,np.float32).ravel(),
                                   np.asarray(triangles,np.uint32).ravel(), params)

    def ground_height_at(self, xy):
        cfg = self.cfg.stairs
        index = (xy[..., 1]/cfg.lane_spacing_m).round().long().clamp(0, len(self.stair_lanes)-1)
        spec = self.stair_lanes[index]
        in_lane = (xy[..., 1]-index*cfg.lane_spacing_m).abs() < cfg.half_width_m
        return spec[..., 0] * (in_lane & (xy[..., 0] >= 0.) & (xy[..., 0] <= spec[..., 1]))

    def _get_env_origins(self):
        cfg = self.cfg.stairs
        ids = torch.arange(self.num_envs, device=self.device)
        lane = 1 + ids % (len(self.stair_lanes)-1) if self.cfg.env.test else ids % len(self.stair_lanes)
        self.stair_lane = lane
        self.rough_env = lane > 0
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs,3,device=self.device)
        self.env_origins[:, 0] = -cfg.approach_m
        self.env_origins[:, 1] = lane * cfg.lane_spacing_m

    def _init_buffers(self):
        super()._init_buffers()
        self.stair_events = torch.zeros(7, device=self.device)
        self.stair_stage = torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.stair_top_start = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)

    def reset_idx(self, ids):
        if not self.cfg.env.test:
            self.env_origins[ids, 0] = -self.cfg.stairs.approach_m + self._uniform(
                (len(ids),), [-self.cfg.stairs.spawn_jitter_m,self.cfg.stairs.spawn_jitter_m])
            self.env_origins[ids, 2] = 0.
            top = (torch.rand(len(ids),device=self.device)<self.cfg.stairs.top_start_fraction) & self.rough_env[ids]
            self.stair_top_start[ids] = top
            selected = ids[top]
            platform = self.stair_lanes[self.stair_lane[selected]]
            self.env_origins[selected, 0] = .5*platform[:,1]
            self.env_origins[selected, 2] = platform[:,0]
        super().reset_idx(ids)
        self.stair_stage[ids] = self.stair_top_start[ids].long()

    def _resample_commands(self, ids):
        super()._resample_commands(ids)
        if self.command_override is None:
            course = ids[self.rough_env[ids]]
            self.target_commands[course] = self.commands.new_tensor([self.cfg.stairs.speed_m_s,0.,0.])
            self.target_gait[course] = True

    def edge_distance(self, xy):
        length = self.stair_lanes[self.stair_lane,1]
        while length.ndim < xy[...,0].ndim:
            length = length.unsqueeze(-1)
        return torch.minimum(xy[...,0].abs(), (xy[...,0]-length).abs())

    def _reward_gait_quality(self):
        nominal = super()._reward_gait_quality()
        near_edge = self.rough_env & (self.edge_distance(self.root_states[:, :2]) < .33)
        # Mixed-height supports may legitimately switch away from the flat
        # diagonal clock. Still discourage losing load-bearing contact.
        count = self.get_foot_contact_mask().sum(1)
        return torch.where(near_edge, (2-count).clamp(min=0).float(), nominal)

    def _reward_stair_edge_contact(self):
        feet = self.feet_pos[:, :, :2]
        near = (self.edge_distance(feet) < self.cfg.stairs.edge_clearance_m)
        return (near & self.get_foot_contact_mask()).float().mean(1)*self.rough_env

    def check_termination(self):
        super().check_termination()
        length = self.stair_lanes[self.stair_lane, 1]
        contact = self.get_foot_contact_mask()
        on_top = self.ground_height_at(self.feet_pos[:, :, :2]) > .001
        climbed = (on_top & contact).all(1)
        x = self.root_states[:,0]
        first_top = self.rough_env & (self.stair_stage==0) & climbed & ~self.reset_buf
        self.stair_stage[first_top] = 1
        exit_contact = (self.feet_pos[:,:,0] > length[:,None]+.025).all(1) & (contact.sum(1)>=2)
        cleared = (self.rough_env & (self.stair_stage==1) & (x>length+self.cfg.stairs.exit_m)
                   & exit_contact & ~self.reset_buf)
        self.stair_stage[cleared] = 2
        self.stair_events[0] += first_top.sum()
        self.stair_events[1] += (cleared & ~self.stair_top_start).sum()
        self.stair_events[2] += (self.rough_env & self.reset_buf & ~self.time_out_buf).sum()
        self.stair_events[3] += (self.rough_env & self.time_out_buf & (self.stair_stage!=2) & ~self.stair_top_start).sum()
        self.stair_events[4] += (self.rough_env & self.reset_buf & ~self.stair_top_start).sum()
        self.stair_events[5] += (cleared & self.stair_top_start).sum()
        self.stair_events[6] += (self.rough_env & self.reset_buf & self.stair_top_start).sum()

    def step(self, actions):
        old_edge = self.edge_distance(self.root_states[:, :2]) < .33
        result = super().step(actions)
        moved_edge = self.edge_distance(self.root_states[:, :2]) < .33
        result[4]['amp']['stair_edge'] = self.rough_env & (old_edge | moved_edge)
        return result

    def compute_observations(self):
        super().compute_observations()
        scan = stair_scan(self.root_states[:,:2], yaw_from_quat(self.root_states[:,3:7]), self.ground_height_at)
        measured = scan
        if self.add_noise:
            noise = self._uniform(scan.shape, [-self.cfg.stairs.scan_noise_m,self.cfg.stairs.scan_noise_m])/.14
            active = (scan.abs().amax(1) > .01) & ~self.nominal_mask
            measured = (scan + active[:,None]*noise).clamp(-1.1,1.1)
        # Keep the previous AMP/CTS history and privileged state intact.
        old=self.obs_buf
        self.obs_buf=torch.cat((old[:,:BASE_PUBLIC],measured,old[:,BASE_PUBLIC:-1],scan,old[:,-1:]),1)
        self.privileged_obs_buf=self.obs_buf
