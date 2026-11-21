"""Practice command transitions while keeping fixed-speed V4 behavior available."""
import torch

from .amp_rough_config import ROOT, Urdf4AMPRoughCfg, Urdf4AMPRoughPPO
from .amp_transition_env import Urdf4AMPTransitionRobot
from .amp_cts_env import yaw_from_quat
from .gait_clock import CycleMean
from .path_tracking import rotate_xy


class Urdf4AMPTransitionTrainingCfg(Urdf4AMPRoughCfg):
    class transition:
        # Use half the environments for ordered transitions. The other half
        # keep the full-direction fixed-speed sample distribution.
        sequence_fraction = .5
        settling_tail_s = .35
        body_rate_free_rad_s = .35
        sequences = [
            [[.18, 0., 0.], [.50, 0., 0.], [.25, 0., 0.], [0., 0., 0.]],
            [[.25, 0., 0.], [.50, 0., 0.], [.18, 0., 0.], [-.15, 0., 0.]],
            [[-.15, 0., 0.], [-.30, 0., 0.], [-.15, 0., 0.], [0., 0., 0.]],
            [[0., .10, 0.], [0., .20, 0.], [0., -.10, 0.], [0., -.20, 0.]],
            [[0., 0., .40], [0., 0., .60], [0., 0., -.40], [0., 0., -.60]],
            [[.15, .08, 0.], [.15, -.08, 0.], [.20, 0., .30], [.20, 0., -.30]],
        ]

    class amp_cts(Urdf4AMPRoughCfg.amp_cts):
        reward_revision = 'v4_transition_sequence_and_body_stability_v1'

    class rewards(Urdf4AMPRoughCfg.rewards):
        class scales(Urdf4AMPRoughCfg.rewards.scales):
            transition_body_rate = -.10


class Urdf4AMPTransitionTrainingPPO(Urdf4AMPRoughPPO):
    class runner(Urdf4AMPRoughPPO.runner):
        cts_transition_training_enabled = True
        experiment_name = 'urdf4_amp_cts_transition_training'
        training_objective = 'variable_speed_omni_body_stability'
        v3_preflight = str(ROOT / 'artifacts/urdf4_transition_training_20261119/preflight.json')


def restore_transition_contract(cfg, train, saved):
    contract = saved['contract']
    for key, value in contract['transition_training'].items():
        setattr(cfg.transition, key, value)
    if 'transition_control' in contract:
        cfg.transition_control = dict(contract['transition_control'])
        trust = contract['transition_seed_trust']
        train.runner.seed_trust_coef = trust['coefficient']
        train.runner.seed_action_deadband = trust['deadband']
        train.runner.transition_sampling_std_max = trust['sampling_std_max']
    if 'transition_reference_sha256' in contract:
        train.runner.transition_reference = saved['experiment']['train']['runner']['transition_reference']


class Urdf4AMPTransitionTrainingRobot(Urdf4AMPTransitionRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.training_forward_limit = (self.transition_control or {}).get('training_start_forward_m_s', .50)
        self.sequence_commands = self.target_commands.new_tensor(self.cfg.transition.sequences)
        self.sequence_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.transition_error = CycleMean(self.num_envs, 3, max(1, round(.12/self.dt)), self.device)
        self.transition_window = torch.zeros_like(self.cold_start)
        self.fast_transition_window = torch.zeros_like(self.cold_start)
        self.transition_meter = torch.zeros(7, device=self.device)
        if self.torque_burst is not None:
            self.burst_meter = torch.zeros(3, device=self.device)
        self.coordination = (self.transition_control or {}).get('coordination')
        if self.coordination is not None:
            self.coordination_slots = [self.foot_legs.index(leg) for leg in ('FL', 'FR', 'RL', 'RR')]
            self.diagonal_split_time = torch.zeros(self.num_envs, 2, device=self.device)

    def reset_idx(self, ids):
        # Select a new family before the inherited reset samples its first
        # command. Sampling later would repeat the previous episode's family.
        if len(ids) and not self.cfg.env.test and self.command_override is None:
            assigned = torch.rand(len(ids), device=self.device) < self.cfg.transition.sequence_fraction
            families = torch.randint(len(self.sequence_commands), (len(ids),), device=self.device)
            self.sequence_id[ids] = torch.where(assigned, families, -1)
        super().reset_idx(ids)
        self.transition_error.reset(ids)
        self.transition_window[ids] = False
        self.fast_transition_window[ids] = False
        if self.coordination is not None:
            self.diagonal_split_time[ids] = 0.

    def _resample_commands(self, ids):
        super()._resample_commands(ids)
        if not len(ids) or self.cfg.env.test or self.command_override is not None:
            return
        selected = ids[self.sequence_id[ids] >= 0]
        if len(selected):
            # Gym samples the initial command before it zeros the episode clock.
            index = self.episode_length_buf[selected] // self.command_resample_steps
            index = torch.where(self.reset_buf[selected].bool(), 0, index)
            index = index.clamp(max=self.sequence_commands.shape[1]-1).long()
            command = self.sequence_commands[self.sequence_id[selected], index]
            self.target_commands[selected] = command
            self.target_gait[selected] = command.abs().sum(1) > .001
            self.command_frontier[selected] = True
        if self.training_forward_limit < .50:
            pure_forward = self.target_commands[ids, 1:].abs().sum(1) < .001
            forward = ids[pure_forward]
            self.target_commands[forward, 0] = self.target_commands[forward, 0].clamp(max=self.training_forward_limit)

    def _post_physics_step_callback(self):
        # Match the just-executed command, before the ramp advances this step.
        along_velocity = rotate_xy(-self.path.reference_yaw, self.root_states[:, 7:9])
        rate = self.root_states[:, 12:13]
        target = self.external_commands
        if self.transition_control is not None:
            # Match the bounded recentering requested by the path controller.
            # Forward speed still uses the user ramp, measured on the route.
            yaw = yaw_from_quat(self.root_states[:, 3:7])
            body_to_route = yaw - self.path.reference_yaw
            correction = rotate_xy(body_to_route, self.commands[:, :2])
            route_target = torch.stack((target[:, 0], correction[:, 1].clamp(-.05, .05),
                                        self.commands[:, 2].clamp(-.15, .15)), 1)
            along_route = (self.path.track_transition & self.fast_transition_window)
            target = torch.where(along_route[:, None], route_target, target)
        executed_error = target - torch.cat((along_velocity, rate), 1)
        self.transition_error.push(executed_error)
        super()._post_physics_step_callback()
        change = (self.command_start - self.target_commands).abs().amax(1) > .015
        tail = self.cfg.transition.settling_tail_s
        # Use the ramp state and include its first settling samples afterward.
        target_fast = self._pure_fast(self.target_commands)
        start_fast = self._pure_fast(self.command_start)
        from_stop = self.command_start.abs().sum(1) < .001
        duration = self._command_ramp(target_fast, start_fast, from_stop)
        self.transition_window[:] = change & ~self.cold_start & (self.command_elapsed < duration + tail)
        self.fast_transition_window[:] = self.transition_window & (target_fast | start_fast)

    def _reward_command_tracking(self):
        steady = super()._reward_command_tracking()
        cfg = self.cfg.rewards
        error = self.transition_error.mean() / self.commands.new_tensor([
            cfg.command_linear_sigma_m_s, cfg.command_linear_sigma_m_s, cfg.command_yaw_sigma_rad_s])
        error = error.abs()
        transition_loss = torch.where(error <= 1., .5*error.square(), error-.5).sum(1)
        reward = torch.where(self.transition_window, -transition_loss, steady)
        self.command_quality = reward.exp()
        return reward

    def _reward_gait_quality(self):
        penalty = super()._reward_gait_quality()
        if self.coordination is None:
            return penalty
        fl, fr, rl, rr = self.get_foot_contact_mask()[:, self.coordination_slots].unbind(1)
        split = torch.stack((fl ^ rr, fr ^ rl), 1)
        fast = self.gait_enabled & (self._pure_fast(self.external_commands) | self.fast_transition_window)
        active = split & fast[:, None]
        self.diagonal_split_time[:] = torch.where(active, self.diagonal_split_time+self.dt, 0.)
        late = ((self.diagonal_split_time-.04)/.06).clamp(0., 1.).mean(1)
        return penalty + self.coordination['handoff_cost']*late*self.recovery_factor()

    def _reward_transition_body_rate(self):
        body_rate = self.base_ang_vel[:, :2].norm(dim=1)
        excess = (body_rate - self.cfg.transition.body_rate_free_rad_s).clamp(min=0.)
        mask = self.transition_window
        student = mask & ~self.teacher_mask
        gyro = self.projected_gravity
        tilt = torch.stack((torch.atan2(-gyro[:, 1], -gyro[:, 2]).abs(), gyro[:, 0].clamp(-1, 1).asin().abs()), 1)
        tilt = torch.rad2deg(tilt)*student[:, None]
        self.transition_meter[0] += student.sum()
        self.transition_meter[1] += (student & self.fast_transition_window).sum()
        self.transition_meter[2:4] = torch.maximum(self.transition_meter[2:4], tilt.amax(0))
        self.transition_meter[4] += (student & (self.get_foot_contact_mask().sum(1) == 1)).sum()
        self.transition_meter[5] += (student & self.reset_buf.bool() & ~self.time_out_buf).sum()
        self.transition_meter[6] = torch.maximum(self.transition_meter[6], (self.path.cross_error.abs()*student).amax())
        if self.coordination is not None:
            mask = mask | (self.gait_enabled & self._pure_fast(self.external_commands))
        return mask.float() * excess.square().clamp(max=4.)

    def step(self, actions):
        fast_window = self.fast_transition_window.clone()
        start, goal = self.command_start.clone(), self.target_commands.clone()
        result = super().step(actions)
        if self.torque_burst is not None:
            steady = self.cfg.rs01_actuator.peak_torque_limit_nm
            motor_peak = self.motor_torques.abs().amax(dim=1)
            self.burst_meter[0] += (self.peak_torque[:, 0] > steady + .01).sum()
            self.burst_meter[1] += (motor_peak > steady + .01).sum()
            self.burst_meter[2] = torch.maximum(self.burst_meter[2], motor_peak.max())
        amp = result[4]['amp']
        amp['fast_transition'] = fast_window
        pure = (start[:, 1:].abs().sum(1) + goal[:, 1:].abs().sum(1)) < .001
        rising = ((start[:, 0]-.18).abs() < .015) & ((goal[:, 0]-.50).abs() < .015)
        braking = ((start[:, 0]-.50).abs() < .015) & ((goal[:, 0]-.25).abs() < .015)
        changing = self.external_commands[:, 0] - amp['style_command'][:, 0]
        kind = (rising & (changing > 1.e-5)).float() - (braking & (changing < -1.e-5)).float()
        amp['transition_kind'] = kind * (fast_window & pure).float()
        return result
