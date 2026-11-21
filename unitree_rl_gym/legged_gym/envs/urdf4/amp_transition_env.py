"""Opt-in command-transition controller for the existing rough V4 student."""
import json
from pathlib import Path

import numpy as np
import torch

from .amp_rough_env import RoughPathController, Urdf4AMPRoughRobot
from .amp_cts_env import yaw_from_quat
from .path_tracking import rotate_xy, wrap
from .urdf4_omni_env import Urdf4OmniRobot


class TransitionPathController(RoughPathController):
    """Keep the route servo active while releasing the longitudinal PI bias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_speed_bias = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.speed_release = torch.zeros_like(self.speed_bias)
        self.entry_bias_cap = torch.full_like(self.speed_bias, float('inf'))
        self.track_transition = torch.zeros_like(self.release_speed_bias)
        self.route_yaw = torch.zeros_like(self.speed_bias)
        self.transition_steering = None

    def reset_ids(self, ids, position, yaw):
        super().reset_ids(ids, position, yaw)
        self.release_speed_bias[ids] = False
        self.speed_release[ids] = 0.
        self.entry_bias_cap[ids] = float('inf')
        self.track_transition[ids] = False
        self.route_yaw[ids] = 0.

    def update(self, external, position, yaw, velocity_world, yaw_rate):
        self.speed_integral[self.release_speed_bias] *= .85
        self.speed_integral[:] = torch.minimum(self.speed_integral, self.entry_bias_cap)
        output = super().update(external, position, yaw, velocity_world, yaw_rate)
        # The inherited lateral and yaw corrections remain applied. Only the
        # separate speed PI contribution is faded out during a gear change.
        step = self.dt / .22
        desired = self.release_speed_bias.float()
        self.speed_release += (desired - self.speed_release).clamp(-step, step)
        bias_excess = (self.speed_bias - self.entry_bias_cap).clamp(min=0.)
        self.speed_bias -= bias_excess
        along = torch.stack((bias_excess, torch.zeros_like(bias_excess)), 1)
        capped = rotate_xy(self.reference_yaw - yaw, along)
        removed = capped + self.speed_release[:, None] * (self.speed_applied - capped)
        output[:, :2] -= removed
        self.speed_applied -= removed
        if self.transition_steering is not None:
            gain, damping = self.transition_steering
            cross_velocity = rotate_xy(-self.reference_yaw, self.filtered_velocity)[:, 1]
            target = (-gain*self.cross_error - damping*cross_velocity).clamp(-.12, .12)
            desired = torch.where(self.track_transition, target, self.cross_yaw)
            yaw_step = self.gains.yaw_acceleration*self.dt
            self.route_yaw += (desired-self.route_yaw).clamp(-yaw_step, yaw_step)
            limit = self.gains.output_yaw_limit_rad_s
            output[:, 2] = (output[:, 2]-self.cross_yaw+self.route_yaw).clamp(-limit, limit)
        self.command_output.copy_(output)
        return output


class Urdf4AMPTransitionRobot(Urdf4AMPRoughRobot):
    """Retimes user commands and shares the same normalized gait phase."""

    def _init_buffers(self):
        super()._init_buffers()
        self.path = TransitionPathController(self.num_envs, self.dt, self.device,
            gains=self.path.gains, speed_enabled=self.path.speed_enabled, speed_settings=self.path.speed_settings,
            rough_mask=self.rough_env, rough_cfg=self.cfg.rough)
        self._init_transition_control()

    def _init_transition_control(self):
        self.transition_goal = torch.zeros_like(self.target_commands)
        self.transition_ready_time = torch.zeros_like(self.command_elapsed)
        self.handoff_ready = torch.zeros_like(self.cold_start)
        self.ramp_wait = torch.zeros_like(self.command_elapsed)
        self.ramp_pending = torch.zeros_like(self.cold_start)
        self.acceleration_bias_limit = .08
        self.transition_control = getattr(self.cfg, 'transition_control', None)
        self.torque_burst = (self.transition_control or {}).get('torque_burst')
        if self.torque_burst is not None:
            burst = self.torque_burst
            if not (0. < burst['peak_nm'] <= 17. and 0. < burst['half_width_cycles'] <= .12):
                raise ValueError('Transition torque burst must stay within 17 Nm and 0.12 cycles per side')
        if self.transition_control is not None:
            self.acceleration_bias_limit = self.transition_control['acceleration_bias_m_s']
            self.path.transition_steering = (self.transition_control['cross_gain'], self.transition_control['cross_damping'])
            if 'contact_phase_gain_hz' in self.transition_control:
                self.contact_phase = self._reference_touchdown_phases()
                self.cadence_foot_slots = [self.foot_legs.index(leg) for leg in ('FL', 'FR', 'RL', 'RR')]
                self.last_diagonal = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
                self.cadence_correction = torch.zeros_like(self.motion_frequency)

    def _reference_touchdown_phases(self):
        reference = Path(self.cfg.amp_cts.reference)
        manifest = json.loads((reference/'manifest.json').read_text())
        phases = []
        for clip in manifest['clips']:
            with np.load(reference/(clip['clip']+'.npz'), allow_pickle=False) as record:
                contact = record['contact'].astype(bool)
                phase = record['leg_phase'][:, 0]
            touchdown = []
            for legs in ((0, 3), (1, 2)):
                loaded = contact[:, legs].all(1)
                events = phase[1:][loaded[1:] & ~loaded[:-1]]
                touchdown.append(float(np.angle(np.exp(2j*np.pi*events).mean())/(2*np.pi)) if len(events) else 0.)
            phases.append(touchdown)
        return self.motion_phase.new_tensor(phases)

    def _contact_cadence(self, contact, active, reference):
        fl, fr, rl, rr = contact[:, self.cadence_foot_slots].unbind(1)
        loaded = torch.stack((fl & rr, fr & rl), 1)
        touchdown = loaded & ~self.last_diagonal
        self.last_diagonal.copy_(loaded)
        error = self.contact_phase[reference] - self.motion_phase[:, None]
        offset = torch.as_tensor(self.transition_control.get('contact_phase_offset_cycles', 0.),
                                 device=error.device, dtype=error.dtype)
        error += offset.reshape(-1, 1)
        error = torch.remainder(error+.5, 1.)-.5
        useful = touchdown & (error.abs() < .18) & active[:, None]
        update = useful.any(1)
        desired = (error*useful).sum(1)/useful.sum(1).clamp(min=1)
        desired *= self.transition_control['contact_phase_gain_hz']
        self.cadence_correction *= 1-self.dt/.32
        self.cadence_correction[update] = desired[update].clamp(-.18, .18)
        return self.cadence_correction * active

    def reset_idx(self, ids):
        super().reset_idx(ids)
        self.transition_goal[ids] = 0.
        self.transition_ready_time[ids] = 0.
        self.handoff_ready[ids] = False
        self.ramp_wait[ids] = 0.
        self.ramp_pending[ids] = False
        if hasattr(self, 'contact_phase'):
            self.last_diagonal[ids] = False
            self.cadence_correction[ids] = 0.

    @staticmethod
    def _pure_fast(command):
        return (command[:, 0] > .35) & (command[:, 1:].abs().sum(1) < .001)

    def _command_ramp(self, target_fast, start_fast, from_stop):
        ramp = torch.where(from_stop, 1.0, 1.15)
        control = self.transition_control or {}
        ramp = torch.where(start_fast, control.get('brake_s', 1.8), ramp)
        ramp = torch.where(target_fast, control.get('accelerate_s', 3.2), ramp)
        return torch.where(target_fast & from_stop, self.cfg.commands.fast_ramp_s, ramp)

    def _transition_torque_cap(self, steady_cap):
        burst = self.torque_burst
        target, start = self.target_commands, self.command_start
        target_fast, start_fast = self._pure_fast(target), self._pure_fast(start)
        ramp = self._command_ramp(target_fast, start_fast, start.abs().sum(1) < .001)
        pure = (start[:, 1:].abs() + target[:, 1:].abs()).sum(1) < .001
        active = pure & (target_fast ^ start_fast) & self.gait_enabled & ~self.ramp_pending
        active &= (start[:, 0] > .02) & (target[:, 0] > .02)
        # At 2 Hz, a half-width of .10 cycles gives at most about 100 ms
        # of additional headroom per diagonal exchange.
        phase = torch.remainder(self.motion_phase - burst['phase_cycles'] + .25, .50) - .25
        phase_weight = (1.-phase.abs()/burst['half_width_cycles']).clamp(0., 1.)
        fade = (torch.minimum(self.command_elapsed, ramp-self.command_elapsed)/.12).clamp(0., 1.)
        blend = phase_weight.square() * (3.-2.*phase_weight) * fade * active
        return steady_cap + max(0., burst['peak_nm']-steady_cap)*blend

    def step(self, actions):
        if self.torque_burst is not None:
            cap = self._transition_torque_cap(self.cfg.rs01_actuator.peak_torque_limit_nm)
            self.peak_torque.copy_(cap[:, None].expand_as(self.peak_torque))
        return super().step(actions)

    def _post_physics_step_callback(self):
        self.speed_bias_cycle.push(self.path.speed_applied.clone())
        Urdf4OmniRobot._post_physics_step_callback(self)
        self._update_transition_command()

    def _update_transition_command(self):
        target = self.target_commands
        changed = (target - self.previous_target).abs().amax(1) > 1.e-6
        self.command_start[changed] = self.external_commands[changed]
        self.command_elapsed[changed] = 0.
        self.previous_target[changed] = target[changed]
        self.handoff_ready[changed] = False

        # Observe several quiet samples after the initial settling transient.
        moving = target.abs().sum(1) > .001
        quiet = (self.projected_gravity[:, 2] < -.995) & (self.base_ang_vel.norm(dim=1) < .35)
        quiet &= (self.base_lin_vel.norm(dim=1) < .06) & (self.get_foot_contact_mask().sum(1) == 4)
        self.transition_ready_time = torch.where(quiet, self.transition_ready_time + self.dt, 0.)
        settling = self.cold_start & ((self.episode_steps * self.dt < .28) | (self.transition_ready_time < .12))
        self.cold_start &= settling
        target_fast = self._pure_fast(target)
        start_fast = self._pure_fast(self.command_start)
        from_stop = self.command_start.abs().sum(1) < .001
        control = self.transition_control or {}
        if 'ramp_start_phase' in control:
            moving_forward = (self.command_start[:, 0] > .02) & (target[:, 0] > .02)
            pure = (self.command_start[:, 1:].abs() + target[:, 1:].abs()).sum(1) < .001
            changing_gear = (target_fast ^ start_fast) & moving_forward & pure
            self.ramp_wait[changed] = 0.
            self.ramp_pending[changed] = changing_gear[changed]
            onset = torch.where(target_fast, control['ramp_start_phase'],
                                control.get('brake_start_phase', control['ramp_start_phase']))
            phase_error = torch.remainder(self.motion_phase-onset+.25, .50) - .25
            self.ramp_wait += self.ramp_pending.float()*self.dt
            self.ramp_pending &= (phase_error.abs() > .023) & (self.ramp_wait < .28)
        self.command_elapsed += (~settling & ~self.ramp_pending).float() * self.dt

        ramp = self._command_ramp(target_fast, start_fast, from_stop)
        u = (self.command_elapsed / ramp).clamp(0, 1)
        blend = u.square() * (3 - 2 * u)
        if 'ramp_waypoint_m_s' in control:
            # Give the slow gait a brief constant-speed handoff before
            # commanding the remainder of the long stride.
            start = self.command_start[:, 0]
            middle = control['ramp_waypoint_m_s']
            waypoint = ((middle-start)/(target[:, 0]-start).clamp(min=.01)).clamp(0, 1)
            first = control.get('ramp_waypoint_fraction', .42)
            pause_end = first + control.get('ramp_waypoint_hold_s', .25)/ramp
            entry = (u/first).clamp(0, 1)
            exit = ((u-pause_end)/(1-pause_end)).clamp(0, 1)
            staged = waypoint*entry.square()*(3-2*entry) + (1-waypoint)*exit.square()*(3-2*exit)
            use_waypoint = target_fast & ~start_fast & ~from_stop & (start < middle)
            blend = torch.where(use_waypoint, staged, blend)
        external = self.command_start + (target - self.command_start) * blend[:, None]
        self.external_commands[:] = torch.where(settling[:, None], 0., external)

        xy, yaw, velocity = self._localized_pose()
        along = rotate_xy(-self.path.reference_yaw, self.path.filtered_velocity)[:, 0]
        slow_entry = target_fast & ~start_fast & ~from_stop
        fast_exit = start_fast & ~target_fast
        # For acceleration, wait for the ramp to enter the viable fast range.
        # For braking, retain the high-speed template until the body slows.
        goal = torch.where(moving[:, None], target, self.transition_goal)
        entry_goal = torch.minimum(self.external_commands[:, 0] + control.get('entry_goal_lead_m_s', .10), target[:, 0])
        contact = self.get_foot_contact_mask()
        contact_overlap = contact.sum(1) >= 3
        may_enter = slow_entry & (entry_goal > .35) & (along > .12)
        may_leave = fast_exit & (self.external_commands[:, 0] < .45) & (along < .45)
        eligible = may_enter | may_leave
        if self.transition_control is not None:
            # Check normalized gait phase as well as actual contact. The two
            # diagonal exchanges are half a cycle apart. Keep advancing the
            # clock until the exchange so the current stance can support it.
            phase = torch.remainder(self.motion_phase - control['handoff_phase'] + .25, .50) - .25
            in_overlap = phase.abs() < control['handoff_window']
            unblocked = self.command_elapsed > ramp + .20
            eligible &= in_overlap | unblocked
        self.handoff_ready |= contact_overlap & eligible
        entry_goal = torch.where(self.handoff_ready, entry_goal, entry_goal.clamp(max=.35))
        minimum_goal = torch.where(slow_entry, entry_goal, goal[:, 0])
        braking_goal = torch.maximum(self.external_commands[:, 0], along - .02)
        braking_goal = torch.minimum(braking_goal, self.command_start[:, 0])
        braking_goal = torch.where(self.handoff_ready, braking_goal, braking_goal.clamp(min=.45))
        goal[:, 0] = torch.where(fast_exit, torch.maximum(braking_goal, goal[:, 0]), minimum_goal)
        # Once tracking has settled at a nonzero setpoint use the exact
        # validated endpoint template. Keep that template during a stop ramp.
        goal = torch.where((fast_exit & ~moving)[:, None], self.transition_goal, goal)
        if 'coordination_goal_slew_m_s2' in control:
            # A detected handoff may release the fast template partway through
            # the velocity ramp. Limit the change in blend at that release.
            coast = ((slow_entry | fast_exit) & (target[:, 0] > .02)
                     & (target[:, 1:].abs().sum(1) < .001))
            previous = self.transition_goal[:, 0]
            step = self.dt * control['coordination_goal_slew_m_s2']
            continuous = previous + (goal[:, 0]-previous).clamp(-step, step)
            goal[:, 0] = torch.where(coast, continuous, goal[:, 0])
        self.transition_goal[:] = goal

        reference = self._select_reference(goal)
        self.reference_id[:] = reference
        active = ~settling & (moving | (self.external_commands.abs().sum(1) > .001))
        self.gait_enabled[:] = ~settling & (self.external_commands.abs().sum(1) > .001)
        start_phase = active & ~self.phase_was_active
        phase_offset = self.reference_phase_offsets[reference]
        self.motion_phase[start_phase] = phase_offset[start_phase] % 1
        frequency = self.reference_frequencies[reference].clamp(1.8, 2.3)
        if 'coordination_fast_cadence' in control:
            load = ((goal[:, 0]-.35)/.10).clamp(0., 1.)
            load *= (goal[:, 1:].abs().sum(1) < .001)
            frequency *= 1. + (control['coordination_fast_cadence']-1.)*load
        if 'contact_phase_gain_hz' in control:
            correcting = active & (goal[:, 0] >= .38) & (goal[:, 1:].abs().sum(1) < .001)
            frequency += self._contact_cadence(contact, correcting, reference)
        self.motion_frequency[start_phase] = frequency[start_phase]
        self.motion_frequency += (frequency - self.motion_frequency).clamp(-.8*self.dt, .8*self.dt)
        self.motion_phase[:] = torch.remainder(self.motion_phase + self.motion_frequency*self.dt*active, 1.)
        self.motion_phase[~active] = 0.
        self.phase_was_active[:] = active

        self.path.release_speed_bias[:] = start_fast & (target[:, 0] < self.command_start[:, 0] - .02) & (u < 1.) & ~self.ramp_pending
        if self.transition_control is not None:
            pure_forward = (target[:, 0] > .02) & (target[:, 1:].abs().sum(1) < .001)
            self.path.track_transition[:] = pure_forward & (slow_entry | fast_exit) & (self.command_elapsed < ramp + 3.)
        bias_cap = self.acceleration_bias_limit
        if 'settled_bias_m_s' in control:
            # The entry cap protects the gear change. Restore PI authority
            # gradually once the gait has accepted the new cadence.
            recovered = ((self.command_elapsed-ramp)/control.get('settled_bias_ramp_s', 1.)).clamp(0, 1)
            bias_cap = bias_cap + recovered*(control['settled_bias_m_s']-bias_cap)
        self.path.entry_bias_cap[:] = torch.where(slow_entry, bias_cap, float('inf'))
        gyro = self.obs_history[:, 0, 2] / self.obs_scales.ang_vel
        self.commands[:, :3] = self.path.update(self.external_commands, xy, yaw, velocity, gyro)
        self.commands[settling, :3] = 0.
        self.heading_target[:] = self.path.reference_yaw
        self.heading_error[:] = wrap(self.heading_target - yaw_from_quat(self.root_states[:, 3:7]))
        self.public_heading_target[:] = wrap(self.path.reference_yaw + self.path_public_offset)
        self.yaw_drift += (~self.nominal_mask).float() * self.imu.gyro_bias[:, 2] * self.dt
        self.yaw_drift.clamp_(-.035, .035)
        self.recovery_left.sub_(self.dt).clamp_(min=0.)

    def compute_observations(self):
        super().compute_observations()
        self.obs_buf[:, 1020:1023] = self.transition_goal
        self.obs_buf[:, 1028] = self._transition_amplitude()

    def _transition_amplitude(self):
        amplitude = self.external_commands.norm(dim=1) / self.transition_goal.norm(dim=1).clamp(min=1.e-6)
        if self.transition_control is not None and 'amplitude_min' in self.transition_control:
            during_handoff = self.path.track_transition & self.gait_enabled
            amplitude = torch.where(during_handoff, torch.maximum(amplitude, torch.as_tensor(
                self.transition_control['amplitude_min'], device=amplitude.device, dtype=amplitude.dtype)), amplitude)
        return amplitude.clamp(0, 1)
