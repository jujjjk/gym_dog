"""Guard5 robustness: equivalent trunk payload and smooth external force pulses.

No root-velocity edits, new rewards, ideal motors or actor-only privileged inputs.
"""
import math
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from .rs01_v22_phase_guard import (
    Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard5PPO)


def pulse_envelope(elapsed, duration):
    phase = elapsed / duration.clamp(min=1e-6)
    return torch.where((phase >= 0.) & (phase < 1.),
                       torch.sin(math.pi*phase.clamp(0.,1.)).square(),
                       torch.zeros_like(phase))


class Rs01V22GuardRobustCfg(Rs01V22PhaseGuard5Cfg):
    class domain_rand(Rs01V22PhaseGuard5Cfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [0., 2.]
        push_robots = False  # Never enable the legacy velocity overwrite.

    class robust_push:
        enabled = True
        nominal_fraction = .25
        peak_force_n = [6., 18.]
        duration_s = [.10, .20]
        interval_s = [6., 10.]
        first_delay_s = [4., 7.]
        application_height_m = .06  # World-vertical offset above trunk COM.
        initial_strength = .5
        ramp_s = 120.  # Simulated seconds per environment, not wall time.


class Rs01V22GuardRobustPPO(Rs01V22PhaseGuard5PPO):
    class runner(Rs01V22PhaseGuard5PPO.runner):
        experiment_name = 'rs01_v22_guard5_robust'


class Rs01V22GuardRobustRobot(Rs01V22PhaseGuardRobot):
    def __init__(self, *args, **kwargs):
        self._robust_ready = False
        self._payload_samples = []
        self._nominal_samples = []
        super().__init__(*args, **kwargs)
        c = self.cfg.robust_push
        if self.cfg.domain_rand.push_robots:
            raise ValueError('Legacy velocity pushes must stay disabled')
        if not 0. <= c.nominal_fraction < 1. or not 0. <= c.initial_strength <= 1. or c.ramp_s < 0.:
            raise ValueError('Invalid disturbance mixture/ramp')
        for bounds in (c.peak_force_n, c.duration_s, c.interval_s, c.first_delay_s):
            if not 0. < bounds[0] <= bounds[1]:
                raise ValueError('Invalid positive disturbance bounds')
        self.push_body = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], 'Trunk')
        if self.push_body != 0:
            raise ValueError('Payload contract requires Trunk at body0')
        self.payload_kg = torch.tensor(self._payload_samples,device=self.device)
        self.nominal_environment = torch.tensor(self._nominal_samples,device=self.device,dtype=torch.bool)
        self.push_force_tensor = torch.zeros(self.num_envs,self.num_bodies,3,device=self.device)
        self.push_torque_tensor = torch.zeros_like(self.push_force_tensor)
        self.push_vector = torch.zeros(self.num_envs,3,device=self.device)
        self.push_elapsed = torch.ones(self.num_envs,device=self.device)
        self.push_duration = torch.full_like(self.push_elapsed,.1)
        self.push_wait = torch.zeros_like(self.push_elapsed)
        self.push_count = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.push_impulse_ns = torch.zeros_like(self.push_elapsed)
        self.push_count_at_reset = torch.zeros_like(self.push_count)
        self.push_impulse_at_reset = torch.zeros_like(self.push_impulse_ns)
        self._robust_ready = True
        self._clear_push(torch.arange(self.num_envs,device=self.device))

    def _process_rigid_body_props(self, props, env_id):
        original_mass = props[0].mass
        props = super()._process_rigid_body_props(props, env_id)
        nominal = np.random.random() < self.cfg.robust_push.nominal_fraction
        if nominal:
            props[0].mass = original_mass
        self._payload_samples.append(float(props[0].mass-original_mass))
        self._nominal_samples.append(nominal)
        return props

    def _uniform_push(self, count, bounds):
        return bounds[0] + (bounds[1]-bounds[0])*torch.rand(count,device=self.device)

    def _clear_push(self, ids):
        self.push_elapsed[ids] = 1.
        self.push_duration[ids] = .1
        self.push_wait[ids] = self._uniform_push(len(ids),self.cfg.robust_push.first_delay_s)
        self.push_vector[ids] = 0.
        self.push_force_tensor[ids] = 0.
        self.push_torque_tensor[ids] = 0.

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if self._robust_ready and len(ids):
            c = self.cfg.robust_push
            blend = 1. if c.ramp_s == 0. else min(1., self.common_step_counter*self.dt/c.ramp_s)
            self.extras['episode'].update({
                'robust_payload_kg': self.payload_kg[ids].mean(),
                'robust_pushes': (self.push_count[ids]-self.push_count_at_reset[ids]).float().mean(),
                'robust_impulse_ns': (self.push_impulse_ns[ids]-self.push_impulse_at_reset[ids]).mean(),
                'robust_force_scale': c.initial_strength+(1.-c.initial_strength)*blend,
            })
            self.push_count_at_reset[ids] = self.push_count[ids]
            self.push_impulse_at_reset[ids] = self.push_impulse_ns[ids]
            self._clear_push(ids)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        if not self._robust_ready:
            return torques
        c = self.cfg.robust_push
        dt = self.cfg.sim.dt
        self.push_wait.sub_(dt)
        start = ((self.push_wait <= 0.) & ~self.nominal_environment).nonzero().flatten()
        if c.enabled and len(start):
            time_s = self.common_step_counter*self.dt
            blend = 1. if c.ramp_s == 0. else min(1., time_s/c.ramp_s)
            strength = c.initial_strength+(1.-c.initial_strength)*blend
            angle = torch.rand(len(start),device=self.device)*2.*math.pi
            peak = strength*self._uniform_push(len(start),c.peak_force_n)
            self.push_vector[start,0] = peak*torch.cos(angle)
            self.push_vector[start,1] = peak*torch.sin(angle)
            self.push_duration[start] = self._uniform_push(len(start),c.duration_s)
            self.push_elapsed[start] = 0.
            self.push_wait[start] = self.push_duration[start]+self._uniform_push(len(start),c.interval_s)
            self.push_count[start] += 1
        envelope = pulse_envelope(self.push_elapsed+.5*dt,self.push_duration)
        force = self.push_vector*envelope[:,None] if c.enabled else torch.zeros_like(self.push_vector)
        self.push_force_tensor[:,self.push_body] = force
        # Equivalent to F applied 6cm above COM: torque = r cross F.
        self.push_torque_tensor[:,self.push_body,0] = -c.application_height_m*force[:,1]
        self.push_torque_tensor[:,self.push_body,1] = c.application_height_m*force[:,0]
        self.gym.apply_rigid_body_force_tensors(self.sim,
            gymtorch.unwrap_tensor(self.push_force_tensor),gymtorch.unwrap_tensor(self.push_torque_tensor),
            gymapi.ENV_SPACE)
        self.push_impulse_ns += torch.linalg.vector_norm(force,dim=1)*dt
        self.push_elapsed.add_(dt)
        return torques
