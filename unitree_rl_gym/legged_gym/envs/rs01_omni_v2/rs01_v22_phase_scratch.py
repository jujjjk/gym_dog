"""B23000/V22 unchanged plant and rewards, with per-foot exchange detection."""
import torch

from .rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CfgPPO
from .rs01_omni_v22_env import Rs01OmniV22Robot


class Rs01V22PhaseScratchCfg(Rs01OmniV22Cfg):
    class phase_exchange:
        # Detect a real lift, not an unloaded/sliding foot or contact chatter.
        minimum_clearance_m = .008
        minimum_lift_s = .040
        # One full cycle must remain legal: the two diagonals are 0.5 apart.
        maximum_no_lift_cycles = 1.15
        penalty_ramp_cycles = .50


class Rs01V22PhaseScratchPPO(Rs01OmniV22CfgPPO):
    class runner(Rs01OmniV22CfgPPO.runner):
        experiment_name = 'rs01_v22_phase_scratch'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


def update_exchange(no_lift_cycles, lift_time_s, contact, clearance_m,
                    phase_increment, active, dt, cfg):
    """Pure tensor update; only measured liftoff can reset a foot's clock."""
    lifted = (~contact) & (clearance_m >= cfg.minimum_clearance_m)
    previous_lift_time = lift_time_s
    lift_time_s = torch.where(lifted & active[:, None], lift_time_s + dt, 0.)
    qualified = lift_time_s >= cfg.minimum_lift_s - 1.e-7
    if getattr(cfg, 'require_lift_event', False):
        # Credit one threshold crossing, not every frame of a hanging foot.
        qualified &= previous_lift_time < cfg.minimum_lift_s - 1.e-7
    no_lift_cycles = torch.where(
        active[:, None] & ~qualified,
        no_lift_cycles + phase_increment[:, None], 0.)
    penalty = ((no_lift_cycles.amax(dim=1) - cfg.maximum_no_lift_cycles)
               / cfg.penalty_ramp_cycles).clamp(0., 1.)
    return no_lift_cycles, lift_time_s, penalty


class Rs01V22PhaseScratchRobot(Rs01OmniV22Robot):
    def __init__(self, *args, **kwargs):
        self._exchange_ready = False
        super().__init__(*args, **kwargs)
        if self.cfg.terrain.mesh_type != 'plane':
            raise ValueError('V22 exchange height assumes the original flat ground')
        self.no_lift_cycles = torch.zeros((self.num_envs, 4), device=self.device)
        self.exchange_lift_time_s = torch.zeros_like(self.no_lift_cycles)
        self.exchange_penalty = torch.zeros(self.num_envs, device=self.device)
        self._exchange_ready = True

    def _post_physics_step_callback(self):
        if self._exchange_ready:
            phase_before = self._gait_phase().clone()
        super()._post_physics_step_callback()
        if not self._exchange_ready:
            return
        # Measure the original continuous clock, including speed transitions;
        # never use elapsed_seconds * the newly sampled frequency.
        delta = (self._gait_phase() - phase_before).remainder(1.)
        clearance = self.feet_pos[:, :, 2] - self.cfg.rewards.foot_collision_radius_m
        self.no_lift_cycles, self.exchange_lift_time_s, self.exchange_penalty = update_exchange(
            self.no_lift_cycles, self.exchange_lift_time_s,
            self.get_foot_contact_mask(), clearance, delta,
            self._walking_command_gate() > .5, self.dt, self.cfg.phase_exchange)

    def _reward_prolonged_all_feet_contact(self):
        original = super()._reward_prolonged_all_feet_contact()
        if not self._exchange_ready:
            return original
        # REPLACE the coverage of the existing term, not its weight or range.
        # max avoids charging twice when all four feet remain planted.
        return torch.maximum(original, self.exchange_penalty * self._walking_command_gate())

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._exchange_ready:
            self.no_lift_cycles[env_ids] = 0.
            self.exchange_lift_time_s[env_ids] = 0.
            self.exchange_penalty[env_ids] = 0.
