"""One continuous gait clock; replace width cost with landing-first geometry."""
import torch
from .rs01_omni_v19_env import huber_positive
from .rs01_omni_v24_env import Rs01OmniV24Robot


def direction_blend(command, lateral_full, yaw_full):
    x=torch.maximum(command[:,1].abs()/lateral_full,command[:,2].abs()/yaw_full).clamp(0.,1.)
    return x.square()*(3.-2.*x)


def landing_region_cost(y, contact, touchdown, progress, cfg):
    late=((progress-cfg.late_swing_start)/(1.-cfg.late_swing_start)).clamp(0.,1.)
    landing_weight=torch.maximum(late*(~contact).float(),touchdown.float())
    landing=huber_positive((cfg.landing_inner_m-y)/cfg.support_region_scale_m)
    landing+=huber_positive((y-cfg.support_outer_m)/cfg.support_region_scale_m)
    # During support allow normal body-relative travel; only severe narrowing/overwidth.
    stance=huber_positive((cfg.stance_inner_m-y)/cfg.support_region_scale_m)
    stance+=huber_positive((y-cfg.support_outer_m)/cfg.support_region_scale_m)
    weight=cfg.stance_region_weight*contact.float()*(~touchdown).float()
    return (landing*landing_weight+stance*weight).sum(1)/(landing_weight+weight).sum(1).clamp(min=1.)


class Rs01OmniV25Robot(Rs01OmniV24Robot):
    def _directional_blend(self):
        cfg=self.cfg.rewards
        return direction_blend(self.commands[:,:3],cfg.cadence_lateral_full_m_s,cfg.cadence_yaw_full_rad_s)

    def _command_gait_frequency(self):
        original=super()._command_gait_frequency()
        floor=self.cfg.rewards.lateral_frequency_floor_hz
        return original+self._directional_blend()*(floor-original).clamp(min=0.)

    def _advance_wide_gait_phase(self):
        target=self._command_gait_frequency()
        if not hasattr(self,'v25_frequency_hz'):
            self.v25_frequency_hz=target.clone()
        limit=self.cfg.rewards.cadence_slew_hz_s*self.dt
        self.v25_frequency_hz+=(target-self.v25_frequency_hz).clamp(-limit,limit)
        self.wide_gait_phase.add_(self.dt*self.v25_frequency_hz).remainder_(1.)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self,'v25_frequency_hz'):
            self.v25_frequency_hz[env_ids]=self._command_gait_frequency()[env_ids]

    def _swing_progress(self):
        phase=self._gait_phase()
        local=torch.where(self.diagonal_a_contact_mask[None,:],phase[:,None],
                          torch.remainder(phase[:,None]+.5,1.))
        duty=self.cfg.rewards.gait_stance_ratio
        return ((local-duty)/(1.-duty)).clamp(0.,1.)

    def _width_geometry_cost(self, widths, straight):
        # Parent only populates unified contact/landing diagnostics. Discard its cost.
        super()._width_geometry_cost(widths,straight)
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        cost=landing_region_cost(self.v24_signed_y,self.v24_support_contact,
            self.v24_touchdown,self._swing_progress()[:,ids],self.cfg.rewards)
        self.v24_region_cost=cost.detach()
        return cost

    def _phase_swing_clearance_error(self, selected_feet=None):
        cfg=self.cfg.rewards
        # Raise walking, not marching; retain 25mm for high-cadence lateral/turn modes.
        forward=(self.commands[:,0].abs()/.10).clamp(0.,1.)
        clearance=cfg.swing_clearance_m+(cfg.straight_clearance_m-cfg.swing_clearance_m)*forward*(1.-self._directional_blend())
        target,swing=self._swing_height_target_from_phase(self._gait_phase(),
            self.diagonal_a_contact_mask,self.diagonal_b_contact_mask,cfg.gait_stance_ratio,
            cfg.foot_collision_radius_m,clearance[:,None])
        if selected_feet is not None:swing=swing&selected_feet[None,:].bool()
        shortfall=((target-self.feet_pos[:,:,2])/clearance[:,None]).clamp(min=0.)
        return (shortfall.square()*swing.float()).sum(1)/swing.sum(1).clamp(min=1)
