"""Reward-only filtered placement prior. No mirrored action or actuator change."""
import math
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from .rs01_omni_v18_env import Rs01OmniV18Robot


def huber_positive(x):
    x=x.clamp(min=0.)
    return torch.where(x<1.,.5*x.square(),x-.5)


def placement_cost(widths, filtered_dx, raw, age, cfg):
    straight=(1.-raw[:,1].abs()/.08-raw[:,2].abs()/.20).clamp(0.,1.)
    minimum=cfg.footprint_omni_min_width_m+straight*(
        cfg.footprint_straight_min_width_m-cfg.footprint_omni_min_width_m)
    width_cost=huber_positive((minimum[:,None]-widths)/cfg.footprint_soft_scale_m).mean(1)
    # Width remains required on reverse; only sagittal symmetry is weaker.
    bias_gate=straight*torch.where(raw[:,0]<-.02,.5,1.)
    ready=(age/cfg.placement_filter_tau_s).clamp(0.,1.)
    bias=huber_positive((filtered_dx.abs()-cfg.placement_deadband_m)/cfg.placement_scale_m).mean(1)
    return width_cost+bias_gate*ready*bias


class Rs01OmniV19Robot(Rs01OmniV18Robot):
    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        if hasattr(self,'placement_dx'):
            self.placement_dx[env_ids]=0.
            self.placement_age[env_ids]=0.
            self.placement_commands[env_ids]=float('nan')

    def _footprint_quality(self):
        relative=self.feet_pos-self.root_states[:,None,:3]
        quat=self.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4)
        feet=quat_rotate_inverse(quat,relative.reshape(-1,3)).reshape(-1,4,3)
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        feet=feet[:,ids]
        widths=torch.stack((feet[:,0,1]-feet[:,1,1],feet[:,2,1]-feet[:,3,1]),1)
        dx=torch.stack((feet[:,0,0]-feet[:,1,0],feet[:,2,0]-feet[:,3,0]),1)
        raw=self.commands[:,:3]
        if not hasattr(self,'placement_dx'):
            self.placement_dx=torch.zeros_like(dx)
            self.placement_age=torch.zeros_like(raw[:,0])
            self.placement_commands=torch.full_like(raw,float('nan'))
        changed=~torch.isclose(raw,self.placement_commands,atol=1e-4,rtol=0.).all(1)
        self.placement_dx[changed]=0.
        self.placement_age[changed]=0.
        cfg=self.cfg.rewards
        alpha=math.exp(-self.dt/cfg.placement_filter_tau_s)
        self.placement_dx.mul_(alpha).add_(dx,alpha=1.-alpha)
        self.placement_age.add_(self.dt)
        self.placement_commands.copy_(raw)
        self.v16_foot_widths=widths.detach()
        self.v19_placement_cost=placement_cost(widths,self.placement_dx,raw,self.placement_age,cfg)
        # Remove the old multiplicative width objective. Support quality
        # must retain its incentive even while geometry is being corrected.
        return torch.ones_like(raw[:,0])

    def _placement_reward_weight(self):
        return self.cfg.rewards.placement_weight

    def _reward_phase_two_contact_quality(self):
        reward=super()._reward_phase_two_contact_quality()
        reward=reward-self._placement_reward_weight()*self.v19_placement_cost*self._walking_command_gate()
        self.v11_contact_quality_reward=reward.detach()
        return reward
