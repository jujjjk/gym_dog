"""Replace V20 target compression, upstream of unchanged real actuator chain.

Not a physical-state clamp. Not action mirroring. Deployment MUST reproduce this
map and V25 cadence; an actor alone is not an executable V26 controller.
"""
import torch
from .rs01_omni_v25_env import Rs01OmniV25Robot


def landing_bound_weight(phase, diagonal_a, duty):
    local=torch.where(diagonal_a[None,:],phase[:,None],torch.remainder(phase[:,None]+.5,1.))
    swing=((local-duty)/(1.-duty)).clamp(0.,1.)
    # Late swing tightens, touchdown stays tight, handoff releases continuously.
    swing=((swing-.5)/.5).clamp(0.,1.)
    handoff=(local/(duty-.5)).clamp(0.,1.)
    smooth=lambda x:x.square()*(3.-2.*x)
    return torch.where(local<duty,1.-smooth(handoff),smooth(swing))


def map_omni_hip_targets(target, commands, hip_ids, sides, full_range, straight_limit, omni_limit, landing_weight=None):
    straight=(1.-commands[:,1].abs()/.08-commands[:,2].abs()/.20).clamp(0.,1.)
    weight=torch.ones_like(target[:,hip_ids]) if landing_weight is None else landing_weight
    limit=straight_limit+(omni_limit-straight_limit)*(1.-straight[:,None])*weight
    hip=target[:,hip_ids]
    out=target.clone()
    out[:,hip_ids]=torch.where(hip*sides<0.,hip*limit/full_range,hip)
    return out


class Rs01OmniV26Robot(Rs01OmniV25Robot):
    def _project_rs01_policy_target(self, target):
        if not hasattr(self,'v26_hip_ids'):
            self.v26_hip_ids=[self.dof_names.index(x+'_hip_joint') for x in ('FL','FR','RL','RR')]
            self.v26_sides=target.new_tensor([1.,-1.,1.,-1.])
            if not torch.allclose(self.default_dof_pos[:,self.v26_hip_ids],torch.zeros_like(self.default_dof_pos[:,self.v26_hip_ids])):
                raise ValueError('V26 requires zero default hip angles')
            ranges=self.rs01_action_scale_rad[self.v26_hip_ids]*self.cfg.normalization.clip_actions
            if not 0.<self.cfg.control.omni_inward_target_rad<=self.cfg.control.straight_inward_target_rad or bool((ranges<self.cfg.control.straight_inward_target_rad).any()):
                raise ValueError('Invalid inward target range')
        ranges=self.rs01_action_scale_rad[self.v26_hip_ids]*self.cfg.normalization.clip_actions
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        weight=landing_bound_weight(self._gait_phase(),self.diagonal_a_contact_mask,
                                   self.cfg.rewards.gait_stance_ratio)[:,ids]
        result=map_omni_hip_targets(target,self.commands,self.v26_hip_ids,self.v26_sides,ranges,
            self.cfg.control.straight_inward_target_rad,self.cfg.control.omni_inward_target_rad,weight)
        self.v26_projection_abs_rad=(result-target).abs().detach()
        return result
