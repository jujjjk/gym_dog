"""Reward-only measured phase history, never actor/action mirroring."""
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from .rs01_omni_v19_env import huber_positive
from .rs01_omni_v20_env import Rs01OmniV20BoundedRobot


def half_cycle_sample(history, stamps, phase):
    query=phase-.5
    older=torch.where(stamps<=query[:,None],stamps,torch.full_like(stamps,-float('inf')))
    newer=torch.where(stamps>=query[:,None],stamps,torch.full_like(stamps,float('inf')))
    lo,li=older.max(1);hi,hi_idx=newer.min(1)
    valid=torch.isfinite(lo)&torch.isfinite(hi)
    alpha=torch.where(valid,(query-lo)/(hi-lo).clamp(min=1e-6),torch.zeros_like(query))
    rows=torch.arange(history.shape[0],device=history.device)
    out=history[rows,li]*(1-alpha[:,None,None])+history[rows,hi_idx]*alpha[:,None,None]
    return out,valid


class Rs01OmniV21Robot(Rs01OmniV20BoundedRobot):
    def _width_geometry_cost(self, widths, straight):
        cfg=self.cfg.rewards
        minimum=cfg.footprint_omni_min_width_m+straight*(cfg.footprint_straight_min_width_m-cfg.footprint_omni_min_width_m)
        return huber_positive((minimum[:,None]-widths)/cfg.footprint_soft_scale_m).mean(1)

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        if hasattr(self,'coord_stamps'):
            self.coord_stamps[env_ids]=-float('inf')
            self.coord_command[env_ids]=float('nan')
            self.coord_clock[env_ids]=0.

    def _footprint_quality(self):
        relative=self.feet_pos-self.root_states[:,None,:3]
        quat=self.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4)
        feet=quat_rotate_inverse(quat,relative.reshape(-1,3)).reshape(-1,4,3)
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        feet=feet[:,ids];raw=self.commands[:,:3];phase=self._gait_phase()
        widths=torch.stack((feet[:,0,1]-feet[:,1,1],feet[:,2,1]-feet[:,3,1]),1)
        # X relative to body; Z above flat world ground avoids matching body-roll artifacts.
        trajectory=torch.stack((feet[:,:,0],self.feet_pos[:,ids,2]),-1)
        if not hasattr(self,'coord_stamps'):
            self.coord_stamps=torch.full((self.num_envs,32),-float('inf'),device=self.device)
            self.coord_history=torch.zeros((self.num_envs,32,4,2),device=self.device)
            self.coord_command=torch.full_like(raw,float('nan'))
            self.coord_clock=torch.zeros_like(phase)
            self.coord_previous_phase=phase.clone()
        changed=~torch.isclose(raw,self.coord_command,atol=1e-4,rtol=0.).all(1)
        self.coord_stamps[changed]=-float('inf');self.coord_clock[changed]=0.
        delta=torch.remainder(phase-self.coord_previous_phase,1.)
        self.coord_clock+=torch.where(changed,torch.zeros_like(delta),delta)
        self.coord_previous_phase.copy_(phase);self.coord_command.copy_(raw)
        self.coord_stamps=torch.roll(self.coord_stamps,1,1)
        self.coord_history=torch.roll(self.coord_history,1,1)
        self.coord_stamps[:,0]=self.coord_clock;self.coord_history[:,0]=trajectory
        previous,valid=half_cycle_sample(self.coord_history,self.coord_stamps,self.coord_clock)
        error=(trajectory-previous[:,[1,0,3,2]]).abs()
        cfg=self.cfg.rewards
        dead=error.new_tensor([cfg.coordination_deadband_x_m,cfg.coordination_deadband_z_m])
        scale=error.new_tensor([cfg.coordination_scale_x_m,cfg.coordination_scale_z_m])
        coordination=huber_positive((error-dead)/scale).mean((1,2))
        straight=(1-raw[:,1].abs()/.08-raw[:,2].abs()/.20).clamp(0,1)
        gate=straight*torch.where(raw[:,0]<-.02,.5,1.)*valid.float()
        width_cost=self._width_geometry_cost(widths,straight)
        self.v16_foot_widths=widths.detach()
        self.v21_coordination_error=error.detach()
        self.v21_coordination_valid=valid
        # Replaces V19 mean-dx cost entirely; width and parent roll/contact costs stay.
        self.v19_placement_cost=width_cost+cfg.coordination_weight*gate*coordination
        return torch.ones_like(phase)
