"""Command curriculum only; unchanged V25 controller and V27 geometry reward."""
import torch
from .rs01_omni_v25_env import Rs01OmniV25Robot


def curriculum_command(command, lateral_cap, reverse_caps):
    out=command.clone()
    lateral=(out[:,0].abs()<1e-7)&(out[:,2].abs()<1e-7)
    out[:,1]=torch.where(lateral,out[:,1].clamp(-lateral_cap,lateral_cap),out[:,1])
    reverse=(out[:,0]<0)&((out[:,1].abs()>1e-7)|(out[:,2].abs()>1e-7))
    norm=torch.linalg.vector_norm(out/out.new_tensor(reverse_caps),dim=1).clamp(min=1.)
    return torch.where(reverse[:,None],out/norm[:,None],out)


class Rs01OmniV28Robot(Rs01OmniV25Robot):
    def _set_command_modes(self, env_ids, modes):
        super()._set_command_modes(env_ids,modes)
        if len(env_ids):
            cfg=self.cfg.commands
            self.commands[env_ids,:3]=curriculum_command(self.commands[env_ids,:3],
                cfg.lateral_speed_range_m_s[1],cfg.reverse_combined_axis_caps)
            self._update_turn_mode(env_ids)

    def _command_gait_frequency(self):
        cfg=self.cfg.commands;r=self.cfg.rewards
        xcap=torch.where(self.commands[:,0]>=0,cfg.forward_velocity_range_m_s[1],cfg.backward_speed_range_m_s[1])
        intensity=torch.stack((self.commands[:,0].abs()/xcap,
            self.commands[:,1].abs()/cfg.cadence_lateral_reference_m_s,
            self.commands[:,2].abs()/cfg.yaw_speed_range_rad_s[1]),1).amax(1)
        blend=((intensity-r.wide_gait_speed_deadband)/(1.-r.wide_gait_speed_deadband)).clamp(0.,1.)
        old=1./r.gait_period_s+(r.wide_gait_max_frequency_hz-1./r.gait_period_s)*blend
        return old+self._directional_blend()*(r.lateral_frequency_floor_hz-old).clamp(min=0.)
