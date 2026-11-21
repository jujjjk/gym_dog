"""Keep one calibrated fast clock; measured command-conditioned feedback steers."""
import torch
from .amp_v4_env import Urdf4AMPV4Robot,V4PathController
from .path_tracking import rotate_xy


class V5PathController(V4PathController):
    def __init__(self,*args,goal,mid_gain,mid_damping,mid_limit,mid_integral=0.,**kwargs):
        super().__init__(*args,**kwargs);self.goal=goal;self.mid_gain=mid_gain;self.mid_damping=mid_damping;self.mid_limit=mid_limit
        self.mid_integral=mid_integral
        self.mid_yaw=torch.zeros(self.n,device=self.device)

    def reset_ids(self,ids,position,yaw):
        super().reset_ids(ids,position,yaw);self.mid_yaw[ids]=0.

    def update(self,external,position,yaw,velocity_world,yaw_rate):
        output=super().update(external,position,yaw,velocity_world,yaw_rate)
        active=(self.goal[:,0]>.35)&(self.goal[:,0]<.5)&(self.goal[:,1:].abs().sum(1)<.001)
        normal_velocity=rotate_xy(-self.reference_yaw,self.filtered_velocity)[:,1]
        recovery=(-self.mid_gain*self.cross_error-self.mid_damping*normal_velocity).clamp(-self.mid_limit,self.mid_limit)*active.float()
        # Cross-track yaw recovery is a desired heading correction. Integrating
        # route-yaw error alone otherwise cancels it as a persistent yaw bias.
        # Match the heading integral to the combined V4+V5 cross recovery;
        # keep the fixed route and all non-mid-speed commands unchanged.
        offset=(-(2.+self.mid_gain)*self.cross_error/self.gains.heading_gain).clamp(-.06,.06)
        self.heading_integral+=self.mid_integral*offset*self.dt*active.float()
        self.heading_integral.clamp_(-self.gains.heading_integral_limit_rad_s,self.gains.heading_integral_limit_rad_s)
        slew=self.gains.yaw_acceleration*self.dt;self.mid_yaw+=(recovery-self.mid_yaw).clamp(-slew,slew)
        self.mid_yaw[external.abs().sum(1)<.001]=0.;output[:,2]+=self.mid_yaw
        output[:,2].clamp_(-self.gains.output_yaw_limit_rad_s,self.gains.output_yaw_limit_rad_s)
        self.command_output.copy_(output)
        return output


class Urdf4AMPV5Robot(Urdf4AMPV4Robot):
    def _init_buffers(self):
        self.feedback_history=torch.zeros(self.num_envs,20,3,device=self.device)
        self.feedback_history_stale=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        super()._init_buffers();self.path=V5PathController(self.num_envs,self.dt,self.device,gains=self.path.gains,
            speed_enabled=self.cfg.direction.speed_feedback_enabled,speed_settings=self.cfg.direction.speed_feedback,
            goal=self.target_commands,mid_gain=self.cfg.direction.mid_cross_yaw_gain,mid_damping=self.cfg.direction.mid_cross_yaw_damping,
            mid_limit=self.cfg.direction.mid_cross_yaw_limit_rad_s,mid_integral=self.cfg.direction.mid_cross_integral_gain)

    def reset_idx(self,ids):
        super().reset_idx(ids);self.feedback_history_stale[ids]=True

    def compute_observations(self):
        super().compute_observations()
        if not self.cfg.amp_cts.public_feedback_history:return
        current=self.obs_buf[:,1026:1029]
        self.feedback_history=torch.roll(self.feedback_history,1,1);self.feedback_history[:,0]=current
        stale=self.feedback_history_stale
        self.feedback_history[stale]=current[stale,None,:];stale[:]=False
        self.obs_buf=torch.cat((self.obs_buf[:,:1029],self.feedback_history.flatten(1),self.obs_buf[:,1029:]),1)
        self.privileged_obs_buf=self.obs_buf

    def _select_reference(self,command):
        ids=super()._select_reference(command)
        fast=(command[:,0]>.35)&(command[:,1:].abs().sum(1)<.001)
        selected=(self.reference_commands-command.new_tensor([.5,0.,0.])).square().sum(1).argmin()
        ids[fast]=selected
        return ids

    def _resample_commands(self,ids):
        super()._resample_commands(ids)
        if not len(ids) or self.command_override is not None or self.curriculum_stage<1:return
        chosen=ids[torch.rand(len(ids),device=self.device)<.25]
        self.target_commands[chosen]=0.;self.target_commands[chosen,0]=.36+.14*torch.rand(len(chosen),device=self.device)
        self.target_gait[chosen]=True;self.command_frontier[chosen]=True
