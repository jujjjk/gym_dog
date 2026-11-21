"""Expose existing corrected-localization velocity and the public command plan."""
import torch
from .amp_v3_env import Urdf4AMPV3Robot,V3PathController
from .path_tracking import rotate_xy
from .gait_clock import CycleMean


class V4PathController(V3PathController):
    def __init__(self,*args,speed_enabled=False,speed_settings=None,**kwargs):
        super().__init__(*args,**kwargs);self.speed_enabled=speed_enabled;self.speed_settings=speed_settings or {}
        self.speed_integral=torch.zeros(self.n,device=self.device);self.speed_bias=torch.zeros_like(self.speed_integral)
        self.speed_applied=torch.zeros(self.n,2,device=self.device);self.command_output=torch.zeros(self.n,3,device=self.device)
        self.cross_yaw=torch.zeros(self.n,device=self.device)

    def reset_ids(self,ids,position,yaw):
        super().reset_ids(ids,position,yaw)
        self.speed_integral[ids]=0.;self.speed_bias[ids]=0.;self.speed_applied[ids]=0.;self.command_output[ids]=0.
        self.cross_yaw[ids]=0.

    def update(self,external,position,yaw,velocity_world,yaw_rate):
        output=super().update(external,position,yaw,velocity_world,yaw_rate)
        if not self.speed_enabled:self.speed_applied.zero_();self.command_output.copy_(output);return output
        c=self.speed_settings;active=(external[:,0]>.35)&(external[:,1:].abs().sum(1)<.001)
        parallel=rotate_xy(-self.reference_yaw,self.filtered_velocity)[:,0];error=external[:,0]-parallel
        limit=c['correction_limit'];saturated=(self.speed_bias.abs()>=limit-1e-6)&(self.speed_bias*error>0)
        self.speed_integral+=c['integral']*error*self.dt*(active&~saturated).float();self.speed_integral.clamp_(-limit,limit)
        self.speed_integral[~active]=0.
        desired=(c['proportional']*error+self.speed_integral).clamp(-limit,limit)*active.float()
        step=c['correction_acceleration']*self.dt;self.speed_bias+=(desired-self.speed_bias).clamp(-step,step)
        stopped=external.abs().sum(1)<.001;self.speed_bias[stopped]=0.
        # Fast forward cannot sidestep as readily as the low-speed gait.
        # Couple cross-track recovery to the existing yaw servo as well:
        # 2 (rad/s)/m, bounded to 0.12 rad/s and the original yaw slew limit.
        desired_yaw=(-2.*self.cross_error).clamp(-.12,.12)*active.float()
        self.cross_yaw+=(desired_yaw-self.cross_yaw).clamp(-self.gains.yaw_acceleration*self.dt,self.gains.yaw_acceleration*self.dt)
        self.cross_yaw[stopped]=0.
        output[:,2]=(output[:,2]+self.cross_yaw).clamp(-self.gains.output_yaw_limit_rad_s,self.gains.output_yaw_limit_rad_s)
        along=torch.stack((self.speed_bias,torch.zeros_like(self.speed_bias)),1)
        correction=rotate_xy(self.reference_yaw-yaw,along);before=output[:,:2].clone();output[:,:2]+=correction
        output[:,0].clamp_(-.45,c['output_x_limit']);output[:,1].clamp_(-.30,.30)
        self.speed_applied[:]=output[:,:2]-before;self.command_output.copy_(output)
        return output


class Urdf4AMPV4Robot(Urdf4AMPV3Robot):
    def _init_buffers(self):
        self.motion_goal=torch.zeros(self.num_envs,3,device=self.device)
        self.measured_planar_velocity=torch.zeros(self.num_envs,2,device=self.device)
        self.speed_bias_cycle=CycleMean(self.num_envs,2,25,self.device)
        public=self.cfg.env.public_obs_size;self.cfg.env.public_obs_size=1020
        try:super()._init_buffers()
        finally:self.cfg.env.public_obs_size=public
        self.path=V4PathController(self.num_envs,self.dt,self.device,gains=self.path.gains,
            speed_enabled=self.cfg.direction.speed_feedback_enabled,speed_settings=self.cfg.direction.speed_feedback)

    def _localized_pose(self):
        xy,yaw,velocity=super()._localized_pose()
        self.measured_planar_velocity[:]=rotate_xy(-yaw,velocity)
        return xy,yaw,velocity

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if len(ids):self.motion_goal[ids]=self.target_commands[ids];self.speed_bias_cycle.reset(ids)

    def _post_physics_step_callback(self):
        previous=getattr(self.path,'speed_applied',torch.zeros_like(self.measured_planar_velocity)).clone()
        self.speed_bias_cycle.push(previous)
        super()._post_physics_step_callback()
        if isinstance(self.path,V4PathController) and self.path.speed_enabled:
            self.commands[:,0]=self.path.command_output[:,0]

    def compute_reward(self):
        # A speed-setpoint bias compensates the inverse controller. It must
        # not change the user's longitudinal task objective or curriculum.
        correction=self.speed_bias_cycle.mean();self.velocity_drift[:,:2]-=correction
        try:super().compute_reward()
        finally:self.velocity_drift[:,:2]+=correction

    def compute_observations(self):
        super().compute_observations();old=self.obs_buf
        active=self.target_commands.abs().sum(1)>.001
        self.motion_goal[active]=self.target_commands[active]
        amplitude=(self.external_commands.norm(dim=1)/self.motion_goal.norm(dim=1).clamp(min=1e-6)).clamp(0,1)
        extra=torch.cat((self.motion_goal,self.external_commands,self.measured_planar_velocity,amplitude[:,None]),1)
        self.obs_buf=torch.cat((old[:,:1020],extra,old[:,1020:]),1)
        self.privileged_obs_buf=self.obs_buf
