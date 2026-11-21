"""Planar path servo and public-history dead reckoning; no simulator imports.

Pose inputs are in a fixed frame established at the start of the route.
Reference heading follows USER yaw commands, never corrective yaw commands.
"""
from dataclasses import dataclass,asdict
import torch


def wrap(angle):
    return torch.atan2(angle.sin(),angle.cos())


def rotate_xy(yaw,xy):
    c,s=yaw.cos(),yaw.sin()
    return torch.stack((c*xy[:,0]-s*xy[:,1],s*xy[:,0]+c*xy[:,1]),1)


@dataclass
class PathGains:
    cross_gain:float=.8
    cross_damping:float=.2
    heading_gain:float=2.5
    yaw_damping:float=0.
    lateral_limit_m_s:float=.04
    yaw_limit_rad_s:float=.15
    filter_tau_s:float=.5
    update_period_s:float=.10
    lateral_acceleration:float=.15
    yaw_acceleration:float=.8
    arc_along_gain:float=1.2
    arc_speed_damping:float=.2
    along_limit_m_s:float=.06
    straight_along_gain:float=0.
    straight_speed_damping:float=0.
    arc_left_bias:tuple=(0.,0.,0.)
    arc_right_bias:tuple=(0.,0.,0.)
    transverse_cross_gain:float=0.
    nonforward_heading_integral_gain:float=0.
    heading_integral_limit_rad_s:float=.10
    output_yaw_limit_rad_s:float=.40
    left_command_scale:float=1.
    turn_position_gain:float=0.
    nonforward_yaw_limit_rad_s:float=0.
    output_lateral_limit_m_s:float=.10
    forward_heading_integral_gain:float=0.


class PlanarPathController:
    def __init__(self,num_envs,dt,device,mode=2,gains=None):
        self.n,self.dt,self.device=num_envs,float(dt),device
        self.gains=gains or PathGains()
        self.mode=torch.as_tensor(mode,device=device).expand(num_envs).clone()
        self.reference_xy=torch.zeros(num_envs,2,device=device)
        self.reference_yaw=torch.zeros(num_envs,device=device)
        self.reference_yaw_rate=torch.zeros_like(self.reference_yaw)
        self.filtered_yaw=torch.zeros_like(self.reference_yaw)
        self.filtered_reference_yaw=torch.zeros_like(self.reference_yaw)
        self.filtered_rate=torch.zeros_like(self.reference_yaw)
        self.filtered_velocity=torch.zeros_like(self.reference_xy)
        self.last_external=torch.zeros(num_envs,3,device=device)
        self.output=torch.zeros_like(self.last_external)
        self.desired=torch.zeros_like(self.output)
        self.cross_error=torch.zeros_like(self.reference_yaw)
        self.heading_error=torch.zeros_like(self.reference_yaw)
        self.heading_integral=torch.zeros_like(self.reference_yaw)
        self.steps=0
        self.initialized=False

    def reset(self,position,yaw):
        self.reference_xy.copy_(position);self.reference_yaw.copy_(yaw)
        self.reference_yaw_rate.zero_()
        self.filtered_yaw.copy_(yaw);self.filtered_velocity.zero_();self.filtered_rate.zero_()
        self.filtered_reference_yaw.copy_(yaw)
        self.output.zero_();self.desired.zero_();self.last_external.zero_()
        self.cross_error.zero_();self.heading_error.zero_()
        self.heading_integral.zero_()
        self.steps=0;self.initialized=True

    def reset_ids(self,ids,position,yaw):
        """Reset selected episodes without altering other robots' references."""
        self.reference_xy[ids]=position;self.reference_yaw[ids]=yaw
        self.filtered_yaw[ids]=yaw;self.filtered_reference_yaw[ids]=yaw
        for value in (self.reference_yaw_rate,self.filtered_rate,self.filtered_velocity,
                      self.last_external,self.output,self.desired,self.cross_error,
                      self.heading_error,self.heading_integral):value[ids]=0.
        self.initialized=True

    def update(self,external,position,yaw,velocity_world,yaw_rate):
        if not bool(torch.isfinite(torch.cat((external,position,yaw[:,None],velocity_world,yaw_rate[:,None]),1)).all()):
            raise ValueError('Path inputs contain NaN or Inf')
        if not self.initialized:self.reset(position,yaw)
        g=self.gains;dt=self.dt
        moving=external[:,:2].norm(dim=1)>.02
        previous=self.last_external[:,:2].norm(dim=1)>.02
        unit=external[:,:2]/external[:,:2].norm(dim=1,keepdim=True).clamp(min=1.e-6)
        old_unit=self.last_external[:,:2]/self.last_external[:,:2].norm(dim=1,keepdim=True).clamp(min=1.e-6)
        changed=(moving&~previous)|(((unit*old_unit).sum(1)<.98)&moving&previous)|((external[:,2]-self.last_external[:,2]).abs()>.02)
        self.reference_xy[changed]=position[changed]
        alpha=dt/(g.filter_tau_s+dt)
        self.filtered_yaw+=alpha*wrap(yaw-self.filtered_yaw)
        self.filtered_reference_yaw+=alpha*wrap(self.reference_yaw-self.filtered_reference_yaw)
        self.filtered_rate+=alpha*(yaw_rate-self.filtered_rate)
        self.filtered_velocity+=alpha*(velocity_world-self.filtered_velocity)
        world_target=rotate_xy(self.reference_yaw,external[:,:2])
        tangent=world_target/world_target.norm(dim=1,keepdim=True).clamp(min=1.e-6)
        normal=torch.stack((-tangent[:,1],tangent[:,0]),1)
        self.cross_error=((position-self.reference_xy)*normal).sum(1)
        stationary_turn=~moving&(external[:,2].abs()>.02)
        self.cross_error=torch.where(stationary_turn,(position-self.reference_xy).norm(dim=1),self.cross_error)
        self.heading_error=wrap(self.filtered_reference_yaw-self.filtered_yaw)
        pure_forward=(external[:,0]>.02)&(external[:,1].abs()<1.e-6)&(external[:,2].abs()<1.e-6)
        integrating=(self.mode>0)&(~pure_forward|(g.forward_heading_integral_gain>0))&((external.abs().sum(1)>.02))
        # Integral action removes a persistent turn-rate bias without changing
        # the already qualified pure-forward profile. Freeze at output saturation.
        saturated=(self.output[:,2].abs()>=g.output_yaw_limit_rad_s-1.e-5)&(self.output[:,2]*self.heading_error>0)
        integral_gain=torch.where(pure_forward,g.forward_heading_integral_gain,g.nonforward_heading_integral_gain)
        self.heading_integral+=integral_gain*self.heading_error*dt*(integrating&~saturated).float()
        self.heading_integral.clamp_(-g.heading_integral_limit_rad_s,g.heading_integral_limit_rad_s)
        self.heading_integral[~integrating]=0.
        self.reference_yaw_rate+=(external[:,2]-self.reference_yaw_rate).clamp(-g.yaw_acceleration*dt,g.yaw_acceleration*dt)
        if self.steps%max(1,round(g.update_period_s/dt))==0 or bool(changed.any()):
            transverse=(external[:,2].abs()<.02)&((external[:,0]<=0)|(external[:,1].abs()>.02))
            cross_gain=torch.where(transverse,g.transverse_cross_gain if g.transverse_cross_gain>0 else g.cross_gain,g.cross_gain)
            cross=(-cross_gain*self.cross_error-g.cross_damping*(self.filtered_velocity*normal).sum(1)).clamp(-g.lateral_limit_m_s,g.lateral_limit_m_s)
            # Straight segments constrain cross-track only. Arcs also need
            # along-track correction to retain their commanded radius/timing.
            along_error=((position-self.reference_xy)*tangent).sum(1)
            along_speed=((self.filtered_velocity-world_target)*tangent).sum(1)
            arc=external[:,2].abs()>.02
            forward=(external[:,0]>0)&(external[:,1].abs()<1.e-6)
            along=-(torch.where(arc,g.arc_along_gain,torch.where(forward,g.straight_along_gain,0.))*along_error+
                torch.where(arc,g.arc_speed_damping,torch.where(forward,g.straight_speed_damping,0.))*along_speed)
            along=along.clamp(-g.along_limit_m_s,g.along_limit_m_s)
            current_yaw=self.filtered_yaw+wrap(self.reference_yaw-self.filtered_reference_yaw)
            # While following an arc, yaw alignment has priority. Do not ask
            # the gait to sidestep aggressively to compensate a heading lag.
            current_yaw=torch.where(arc,self.reference_yaw,current_yaw)
            position_gate=torch.where(arc,(1-self.heading_error.abs()/.10).clamp(0,1),torch.ones_like(arc,dtype=external.dtype))
            corrected=rotate_xy(-current_yaw,world_target+(normal*cross[:,None]+tangent*along[:,None])*position_gate[:,None])
            turn_in_place=~moving&(external[:,2].abs()>.02)
            hold=-g.turn_position_gain*(position-self.reference_xy)
            hold*= (g.lateral_limit_m_s/hold.norm(dim=1).clamp(min=g.lateral_limit_m_s))[:,None]
            corrected=torch.where(turn_in_place[:,None],rotate_xy(-current_yaw,hold),corrected)
            turn_limit=torch.where(pure_forward,g.yaw_limit_rad_s,
                g.nonforward_yaw_limit_rad_s if g.nonforward_yaw_limit_rad_s>0 else g.yaw_limit_rad_s)
            turn=g.heading_gain*self.heading_error+self.heading_integral-g.yaw_damping*(self.filtered_rate-self.reference_yaw_rate)
            turn=torch.maximum(torch.minimum(turn,turn_limit),-turn_limit)
            self.desired[:,:2]=torch.where((self.mode==2)[:,None],corrected,external[:,:2])
            self.desired[:,2]=self.reference_yaw_rate+torch.where(self.mode>0,turn,torch.zeros_like(turn))
            # Optional measured feedforward calibration of the FROZEN gait.
            # These body-command offsets are tied to the policy hash/profile.
            load=(external[:,0]/.22).clamp(0,1)*(external[:,2].abs()/.30).clamp(0,1)*(self.mode==2).float()
            bias=torch.where((external[:,2]>0)[:,None],external.new_tensor(g.arc_left_bias)[None],external.new_tensor(g.arc_right_bias)[None])*load[:,None]
            bias[:,2]*=(self.reference_yaw_rate.abs()/external[:,2].abs().clamp(min=.02)).clamp(max=1)
            self.desired+=bias
            pure_left=(external[:,0].abs()<1.e-6)&(external[:,1]>.02)&(external[:,2].abs()<1.e-6)&(self.mode==2)
            self.desired[pure_left,1]*=g.left_command_scale
            scale=(g.output_lateral_limit_m_s/self.desired[:,1].abs().clamp(min=g.output_lateral_limit_m_s)).clamp(max=1.)
            self.desired[:,:2]*=scale[:,None]
        limit=external.new_tensor([g.lateral_acceleration,g.lateral_acceleration,g.yaw_acceleration])*dt
        self.output+=(self.desired-self.output).clamp(-limit,limit)
        # Startup forward acceleration is already handled by the unchanged
        # actuator target limits; only corrective components need extra slew.
        self.output[:,0]=self.desired[:,0]
        self.output[:,1].clamp_(-g.output_lateral_limit_m_s,g.output_lateral_limit_m_s);self.output[:,2].clamp_(-g.output_yaw_limit_rad_s,g.output_yaw_limit_rad_s)
        self.output[self.mode==0]=external[self.mode==0]
        stopped=~moving&(external[:,2].abs()<.02)
        self.output[stopped]=0.;self.desired[stopped]=0.
        self.reference_xy+=world_target*dt
        reference_rate=torch.where(self.mode==0,external[:,2],self.reference_yaw_rate)
        self.reference_yaw=wrap(self.reference_yaw+reference_rate*dt)
        self.last_external.copy_(external);self.steps+=1
        return self.output.clone()

    def metadata(self):
        return dict(gains=asdict(self.gains),frame='fixed route-start frame',
            reference_heading='integral of slew-limited external yaw only',modes=['off','heading','path'])


class StudentDeadReckoning:
    """Diagnostic odometry from CTS-predicted velocity and public heading error.

    This estimator has NO absolute position correction. Its drift must be
    measured; it is not a replacement for externally corrected odometry.
    """
    def __init__(self,encoder,num_envs,dt,device):
        self.encoder=encoder;self.dt=float(dt)
        self.position=torch.zeros(num_envs,2,device=device)

    def update(self,public_history,reference_yaw):
        if public_history.shape[1]!=1020:raise ValueError('Expected only public 1020D history')
        frame=public_history[:,:51]
        error=torch.atan2(frame[:,48],frame[:,49]+1.)
        yaw=wrap(reference_yaw-error)
        body=self.encoder(public_history)[:,:3]
        velocity=rotate_xy(yaw,body[:,:2])
        position=self.position.clone()
        self.position+=velocity*self.dt
        return position,yaw,velocity


class FixedOdometryFrame:
    """Align an absolute external odometry stream once, without chasing drift."""
    def __init__(self,origin_xy,origin_yaw):
        self.origin_xy=origin_xy.clone();self.origin_yaw=origin_yaw.clone()

    def transform(self,position_xy,yaw,velocity_world):
        return (rotate_xy(-self.origin_yaw,position_xy-self.origin_xy),
                wrap(yaw-self.origin_yaw),rotate_xy(-self.origin_yaw,velocity_world))
