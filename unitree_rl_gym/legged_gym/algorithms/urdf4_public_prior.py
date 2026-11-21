"""Stateless, exportable public centroidal prior for the calibrated fast gait.

Only the declared 1029 sensor/command history is used. This model supplies a
bounded NN's physical action prior, not privileged simulator state.
"""
import math
import torch
from torch import nn
from .urdf4_public_dynamics import PublicLegDynamics


class PublicCentroidalPrior(nn.Module):
    def __init__(self, urdf, joint_names, default_q, archive):
        super().__init__()
        self.dynamics=PublicLegDynamics(urdf,joint_names,default_q)
        stored=torch.load(archive,map_location='cpu') if isinstance(archive,str) or hasattr(archive,'read_bytes') else archive
        values=stored.get('upper',stored);lower=stored.get('lower',values)
        assert values['urdf_sha256']==self.dynamics.observer.urdf_sha256
        assert values['yaw_step_integral']==0., 'Use the public outer direction integrator only'
        for name in ('geometry','yaw_derivative','stand','leads','duties','action_scale'):
            self.register_buffer(name,values[name].float())
            if name!='action_scale':self.register_buffer('lower_'+name,lower[name].float())
        self.height=float(values['height']);self.period=float(values['period']);self.offset=float(values['offset'])
        self.feed=float(values['feed']);self.velocity_gain=float(values['velocity_gain'])
        self.yaw_gain=float(values['yaw_gain']);self.yaw_step=float(values['yaw_step'])
        self.alpha=float(values['alpha']);self.support_ramp=float(values['support_ramp'])
        assert lower['period']==values['period'] and lower['alpha']==values['alpha'] and lower['yaw_step_integral']==0.
        self.lower_height=float(lower['height']);self.lower_feed=float(lower['feed'])
        self.lower_velocity_gain=float(lower['velocity_gain']);self.lower_yaw_gain=float(lower['yaw_gain'])
        self.lower_yaw_step=float(lower['yaw_step'])
        self.lower_goal=float(stored.get('lower_goal',.4));self.upper_goal=float(stored.get('upper_goal',.45))

    @torch.jit.export
    def instantaneous_torque(self,frame,velocity,amplitude,phase,mixture):
        feet,jacobian,com,bias=self.dynamics(frame)
        height_velocity=self.dynamics.observer(frame)
        gravity=frame[:,3:6];gravity=gravity/gravity.norm(p=2,dim=1,keepdim=True).clamp(min=1.e-6)
        roll=torch.atan2(-gravity[:,1],-gravity[:,2]);pitch=torch.asin(gravity[:,0].clamp(-1,1))
        cr=torch.cos(roll);sr=torch.sin(roll);cp=torch.cos(pitch);sp=torch.sin(pitch)
        zero=torch.zeros_like(cr);one=torch.ones_like(cr)
        rotation=torch.stack((cp,sp*sr,sp*cr,zero,cr,-sr,-sp,cp*sr,cp*cr),1).reshape(-1,3,3)
        planned=frame[:,6:9]/torch.tensor([2.,2.,.25],device=frame.device,dtype=frame.dtype)
        gyro=frame[:,:3]/.25
        height=self.height+mixture*(self.lower_height-self.height)
        velocity_gain=self.velocity_gain+mixture*(self.lower_velocity_gain-self.velocity_gain)
        yaw_gain=self.yaw_gain+mixture*(self.lower_yaw_gain-self.yaw_gain)
        force=self.dynamics.total_mass*torch.cat((velocity_gain[:,None]*(planned[:,:2]-velocity),
            (9.81+100*(height-height_velocity[:,0])-15*height_velocity[:,1])[:,None]),1)
        moment_body=torch.stack((-8*roll-.8*gyro[:,0],-8*pitch-.8*gyro[:,1],yaw_gain*(planned[:,2]-gyro[:,2])),1)
        moment=(rotation@moment_body[:,:,None]).squeeze(-1)
        relative=(rotation[:,None,:,:]@(feet-com[:,None,:])[:,:,:,None]).squeeze(-1)
        x=relative[:,:,0];y=relative[:,:,1];z=relative[:,:,2];o=torch.zeros_like(x)
        skew=torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(-1,4,3,3)
        top=torch.eye(3,device=frame.device,dtype=frame.dtype).repeat(1,4).expand(frame.shape[0],-1,-1)
        matrix=torch.cat((top,skew.permute(0,2,1,3).flatten(2)),1)
        leads=self.leads+mixture[:,None]*(self.lower_leads-self.leads)
        duties=self.duties+mixture[:,None]*(self.lower_duties-self.duties)
        p=torch.remainder(phase[:,None]+leads/self.period+torch.tensor([0.,.5,.5,0.],device=frame.device,dtype=frame.dtype),1.)
        support=torch.minimum(p/self.support_ramp,(duties-p)/self.support_ramp).clamp(0,1)
        weight=((1-amplitude[:,None])+amplitude[:,None]*support).repeat_interleave(3,1)
        regularizer=torch.eye(6,device=frame.device,dtype=frame.dtype)*1.e-5
        wrench=torch.cat((force,moment),1)[:,:,None]
        contact=(weight[:,:,None]*(matrix.transpose(1,2)@torch.linalg.solve((matrix*weight[:,None,:])@matrix.transpose(1,2)+regularizer,wrench))).reshape(-1,4,3)
        fz=contact[:,:,2].clamp(0,100)
        contact=torch.cat((torch.maximum(torch.minimum(contact[:,:,:2],.6*fz[:,:,None]),-.6*fz[:,:,None]),fz[:,:,None]),2)
        body_force=(rotation.transpose(1,2)[:,None,:,:]@contact[:,:,:,None]).squeeze(-1)
        torque=bias-(jacobian.transpose(2,3)@body_force[:,:,:,None]).squeeze(-1).flatten(1)
        world_yaw=(rotation@gyro[:,:,None]).squeeze(-1)[:,2]
        return torque,world_yaw

    def forward(self,public):
        count=public.shape[0];history=public[:,:1020].reshape(count,20,51)
        phase=torch.remainder(torch.atan2(history[:,:,45],history[:,:,46])/(2*math.pi)-self.offset,1.)
        amplitude=public[:,1028].clamp(0,1)
        mixture=((self.upper_goal-public[:,1020])/(self.upper_goal-self.lower_goal)).clamp(0,1)
        velocity=public[:,None,1026:1028].expand(-1,20,-1)
        historical_amplitude=amplitude[:,None].expand(-1,20)
        if public.shape[1]>=1089:
            measured_history=public[:,1029:1089].reshape(count,20,3)
            velocity=measured_history[:,:,:2];historical_amplitude=measured_history[:,:,2]
        torque,yaw=self.instantaneous_torque(history.reshape(-1,51),velocity.reshape(-1,2),historical_amplitude.reshape(-1),phase.flatten(),mixture[:,None].expand(-1,20).reshape(-1))
        ages=torch.arange(20,device=public.device,dtype=public.dtype)
        weights=self.alpha*torch.pow(torch.ones_like(ages)*(1-self.alpha),ages);weights/=weights.sum()
        filtered=(torque.reshape(count,20,12)*weights[None,:,None]).sum(1).clamp(-10,10)
        yaw_alpha=.02/.27;yaw_weights=yaw_alpha*torch.pow(torch.ones_like(ages)*(1-yaw_alpha),ages);yaw_weights/=yaw_weights.sum()
        yaw_mean=(yaw.reshape(count,20)*yaw_weights).sum(1)
        planned_yaw=public[:,8]/.25
        yaw_step=self.yaw_step+mixture*(self.lower_yaw_step-self.yaw_step)
        yaw_trim=(yaw_step*(planned_yaw-yaw_mean)).clamp(-.3,.3)
        leads=self.leads+mixture[:,None]*(self.lower_leads-self.leads)
        index=torch.remainder(phase[:,0,None]+leads/self.period,1.)*self.geometry.shape[0]
        low=index.floor().long();high=low+1;high=torch.where(high>=self.geometry.shape[0],torch.zeros_like(high),high)
        blend=(index-index.floor())[:,:,None];leg=torch.arange(4,device=public.device)
        geom=(1-blend)*self.geometry.reshape(-1,4,3)[low,leg]+blend*self.geometry.reshape(-1,4,3)[high,leg]
        derivative=(1-blend)*self.yaw_derivative.reshape(-1,4,3)[low,leg]+blend*self.yaw_derivative.reshape(-1,4,3)[high,leg]
        lower_geom=(1-blend)*self.lower_geometry.reshape(-1,4,3)[low,leg]+blend*self.lower_geometry.reshape(-1,4,3)[high,leg]
        lower_derivative=(1-blend)*self.lower_yaw_derivative.reshape(-1,4,3)[low,leg]+blend*self.lower_yaw_derivative.reshape(-1,4,3)[high,leg]
        geom=geom+mixture[:,None,None]*(lower_geom-geom)
        derivative=derivative+mixture[:,None,None]*(lower_derivative-derivative)
        stand=self.stand+mixture[:,None]*(self.lower_stand-self.stand)
        feed=self.feed+mixture*(self.lower_feed-self.feed)
        desired=stand+amplitude[:,None]*(geom.flatten(1)+yaw_trim[:,None]*derivative.flatten(1)-stand)
        return (desired+feed[:,None]*filtered/40-self.dynamics.observer.default_q)/(.928*self.action_scale)


class ProtectedPublicPrior(nn.Module):
    def __init__(self,protected_policy,prior):
        super().__init__();self.protected=torch.jit.load(str(protected_policy));self.prior=prior
    def forward(self,public):
        mid=(public[:,1020]>.35)&(public[:,1020]<.5)&(public[:,1021:1023].abs().sum(1)<.001)
        return torch.where(mid[:,None],self.prior(public),self.protected(public))
