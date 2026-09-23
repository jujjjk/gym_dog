"""Causal, independent motor/IMU acquisition queues on the physics clock."""
import math
import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul, quat_rotate_inverse


class SensorSnapshot:
    def __init__(self, n, device, dt, cfg):
        self.n, self.device, self.dt, self.cfg = n, device, dt, cfg
        self.size = 32  # 80 ms at 2.5 ms: exceeds 40 ms transport + 10 ms sampling.
        self.motor = torch.zeros(n, self.size, 36, device=device)
        self.imu = torch.zeros(n, self.size, 7, device=device)
        self.ms = torch.full((n,self.size), -1000000000, device=device, dtype=torch.long)
        self.is_ = self.ms.clone(); self.ma = self.ms.clone(); self.ia = self.ms.clone()
        self.motor_value = torch.zeros(n,36,device=device)
        self.imu_value = torch.zeros(n,7,device=device); self.imu_value[:,6]=1.
        self.motor_stamp = torch.zeros(n,device=device,dtype=torch.long)
        self.imu_stamp = self.motor_stamp.clone()
        self.clean = torch.ones(n,device=device,dtype=torch.bool)
        self.robust = torch.zeros_like(self.clean)
        self.mount = torch.zeros(n,4,device=device); self.mount[:,3]=1.
        self.bias = torch.zeros(n,3,device=device)
        self.motor_delay = self.motor_stamp.clone(); self.imu_delay=self.motor_stamp.clone()
        self.motor_phase=self.motor_stamp.clone(); self.imu_phase=self.motor_stamp.clone()
        self.motor_period=max(1,round(cfg.motor_period_s/dt))
        self.imu_period=max(1,round(cfg.imu_period_s/dt))

    def reset(self, ids, tick, q, dq, quat, gyro, torque):
        count=len(ids)
        if not count: return
        c=self.cfg
        self.clean[ids] = (torch.rand(count,device=self.device)<c.clean_fraction) if c.enabled else True
        self.robust[ids] = (~self.clean[ids]) & (torch.rand(count,device=self.device)<c.robust_fraction)
        amplitude=torch.where(self.robust[ids],c.mount_robust_deg,c.mount_normal_deg)*math.pi/180
        angles=(2*torch.rand(count,2,device=self.device)-1)*amplitude[:,None]
        angles[self.clean[ids]]=0
        self.mount[ids]=quat_from_euler_xyz(angles[:,0],angles[:,1],torch.zeros(count,device=self.device))
        self.bias[ids]=(2*torch.rand(count,3,device=self.device)-1)*c.gyro_bias_rad_s
        self.bias[ids[self.clean[ids]]]=0.
        md=torch.where(self.robust[ids],c.motor_robust_max_s,c.motor_normal_max_s)
        im=torch.where(self.robust[ids],c.imu_robust_max_s,c.imu_normal_max_s)
        self.motor_delay[ids]=torch.round(torch.rand(count,device=self.device)*md/self.dt).long()
        self.imu_delay[ids]=torch.round((c.imu_min_s+torch.rand(count,device=self.device)*(im-c.imu_min_s))/self.dt).long()
        self.motor_delay[ids[self.clean[ids]]]=0;self.imu_delay[ids[self.clean[ids]]]=0
        self.motor_phase[ids]=torch.randint(self.motor_period,(count,),device=self.device)
        self.imu_phase[ids]=torch.randint(self.imu_period,(count,),device=self.device)
        self.ms[ids]=-1000000000;self.is_[ids]=-1000000000
        self.ma[ids]=-1000000000;self.ia[ids]=-1000000000
        # Bootstrap at reset time, no invented pre-reset packet history.
        m,i=self.measure(q,dq,quat,gyro,torque)
        self.motor_value[ids]=m[ids];self.imu_value[ids]=i[ids]
        self.motor_stamp[ids]=tick;self.imu_stamp[ids]=tick

    def measure(self,q,dq,quat,gyro,torque):
        c=self.cfg; active=(~self.clean).float()[:,None]
        qn=(torch.randn_like(q)*c.q_sigma).clamp(-c.q_clip,c.q_clip)*active
        dclip=torch.where(self.robust,c.dq_robust_clip,c.dq_clip)[:,None]
        dn=torch.maximum(torch.minimum(torch.randn_like(dq)*c.dq_sigma,dclip),-dclip)*active
        gn=(torch.randn_like(gyro)*c.gyro_sigma).clamp(-c.gyro_clip,c.gyro_clip)*active
        orientation=quat_mul(quat,self.mount)
        measured_gyro=quat_rotate_inverse(self.mount,gyro)+self.bias+gn
        return torch.cat((q+qn,dq+dn,torque),1),torch.cat((measured_gyro,orientation),1)

    def acquire(self,tick,q,dq,quat,gyro,torque):
        m,i=self.measure(q,dq,quat,gyro,torque)
        mdue=self.clean | ((tick+self.motor_phase)%self.motor_period==0)
        idue=self.clean | ((tick+self.imu_phase)%self.imu_period==0)
        slot=tick%self.size
        self.motor[:,slot]=m;self.imu[:,slot]=i
        self.ms[:,slot]=torch.where(mdue,tick,-1000000000)
        self.is_[:,slot]=torch.where(idue,tick,-1000000000)
        self.ma[:,slot]=tick+self.motor_delay;self.ia[:,slot]=tick+self.imu_delay

    def read(self,tick):
        for value,stamp,data,stamps,arrivals in (
            (self.motor_value,self.motor_stamp,self.motor,self.ms,self.ma),
            (self.imu_value,self.imu_stamp,self.imu,self.is_,self.ia)):
            eligible=torch.where(arrivals<=tick,stamps,-1000000000)
            latest,index=eligible.max(1)
            update=latest>stamp
            selected=data[torch.arange(self.n,device=self.device),index]
            value.copy_(torch.where(update[:,None],selected,value))
            stamp.copy_(torch.maximum(stamp,latest))
        self.q=self.motor_value[:,:12];self.dq=self.motor_value[:,12:24]
        self.torque=self.motor_value[:,24:36]
        self.gyro=self.imu_value[:,:3];self.quat=self.imu_value[:,3:7]
        gravity=torch.zeros(self.n,3,device=self.device);gravity[:,2]=-1.
        self.gravity=quat_rotate_inverse(self.quat,gravity)
        x,y,z,w=self.quat.unbind(1)
        self.yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.motor_age_s=(tick-self.motor_stamp)*self.dt
        self.imu_age_s=(tick-self.imu_stamp)*self.dt
        self.skew_s=(self.motor_stamp-self.imu_stamp)*self.dt


def assemble61(sensor,velocity,confidence,target,command,default,scales,command_scale,previous,phase,heading,gait):
    """No simulator state accepted here. Slots match B18000/A20500."""
    n=velocity.shape[0]; zero=torch.zeros(n,device=velocity.device)
    return torch.cat((velocity*scales.lin_vel,sensor.gyro*scales.ang_vel,sensor.gravity,
        target*command_scale,(sensor.q-default)*scales.dof_pos,sensor.dq*scales.dof_vel,previous,
        torch.stack((torch.sin(2*math.pi*phase),torch.cos(2*math.pi*phase)),1),
        torch.stack((torch.sin(heading),torch.cos(heading),zero,
            ((velocity[:,1]-target[:,1])*2).clamp(-10,10),gait,zero,
            ((velocity[:,0]-target[:,0])*2).clamp(-10,10)),1),
        command*command_scale,confidence.clamp(0,1)[:,None]),1)
