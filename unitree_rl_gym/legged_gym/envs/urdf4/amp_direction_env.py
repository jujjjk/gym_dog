"""AMP/CTS command learning inside an explicit external-localization path loop.

Only the localization simulator reads world pose. The student observes the same
1020 public values as before; pose enters through bounded twist corrections.
No position observation is smuggled into the history or the actor network.
"""
import torch
from .amp_expand_env import Urdf4AMPExpandRobot
from .amp_cts_env import Urdf4AMPCTSRobot,yaw_from_quat
from .urdf4_omni_env import Urdf4OmniRobot
from .path_tracking import PlanarPathController,PathGains,wrap
from .expand_support import command_classes


class Urdf4AMPDirectionRobot(Urdf4AMPExpandRobot):
    def _init_buffers(self):
        super()._init_buffers()
        n,dev=self.num_envs,self.device
        self.path=PlanarPathController(n,self.dt,dev,gains=PathGains(**self.cfg.direction.gains))
        self.external_commands=torch.zeros(n,3,device=dev)
        self.path_public_offset=torch.zeros(n,device=dev)
        self.path_origin=torch.zeros(n,2,device=dev)
        self.path_origin_yaw=torch.zeros(n,device=dev)
        self.pose_queue=torch.zeros(self.cfg.direction.pose_latency_steps+1,n,5,device=dev)
        self.direction_meter=torch.zeros(8,3,device=dev)
        self.direction_enabled=True

    def _resample_commands(self,ids):
        super()._resample_commands(ids)
        if not len(ids) or self.command_override is not None:return
        # Preserve each nonzero V2 direction; zero means stand, never march.
        cmd=self.target_commands[ids].clone()
        march=(cmd.abs().sum(1)<1.e-6)&self.target_gait[ids]
        valid=self.reference_commands.abs().sum(1)>1.e-6
        bank=self.reference_commands[valid]
        cmd[march]=bank[torch.randint(len(bank),(int(march.sum()),),device=self.device)]
        pure_forward=(cmd[:,0]>.0)&(cmd[:,1:].abs().sum(1)<1.e-6)
        cmd[pure_forward,0].clamp_(max=self.cfg.expand.stages[self.curriculum_stage][1])
        cmd[:,0]=torch.where(pure_forward,cmd[:,0].clamp(max=self.cfg.expand.stages[self.curriculum_stage][1]),cmd[:,0])
        self.target_commands[ids]=cmd

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if not len(ids) or not hasattr(self,'path'):return
        self.reset_path(ids)

    def reset_path(self,ids):
        xy=self.root_states[ids,:2].clone();yaw=yaw_from_quat(self.root_states[ids,3:7])
        self.path.reset_ids(ids,xy,yaw)
        self.external_commands[ids]=0.
        self.path_origin[ids]=xy;self.path_origin_yaw[ids]=yaw
        self.path_public_offset[ids]=wrap(self._public_yaw()[ids]-yaw)
        sample=torch.cat((xy,yaw[:,None],self.root_states[ids,7:9]),1)
        self.pose_queue[:,ids]=sample[None]

    def set_command_override(self,commands,gait_enabled=True):
        super().set_command_override(commands,gait_enabled)
        if hasattr(self,'external_commands'):self.external_commands.copy_(commands)

    def _localized_pose(self):
        sample=torch.cat((self.root_states[:,:2],yaw_from_quat(self.root_states[:,3:7])[:,None],self.root_states[:,7:9]),1)
        cfg=self.cfg.direction
        std=sample.new_tensor([cfg.position_noise_m,cfg.position_noise_m,
            cfg.yaw_noise_deg*torch.pi/180,cfg.velocity_noise_m_s,cfg.velocity_noise_m_s])
        if self.add_noise:sample=sample+torch.randn_like(sample)*std*(~self.nominal_mask)[:,None]
        self.pose_queue=torch.roll(self.pose_queue,-1,0);self.pose_queue[-1]=sample
        # Nominal replicas have exact diagnostic localization; robust episodes
        # assume a globally corrected, delayed external pose stream.
        measured=torch.where(self.nominal_mask[:,None],sample,self.pose_queue[0])
        return measured[:,:2],measured[:,2],measured[:,3:5]

    def _post_physics_step_callback(self):
        # The base callback updates contact/velocity statistics for the twist
        # actually executed, and resamples USER commands only. Bypass the old
        # expand callback that would integrate corrective yaw into the route.
        Urdf4OmniRobot._post_physics_step_callback(self)
        settling=self.cold_start&(self.episode_steps*self.dt<self.cfg.expand.cold_settle_s)
        limit=self.commands.new_tensor(self.cfg.expand.command_acceleration)*self.dt
        self.external_commands+=(self.target_commands-self.external_commands).clamp(-limit,limit)
        self.external_commands[settling]=0.
        self.gait_enabled[:]=~settling&(self.target_gait|(self.external_commands.abs().sum(1)>.001))
        self.cold_start&=settling
        xy,yaw,velocity=self._localized_pose()
        gyro=self.obs_history[:,0,2]/self.obs_scales.ang_vel
        corrected=self.path.update(self.external_commands,xy,yaw,velocity,gyro)
        if self.direction_enabled:self.commands[:,:3]=corrected
        else:self.commands[:,:3]=self.external_commands
        self.commands[settling,:3]=0.
        scale=self.commands.new_tensor([.3,.15,.5])
        self.reference_id[:]=torch.cdist(self.commands[:,:3]/scale,self.reference_commands/scale).argmin(1)
        self.heading_target[:]=self.path.reference_yaw
        self.heading_error[:]=wrap(self.heading_target-yaw_from_quat(self.root_states[:,3:7]))
        self.public_heading_target[:]=wrap(self.path.reference_yaw+self.path_public_offset)
        self.yaw_drift+=(~self.nominal_mask).float()*self.imu.gyro_bias[:,2]*self.dt
        self.yaw_drift.clamp_(-.035,.035)
        self.recovery_left.sub_(self.dt).clamp_(min=0.)

    def compute_reward(self):
        # Exactly the existing six terms, with one corrected-twist target.
        Urdf4AMPCTSRobot.compute_reward(self)
        stable=(self.external_commands-self.target_commands).abs().amax(1)<1.e-5
        mask=self.nominal_mask&~self.teacher_mask&self.gait_enabled&stable&(self.episode_steps>150)
        mask&=(self.external_commands[:,:2].norm(dim=1)>.03)|(self.external_commands[:,2].abs()>.08)
        groups=command_classes(self.external_commands)
        contact=self.get_foot_contact_mask();slots=[self.foot_legs.index(x) for x in ['FL','FR','RL','RR']];c=contact[:,slots]
        first=c[:,0]&c[:,3]&~c[:,1]&~c[:,2];second=c[:,1]&c[:,2]&~c[:,0]&~c[:,3]
        values=torch.stack((self.velocity_drift[:,:2].norm(dim=1),self.velocity_drift[:,2].abs(),
            (self._reward_illegal_support()>0).float(),(self.reset_buf.bool()&~self.time_out_buf).float(),torch.ones(self.num_envs,device=self.device),
            (contact!=self._scheduled_stance()).float().mean(1),first.float(),second.float()),1)
        self.curriculum_meter.index_add_(0,groups[mask],values[mask])
        direction=torch.stack((self.path.cross_error.abs(),self.path.heading_error.abs(),torch.ones(self.num_envs,device=self.device)),1)
        self.direction_meter.index_add_(0,groups[mask],direction[mask])

    def step(self,actions):
        previous_reset=self.reset_buf.bool().clone()
        user_target=self.target_commands.clone()
        enabled=self.gait_enabled.clone()
        result=super().step(actions)
        # Corrective twists vary every servo tick. They are NOT a new motion
        # class. Only real user-target/gait changes or resets break an AMP pair.
        valid=(~previous_reset&~result[3].bool()
            &torch.isclose(user_target,self.target_commands,atol=1.e-6,rtol=0).all(1)
            &(enabled==self.amp_terminal_enabled))
        a=result[4]['amp'];a['task_valid']=valid
        a['valid']=valid&enabled&self.amp_terminal_enabled
        a['style_command']=user_target
        return result
