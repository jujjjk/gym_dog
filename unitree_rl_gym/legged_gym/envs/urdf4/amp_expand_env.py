"""Continuous commands, observable heading and light robustness for AMP/CTS."""
import torch
from isaacgym import gymapi,gymtorch
from .amp_cts_env import Urdf4AMPCTSRobot,yaw_from_quat
from .disturbances import quat_mul,quat_from_rpy,quat_rotate_inverse
from .expand_support import reference_coverage,command_classes

class Urdf4AMPExpandRobot(Urdf4AMPCTSRobot):
    def _nominal(self,ids):
        if self.cfg.env.test or not self.cfg.expand.robust_training:return torch.ones_like(ids,dtype=torch.bool)
        return ids%self.cfg.expand.nominal_stride==0

    def _init_buffers(self):
        n,dev=self.num_envs,self.device
        self.nominal_mask=self._nominal(torch.arange(n,device=dev))
        self.curriculum_stage=min(getattr(self.cfg.expand,'initial_stage',0),self.cfg.expand.max_stage);self.curriculum_pass_windows=0
        self.curriculum_meter=torch.zeros(8,8,device=dev)
        self.target_commands=torch.zeros(n,3,device=dev)
        self.target_gait=torch.ones(n,device=dev,dtype=torch.bool)
        self.command_frontier=torch.zeros(n,device=dev,dtype=torch.bool)
        self.cold_start=torch.ones(n,device=dev,dtype=torch.bool)
        self.recovery_left=torch.zeros(n,device=dev)
        self.public_heading_target=torch.zeros(n,device=dev)
        self.public_heading_error=torch.zeros(n,device=dev)
        self.yaw_drift=torch.zeros(n,device=dev)
        super()._init_buffers()
        # The validated bank retains the longer original delay queue. Keep its
        # full chronology even though this stage samples a narrower delay range.
        if self.reference_bank is not None:
            length=self.reference_bank['snapshots']['delay_buffer'].shape[1]
            if self.delay_buffer.shape[0]<length:
                self.delay_buffer=self.default_dof_pos.expand(length,n,-1).clone()

    def _process_rigid_shape_props(self,props,env_id):
        props=super()._process_rigid_shape_props(props,env_id)
        if self.cfg.env.test or not self.cfg.expand.robust_training or env_id%self.cfg.expand.nominal_stride==0:
            self._shape_friction[env_id]=self.cfg.terrain.static_friction
            self._shape_restitution[env_id]=0.
            for p in props:p.friction=self.cfg.terrain.static_friction;p.restitution=0.
        return props

    def _process_rigid_body_props(self,props,env_id):
        original=[(p.mass,p.com.x,p.com.y,p.com.z) for p in props]
        props=super()._process_rigid_body_props(props,env_id)
        if self.cfg.env.test or not self.cfg.expand.robust_training or env_id%self.cfg.expand.nominal_stride==0:
            for p,(mass,x,y,z) in zip(props,original):p.mass=mass;p.com=gymapi.Vec3(x,y,z)
            self._payload_kg[env_id]=0.;self._com_offset_m[env_id]=0.
        return props

    def _randomize_episode(self,ids):
        super()._randomize_episode(ids)
        clean=ids[self._nominal(ids)];act=self.cfg.rs01_actuator
        for name,value in [('kp',self.p_gains),('kd',self.d_gains),('motor_strength',1.),
            ('response_gain',act.response_gain),('time_constant',act.time_constant_s),
            ('coulomb_friction',act.coulomb_friction_nm),('delay_steps',self.nominal_delay_steps),
            ('imu_latency_steps',0),('joint_latency_steps',0),('encoder_offset',0.)]:
            getattr(self,name)[clean]=value
        self.imu.clean[clean]=True;self.imu.mount_quat[clean]=0.;self.imu.mount_quat[clean,3]=1.
        self.imu.gyro_bias[clean]=0.;self.imu.gyro_scale[clean]=1.;self.imu.tilt_error[clean]=0.
        self.yaw_drift[ids]=self._uniform((len(ids),),[-.003,.003])*(~self._nominal(ids)).float()

    def _public_yaw(self,imu_sample=None):
        if not self.add_noise:return yaw_from_quat(self.root_states[:,3:7])
        sample=self.imu_line.read(self.imu_latency_steps) if imu_sample is None else imu_sample
        drift=quat_from_rpy(self.imu.tilt_error[:,0],self.imu.tilt_error[:,1],self.yaw_drift)
        reported=quat_mul(sample[:,3:7],quat_mul(self.imu.mount_quat,drift))
        return yaw_from_quat(reported)

    def _reset_sensors(self,ids):
        super()._reset_sensors(ids)
        self.public_heading_target[ids]=self._public_yaw()[ids]
        self.public_heading_error[ids]=0.

    def _clear_push(self,ids):
        super()._clear_push(ids)
        self.push_wait[ids[self._nominal(ids)]]=1.e9
        self.recovery_left[ids]=0.

    def _apply_push(self):
        if not self.cfg.push.enabled:return
        starts=self.push_wait<=self.physics_dt
        super()._apply_push()
        self.recovery_left[starts]=self.push_duration[starts]+self.cfg.expand.recovery_grace_s

    def recovery_factor(self):
        return (1-self.recovery_left/self.cfg.expand.recovery_grace_s).clamp(0,1)

    def _resample_commands(self,ids):
        if not len(ids):return
        if self.command_override is not None:
            self.target_commands[ids]=self.command_override[ids]
            self.target_gait[ids]=self.override_gait_enabled
            self.command_frontier[ids]=False
            return
        cfg=self.cfg.commands;n=len(ids);old=self.cfg.expand.stages[0]
        current=self.cfg.expand.stages[self.curriculum_stage]
        if getattr(self.cfg.expand,'x_only',False):
            # Retain the twelve validated commands. The only new task is
            # straight forward speed; do not relearn every continuous mixture.
            choose=torch.rand(n,device=self.device)
            rid=torch.randint(len(self.reference_commands),(n,),device=self.device)
            cmd=self.reference_commands[rid].clone()
            frontier=choose>=1-self.cfg.expand.frontier_probability
            cmd[frontier]=0.
            cmd[frontier,0]=self._uniform((int(frontier.sum()),),[old[1],current[1]])
            stand=(choose>=cfg.reference_probability)&(choose<cfg.reference_probability+cfg.stand_probability)
            march=(choose>=cfg.reference_probability+cfg.stand_probability)&~frontier
            cmd[stand|march]=0.
            self.target_commands[ids]=cmd;self.target_gait[ids]=~stand
            self.command_frontier[ids]=frontier
            return
        draw=lambda b:torch.stack((self._uniform((n,),b[:2]),self._uniform((n,),[-b[2],b[2]]),self._uniform((n,),[-b[3],b[3]])),1)
        cmd=draw(old);new=draw(current)
        single=torch.rand(n,device=self.device)<cfg.single_axis_probability
        axes=torch.randint(3,(n,),device=self.device);one=torch.nn.functional.one_hot(axes,3).bool()
        cmd*=one|~single[:,None];new*=one|~single[:,None]
        choose=torch.rand(n,device=self.device)
        frontier=choose>=1-self.cfg.expand.frontier_probability
        # Sample the new boundary, retaining small mixed-command components.
        sign=torch.where(torch.rand(n,device=self.device)>.5,1.,-1.)
        old_caps=cmd.new_tensor([old[1],old[2],old[3]])[axes]
        new_caps=cmd.new_tensor([current[1],current[2],current[3]])[axes]
        old_caps=torch.where((axes==0)&(sign<0),-old[0],old_caps)
        new_caps=torch.where((axes==0)&(sign<0),-current[0],new_caps)
        amplitude=(old_caps+torch.rand(n,device=self.device)*(new_caps-old_caps))*sign
        new*=.5;new[torch.arange(n,device=self.device),axes]=amplitude
        cmd[frontier]=new[frontier]
        caps=torch.stack((torch.where(cmd[:,0]>=0,current[1],-current[0]),cmd.new_full((n,),current[2]),cmd.new_full((n,),current[3])),1)
        norm=(cmd/caps).norm(dim=1).clamp(min=1);cmd/=norm[:,None]
        ref=choose<cfg.reference_probability
        rid=torch.randint(len(self.reference_commands),(n,),device=self.device);cmd[ref]=self.reference_commands[rid[ref]]
        stand=(choose>=cfg.reference_probability)&(choose<cfg.reference_probability+cfg.stand_probability)
        march=(choose>=cfg.reference_probability+cfg.stand_probability)&(choose<cfg.reference_probability+cfg.stand_probability+cfg.march_probability)
        tiny=(cmd[:,:2].norm(dim=1)<.03)&(cmd[:,2].abs()<.08)&~ref&~march
        stand|=tiny;cmd[stand|march]=0.
        self.target_commands[ids]=cmd;self.target_gait[ids]=~stand
        self.command_frontier[ids]=frontier&~stand&~march

    def reset_idx(self,ids):
        probability=self.reference_reset_probability;self.reference_reset_probability=0.
        try:super().reset_idx(ids)
        finally:self.reference_reset_probability=probability
        if not len(ids):return
        self.commands[ids,:3]=0.;self.gait_enabled[ids]=False;self.cold_start[ids]=True
        target=self.target_commands[ids]
        distance,index=torch.cdist(target,self.reference_commands).min(1)
        eligible=self._nominal(ids)&self.target_gait[ids]&(distance<1.e-6)
        chosen=ids[eligible&(torch.rand(len(ids),device=self.device)<probability)]
        if self.reference_bank is not None and len(chosen):
            self.commands[chosen,:3]=self.target_commands[chosen];self.gait_enabled[chosen]=True
            scale=target.new_tensor([.3,.15,.5])
            self.reference_id[chosen]=torch.cdist(self.commands[chosen,:3]/scale,self.reference_commands/scale).argmin(1)
            table=self.reference_bank['clip_indices'];pick=torch.randint(table.shape[1],(len(chosen),),device=self.device)
            self.restore_reference_states(chosen,table[self.reference_id[chosen],pick],write_sim=False);self.cold_start[chosen]=False
            self.public_heading_target[chosen]=self.heading_target[chosen]
        self.recovery_left[ids]=0.
        # Commit the union once. Multiple indexed writes before a GPU physics
        # step can otherwise leave the non-RSI actors at their creation pose.
        ids32=ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(ids32),len(ids32))
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(ids32),len(ids32))

    def _reset_dofs(self,ids):
        noise=0. if self.cfg.env.test else self.cfg.init_state.reset_dof_position_noise_rad
        q=self.default_dof_pos+self._uniform((len(ids),self.num_dof),[-noise,noise])
        self.dof_pos[ids]=torch.maximum(torch.minimum(q,self.dof_pos_limits[:,1]),self.dof_pos_limits[:,0])
        self.dof_vel[ids]=0.;self.position_target[ids]=self.default_dof_pos;self.target_rate[ids]=0.
        self.response_target[ids]=self.default_dof_pos;self.delay_buffer[:,ids]=self.default_dof_pos
        self.raw_pd_torques[ids]=0.;self.motor_torques[ids]=0.

    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state;self.root_states[ids,:3]+=self.env_origins[ids]
        init=self.cfg.init_state
        if not self.cfg.env.test:
            if init.randomize_yaw:
                zero=torch.zeros(len(ids),device=self.device)
                self.root_states[ids,3:7]=quat_from_rpy(zero,zero,self._uniform((len(ids),),[-torch.pi,torch.pi]))
            self.root_states[ids,7:9]=self._uniform((len(ids),2),[-init.reset_linear_velocity_m_s,init.reset_linear_velocity_m_s])
            self.root_states[ids,10:13]=self._uniform((len(ids),3),[-init.reset_angular_velocity_rad_s,init.reset_angular_velocity_rad_s])
        self.heading_target[ids]=yaw_from_quat(self.root_states[ids,3:7]);self.heading_error[ids]=0.

    def set_command_override(self,commands,gait_enabled=True):
        super().set_command_override(commands,gait_enabled)
        self.target_commands[:]=self.commands[:,:3];self.target_gait[:]=gait_enabled
        self.cold_start[:]=False;self.command_frontier[:]=False

    def _post_physics_step_callback(self):
        settling=self.cold_start&(self.episode_steps*self.dt<self.cfg.expand.cold_settle_s)
        limit=self.commands.new_tensor(self.cfg.expand.command_acceleration)*self.dt
        self.commands[:,:3]+=(self.target_commands-self.commands[:,:3]).clamp(-limit,limit)
        self.commands[settling,:3]=0.
        self.gait_enabled[:]=~settling&(self.target_gait|(self.commands[:,:3].abs().sum(1)>.001))
        self.cold_start&=settling
        scale=self.commands.new_tensor([.3,.15,.5])
        self.reference_id[:]=torch.cdist(self.commands[:,:3]/scale,self.reference_commands/scale).argmin(1)
        super()._post_physics_step_callback()
        self.public_heading_target+=self.commands[:,2]*self.dt
        self.public_heading_target=torch.atan2(self.public_heading_target.sin(),self.public_heading_target.cos())
        self.yaw_drift+=(~self.nominal_mask).float()*self.imu.gyro_bias[:,2]*self.dt
        self.yaw_drift.clamp_(-.035,.035)
        self.recovery_left.sub_(self.dt).clamp_(min=0.)

    def _assemble_frame(self,gyro,gravity,q,dq,yaw_mean,heading):
        base=super(Urdf4AMPCTSRobot,self)._frame(gyro,gravity,q,dq,yaw_mean)
        return torch.cat((base,heading.sin()[:,None],(heading.cos()-1)[:,None],self.gait_enabled.float()[:,None]),1)

    def compute_observations(self):
        held=self._rsi_hold_history.nonzero().flatten();saved=self.obs_history[held].clone()
        mean=self.measured_yaw_error;ix=(mean.head-torch.arange(mean.window.shape[1],device=self.device))%mean.window.shape[1]
        saved_yaw=mean.window[held[:,None],ix[None,:]].clone()
        quat=self.root_states[:,3:7];true_gyro=quat_rotate_inverse(quat,self.root_states[:,10:13]);gravity=quat_rotate_inverse(quat,self.gravity_vec)
        if self.add_noise:
            sample=self.imu_line.read(self.imu_latency_steps);gyro,measured_gravity=self.imu.measure(sample[:,3:7],sample[:,:3])
            joints=self.joint_line.read(self.joint_latency_steps);active=(~self.nominal_mask).float()[:,None]
            sensor=self.cfg.joint_sensor
            q=joints[:,:12]+self.encoder_offset+active*self._uniform((self.num_envs,12),[-sensor.position_noise_rad,sensor.position_noise_rad])
            dq=joints[:,12:]+active*self._uniform((self.num_envs,12),[-sensor.velocity_noise_rad_s,sensor.velocity_noise_rad_s])
            yaw=self._public_yaw(sample)
        else:gyro,measured_gravity,q,dq,yaw=true_gyro,gravity,self.dof_pos,self.dof_vel,yaw_from_quat(quat)
        error=self.public_heading_target-yaw;self.public_heading_error=torch.atan2(error.sin(),error.cos())
        self.measured_yaw_error.push(self.commands[:,2:3]-gyro[:,2:3])
        frame=self._assemble_frame(gyro,measured_gravity,q,dq,self.measured_yaw_error.mean(),self.public_heading_error)
        self.obs_history=torch.roll(self.obs_history,1,1);self.obs_history[:,0]=frame
        self.obs_history[self._history_stale]=frame[self._history_stale,None,:];self._history_stale[:]=False
        self.obs_history[held]=saved;self._restore_cycle_mean(mean,held,saved_yaw);self._rsi_hold_history[held]=False
        privileged=torch.cat((self._assemble_frame(true_gyro,gravity,self.dof_pos,self.dof_vel,self.velocity_drift[:,2:3],self.heading_error),
            quat_rotate_inverse(quat,self.root_states[:,7:10])*2.,(self.root_states[:,2:3]-self.cfg.rewards.base_height_target)*5.,
            self.get_foot_contact_mask().float(),self._scheduled_stance().float(),self.feet_pos[:,:,2]*10.,self.push_force/20.,
            self.friction_coeff,self.payload_kg,self.com_offset_m*10.,self.torques*.05,(self.motor_strength-1)*5.),1)
        anchor=reference_coverage(self.commands[:,:3],self.reference_commands)*self.recovery_factor()*self.nominal_mask.float()*self.gait_enabled.float()
        cmd=self.commands[:,:3]
        old_box=(cmd[:,0]>=-.180001)&(cmd[:,0]<=.280001)&(cmd[:,1].abs()<=.100001)&(cmd[:,2].abs()<=.400001)
        anchor*=old_box.float()
        self.obs_buf=torch.cat((self.obs_history.reshape(self.num_envs,-1),privileged,anchor[:,None],self.teacher_mask.float()[:,None]),1)
        self.privileged_obs_buf=self.obs_buf

    def _reward_command_tracking(self):
        cfg=self.cfg.rewards
        correction=(cfg.heading_gain*self.heading_error).clamp(-cfg.heading_correction_limit_rad_s,cfg.heading_correction_limit_rad_s)
        error=torch.cat((self.velocity_drift[:,:2]/cfg.command_linear_sigma_m_s,
            ((self.velocity_drift[:,2]+correction)/cfg.command_yaw_sigma_rad_s)[:,None]),1).abs()
        loss=torch.where(error<=1,.5*error.square(),error-.5).sum(1)
        self.command_quality=torch.exp(-loss);return -loss

    def _reward_gait_quality(self):
        contact=self.get_foot_contact_mask()
        unsafe=self._reward_illegal_support()
        mismatch=(contact!=self._scheduled_stance()).float().mean(1)
        stand=(~contact).float().mean(1)+.05*self.dof_vel.square().mean(1)
        return unsafe+torch.where(self.gait_enabled,.5*mismatch*self.recovery_factor(),stand)

    def compute_reward(self):
        super().compute_reward()
        stable=(self.commands[:,:3]-self.target_commands).abs().amax(1)<1.e-5
        assessed=self.command_frontier|getattr(self.cfg.expand,'x_only',False)
        mask=self.nominal_mask&~self.teacher_mask&self.gait_enabled&assessed&stable&(self.episode_steps>50)
        xy=self.commands[:,:2].norm(dim=1);mask&=(xy>.03)|(self.commands[:,2].abs()>.08)
        groups=command_classes(self.commands[:,:3])
        contact=self.get_foot_contact_mask()
        slots=[self.foot_legs.index(x) for x in ['FL','FR','RL','RR']];c=contact[:,slots]
        first=c[:,0]&c[:,3]&~c[:,1]&~c[:,2];second=c[:,1]&c[:,2]&~c[:,0]&~c[:,3]
        values=torch.stack((self.velocity_drift[:,:2].norm(dim=1),self.velocity_drift[:,2].abs(),
            (self._reward_illegal_support()>0).float(),(self.reset_buf.bool()&~self.time_out_buf).float(),torch.ones(self.num_envs,device=self.device),
            (contact!=self._scheduled_stance()).float().mean(1),first.float(),second.float()),1)
        self.curriculum_meter.index_add_(0,groups[mask],values[mask])

    def step(self,actions):
        result=super().step(actions)
        result[4]['amp']['style_factor']=self.recovery_factor().clone()
        return result
