"""Truth-only plant/reward state; sensor-only actor, odometry and host guard."""
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from .rs01_omni_v21_env import Rs01OmniV21Robot
from .rs01_omni_v16_env import guarded_target, update_guard
from .rs01_sensor_snapshot import SensorSnapshot, assemble61


class Rs01OmniV22Robot(Rs01OmniV21Robot):
    def __init__(self,*args,**kwargs):
        self.sensor_ready=False
        super().__init__(*args,**kwargs)
        self.sensor_tick=0;self.sensor_captured_tick=-1
        self.sensor=SensorSnapshot(self.num_envs,self.device,self.cfg.sim.dt,self.cfg.sensor)
        ids=torch.arange(self.num_envs,device=self.device)
        self.sensor.reset(ids,0,*self._fresh_sensor_truth())
        self.sensor.read(0);self.sensor_ready=True
        self._start_direction_segment(ids)
        self._update_rs01_observation_estimator()
        self.compute_observations()

    def _fresh_sensor_truth(self):
        # base_ang_vel/projected_gravity update only at policy rate: NEVER use them here.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        quat=self.root_states[:,3:7]
        gyro=quat_rotate_inverse(quat,self.root_states[:,10:13])
        return self.dof_pos,self.dof_vel,quat,gyro,self.motor_electromagnetic_torques

    def _capture_sensor_substep(self):
        if self.sensor_ready and self.sensor_captured_tick!=self.sensor_tick:
            self.sensor.acquire(self.sensor_tick,*self._fresh_sensor_truth())
            self.sensor_captured_tick=self.sensor_tick

    def _compute_torques(self,actions):
        if self.sensor_ready:
            self._capture_sensor_substep()
            if getattr(self,'_v16_project_pending',False):
                cfg=self.cfg.rs01_actuator
                if not hasattr(self,'guard_rms_sq'):
                    self.guard_rms_sq=torch.zeros_like(self.dof_pos)
                    self.guard_active_limit=torch.full_like(self.dof_pos,cfg.peak_torque_limit_nm)
                # Host guard uses the SAME policy snapshot; motor's internal PD keeps physical feedback.
                self.guard_target,self.guard_raw_pd,self.guard_safe_pd=guarded_target(
                    self.rs01_limited_position_target_rad,self.sensor.q,self.sensor.dq,
                    self.p_gains,self.d_gains,self.guard_active_limit,self.guard_joint_lower,self.guard_joint_upper)
                self.guard_rms_sq,self.guard_active_limit=update_guard(self.guard_rms_sq,
                    torch.maximum(self.guard_safe_pd.abs(),self.sensor.torque.abs()),self.dt,
                    cfg.continuous_torque_nm,cfg.peak_torque_limit_nm,cfg.guard_derate_full_rms_nm,cfg.guard_time_constant_s)
                self._v16_project_pending=False
        result=super()._compute_torques(actions)
        if self.sensor_ready:self.sensor_tick+=1
        return result

    def _post_physics_step_callback(self):
        if self.sensor_ready:
            self._capture_sensor_substep()  # Final substep at t+20 ms, not t+17.5 ms.
            self.sensor.read(self.sensor_tick)
        super()._post_physics_step_callback()

    def _update_rs01_observation_estimator(self):
        if not self.sensor_ready:return
        r=self.rs01_leg_odometry.estimate(self.sensor.q,self.sensor.dq,self.sensor.gyro)
        self.estimated_base_lin_vel.copy_(r['base_linear_velocity'])
        self.estimated_odom_confidence.copy_(r['confidence'])
        self.estimated_odom_stance_mask.copy_(r['stance_mask'])
        self.estimated_velocity_by_foot_m_s.copy_(r['velocity_by_foot'])

    def _start_direction_segment(self,ids):
        super()._start_direction_segment(ids)
        if self.sensor_ready:
            self.straight_heading_target_rad[ids]=self.sensor.yaw[ids]

    def _straight_heading_error(self):
        if not self.sensor_ready:return torch.zeros(self.num_envs,device=self.device)
        delta=self.straight_heading_target_rad-self.sensor.yaw
        return torch.atan2(torch.sin(delta),torch.cos(delta))

    def compute_observations(self):
        # Do not invoke any ancestor observation builder, including during construction.
        if not self.sensor_ready:
            self.obs_buf=torch.zeros(self.num_envs,61,device=self.device)
            return
        self.sensor_effective_command=self._direction_velocity_target()
        self.obs_buf=assemble61(self.sensor,self.estimated_base_lin_vel,self.estimated_odom_confidence,
            self.sensor_effective_command,self.commands[:,:3],self.default_dof_pos,self.obs_scales,
            self.commands_scale,self.actions,self._gait_phase(),self._straight_heading_error(),self.gait_enable)
        self.obs_buf.clamp_(-self.cfg.normalization.clip_observations,self.cfg.normalization.clip_observations)

    def step(self,actions):
        if self.sensor_ready:
            # Freeze the target seen by the actor for THIS transition's reward.
            self.executed_sensor_target=self.sensor_effective_command.clone()
        return super().step(actions)

    def _reward_tracking_command_velocity(self):
        target=self.executed_sensor_target
        error=((target[:,:2]-self.base_lin_vel[:,:2]).square().sum(1)/self.cfg.rewards.command_planar_tracking_sigma
               +(target[:,2]-self.base_ang_vel[:,2]).square()/self.cfg.rewards.command_yaw_tracking_sigma)
        reward=1/(1+error)
        self.direction_reward_target=target.detach().clone()
        self.v11_tracking_accuracy=reward.detach();self.v11_tracking_reward=reward.detach()
        self.v11_legal_contact_gate=self._legal_task_contact_gate().detach()
        return reward

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if self.sensor_ready and len(ids):
            self.sensor.reset(ids,self.sensor_tick,*self._fresh_sensor_truth())
            self.sensor.read(self.sensor_tick)
            self._start_direction_segment(ids)
