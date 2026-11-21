"""Frozen AMP/CTS gait + path-command adapter.

Oracle pose is an explicitly named diagnostic backend. External odometry is
provided by the caller; no fallback to simulator position is allowed.
"""
import torch
from .amp_cts_env import yaw_from_quat
from .path_tracking import PlanarPathController,StudentDeadReckoning,rotate_xy,wrap


class PathTrackingSession:
    def __init__(self,env,policy,mode=2,pose_source='external',pose_provider=None,gains=None,control_scope='forward'):
        if pose_source not in ('oracle','student','external'):raise ValueError(pose_source)
        if pose_source=='external' and pose_provider is None:raise ValueError('External odometry provider is required; no truth fallback')
        self.env,self.policy,self.pose_source,self.pose_provider=env,policy,pose_source,pose_provider
        if control_scope not in ('forward','experimental_all'):raise ValueError(control_scope)
        self.control_scope=control_scope
        self.controller=PlanarPathController(env.num_envs,env.dt,env.device,mode,gains)
        self.requested_mode=self.controller.mode.clone()
        self.estimate=StudentDeadReckoning(policy.encoder,env.num_envs,env.dt,env.device)
        self.truth_origin=env.root_states[:,:2].clone()
        self.truth_heading=yaw_from_quat(env.root_states[:,3:7]).clone()
        self.reported_heading=env._public_yaw().clone()
        zero=torch.zeros_like(self.truth_origin)
        self.controller.reset(zero,zero[:,0])
        self.original_callback=env._post_physics_step_callback
        env._post_physics_step_callback=self._post_physics

    def truth_pose(self):
        env=self.env
        position=rotate_xy(-self.truth_heading,env.root_states[:,:2]-self.truth_origin)
        yaw=wrap(yaw_from_quat(env.root_states[:,3:7])-self.truth_heading)
        velocity=rotate_xy(-self.truth_heading,env.root_states[:,7:9])
        return position,yaw,velocity

    def _post_physics(self):
        self.original_callback()
        env=self.env;reference=self.controller.reference_yaw
        # Do not integrate corrective yaw into the user's desired heading.
        env.heading_target[:]=wrap(self.truth_heading+reference)
        env.heading_error[:]=wrap(env.heading_target-yaw_from_quat(env.root_states[:,3:7]))
        env.public_heading_target[:]=wrap(self.reported_heading+reference)

    def prepare_command(self,external,obs,moving=None):
        env=self.env;public=obs[:,:1020]
        if self.control_scope=='forward':
            forward=(external[:,0]>.02)&(external[:,1].abs()<1.e-6)&(external[:,2].abs()<1.e-6)
            if bool((forward&(external[:,0]>.250001)&(self.requested_mode>0)).any()):
                raise ValueError('Forward path profile is validated at 0.18/0.25 m/s; larger commands require experimental_all')
            self.controller.mode[:]=torch.where(forward,self.requested_mode,torch.zeros_like(self.requested_mode))
            self.controller.reference_yaw_rate[forward]=0.
        if self.pose_source=='oracle':position,yaw,velocity=self.truth_pose()
        elif self.pose_source=='student':position,yaw,velocity=self.estimate.update(public,self.controller.reference_yaw)
        else:position,yaw,velocity=self.pose_provider(public,env.dt)
        # Public gyro is scaled by 0.25 in the existing observation contract.
        rate=public[:,2]/env.obs_scales.ang_vel
        command=self.controller.update(external,position,yaw,velocity,rate)
        enabled=(external.abs().sum(1)>.02) if moving is None else moving
        env.set_command_override(command,gait_enabled=True)
        env.gait_enabled[:]=enabled;env.target_gait[:]=enabled
        # Patch the current frame only. Recomputing observations here would
        # double-advance sensor noise, cycle windows, and the 20-frame history.
        scaled=command*env.commands_scale
        env.obs_history[:,0,6:9]=scaled;env.obs_history[:,0,50]=enabled.float()
        env.obs_buf[:,6:9]=scaled;env.obs_buf[:,50]=enabled.float()
        env.obs_buf[:,1026:1029]=scaled;env.obs_buf[:,1070]=enabled.float()
        return dict(command=command,estimated_xy=position,controller_active=self.controller.mode>0,
            cross_error=self.controller.cross_error.clone(),heading_error=self.controller.heading_error.clone())

    def step(self,external,obs,moving=None):
        telemetry=self.prepare_command(external,obs,moving)
        result=self.env.step(self.policy(self.env.obs_buf[:,:1020]))
        return result,telemetry

    def close(self):
        self.env._post_physics_step_callback=self.original_callback
