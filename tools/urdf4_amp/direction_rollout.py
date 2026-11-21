"""Evaluation adapter for the path controller already INSIDE the training env."""
import torch
from legged_gym.envs.urdf4.path_tracking import rotate_xy,wrap
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat


class DirectionRolloutSession:
    def __init__(self,env,policy,mode,**kwargs):
        self.env,self.policy=env,policy
        env.reset_path(torch.arange(env.num_envs,device=env.device))
        self.controller=env.path
        self.controller.mode[:]=mode
        self.origin=env.path_origin.clone();self.yaw0=env.path_origin_yaw.clone()
        self.last=None

    def truth_pose(self):
        env=self.env
        return (env.root_states[:,:2],yaw_from_quat(env.root_states[:,3:7]),env.root_states[:,7:9])

    def step(self,external,obs,moving=None):
        env=self.env
        # Overrides are user targets, not another path-control layer.
        if self.last is None or not torch.equal(external,self.last):
            env.set_command_override(external)
            scaled=external*env.commands_scale
            env.obs_history[:,0,6:9]=scaled;env.obs_buf[:,6:9]=scaled;env.obs_buf[:,1026:1029]=scaled
            self.last=external.clone()
        if moving is not None:
            env.target_gait[:]=moving;env.gait_enabled[:]=moving
            env.obs_history[:,0,50]=moving.float();env.obs_buf[:,50]=moving.float();env.obs_buf[:,1070]=moving.float()
        pose=torch.where(env.nominal_mask[:,None],env.pose_queue[-1,:,:2],env.pose_queue[0,:,:2]).clone()
        command=env.commands[:,:3].clone()
        result=env.step(self.policy(env.obs_buf[:,:1020]))
        return result,dict(command=command,estimated_xy=pose)

    def close(self):
        pass
