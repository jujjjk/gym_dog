import torch
from .rs01_amp_scratch_env import Rs01AMPScratchRobot


def progress_over_stationary(actual,target):
    scale=actual.new_tensor([.10,.10,.30])
    return 1/(1+((actual-target)/scale).square().sum(1))-1/(1+(target/scale).square().sum(1))


class Rs01AMPCoreRobot(Rs01AMPScratchRobot):
    def reset_idx(self,env_ids):
        super().reset_idx(env_ids)
        if hasattr(self,'amp_recent_lift'):self.amp_recent_lift[env_ids]=0.

    def _reward_tracking_command_velocity(self):
        actual=torch.cat((self.base_lin_vel[:,:2],self.base_ang_vel[:,2:3]),1)
        return progress_over_stationary(actual,self.executed_sensor_target)

    def compute_reward(self):
        super().compute_reward()
        # One style modifier distinguishes marching from stationary zero-command posture.
        # No prescribed trajectory/phase. Physical stance and flight still have their own legality cost.
        height=self.feet_pos[:,:,2]-self.env_origins[:,None,2]-self.cfg.rewards.foot_collision_radius_m
        if not hasattr(self,'amp_recent_lift'):self.amp_recent_lift=torch.zeros_like(height)
        decay=torch.exp(height.new_tensor(-self.dt/self.cfg.rewards.gait_period_s))
        self.amp_recent_lift=torch.maximum(self.amp_recent_lift*decay,height.clamp(min=0,max=.05))
        activity=(self.amp_recent_lift.min(1).values/.015).clamp(0,1)
        upright=(self.projected_gravity[:,2]<-.94).float()
        self.amp_gate=(.1+.9*activity)*self._legal_task_contact_gate()*upright
