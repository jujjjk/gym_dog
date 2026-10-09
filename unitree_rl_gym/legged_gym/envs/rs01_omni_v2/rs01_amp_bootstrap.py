"""Temporary, explicitly supervised motion acquisition; NOT a pure AMP result."""
import torch
from .rs01_amp_style_first import (
    Rs01AMPStyleFirstRobot, Rs01AMPStyleFirstCfg, Rs01AMPStyleFirstSlowCfg,
    Rs01AMPStyleFirstPPO, Rs01AMPStyleFirstSlowPPO)


def nearest_phase_frame(phase, reference_phase):
    delta = (phase[:,None] - reference_phase + .5).remainder(1.) - .5
    return delta.abs().argmin(1)


class Rs01AMPBootstrapRobot(Rs01AMPStyleFirstRobot):
    def _reward_reference_motion(self):
        if not self.reference_reset_ready:return torch.zeros(self.num_envs,device=self.device)
        r=self.reference_reset
        case=torch.cdist(self.commands[:,:3],r['command'][:,0]).argmin(1)
        frame=nearest_phase_frame(self._gait_phase(),r['leg_phase'][case,:,0])
        target=r['joint_pos_rad'][case,frame]
        velocity=r['joint_vel_rad_s'][case,frame]
        # One temporary motion target, no extra clearance/width/mirror rewards.
        error=((self.dof_pos-target)/.12).square().mean(1)
        error+=.15*((self.dof_vel-velocity)/3.).square().mean(1)
        return torch.exp(-error)


class BootstrapRewards(Rs01AMPStyleFirstCfg.rewards):
    class scales(Rs01AMPStyleFirstCfg.rewards.scales):
        reference_motion=8.


class Rs01AMPBootstrapCfg(Rs01AMPStyleFirstCfg):
    class rewards(BootstrapRewards):pass


class Rs01AMPBootstrapSlowCfg(Rs01AMPStyleFirstSlowCfg):
    class rewards(BootstrapRewards):
        gait_period_s=1./1.5


class Rs01AMPBootstrapPPO(Rs01AMPStyleFirstPPO):
    class algorithm(Rs01AMPStyleFirstPPO.algorithm):
        learning_rate=3e-4
        schedule='fixed'
    class runner(Rs01AMPStyleFirstPPO.runner):
        experiment_name='rs01_amp_bootstrap'
        # Discriminator may collect data, but cannot compete with the temporary teacher.
        amp_style_weight=0.


class Rs01AMPBootstrapSlowPPO(Rs01AMPStyleFirstSlowPPO):
    class algorithm(Rs01AMPBootstrapPPO.algorithm):pass
    class runner(Rs01AMPStyleFirstSlowPPO.runner):
        experiment_name='rs01_amp_bootstrap_slow'
        amp_style_weight=0.
