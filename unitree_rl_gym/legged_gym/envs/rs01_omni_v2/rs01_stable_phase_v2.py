"""Remove stationary phase income and premature gait-timeout resets; unchanged RS01."""
import torch
from .rs01_stable_phase import Rs01StablePhaseRobot,Rs01StablePhaseCfg,Rs01StablePhasePPO


def phase_progress(contact,height,stance,swing,lift):
    middle=torch.sin(torch.pi*swing).square()
    score=torch.where(stance,contact.float(),
        torch.exp(-((height-lift)/.015).square())*(1.-middle*contact.float()))
    stationary=torch.where(stance,torch.ones_like(height),
        torch.exp(-(lift/.015).square())*(1.-middle))
    return (score-stationary).mean(1)


def acquisition_grace(step,training,initial=12.,final=3.,curriculum_steps=24000):
    if not training:return final
    fraction=min(1.,max(0.,float(step)/curriculum_steps))
    return initial+(final-initial)*fraction


class Rs01StablePhaseV2Cfg(Rs01StablePhaseCfg):
    class stability(Rs01StablePhaseCfg.stability):
        training_initial_grace_s=12.
        timeout_curriculum_steps=24000  # 1000 PPO updates at24 policy steps/update.
    class rewards(Rs01StablePhaseCfg.rewards):
        class scales(Rs01StablePhaseCfg.rewards.scales):
            stable_phase=6.
            # Reward scales are multiplied by dt=.02: cost6, not merely.4.
            # Failing must not be cheaper than discounted stationary running costs.
            termination=-300.


class Rs01StablePhaseV2PPO(Rs01StablePhasePPO):
    class runner(Rs01StablePhasePPO.runner):
        experiment_name='rs01_stable_phase_v2'
        save_interval=50


class Rs01StablePhaseV2Robot(Rs01StablePhaseRobot):
    def _reward_stable_phase(self):
        if not self.stable_ready:return torch.zeros(self.num_envs,device=self.device)
        stance,swing,lift=self._phase_targets()
        return phase_progress(self.stable_contact,self.stable_height,stance,swing,lift)*self.gait_enable

    def _gait_timeout_ready(self):
        cfg=self.cfg.stability
        grace=acquisition_grace(self.common_step_counter,not self.cfg.env.test,
                               cfg.training_initial_grace_s,cfg.startup_grace_s,cfg.timeout_curriculum_steps)
        return (self.stable_elapsed_s>grace)&(self.gait_enable>.5)
