"""Independent non-AMP phase/contact/footprint task on the unchanged RS01 plant."""
import torch
from .rs01_omni_v25_env import Rs01OmniV25Robot
from .rs01_omni_v30_config import Rs01OmniV30Cfg, Rs01OmniV30CfgPPO


def advance_lift_clock(age, air_time, contact, height, enabled, dt):
    airborne = (~contact) & (height >= .010)
    air_time = torch.where(airborne, air_time + dt, torch.zeros_like(air_time))
    qualified = air_time >= .04 - 1e-6
    age = torch.where(qualified | ~enabled[:,None], torch.zeros_like(age), age + dt)
    return age, air_time


def corridor_cost(signed_y, weights, inner=.13, outer=.21):
    error = ((inner-signed_y).clamp(min=0) + (signed_y-outer).clamp(min=0))/.03
    return ((error.square().clamp(max=9))*weights).sum(1)/weights.sum(1).clamp(min=1)


class Rs01StablePhaseCfg(Rs01OmniV30Cfg):
    class stability:
        # Zero is marching, never randomly standing. Start with a modest omni envelope.
        commands = [[0.,0.,0.],[.12,0.,0.],[-.10,0.,0.],
                    [0.,.06,0.],[0.,-.06,0.],[0.,0.,.25],[0.,0.,-.25],
                    [.08,.04,0.],[.08,-.04,0.],[.08,.03,.15]]
        startup_grace_s = 3.
        max_no_lift_cycles = 3.
        severe_inward_m = .09
        severe_inward_duration_s = .20
    class commands(Rs01OmniV30Cfg.commands):
        resampling_time = 4.
        moving_to_stand_probability = 0.
    class rewards(Rs01OmniV30Cfg.rewards):
        gait_period_s = .50
        gait_stance_ratio = .70
        swing_clearance_m = .025
        coordination_weight = 0.
        class scales:
            tracking_command_velocity = 3.
            stable_phase = 3.
            stable_corridor = -3.
            overdue_lift = -2.
            orientation = -2.
            lin_vel_z = -.5
            ang_vel_xy = -.05
            stance_foot_slip = -.05
            raw_torque_over_peak = -.5
            collision = -1.
            dof_pos_limits = -2.
            action_rate = -.005
            termination = -20.


class Rs01StablePhasePPO(Rs01OmniV30CfgPPO):
    class algorithm(Rs01OmniV30CfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = 'fixed'
    class runner(Rs01OmniV30CfgPPO.runner):
        experiment_name = 'rs01_stable_phase'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        max_iterations = 10000
        save_interval = 100
        amp_enabled = False
        phase_residual_policy = False
        reference_policy_coef = 0.
        symmetry_coef = 0.
        freeze_action_std = True
        action_std_value = .35
        adapt_observation_input = False


class Rs01StablePhaseRobot(Rs01OmniV25Robot):
    def __init__(self,*args,**kwargs):
        self.stable_ready = False
        super().__init__(*args,**kwargs)
        self.stable_ids = [self.foot_slot_by_leg[l] for l in ('FL','FR','RL','RR')]
        self.no_lift_s = torch.zeros((self.num_envs,4),device=self.device)
        self.qualified_air_s = torch.zeros_like(self.no_lift_s)
        self.contact_duration_s = torch.zeros_like(self.no_lift_s)
        self.inward_duration_s = torch.zeros(self.num_envs,device=self.device)
        self.stable_elapsed_s = torch.zeros(self.num_envs,device=self.device)
        self.stable_ready = True

    def _set_command_modes(self,env_ids,modes):
        if len(env_ids)==0:return
        table=torch.tensor(self.cfg.stability.commands,device=self.device)
        selected=torch.randint(len(table),(len(env_ids),),device=self.device)
        self.commands[env_ids,:3]=table[selected]
        categories=torch.tensor([self.COMMAND_MARCH,self.COMMAND_FORWARD,self.COMMAND_BACKWARD,
            self.COMMAND_LATERAL,self.COMMAND_LATERAL,self.COMMAND_YAW,self.COMMAND_YAW,
            self.COMMAND_COMBINED,self.COMMAND_COMBINED,self.COMMAND_COMBINED],device=self.device)
        self.command_mode[env_ids]=categories[selected]
        self.gait_enable[env_ids]=1.
        self._reset_command_reference(env_ids)
        self._update_turn_mode(env_ids)

    def _command_gait_frequency(self):
        return torch.full_like(self.commands[:,0],1./self.cfg.rewards.gait_period_s)

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if self.stable_ready:
            self.no_lift_s[ids]=0.;self.qualified_air_s[ids]=0.
            self.contact_duration_s[ids]=0.;self.inward_duration_s[ids]=0.
            self.stable_elapsed_s[ids]=0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if not self.stable_ready:return
        self.stable_elapsed_s+=self.dt
        ids=self.stable_ids
        self.stable_contact=self.get_foot_contact_mask()[:,ids]
        self.stable_height=self.feet_pos[:,ids,2]-self.env_origins[:,None,2]-self.cfg.rewards.foot_collision_radius_m
        relative=self.feet_pos[:,ids]-self.root_states[:,None,:3]
        yaw=self.rpy[:,2,None]
        self.stable_y=(-yaw.sin()*relative[:,:,0]+yaw.cos()*relative[:,:,1])*relative.new_tensor([1,-1,1,-1])
        self.no_lift_s,self.qualified_air_s=advance_lift_clock(self.no_lift_s,self.qualified_air_s,
            self.stable_contact,self.stable_height,self.gait_enable>.5,self.dt)
        self.contact_duration_s=torch.where(self.stable_contact,self.contact_duration_s+self.dt,
                                           torch.zeros_like(self.contact_duration_s))
        inward=((self.stable_y<self.cfg.stability.severe_inward_m)&self.stable_contact).any(1)
        self.inward_duration_s=torch.where(inward,self.inward_duration_s+self.dt,torch.zeros_like(self.inward_duration_s))

    def check_termination(self):
        super().check_termination()
        if not self.stable_ready:return
        ready=(self.stable_elapsed_s>self.cfg.stability.startup_grace_s)&(self.gait_enable>.5)
        self.stable_last_age_s=self.no_lift_s.clone()
        self.stable_last_contact_s=self.contact_duration_s.clone()
        period=self.cfg.rewards.gait_period_s
        self.stalled_foot=self._gait_timeout_ready()&((self.no_lift_s.amax(1)>self.cfg.stability.max_no_lift_cycles*period)
                                |(self.qualified_air_s.amax(1)>period))
        self.narrow_failure=ready&(self.inward_duration_s>self.cfg.stability.severe_inward_duration_s)
        self.reset_buf |= self.stalled_foot|self.narrow_failure

    def _gait_timeout_ready(self):
        return (self.stable_elapsed_s>self.cfg.stability.startup_grace_s)&(self.gait_enable>.5)

    def _phase_targets(self):
        phase=(self._gait_phase()[:,None]+self.commands.new_tensor([0.,.5,.5,0.])).remainder(1.)
        duty=self.cfg.rewards.gait_stance_ratio
        swing=((phase-duty)/(1.-duty)).clamp(0.,1.)
        lift=self.cfg.rewards.swing_clearance_m*torch.sin(torch.pi*swing).square()
        return phase<duty,swing,lift

    def _reward_stable_phase(self):
        if not self.stable_ready:return torch.zeros(self.num_envs,device=self.device)
        stance,swing,lift=self._phase_targets()
        height_score=torch.exp(-((self.stable_height-lift)/.015).square())
        # Smooth swing transition, with clearance and unloading in one phase objective.
        middle=torch.sin(torch.pi*swing).square()
        swing_score=height_score*(1.-middle*self.stable_contact.float())
        score=torch.where(stance,self.stable_contact.float(),swing_score)
        return score.mean(1)*self.gait_enable

    def _reward_stable_corridor(self):
        if not self.stable_ready:return torch.zeros(self.num_envs,device=self.device)
        _,swing,_=self._phase_targets()
        weights=torch.maximum(self.stable_contact.float(),((swing-.5)/.5).clamp(0,1))
        return corridor_cost(self.stable_y,weights)*self.gait_enable

    def _reward_overdue_lift(self):
        if not self.stable_ready:return torch.zeros(self.num_envs,device=self.device)
        period=self.cfg.rewards.gait_period_s
        grace=period*self.cfg.rewards.gait_stance_ratio+.10
        return ((self.no_lift_s-grace)/period).clamp(0,1).mean(1)*self.gait_enable

    def _reward_tracking_command_velocity(self):
        actual=torch.cat((self.base_lin_vel[:,:2],self.base_ang_vel[:,2:3]),1)
        target=self.executed_sensor_target
        scale=actual.new_tensor([.10,.10,.30])
        # Stationary motion earns zero task progress, including marching.
        return 1/(1+((actual-target)/scale).square().sum(1))-1/(1+(target/scale).square().sum(1))
