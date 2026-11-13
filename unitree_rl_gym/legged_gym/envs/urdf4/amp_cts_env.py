"""CTS packed observations + causally captured AMP transitions for the new URDF."""
import json
from pathlib import Path
import torch
import numpy as np
from .urdf4_omni_env import Urdf4OmniRobot
from .amp_reference import amp_features, valid_transition

def yaw_from_quat(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

class Urdf4AMPCTSRobot(Urdf4OmniRobot):
    def _init_buffers(self):
        n,device=self.num_envs,self.device
        self.heading_target=torch.zeros(n,device=device)
        self.heading_error=torch.zeros(n,device=device)
        self.gait_enabled=torch.ones(n,device=device,dtype=torch.bool)
        self.override_gait_enabled=True
        self.teacher_mask=torch.arange(n,device=device)<int(n*self.cfg.amp_cts.teacher_fraction)
        if self.cfg.amp_cts.student_only:self.teacher_mask[:]=False
        self.reference_id=torch.zeros(n,device=device,dtype=torch.long)
        manifest=json.loads((Path(self.cfg.amp_cts.reference)/'manifest.json').read_text())
        self.reference_commands=torch.tensor([c['command'] for c in manifest['clips']],device=device)
        self.reference_duties=torch.tensor([c['stance_ratio_target'] for c in manifest['clips']],device=device)
        self.reference_frequencies=torch.tensor([c['cadence_hz'] for c in manifest['clips']],device=device)
        templates=[]
        self.reference_bins=round(self.cfg.gait.period_s/.02)
        for clip in manifest['clips']:
            with np.load(Path(self.cfg.amp_cts.reference)/(clip['clip']+'.npz'),allow_pickle=False) as data:
                phase=data['leg_phase'][:,0]
                bins=np.rint(phase*self.reference_bins).astype(int)%self.reference_bins
                feet=data['amp_features'][:,24:36]
                templates.append(np.stack([feet[bins==i].mean(0) if np.any(bins==i) else feet[np.argmin(np.abs((phase-i/self.reference_bins+.5)%1-.5))]
                                           for i in range(self.reference_bins)]))
        self.reference_feet=torch.tensor(np.stack(templates),device=device)
        if not bool(torch.isclose(self.reference_frequencies,torch.full_like(self.reference_frequencies,1/self.cfg.gait.period_s)).all()):
            raise ValueError('CTS phase clock and V2 cadence differ')
        packed=self.cfg.env.num_observations
        self.cfg.env.num_observations=self.cfg.env.public_obs_size
        try:super()._init_buffers()
        finally:self.cfg.env.num_observations=packed

    def _resample_commands(self,env_ids):
        if not len(env_ids):return
        if self.command_override is not None:
            self.commands[env_ids,:3]=self.command_override[env_ids]
            self.gait_enabled[env_ids]=self.override_gait_enabled
        else:
            n=len(env_ids);cfg=self.cfg.commands;r=self.command_ranges
            cmd=torch.stack([self._uniform((n,),r[k]) for k in ['lin_vel_x','lin_vel_y','ang_vel_yaw']],1)
            single=torch.rand(n,device=self.device)<cfg.single_axis_probability
            axes=torch.randint(0,3,(n,),device=self.device)
            cmd*=torch.nn.functional.one_hot(axes,3).bool()|~single[:,None]
            choose=torch.rand(n,device=self.device)
            ref=choose<cfg.reference_probability
            rid=torch.randint(len(self.reference_commands),(n,),device=self.device)
            cmd[ref]=self.reference_commands[rid[ref]]
            stand=(choose>=cfg.reference_probability)&(choose<cfg.reference_probability+cfg.stand_probability)
            march=(choose>=cfg.reference_probability+cfg.stand_probability)&(choose<cfg.reference_probability+cfg.stand_probability+cfg.march_probability)
            cmd[stand|march]=0.
            # Small random commands become standing, not an unsupported creeping command.
            tiny=(cmd[:,:2].norm(dim=1)<.03)&(cmd[:,2].abs()<.08)&~ref&~march
            stand|=tiny;cmd[stand]=0.
            self.commands[env_ids,:3]=cmd;self.gait_enabled[env_ids]=~stand
        scale=self.commands.new_tensor([.3,.15,.5])
        self.reference_id[env_ids]=torch.cdist(self.commands[env_ids,:3]/scale,self.reference_commands/scale).argmin(1)

    def _moving(self):return self.gait_enabled

    def set_command_override(self,commands,gait_enabled=True):
        super().set_command_override(commands)
        self.override_gait_enabled=bool(gait_enabled)
        self.gait_enabled[:]=gait_enabled
        scale=self.commands.new_tensor([.3,.15,.5])
        self.reference_id[:]=torch.cdist(self.commands[:,:3]/scale,self.reference_commands/scale).argmin(1)

    def _reset_root_states(self,env_ids):
        super()._reset_root_states(env_ids)
        self.heading_target[env_ids]=yaw_from_quat(self.root_states[env_ids,3:7])
        self.heading_error[env_ids]=0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.heading_target+=self.commands[:,2]*self.dt
        self.heading_target=torch.atan2(self.heading_target.sin(),self.heading_target.cos())
        error=self.heading_target-yaw_from_quat(self.root_states[:,3:7])
        self.heading_error=torch.atan2(error.sin(),error.cos())

    def _frame(self,gyro,gravity,joint_position,joint_velocity,yaw_error_mean):
        base=super()._frame(gyro,gravity,joint_position,joint_velocity,yaw_error_mean)
        return torch.cat((base,self.heading_error.sin()[:,None],(self.heading_error.cos()-1)[:,None],self.gait_enabled.float()[:,None]),1)

    def compute_observations(self):
        super().compute_observations()
        public=self.obs_buf;privileged=self.privileged_obs_buf
        if public.shape[1]!=self.cfg.env.public_obs_size or privileged.shape[1]!=self.cfg.env.teacher_state_size:
            raise ValueError('CTS public/privileged feature layout changed')
        packed=torch.cat((public,privileged,self.teacher_mask.float()[:,None]),1)
        self.obs_buf=packed;self.privileged_obs_buf=packed

    def _stance_ratio(self):
        reference=self.reference_duties[self.reference_id]
        # Preserve every V2 clip's timing. Beyond its envelope, demanding .68
        # duty at .6 m/s leaves too little return time under RS01 target limits.
        load=(self.commands[:,:3].abs()/self.commands.new_tensor(self.cfg.gait.expert_command_scale)).amax(1)
        blend=(load-1.).clamp(0,1)
        return reference+(self.cfg.gait.fast_stance_ratio-reference)*blend

    def _scheduled_stance(self):
        return (self._leg_phase()<self._stance_ratio()[:,None])|~self.gait_enabled[:,None]

    def _reward_illegal_support(self):
        c=self.get_foot_contact_mask();ordered=c[:,[self.foot_legs.index(x) for x in ['FL','FR','RL','RR']]];n=ordered.sum(1)
        diagonal=(ordered[:,0]&ordered[:,3])|(ordered[:,1]&ordered[:,2])
        return 2*(n==0).float()+(n==1).float()+((n==2)&~diagonal).float()

    def reference_foot_target(self):
        phase=self._clock_phase()*self.reference_bins
        lower=phase.floor().long()%self.reference_bins
        alpha=(phase-phase.floor())[:,None]
        return ((1-alpha)*self.reference_feet[self.reference_id,lower]
                +alpha*self.reference_feet[self.reference_id,(lower+1)%self.reference_bins])

    def _reward_phase_reference(self):
        # One geometry target per leg, in the body frame. No shared front/rear
        # height target, joint-posture target or competing swing-height reward.
        error=(self.amp_terminal[:,24:36]-self.reference_foot_target()).reshape(-1,4,3)
        energy=error.square().sum(2).mean(1)/self.cfg.rewards.reference_foot_sigma_m**2
        self.phase_reference_quality=torch.exp(-.5*energy)*self.gait_enabled.float()
        return self.phase_reference_quality

    def _reward_command_tracking(self):
        cfg=self.cfg.rewards
        correction=(cfg.heading_gain*self.heading_error).clamp(-cfg.heading_correction_limit_rad_s,cfg.heading_correction_limit_rad_s)
        target_yaw=self.commands[:,2]+correction*(self.commands[:,2].abs()<.08)
        # Track cycle means: natural within-cycle body motion in the V2
        # demonstrations must not compete with the style objective.
        mean_yaw=self.commands[:,2]-self.velocity_drift[:,2]
        linear=self.velocity_drift[:,:2].square().sum(1)/cfg.command_linear_sigma_m_s**2
        yaw=(mean_yaw-target_yaw).square()/cfg.command_yaw_sigma_rad_s**2
        self.command_quality=torch.exp(-.5*(linear+yaw))
        return self.command_quality

    def compute_reward(self):
        self.amp_terminal=amp_features(self).clone()
        super().compute_reward()
        slots=[self.foot_legs.index(x) for x in ['FL','FR','RL','RR']]
        self.amp_terminal_contact=self.get_foot_contact_mask()[:,slots].clone()
        self.amp_terminal_command=self.commands[:,:3].clone()
        self.amp_terminal_enabled=self.gait_enabled.clone()
        self.amp_terminal_heading_error=self.heading_error.clone()

    def step(self,actions):
        previous=amp_features(self).clone();command=self.commands[:,:3].clone();reset=self.reset_buf.bool().clone();enabled=self.gait_enabled.clone()
        result=super().step(actions)
        task_valid=valid_transition(reset,result[3],command,self.amp_terminal_command)&(enabled==self.amp_terminal_enabled)
        valid=task_valid&enabled&self.amp_terminal_enabled
        result[4]['amp']={'previous':previous,'next':self.amp_terminal,'command':command,'valid':valid,
                          'task_valid':task_valid,'enabled':enabled,'heading_error':self.amp_terminal_heading_error}
        result[4]['amp']['contact']=self.amp_terminal_contact
        result[4]['amp']['reference_quality']=self.phase_reference_quality
        result[4]['amp']['command_quality']=self.command_quality
        return result
