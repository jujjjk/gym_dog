import torch
from isaacgym.torch_utils import quat_rotate_inverse
from .rs01_omni_v28_env import Rs01OmniV28Robot


def valid_transition(previous_reset,done,command,next_command):
    return (~done.bool())&(~previous_reset.bool())&torch.isclose(command,next_command,atol=1e-6,rtol=0).all(1)


class Rs01AMPRobot(Rs01OmniV28Robot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        names=[leg+'_'+joint+'_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
        if list(self.dof_names)!=names:raise ValueError('AMP requires explicit FL,FR,RL,RR joint order')

    def _set_command_modes(self,env_ids,modes):
        if not len(env_ids):return
        table=torch.tensor(self.cfg.amp.commands,device=self.device,dtype=torch.float)
        ids=torch.randint(len(table),(len(env_ids),),device=self.device)
        self.commands[env_ids,:3]=table[ids]
        categories=torch.tensor([self.COMMAND_MARCH,self.COMMAND_FORWARD,self.COMMAND_BACKWARD,
            self.COMMAND_LATERAL,self.COMMAND_LATERAL,self.COMMAND_YAW,self.COMMAND_YAW,
            self.COMMAND_COMBINED,self.COMMAND_COMBINED,self.COMMAND_COMBINED],device=self.device)
        self.command_mode[env_ids]=categories[ids];self.gait_enable[env_ids]=1.
        self._reset_command_reference(env_ids)
        self._update_turn_mode(env_ids)

    def _command_gait_frequency(self):
        # A kinematic template has one tempo for every command; a reference recorded in this
        # plant carries the demonstrator's command-dependent clock.
        if self.cfg.amp.command_dependent_clock:return super()._command_gait_frequency()
        return torch.full_like(self.commands[:,0],1. / self.cfg.rewards.gait_period_s)

    def gait_frequencies_for(self,commands):
        """Tempo this env's clock settles at for each constant command row."""
        saved=self.commands[:,:3].clone()
        out=[]
        for row in commands:
            self.commands[:,:3]=row.to(self.device)
            out.append(self._command_gait_frequency()[0])
        self.commands[:,:3]=saved
        return torch.stack(out)

    def amp_features(self):
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        relative=self.feet_pos[:,ids]-self.root_states[:,None,:3]
        quat=self.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4)
        feet=quat_rotate_inverse(quat,relative.reshape(-1,3)).reshape(self.num_envs,12)
        height=(self.root_states[:,2]-self.env_origins[:,2])[:,None]
        return torch.cat((self.dof_pos,self.dof_vel,feet,self.base_lin_vel,
                          self.base_ang_vel,self.projected_gravity,height),dim=1)

    def compute_reward(self):
        super().compute_reward()
        # Captured before automatic reset, NOT next episode's initial state.
        self.amp_next=self.amp_features().clone()
        self.amp_gate=self._legal_task_contact_gate().clone()*self._walking_command_gate()
        self.amp_next_command=self.commands[:,:3].clone()

    def step(self,actions):
        previous=self.amp_features().clone();command=self.commands[:,:3].clone()
        previous_reset=self.reset_buf.bool().clone()
        result=super().step(actions)
        done=result[3].bool()
        valid=valid_transition(previous_reset,done,command,self.amp_next_command)
        result[4]['amp']={'previous':previous,'next':self.amp_next,'command':command,
                          'valid':valid,'gate':self.amp_gate}
        return result
