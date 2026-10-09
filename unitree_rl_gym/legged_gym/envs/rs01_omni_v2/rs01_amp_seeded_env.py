"""Reference-state initialization ONLY at training resets, never injected motion in rollouts."""
import json
from pathlib import Path
import numpy as np
import torch
from isaacgym import gymtorch
from .rs01_amp_core_env import Rs01AMPCoreRobot


class Rs01AMPSeededRobot(Rs01AMPCoreRobot):
    def __init__(self,*args,**kwargs):
        self.reference_reset_ready=False
        super().__init__(*args,**kwargs)
        root=Path(self.cfg.amp.reset_reference)
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['joint_order']!=list(self.dof_names):raise ValueError('Reference reset joint order mismatch')
        arrays={k:[] for k in ['joint_pos_rad','joint_vel_rad_s','command','root_pos_world_m','leg_phase']}
        for clip in manifest['clips']:
            with np.load(root/(clip['clip']+'.npz')) as d:
                for key in arrays:arrays[key].append(d[key])
        self.reference_reset={k:torch.tensor(np.stack(v),device=self.device,dtype=torch.float32) for k,v in arrays.items()}
        if not all(torch.isfinite(v).all() for v in self.reference_reset.values()):raise ValueError('Nonfinite reset reference')
        self.reference_reset_ready=True

    def reset_idx(self,env_ids):
        super().reset_idx(env_ids)
        if not self.reference_reset_ready or self.cfg.env.test or len(env_ids)==0:return
        chosen=env_ids[torch.rand(len(env_ids),device=self.device)<self.cfg.amp.reference_reset_probability]
        if len(chosen)==0:return
        r=self.reference_reset
        case=torch.cdist(self.commands[chosen,:3],r['command'][:,0]).argmin(1)
        frame=torch.randint(r['joint_pos_rad'].shape[1],(len(chosen),),device=self.device)
        q=r['joint_pos_rad'][case,frame];dq=r['joint_vel_rad_s'][case,frame]
        self.dof_pos[chosen]=q;self.dof_vel[chosen]=dq
        self.root_states[chosen,:3]=self.env_origins[chosen]
        self.root_states[chosen,2]+=r['root_pos_world_m'][case,frame,2]
        self.root_states[chosen,3:7]=q.new_tensor([0,0,0,1])
        self.root_states[chosen,7:13]=0.
        self.root_states[chosen,7:9]=r['command'][case,frame,:2]
        self.root_states[chosen,12]=r['command'][case,frame,2]
        # Initialize sensor/target history coherently. All normal plant limits remain active next step.
        self.rs01_limited_position_target_rad[chosen]=q
        self.rs01_response_target_rad[chosen]=q
        self.rs01_target_delay_buffer[:,chosen,:]=q
        self.rs01_target_rate_rad_s[chosen]=0.
        if hasattr(self,'guard_target'):self.guard_target[chosen]=q  # guard tensors appear on first torque call
        previous=((q-self.default_dof_pos)/self.rs01_action_scale_rad).clamp(-1,1)
        self.actions[chosen]=previous;self.last_actions[chosen]=previous
        self.last_dof_vel[chosen]=dq
        self.wide_gait_phase[chosen]=r['leg_phase'][case,frame,0]
        ids=chosen.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(ids),len(ids))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(ids),len(ids))
        self.rs01_leg_odometry.reset(chosen)
        self.sensor.reset(chosen,self.sensor_tick,*self._fresh_sensor_truth())
        self.sensor.read(self.sensor_tick)
        self._start_direction_segment(chosen)
