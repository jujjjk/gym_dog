"""V3 continuous phase/commands, honest stop and existing external-pose servo."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from .amp_direction_env import Urdf4AMPDirectionRobot
from .amp_cts_env import Urdf4AMPCTSRobot,yaw_from_quat
from .urdf4_omni_env import Urdf4OmniRobot
from .path_tracking import PlanarPathController,wrap,rotate_xy
from .expand_support import command_classes


class V3PathController(PlanarPathController):
    def update(self,external,position,yaw,velocity_world,yaw_rate):
        stop=(external.abs().sum(1)<.001)&(self.last_external.abs().sum(1)>=.001)
        if bool(stop.any()):self.reset_ids(stop.nonzero().flatten(),position[stop],yaw[stop])
        moving=(external[:,:2].norm(dim=1)>.02)&(external[:,2].abs()<.02)
        tangent=rotate_xy(self.reference_yaw,external[:,:2]);tangent/=tangent.norm(dim=1,keepdim=True).clamp(min=1e-6)
        self.reference_xy+=(((position-self.reference_xy)*tangent).sum(1)*moving.float())[:,None]*tangent
        return super().update(external,position,yaw,velocity_world,yaw_rate)


class Urdf4AMPV3Robot(Urdf4AMPDirectionRobot):
    def _init_buffers(self):
        n,dev=self.num_envs,self.device
        self.motion_phase=torch.zeros(n,device=dev)
        self.motion_frequency=torch.full((n,),2.,device=dev)
        self.command_start=torch.zeros(n,3,device=dev)
        self.command_elapsed=torch.zeros(n,device=dev)
        self.previous_target=torch.zeros(n,3,device=dev)
        self.phase_was_active=torch.zeros(n,device=dev,dtype=torch.bool)
        super()._init_buffers()
        self.path=V3PathController(n,self.dt,dev,gains=self.path.gains)
        m=json.loads((Path(self.cfg.amp_cts.reference)/'manifest.json').read_text())
        asset=Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.urdf_sha256=hashlib.sha256(asset.read_bytes()).hexdigest()
        if self.urdf_sha256!=m['urdf_sha256']:raise ValueError('V3 URDF/data mass contract mismatch')
        self.reference_phase_offsets=torch.tensor([c['phase_alignment_offset_cycles'] for c in m['clips']],device=dev)
        # Contact duty is measured; a template's per-leg calibration can differ
        # from its nominal scalar stance_ratio_target.
        duties=[]
        for c in m['clips']:
            with np.load(Path(self.cfg.amp_cts.reference)/(c['clip']+'.npz')) as z:duties.append(float(z['contact'].mean()))
        self.reference_duties=torch.tensor(duties,device=dev)
        self.v3_meter=torch.zeros(8,14,device=dev)
        self.stand_meter=torch.zeros(3,device=dev)

    def _clock_phase(self):return self.motion_phase

    def _select_reference(self,command):
        scale=command.new_tensor([.5,.2,.8]);distance=torch.cdist(command/scale,self.reference_commands/scale)
        moving=command.abs().sum(1)>.001;zero=self.reference_commands.abs().sum(1)<1e-6
        distance=distance+moving[:,None]*zero[None,:]*1.e6
        return distance.argmin(1)

    def _process_dof_props(self,props,env_id):
        props=super()._process_dof_props(props,env_id)
        if self.cfg.env.test or env_id%self.cfg.expand.nominal_stride==0:props['armature'][:]=self.cfg.asset.armature
        return props

    def _resample_commands(self,ids):
        if not len(ids):return
        if self.command_override is not None:
            self.target_commands[ids]=self.command_override[ids];self.target_gait[ids]=self.override_gait_enabled;return
        n=len(ids);cfg=self.cfg.commands;lo,hi,side,yaw=self.cfg.expand.stages[self.curriculum_stage]
        cmd=self.commands.new_zeros(n,3);groups=torch.randint(8,(n,),device=self.device);sign=torch.where(torch.rand(n,device=self.device)>.5,1.,-1.)
        fraction=.35+.65*torch.rand(n,3,device=self.device)
        cmd[:,0]=fraction[:,0]*torch.where(groups==1,lo,hi)
        cmd[:,1]=fraction[:,1]*side*torch.where(groups==3,-1.,1.)
        cmd[:,2]=fraction[:,2]*yaw*torch.where(groups==5,-1.,1.)
        cmd[groups<2,1:]=0.;cmd[(groups==2)|(groups==3),0]=0.;cmd[(groups==2)|(groups==3),2]=0.;cmd[(groups==4)|(groups==5),:2]=0.
        mixed=groups>=6;cmd[groups==6,2]=0.;cmd[mixed,0]*=sign[mixed];cmd[groups==6,1]*=torch.where(torch.rand(int((groups==6).sum()),device=self.device)>.5,1.,-1.)
        cmd[groups==7,1]=0.;cmd[groups==7,2]*=torch.where(torch.rand(int((groups==7).sum()),device=self.device)>.5,1.,-1.)
        lateral_turn=(groups==7)&(torch.rand(n,device=self.device)<.25)
        cmd[lateral_turn,0]=0.;cmd[lateral_turn,1]=fraction[lateral_turn,1]*side*sign[lateral_turn]
        caps=cmd.new_tensor([hi,side,yaw]).expand(n,-1).clone();caps[:,0]=torch.where(cmd[:,0]<0,-lo,hi)
        cmd/=torch.maximum((cmd/caps).norm(dim=1),torch.ones(n,device=self.device))[:,None]
        available=(self.reference_commands[:,0]>=lo-1e-6)&(self.reference_commands[:,0]<=hi+1e-6)&(self.reference_commands[:,1].abs()<=side+1e-6)&(self.reference_commands[:,2].abs()<=yaw+1e-6)&(self.reference_commands.abs().sum(1)>.001)
        bank=self.reference_commands[available];draw=torch.rand(n,device=self.device);reference=draw<cfg.reference_probability
        cmd[reference]=bank[torch.randint(len(bank),(int(reference.sum()),),device=self.device)]
        stand=(draw>=cfg.reference_probability)&(draw<cfg.reference_probability+cfg.stand_probability);cmd[stand]=0.
        self.target_commands[ids]=cmd;self.target_gait[ids]=~stand;self.command_frontier[ids]=~reference&~stand

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if not len(ids):return
        self.motion_phase[ids]=0.;self.motion_frequency[ids]=2.;self.command_start[ids]=0.;self.command_elapsed[ids]=0.;self.previous_target[ids]=0.
        self.phase_was_active[ids]=False

    def set_command_override(self,commands,gait_enabled=True):
        self.command_override=commands.to(self.device).reshape(self.num_envs,3).clone();self.override_gait_enabled=bool(gait_enabled)
        self.target_commands.copy_(self.command_override);self.target_gait[:]=gait_enabled

    def _post_physics_step_callback(self):
        # Accumulate errors for the twist that actually governed this physics
        # step, before preparing the next servo command.
        Urdf4OmniRobot._post_physics_step_callback(self)
        changed=(self.target_commands-self.previous_target).abs().amax(1)>1e-6
        start=changed&(self.external_commands.abs().sum(1)<.001)&(self.target_commands.abs().sum(1)>.001)
        self.command_start[changed]=self.external_commands[changed];self.command_elapsed[changed]=0.;self.previous_target[changed]=self.target_commands[changed]
        target_for_phase=torch.where((self.target_commands.abs().sum(1)>.001)[:,None],self.target_commands,self.external_commands)
        self.reference_id[:]=self._select_reference(target_for_phase)
        if hasattr(self,'reference_phase_offsets'):self.motion_phase[start]=self.reference_phase_offsets[self.reference_id[start]]%1
        settling=self.cold_start&(self.episode_steps*self.dt<self.cfg.expand.cold_settle_s)
        self.command_elapsed+=(~settling).float()*self.dt
        duration=torch.where(self.target_commands[:,0].abs()>.35,self.cfg.commands.fast_ramp_s,self.cfg.commands.ramp_s)
        u=(self.command_elapsed/duration).clamp(0,1);blend=u*u*(3-2*u)
        self.external_commands[:]=self.command_start+(self.target_commands-self.command_start)*blend[:,None];self.external_commands[settling]=0.
        self.gait_enabled[:]=~settling&(self.external_commands.abs().sum(1)>.001)
        phase_active=~settling&((self.target_commands.abs().sum(1)>.001)|self.gait_enabled)
        phase_start=phase_active&~self.phase_was_active
        if hasattr(self,'reference_phase_offsets'):self.motion_phase[phase_start]=self.reference_phase_offsets[self.reference_id[phase_start]]%1
        self.phase_was_active[:]=phase_active
        self.motion_frequency[:]=self.reference_frequencies[self.reference_id].clamp(min=1.8,max=2.3)
        self.motion_phase[:]=torch.remainder(self.motion_phase+self.motion_frequency*self.dt*phase_active.float(),1.)
        self.motion_phase[~phase_active]=0.;self.cold_start&=settling
        xy,yaw,velocity=self._localized_pose();gyro=self.obs_history[:,0,2]/self.obs_scales.ang_vel
        self.commands[:,:3]=self.path.update(self.external_commands,xy,yaw,velocity,gyro) if self.direction_enabled else self.external_commands
        self.commands[:,0].clamp_(-.45,.70);self.commands[settling,:3]=0.
        self.heading_target[:]=self.path.reference_yaw;self.heading_error[:]=wrap(self.heading_target-yaw_from_quat(self.root_states[:,3:7]))
        self.public_heading_target[:]=wrap(self.path.reference_yaw+self.path_public_offset)
        self.yaw_drift+=(~self.nominal_mask).float()*self.imu.gyro_bias[:,2]*self.dt;self.yaw_drift.clamp_(-.035,.035)
        self.recovery_left.sub_(self.dt).clamp_(min=0.)

    def _stance_ratio(self):return self.reference_duties[self.reference_id].clamp(.52,.80)

    def _reward_gait_quality(self):
        contact=self.get_foot_contact_mask();unsafe=self._reward_illegal_support()
        phase=(contact!=self._scheduled_stance()).float().mean(1)
        stand=(~contact).float().mean(1)+.05*self.dof_vel.square().mean(1)
        return unsafe+torch.where(self.gait_enabled,.20*phase*self.recovery_factor(),stand)

    def compute_reward(self):
        Urdf4AMPCTSRobot.compute_reward(self)
        if not hasattr(self,'v3_meter'):return
        stable=(self.external_commands-self.target_commands).abs().amax(1)<1e-5
        mask=self.nominal_mask&~self.teacher_mask&stable&(self.episode_steps>200)
        moving=mask&self.gait_enabled;group=command_classes(self.external_commands)
        slots=[self.foot_legs.index(x) for x in ['FL','FR','RL','RR']];contact=self.get_foot_contact_mask()[:,slots]
        air=(~contact)&((self.feet_pos[:,slots,2]-self.env_origins[:,None,2]-.02)>.012)
        first=contact[:,0]&contact[:,3]&~contact[:,1]&~contact[:,2];second=contact[:,1]&contact[:,2]&~contact[:,0]&~contact[:,3]
        values=torch.stack((self.velocity_drift[:,:2].norm(dim=1),self.velocity_drift[:,2].abs(),
            self.path.cross_error.abs(),self.path.heading_error.abs(),(self._reward_illegal_support()>0).float(),
            (self.reset_buf.bool()&~self.time_out_buf).float(),(self.get_foot_contact_mask()!=self._scheduled_stance()).float().mean(1),
            *[air[:,j].float() for j in range(4)],first.float(),second.float(),torch.ones(self.num_envs,device=self.device)),1)
        self.v3_meter.index_add_(0,group[moving],values[moving])
        stand=mask&~self.gait_enabled
        self.stand_meter+=torch.stack(((self.base_lin_vel[:,:2].norm(dim=1)*stand).sum(),
            ((~self.get_foot_contact_mask()).any(1).float()*stand).sum(),stand.sum()))

    def step(self,actions):
        user=self.external_commands.clone();previous_reset=self.reset_buf.bool().clone();result=super().step(actions);a=result[4]['amp']
        valid=~previous_reset&~result[3].bool();a.update(valid=valid,task_valid=valid,style_command=user,command=user,
            style_factor=self.recovery_factor().clone(),legal=(self._reward_illegal_support()==0).float())
        return result
