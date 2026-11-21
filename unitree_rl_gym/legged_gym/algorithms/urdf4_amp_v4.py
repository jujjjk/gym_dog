"""Public velocity feedback and native phase sampling, with an expert anchor."""
import copy,hashlib,inspect,json
from pathlib import Path
from typing import Tuple
import torch
from torch.distributions import Normal
from .urdf4_amp_v3 import V3ActorCritic,V3StudentPolicy,Urdf4AMPV3Runner
from .urdf4_native_prior import conditioned_native_actions
from .urdf4_phase_feedback import balance_actions
from legged_gym import LEGGED_GYM_ROOT_DIR
ROOT=Path(LEGGED_GYM_ROOT_DIR).parent


def measured_context(public,encoded):
    return torch.cat((public[:,1026:1028],encoded[:,2:]),1)


def fast_gate(public):
    u=((public[:,1020:1021]-.35)/.10).clamp(0,1)
    return u*u*(3-2*u)


def reflected_history(public):
    # For this URDF a left/right reflection reverses all three joint axes
    # and advances the diagonal clock by half a cycle.
    n = public.shape[0]
    history = public[:, :1020].reshape(n, 20, 51)
    reflected = history.clone()
    reflected[:, :, 0] *= -1.  # roll rate
    reflected[:, :, 2] *= -1.  # yaw rate
    reflected[:, :, 4] *= -1.  # lateral gravity
    reflected[:, :, 7:9] *= -1.  # lateral command and yaw command
    reflected[:, :, 9:45] = -history[:, :, 9:45].reshape(n, 20, 3, 2, 2, 3).flip([4]).reshape(n, 20, 36)
    reflected[:, :, 45:49] *= -1.  # clock sine/cosine, yaw drift and heading sine
    extra = public[:, 1020:1029].clone()
    extra[:, 1:3] *= -1.
    extra[:, 4:6] *= -1.
    extra[:, 7] *= -1.
    return torch.cat((reflected.flatten(1), extra), 1)


def coordination_mix(public, mirror_share: float, fast_balance: float):
    external = public[:, 1023:1026]
    forward = ((external[:, :1] > .025) & (external[:, 1:].abs().sum(1, keepdim=True) < .001)).float()
    gate = fast_gate(public)
    return mirror_share*(1.-gate)*forward, (fast_balance-1.)*gate*forward, gate*forward


def mix_actions(action, reflection, balance, reflected_balance, share, scale):
    nominal = action + scale*balance
    opposite = reflection + scale*reflected_balance
    return nominal*(1.-share) - opposite.reshape(-1, 2, 2, 3).flip([2]).reshape(-1, 12)*share


def touchdown_assist(public, action, active, settings: Tuple[float, float, float]):
    depth, onset, rear_ratio = settings
    if depth == 0.:
        return action
    phase = torch.atan2(public[:, 45:46], public[:, 46:47])/(2*torch.pi)
    diagonal_offsets = torch.tensor([.5, 0., 0., .5], device=public.device, dtype=public.dtype)
    distance = torch.remainder(phase-onset+diagonal_offsets, 1.)-.5
    pulse = torch.exp(-.5*(distance/.065).square())
    # Neutral-pose URDF Jacobian, in canonical FL, FR, RL, RR order.
    # Counter the knee extension's fore-aft component at each thigh.
    leg_scales = torch.tensor([1., -1., rear_ratio, -rear_ratio], device=public.device, dtype=public.dtype)
    knee = pulse*leg_scales*depth*active/(.1235*.25)
    adjusted = action.clone()
    adjusted[:, 1::3] -= .515*knee
    adjusted[:, 2::3] += knee
    return adjusted


class V4ActorCritic(V3ActorCritic):
    def __init__(self,*args,v3_reference_count,v4_feedback_features=43,
                 coordination_mirror_share=0.,coordination_fast_balance=1.,
                 coordination_landing=(0., -.16, 1.),**kwargs):
        super().__init__(*args,v3_reference_count=v3_reference_count,**kwargs)
        self.coordination_mirror_share = float(coordination_mirror_share)
        self.coordination_fast_balance = float(coordination_fast_balance)
        self.coordination_landing = tuple(coordination_landing)
        self.coord_enabled = (self.coordination_mirror_share > 0. or self.coordination_fast_balance != 1.
                              or self.coordination_landing[0] > 0.)
        self.register_buffer('native_table',torch.zeros(v3_reference_count,26,12))
        self.register_buffer('native_counts',torch.full((v3_reference_count,),25,dtype=torch.long))
        self.register_buffer('native_offsets',torch.zeros(v3_reference_count))
        self.register_buffer('native_stands',torch.zeros(v3_reference_count,12))
        width=v4_feedback_features
        self.register_buffer('balance_reference',torch.zeros(24,width));self.register_buffer('balance_stand',torch.zeros(width))
        self.register_buffer('balance_scales',torch.ones(width));self.register_buffer('balance_offset',torch.tensor(0.))
        self.register_buffer('balance_gain',torch.zeros(9*(width+1),12))
        self.legacy_policy=torch.jit.load(str(ROOT/'artifacts/urdf4_v3_selection_20261117/best/student.pt'))
        for parameter in self.legacy_policy.parameters():parameter.requires_grad_(False)

    def student_context(self,public):return measured_context(public,self.student_encoder(public))

    def context(self,obs):
        public,state,role=self.split(obs)
        return torch.where(role,self.teacher_context(state),self.student_context(public).detach())

    def _mean_parts(self,public,raw):
        prior=conditioned_native_actions(public,self.prior_commands,self.native_table,self.native_counts,self.native_offsets,self.native_stands,self.stand_action)
        balance=balance_actions(public,self.balance_reference,self.balance_stand,self.balance_scales,self.balance_offset,self.balance_gain)
        candidate=prior+balance+(.02+.08*public[:,1028:1029].clamp(0,1))*torch.tanh(raw)
        legacy=self.legacy_policy(public[:,:1020])+.05*torch.tanh(raw);gate=fast_gate(public)
        return legacy+gate*(candidate-legacy), balance

    def executed_mean(self,public,raw,detach_context=True):
        action, balance = self._mean_parts(public, raw)
        if not self.coord_enabled:
            return action
        reflected = reflected_history(public)
        mirrored_context = self.student_context(reflected)
        if detach_context:
            mirrored_context = mirrored_context.detach()
        mirrored_raw = self.actor(torch.cat((self.control_frame(reflected), mirrored_context), 1))
        reflection, reflected_balance = self._mean_parts(reflected, mirrored_raw)
        share, scale, active = coordination_mix(public, self.coordination_mirror_share, self.coordination_fast_balance)
        blended = mix_actions(action, reflection, balance, reflected_balance, share, scale)
        return touchdown_assist(public, blended, active, self.coordination_landing)

    def public_mean(self,public,detach_context=False):
        context=self.student_context(public)
        if detach_context:context=context.detach()
        raw=self.actor(torch.cat((self.control_frame(public),context),1))
        return self.executed_mean(public,raw,detach_context)

    def act_inference(self,obs):return self.public_mean(obs[:,:self.public_size])

    def update_distribution(self,observations):
        raw=self.actor(torch.cat((self.control_frame(observations),self.context(observations)),1))
        mean=self.executed_mean(observations,raw);self.distribution=Normal(mean,self.std.clamp(.025,.10).expand_as(mean))


class V4StudentPolicy(V3StudentPolicy):
    def __init__(self,model):
        super().__init__(model)
        for name in ['native_table','native_counts','native_offsets','native_stands','balance_reference','balance_stand','balance_scales','balance_offset','balance_gain']:
            self.register_buffer(name,getattr(model,name).detach().clone())
        self.legacy_policy=copy.deepcopy(model.legacy_policy)
        self.coordination_mirror_share = model.coordination_mirror_share
        self.coordination_fast_balance = model.coordination_fast_balance
        self.coordination_landing = model.coordination_landing
        self.coord_enabled = model.coord_enabled

    def _mean_parts(self,public):
        frame=public[:,:self.frame_size]
        command=torch.maximum(torch.minimum(frame[:,6:9],self.command_max),self.command_min)
        direct=torch.cat((frame[:,:6],command,frame[:,9:]),1)*self.frame_gain
        context=measured_context(public,self.encoder(public));raw=self.actor(torch.cat((direct,context),1))
        prior=conditioned_native_actions(public,self.prior_commands,self.native_table,self.native_counts,self.native_offsets,self.native_stands,self.stand_action)
        balance=balance_actions(public,self.balance_reference,self.balance_stand,self.balance_scales,self.balance_offset,self.balance_gain)
        candidate=prior+balance+(.02+.08*public[:,1028:1029].clamp(0,1))*torch.tanh(raw)
        legacy=self.legacy_policy(public[:,:1020])+.05*torch.tanh(raw);gate=fast_gate(public)
        return legacy+gate*(candidate-legacy), balance

    def forward(self,history):
        public = history[:, :self.public_size]
        action, balance = self._mean_parts(public)
        if not self.coord_enabled:
            return action
        reflection, reflected_balance = self._mean_parts(reflected_history(public))
        share, scale, active = coordination_mix(public, self.coordination_mirror_share, self.coordination_fast_balance)
        blended = mix_actions(action, reflection, balance, reflected_balance, share, scale)
        return touchdown_assist(public, blended, active, self.coordination_landing)


class Urdf4AMPV4Runner(Urdf4AMPV3Runner):
    actor_class=V4ActorCritic;student_policy_class=V4StudentPolicy
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        train_cfg=copy.deepcopy(train_cfg)
        train_cfg['policy']['v4_feedback_features']=torch.load(env.cfg.amp_cts.balance_archive,map_location='cpu')['reference'].shape[1]
        super().__init__(env,train_cfg,log_dir,device)
        file=Path(env.cfg.amp_cts.native_prior);native=torch.load(file,map_location=self.device);model=self.alg.actor_critic
        assert native['urdf_sha256']==env.urdf_sha256 and torch.equal(native['commands'],env.reference_commands)
        assert native['reference_manifest_sha256']==hashlib.sha256((Path(env.cfg.amp_cts.reference)/'manifest.json').read_bytes()).hexdigest()
        model.native_table.copy_(native['actions']);model.native_counts.copy_(native['counts']);model.native_offsets.copy_(native['offsets']);model.native_stands.copy_(native['stand_actions'])
        balance_file=Path(env.cfg.amp_cts.balance_archive);balance=torch.load(balance_file,map_location=self.device)
        assert balance.get('temporal',False) and balance['urdf_sha256']==env.urdf_sha256
        for field in ['reference','stand','scales','offset']:getattr(model,'balance_'+field).copy_(balance[field])
        model.balance_gain.copy_(balance['gains'][env.cfg.amp_cts.balance_variant])
        model.teacher_encoder.requires_grad_(False)
        self.contract.update(native_prior_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),feedback_residual_limit=.35,
            external_velocity_public=True,public_extra=['motion_goal:3','ramped_external_command:3','localized_heading_frame_velocity:2','motion_amplitude:1'],teacher_context_frozen=True)
        self.contract.update(low_speed_legacy_sha256=hashlib.sha256((ROOT/'artifacts/urdf4_v3_selection_20261117/best/student.pt').read_bytes()).hexdigest(),
            fast_gate_forward_m_s=[.35,.45])
        self.contract.update(balance_archive_sha256=hashlib.sha256(balance_file.read_bytes()).hexdigest(),balance_variant=env.cfg.amp_cts.balance_variant,
            neural_residual_limit=.10,low_speed_neural_limit=.05,balance_feedback_limit=1.,balance_temporal=True,balance_input='public IMU/joints/commands and corrected planar localization velocity')
        data_file=Path(env.cfg.amp_cts.feedback_samples)
        self.contract['feedback_samples_sha256']=hashlib.sha256(data_file.read_bytes()).hexdigest()
        data=torch.load(data_file,map_location=self.device);self.anchor_observations=data['observations'];self.anchor_actions=data['actions']
        assert self.anchor_observations.shape[1]==1130
        clips=data['clip'];counts=torch.bincount(clips);weights=1/counts[clips].float()
        weights*=torch.where(self.anchor_observations[:,1020]>.35,80.,1.);weights[data['holdout']]=0.
        with torch.no_grad():
            legacy=torch.cat([model.legacy_policy(row[:,:1020]) for row in self.anchor_observations.split(2048)])
            gate=fast_gate(self.anchor_observations);self.anchor_actions=legacy+gate*(self.anchor_actions-legacy)
        self.contract['anchor_target']='qualified fast expert; frozen validated V3 public policy below the fast gate'
        self.anchor_weights=weights/weights.sum();self.anchor_holdout=data['holdout'];self.last_anchor_error=0.
        self.alg.auxiliary_actor_loss=self._anchor_loss
        for function in [conditioned_native_actions,balance_actions]:
            file=Path(inspect.getsourcefile(function));self.experiment['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        self._freeze_seed_policy()
        if self.log_dir:(Path(self.log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2,default=str)+'\n')

    def _anchor_loss(self):
        ids=torch.multinomial(self.anchor_weights,512,replacement=True)
        predicted=self.alg.actor_critic.public_mean(self.anchor_observations[ids,:1029],detach_context=True)
        mse=(predicted-self.anchor_actions[ids]).square().mean();self.last_anchor_error=float(mse.detach())
        return self.cfg['anchor_coefficient']*mse

    def warm_start(self,path):
        saved=torch.load(path,map_location=self.device);assert saved['contract']['public_size']==1020
        model=self.alg.actor_critic;current=model.state_dict()
        for key,value in saved['model_state_dict'].items():
            if key.startswith(('actor.','teacher_encoder.','student_encoder.')):
                if key=='student_encoder.0.weight':current[key].zero_();current[key][:,:1020]=value
                else:
                    assert current[key].shape==value.shape;current[key]=value
        model.load_state_dict(current);model.actor[-1].weight.data.zero_();model.actor[-1].bias.data.zero_()
        model.std.data.fill_(.05);model.teacher_encoder.requires_grad_(False)
        self._freeze_seed_policy();self.loaded_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.seed_migration=dict(checkpoint=str(path),sha256=self.loaded_checkpoint_sha256,critic_fresh=True,discriminator_fresh=True,new_public_columns_zero_initialized=True)

    def _after_update(self,metrics):
        result=super()._after_update(metrics);self.alg.actor_critic.teacher_encoder.requires_grad_(False)
        result.update(expert_action_anchor_mse=self.last_anchor_error,teacher_context_frozen=1.)
        return result

    def export_student(self,path):
        super().export_student(path);file=Path(path+'.json');meta=json.loads(file.read_text())
        meta.update(public_extra=self.contract.get('public_extra'),native_temporal_knots=True,
            requires_corrected_external_velocity=True,expert_anchor_at_training_only=True,
            fast_cross_yaw_feedback=dict(gain_rad_s_per_m=2.,limit_rad_s=.12,enabled_with_speed_feedback=True),
            speed_feedback_enabled=self.env.cfg.direction.speed_feedback_enabled,speed_feedback=self.env.cfg.direction.speed_feedback)
        file.write_text(json.dumps(meta,indent=2)+'\n')
