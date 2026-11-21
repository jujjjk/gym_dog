"""Real command-conditioned feedback, trained and anchored in executed space."""
import copy,hashlib,inspect,json,math
from pathlib import Path
import torch
from torch import nn
from .urdf4_amp_v4 import V4ActorCritic,V4StudentPolicy,Urdf4AMPV4Runner,measured_context,ROOT
from .urdf4_amp_v3 import V3Style
from .urdf4_native_prior import conditioned_native_actions
from .urdf4_phase_feedback import filtered_balance_features,balance_actions
from .urdf4_adaptive_phase import feedback_history
from .urdf4_ground_observer import JointGroundObserver
from .urdf4_public_prior import PublicCentroidalPrior
from .urdf4_public_dynamics import PublicLegDynamics


def fast_mask(public):
    return (public[:,1020]>.35)&(public[:,1020]<.5)&(public[:,1021:1023].abs().sum(1)<.001)


def command_features(public,reference,stand,command_reference,command_stand,scales,offset):
    count=public.shape[0];history=public[:,:1020].reshape(count,20,51)
    phase=torch.remainder(torch.atan2(history[:,:,45],history[:,:,46])/(2*math.pi)-offset,1.)
    index=phase*reference.shape[0];lo=index.floor().long();lo=torch.where(lo>=reference.shape[0],torch.zeros_like(lo),lo)
    hi=lo+1;hi=torch.where(hi>=reference.shape[0],torch.zeros_like(hi),hi);weight=(index-index.floor())[:,:,None]
    difference=command_reference-reference
    expected=(1-weight)*difference[lo]+weight*difference[hi]
    amplitude=public[:,1028:1029].clamp(0,1)[:,None,:]
    static=(command_stand-stand)[None,None,:]
    coefficient=((.5-public[:,1020:1021])/.1).clamp(0.,1.5)[:,None,:]
    expected=(static+amplitude*(expected-static))*coefficient
    centered=history.clone();centered[:,:,:3]-=expected[:,:,:3];centered[:,:,3:5]-=expected[:,:,3:5]
    centered[:,:,9:45]-=expected[:,:,5:41]
    extra=public[:,1020:1029].clone();extra[:,6:8]-=expected[:,0,41:43]
    normalized=torch.cat((centered.flatten(1),extra),1)
    return filtered_balance_features(normalized,reference,stand,scales,offset)


def feedback_components(public,features):
    coefficient=((.5-public[:,1020:1021])/.1).clamp(0.,1.5)
    base=features.reshape(-1,9,44).clone()
    # Avoid a fitted action-copy loop. Strong new feedback must react to the
    # measured plant; the bounded neural residual still has full history.
    base[:,:,30:42]*=(1-coefficient.clamp(max=1.))[:,None,:]
    deviations=features.reshape(-1,9,44).clone();deviations[:,:,0]=0.;deviations[:,:,30:42]=0.
    return base.flatten(1),deviations.flatten(1),coefficient


def command_balance(public,reference,stand,command_reference,command_stand,scales,offset,gain,delta_gain,speed_gain,observed_features,observer_gain,adaptive:bool):
    observed=feedback_history(public,reference,stand,offset,.25,1.,.02,.08) if adaptive else public
    features=command_features(observed,reference,stand,command_reference,command_stand,scales,offset)
    phase=torch.remainder(torch.atan2(public[:,45],public[:,46])/(2*math.pi)-offset,1.)
    basis=[torch.ones_like(phase)]
    for k in range(1,5):basis.extend([torch.sin(2*math.pi*k*phase),torch.cos(2*math.pi*k*phase)])
    # Only interpolate towards the independently executed slower reference.
    # Above .50 there is no qualified timing/stride slope to extrapolate.
    base,deviations,coefficient=feedback_components(public,features)
    logits=base@gain+coefficient*(deviations@delta_gain+torch.stack(basis,1)@speed_gain+observed_features@observer_gain)
    return public[:,1028:1029].clamp(0,1)*torch.tanh(logits)


class V5ActorCritic(V4ActorCritic):
    def __init__(self,*args,v5_adaptive_phase=False,v5_protected_policy=None,v5_urdf=None,v5_joint_names=None,v5_default_q=None,v5_public_prior=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.public_size>1029:
            first=self.student_encoder[0]
            self.student_encoder[0]=nn.Linear(1029,first.out_features)
        self.balance_delta=nn.Parameter(torch.zeros_like(self.balance_gain))
        self.speed_gain=nn.Parameter(torch.zeros(9,12));self.adaptive_phase=v5_adaptive_phase
        self.observer=JointGroundObserver(v5_urdf,v5_joint_names,v5_default_q)
        self.observer_gain=nn.Parameter(torch.zeros(18,12))
        self.register_buffer('command_reference',torch.zeros_like(self.balance_reference))
        self.register_buffer('command_stand',torch.zeros_like(self.balance_stand))
        self.protected_policy=torch.jit.load(v5_protected_policy or str(ROOT/'artifacts/urdf4_speed_fix_20261117/selected/exported/student_50.pt'))
        for parameter in self.protected_policy.parameters():parameter.requires_grad_(False)
        self.use_public_prior=v5_public_prior is not None
        self.public_prior=PublicCentroidalPrior(v5_urdf,v5_joint_names,v5_default_q,v5_public_prior) if self.use_public_prior else nn.Identity()
        if self.use_public_prior:
            self.public_prior.requires_grad_(False)
            for parameter in (self.balance_delta,self.speed_gain,self.observer_gain):parameter.requires_grad_(False)

    def executed_mean(self,public,raw):
        mask=fast_mask(public)
        if self.use_public_prior:
            return torch.where(mask[:,None],self.public_prior(public)+.10*torch.tanh(raw),self.protected_policy(public[:,:1029]))
        template=public.clone();template[mask,1020]=.5
        base=self.protected_policy(template[:,:1029])
        original=balance_actions(public,self.balance_reference,self.balance_stand,self.balance_scales,self.balance_offset,self.balance_gain)
        observed=self.observer.feedback_features(public,self.balance_offset)
        balance=command_balance(public,self.balance_reference,self.balance_stand,self.command_reference,self.command_stand,self.balance_scales,self.balance_offset,self.balance_gain,self.balance_delta,self.speed_gain,observed,self.observer_gain,self.adaptive_phase)
        coefficient=((.5-public[:,1020:1021])/.1).clamp(0.,1.5)
        delta=balance-original+.10*coefficient*torch.tanh(raw)
        return base+torch.where(mask[:,None],delta,torch.zeros_like(delta))

    def student_context(self,public):
        return measured_context(public,self.student_encoder(public[:,:1029]))

    def student_loss(self,observations):
        public,state,role=self.split(observations);student=~role.flatten()
        if not bool(student.any()):raise ValueError('CTS needs student rollout states')
        prediction=self.student_encoder(public[student,:1029])
        with torch.no_grad():target=self.teacher_context(state[student])
        velocity=(prediction[:,:3]-target[:,:3]).square().mean();latent=(prediction[:,3:]-target[:,3:]).square().mean()
        return velocity+latent,velocity,latent


class V5StudentPolicy(nn.Module):
    def __init__(self,model):
        super().__init__();self.base=V4StudentPolicy(model);self.public_size=model.public_size
        self.register_buffer('speed_gain',model.speed_gain.detach().clone());self.adaptive_phase=model.adaptive_phase
        self.register_buffer('balance_delta',model.balance_delta.detach().clone())
        for key in ('command_reference','command_stand'):self.register_buffer(key,getattr(model,key).detach().clone())
        self.protected_policy=copy.deepcopy(model.protected_policy)
        self.observer=copy.deepcopy(model.observer);self.register_buffer('observer_gain',model.observer_gain.detach().clone())
        self.use_public_prior=model.use_public_prior;self.public_prior=copy.deepcopy(model.public_prior)

    def forward(self,history):
        public=history[:,:self.public_size];frame=public[:,:51]
        command=torch.maximum(torch.minimum(frame[:,6:9],self.base.command_max),self.base.command_min)
        direct=torch.cat((frame[:,:6],command,frame[:,9:]),1)*self.base.frame_gain
        raw=self.base.actor(torch.cat((direct,measured_context(public,self.base.encoder(public[:,:1029]))),1))
        if self.use_public_prior:
            mask=fast_mask(public)
            return torch.where(mask[:,None],self.public_prior(public)+.10*torch.tanh(raw),self.protected_policy(public[:,:1029]))
        mask=fast_mask(public);template=public.clone();template[mask,1020]=.5
        base=self.protected_policy(template)
        original=balance_actions(public,self.base.balance_reference,self.base.balance_stand,self.base.balance_scales,self.base.balance_offset,self.base.balance_gain)
        observed=self.observer.feedback_features(public,self.base.balance_offset)
        balance=command_balance(public,self.base.balance_reference,self.base.balance_stand,self.command_reference,self.command_stand,self.base.balance_scales,self.base.balance_offset,self.base.balance_gain,self.balance_delta,self.speed_gain,observed,self.observer_gain,self.adaptive_phase)
        coefficient=((.5-public[:,1020:1021])/.1).clamp(0.,1.5)
        delta=balance-original+.10*coefficient*torch.tanh(raw)
        return base+torch.where(mask[:,None],delta,torch.zeros_like(delta))


class V5Style(V3Style):
    def __init__(self,path,env,settings=None,validation_receipt=None):
        super().__init__(path,env,settings,validation_receipt)
        file=Path(env.cfg.amp_cts.extra_reference);extra=torch.load(file,map_location=env.device);manifest=extra['manifest']
        assert manifest['urdf_sha256']==env.urdf_sha256 and manifest['case']['passed'] and manifest['seconds']>=60
        assert manifest['maximum_independent_FK_error_m']<5.e-5
        assert manifest['plant_settings']['actuator']['peak_torque_limit_nm']==env.cfg.rs01_actuator.peak_torque_limit_nm
        pairs=extra['pairs'];command=extra['commands'];start=len(self.expert);count=len(pairs)
        assert pairs.shape[1]==92 and torch.isfinite(pairs).all()
        self.expert=torch.cat((self.expert,pairs));self.expert_commands=torch.cat((self.expert_commands,command))
        self.bins=torch.cat((self.bins,command[:1]));old=self.table
        table=torch.zeros(len(self.counts)+1,max(old.shape[1],count),device=env.device,dtype=torch.long)
        table[:-1,:old.shape[1]]=old;table[-1,:count]=torch.arange(start,start+count,device=env.device)
        self.table=table;self.counts=torch.cat((self.counts,torch.tensor([count],device=env.device)))
        self.pair_count=len(self.expert);self.sha=hashlib.sha256((self.sha+hashlib.sha256(file.read_bytes()).hexdigest()).encode()).hexdigest()


class Urdf4AMPV5Runner(Urdf4AMPV4Runner):
    actor_class=V5ActorCritic;student_policy_class=V5StudentPolicy;style_class=V5Style
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        train_cfg=copy.deepcopy(train_cfg);train_cfg['policy']['v5_adaptive_phase']=env.cfg.amp_cts.adaptive_feedback_phase
        train_cfg['policy']['v5_protected_policy']=env.cfg.amp_cts.protected_policy
        train_cfg['policy']['v5_public_prior']=env.cfg.amp_cts.public_prior_archive if env.cfg.amp_cts.use_public_prior else None
        train_cfg['policy'].update(v5_urdf=env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT/'unitree_rl_gym')),
            v5_joint_names=list(env.dof_names),v5_default_q=env.default_dof_pos[0].tolist())
        super().__init__(env,train_cfg,log_dir,device)
        model=self.alg.actor_critic
        file=Path(env.cfg.amp_cts.extra_reference);extra=torch.load(file,map_location=env.device)
        model.command_reference.copy_(extra['nominal_reference']);model.command_stand.copy_(extra['nominal_stand'])
        model.observer.configure(model.command_reference,model.command_stand,model.balance_reference,model.balance_stand)
        count=len(extra['actions']);held=extra['holdout'];weight=(~held).float();weight/=weight.sum()
        self.anchor_observations=torch.cat((self.anchor_observations,extra['observations']))
        self.anchor_actions=torch.cat((self.anchor_actions,extra['actions']))
        self.anchor_holdout=torch.cat((self.anchor_holdout,held));self.anchor_weights=torch.cat((.6*self.anchor_weights,.4*weight))
        self.contract.update(extra_gym_reference_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            command_conditioned_feedback=True,base_balance_frozen=True,command_delta_matrix_trainable=True,speed_head_shape=[9,12],speed_head_forward_range=[.35,.5],
            conditional_nominal_reference=True,
            public_derived_observer=['URDF FK ground height','joint/IMU kinematic world vertical velocity'],
            adaptive_feedback_phase=env.cfg.amp_cts.adaptive_feedback_phase,fast_template_forward_m_s=.5,
            fast_template_threshold_m_s=.35,fast_sampling_fraction=.25,
            anchor_target='qualified actual .40/.50 commands and frozen legacy low-speed public policy')
        self.contract['protected_public_policy_sha256']=hashlib.sha256(Path(env.cfg.amp_cts.protected_policy).read_bytes()).hexdigest()
        self.contract.pop('feedback_residual_limit',None)
        helper=Path(inspect.getsourcefile(feedback_history));self.experiment['source_sha256'][str(helper)]=hashlib.sha256(helper.read_bytes()).hexdigest()
        helper=Path(inspect.getsourcefile(JointGroundObserver));self.experiment['source_sha256'][str(helper)]=hashlib.sha256(helper.read_bytes()).hexdigest()
        if model.use_public_prior:
            for helper_class in (PublicLegDynamics,PublicCentroidalPrior):
                helper=Path(inspect.getsourcefile(helper_class));self.experiment['source_sha256'][str(helper)]=hashlib.sha256(helper.read_bytes()).hexdigest()
            self.contract.update(public_centroidal_prior=True,public_prior_sha256=hashlib.sha256(Path(env.cfg.amp_cts.public_prior_archive).read_bytes()).hexdigest(),
                anchor_target='zero bounded NN correction around public physical prior; all other commands use frozen accepted V4',
                command_delta_matrix_trainable=False,inner_yaw_integral=False)
            if model.public_size>1029:
                self.contract['public_extra']=self.contract['public_extra']+['localized_heading_velocity_and_amplitude_history:20x3, newest first']
            self.alg.auxiliary_actor_loss=self._public_prior_anchor_loss
        self._freeze_seed_policy()
        if self.log_dir:(Path(self.log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2,default=str)+'\n')

    def warm_start(self,path):
        saved=torch.load(path,map_location=self.device)
        assert saved['contract']['public_size']==1029 and saved['contract']['urdf_sha256']==self.env.urdf_sha256
        assert saved['contract']['native_prior_sha256']==self.contract['native_prior_sha256']
        missing,unexpected=self.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=False)
        assert not unexpected and all(k in ('speed_gain','balance_delta','command_reference','command_stand','observer_gain') or k.startswith(('protected_policy.','observer.','public_prior.')) for k in missing)
        self.alg.actor_critic.actor[-1].weight.data.zero_();self.alg.actor_critic.actor[-1].bias.data.zero_()
        self.alg.actor_critic.teacher_encoder.requires_grad_(False)
        self._freeze_seed_policy();self.loaded_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.seed_migration=dict(checkpoint=str(path),sha256=self.loaded_checkpoint_sha256,
            optimizers_fresh=True,discriminator_fresh=True,command_head_zero_initialized=True)

    def _public_prior_anchor_loss(self):
        ids=torch.multinomial(self.anchor_weights,512,replacement=True)
        public=self.anchor_observations[ids,:1029];model=self.alg.actor_critic
        raw=model.actor(torch.cat((model.control_frame(public),model.student_context(public).detach()),1))
        residual=.10*torch.tanh(raw)*fast_mask(public)[:,None]
        mse=residual.square().mean();self.last_anchor_error=float(mse.detach())
        return self.cfg['anchor_coefficient']*mse

    def export_student(self,path):
        super().export_student(path);file=Path(path+'.json');metadata=json.loads(file.read_text())
        metadata.update(training_objective='command_conditioned_fast_gait_and_direction',public_size=1029,
            fast_straight_reference=[.5,0.,0.],fast_straight_threshold_m_s=.35,
            phase_clock_fast_straight_override=True,executed_action_anchor=True)
        file.write_text(json.dumps(metadata,indent=2)+'\n')
