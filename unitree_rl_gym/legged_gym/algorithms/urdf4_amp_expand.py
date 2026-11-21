"""Coverage-aware AMP and gated speed curriculum; CTS dimensions stay intact."""
import hashlib,json,copy
from pathlib import Path
import torch
from .urdf4_amp_cts import Urdf4AMPCTSRunner,PairStyle,CTSActorCritic
from legged_gym.envs.urdf4.expand_support import reference_coverage

class CoveredStyle(PairStyle):
    def coverage(self,command):
        return reference_coverage(command,self.commands)

class ExpandedActorCritic(CTSActorCritic):
    def __init__(self,*args,command_frame_min,command_frame_max,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('command_min',torch.tensor(command_frame_min))
        self.register_buffer('command_max',torch.tensor(command_frame_max))

    def control_frame(self,obs):
        frame=obs[:,:self.frame_size]
        command=torch.maximum(torch.minimum(frame[:,6:9],self.command_max),self.command_min)
        return torch.cat((frame[:,:6],command,frame[:,9:]),1)*self.frame_gain

class ExpandedStudentPolicy(torch.nn.Module):
    """Bound the direct command view; full commands remain in the encoder."""
    def __init__(self,model):
        super().__init__();self.public_size=model.public_size;self.frame_size=model.frame_size
        self.register_buffer('frame_gain',model.frame_gain.detach().clone())
        self.register_buffer('command_min',model.command_min.detach().clone())
        self.register_buffer('command_max',model.command_max.detach().clone())
        self.encoder=copy.deepcopy(model.student_encoder);self.actor=copy.deepcopy(model.actor)

    def forward(self,history):
        public=history[:,:self.public_size];frame=public[:,:self.frame_size]
        cmd=torch.maximum(torch.minimum(frame[:,6:9],self.command_max),self.command_min)
        direct=torch.cat((frame[:,:6],cmd,frame[:,9:]),1)*self.frame_gain
        return self.actor(torch.cat((direct,self.encoder(public)),1))

class Urdf4AMPExpandRunner(Urdf4AMPCTSRunner):
    style_class=CoveredStyle
    actor_class=ExpandedActorCritic
    student_policy_class=ExpandedStudentPolicy

    def __init__(self,*args,**kwargs):
        self.warm_start_sha256=None;self.curriculum_status={}
        self.fresh_stage_warmstart=False
        super().__init__(*args,**kwargs)

    def _validate_training_request(self,iterations):
        if (self.loaded_checkpoint_sha256 is None or self.cfg.get('training_objective')!='expand_robust'
                or self.env.cfg.rewards.scales.command_tracking<=0
                or self.env.cfg.commands.reference_probability>=1):
            raise ValueError('Expansion requires a loaded seed, enabled tracking and continuous commands')

    def _amp_weights(self,a,style,valid):
        coverage=self.style.coverage(a['command'].to(self.device))
        factor=a['style_factor'].to(self.device)
        weight=coverage*factor
        # Do not let unsupported high-speed/push transients teach the D an
        # easy velocity-only discriminator against low-speed experts.
        replay=valid&(coverage>=.5)&(factor>=.999)
        return weight,replay

    def _after_update(self,metrics):
        env=self.env;cfg=env.cfg.expand
        if self.current_learning_iteration%cfg.curriculum_window==0:
            meter=env.curriculum_meter.clone();env.curriculum_meter.zero_()
            counts=meter[:,4];means=meter[:,[0,1,2,3,5,6,7]]/counts[:,None].clamp(min=1)
            ready=bool((counts>=cfg.curriculum_min_samples_per_direction).all())
            passed=ready and bool((means[:,0]<=cfg.curriculum_linear_error_m_s).all()
                and (means[:,1]<=cfg.curriculum_yaw_error_rad_s).all()
                and (means[:,2]<=cfg.curriculum_illegal_fraction).all()
                and (means[:,3]<=cfg.curriculum_failure_fraction).all()
                and (means[:,4]<=cfg.curriculum_max_phase_error).all()
                and (means[:,5:]>=cfg.curriculum_min_diagonal_fraction).all())
            if self.current_learning_iteration>=cfg.curriculum_min_updates:
                env.curriculum_pass_windows=env.curriculum_pass_windows+1 if passed else 0
            if env.curriculum_pass_windows>=cfg.curriculum_required_windows and env.curriculum_stage<cfg.max_stage:
                env.curriculum_stage+=1;env.curriculum_pass_windows=0
            self.curriculum_status=dict(curriculum_ready=float(ready),curriculum_window_passed=float(passed),
                curriculum_worst_linear_m_s=float(means[:,0].max()),curriculum_worst_yaw_rad_s=float(means[:,1].max()),
                curriculum_min_direction_samples=float(counts.min()),
                curriculum_worst_phase_error=float(means[:,4].max()),curriculum_min_diagonal_fraction=float(means[:,5:].min()))
        return dict(self.curriculum_status,curriculum_stage=env.curriculum_stage,
            curriculum_pass_windows=env.curriculum_pass_windows,nominal_fraction=float(env.nominal_mask.float().mean()),
            push_count=float(env.push_count.sum()),recovery_fraction=float((env.recovery_left>0).float().mean()))

    def load(self,path,load_optimizer=True):
        saved=torch.load(path,map_location=self.device)
        schema=saved.get('contract',{}).get('schema')
        legacy=schema=='v3_rsi_pair_amp'
        x_migration=(schema=='v4_continuous_robust_bounded' and getattr(self.env.cfg.expand,'x_only',False))
        if legacy or x_migration:
            expected=dict(self.contract,schema=schema)
            if saved['contract']!=expected:raise ValueError('Legacy seed has different public/control/data contract')
            model_state=dict(saved['model_state_dict'])
            model_state.update(command_min=self.alg.actor_critic.command_min,command_max=self.alg.actor_critic.command_max)
            self.alg.actor_critic.load_state_dict(model_state,strict=True)
            self.style.load_state_dict(saved['amp_state'])
            # These privileged columns were constant throughout nominal V3
            # training. Their old weights are not calibrated for DR. Rebase
            # friction into the bias and start the newly varying columns at
            # zero, preserving the nominal function while allowing learning.
            unseen=list(range(67,75))+list(range(87,99))
            if legacy:
                with torch.no_grad():
                    for layer in (self.alg.actor_critic.teacher_encoder[0],self.alg.actor_critic.critic[0]):
                        layer.bias.add_(.8*layer.weight[:,70])
                        layer.weight[:,unseen]=0.
            # Anchor the selected 17800 network, not its original BC ancestor.
            self._freeze_seed_policy(saved.get('seed_policy_state_dict') if x_migration else None);self.current_learning_iteration=0
            self.fresh_stage_warmstart=True
            self.loaded_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.warm_start_sha256=self.loaded_checkpoint_sha256
            print('EXPAND WARM START: selected actor/encoders/D loaded; fresh optimizers and curriculum counter.',flush=True)
        else:
            self.fresh_stage_warmstart=False
            super().load(path,load_optimizer=load_optimizer)
            state=saved.get('expand_state',{})
            self.env.curriculum_stage=int(state.get('stage',0));self.env.curriculum_pass_windows=int(state.get('pass_windows',0))
            if self.env.curriculum_stage>self.env.cfg.expand.max_stage:
                if self.env.cfg.env.test:self.env.cfg.expand.max_stage=self.env.curriculum_stage
                else:raise ValueError('Checkpoint curriculum exceeds configured target stage')
            if 'meter' in state:
                if state['meter'].shape==self.env.curriculum_meter.shape:self.env.curriculum_meter.copy_(state['meter'].to(self.env.device))
                else:
                    self.env.curriculum_meter.zero_();self.env.curriculum_pass_windows=0
                    print('Curriculum history reset: checkpoint predates the diagonal-support gates.',flush=True)
            self.warm_start_sha256=saved.get('warm_start_sha256')
        self.env.reset()
        return saved.get('infos')

    def save(self,path,infos=None):
        super().save(path,infos)
        saved=torch.load(path,map_location=self.device)
        saved['expand_state']=dict(stage=self.env.curriculum_stage,pass_windows=self.env.curriculum_pass_windows,
            meter=self.env.curriculum_meter.detach().cpu())
        saved['warm_start_sha256']=self.warm_start_sha256
        torch.save(saved,path)

    def export_student(self,path):
        super().export_student(path)
        meta_path=Path(path+'.json');meta=json.loads(meta_path.read_text())
        meta.update(training_objective='expand_robust',private_training_fields=['PPO_anchor_weight'],
            direct_command_bounds_scaled=[self.alg.actor_critic.command_min.tolist(),self.alg.actor_critic.command_max.tolist()],
            command_context='full requested command remains in all public history frames and CTS context',
            student_heading='integrated command minus delayed, disturbed IMU heading; no simulator truth',
            expansion=dict(current_stage=self.env.curriculum_stage,max_stage=self.env.cfg.expand.max_stage,stages=self.env.cfg.expand.stages,
                command_acceleration=self.env.cfg.expand.command_acceleration,
                x_only=getattr(self.env.cfg.expand,'x_only',False)))
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
