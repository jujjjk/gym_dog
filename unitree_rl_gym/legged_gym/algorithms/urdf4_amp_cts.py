"""Concurrent teacher/student PPO with a shared policy and validated V2 AMP.

The role bit is stored with each transition: PPO recomputes the exact branch
that generated its old likelihood. Student context is trained by supervision
on student-visited states; PPO cannot change the student encoder.
"""
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from legged_gym.envs.urdf4.amp_reference import Urdf4AMPReference


def mlp(sizes):
    layers=[]
    for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(sizes)-2:layers.append(nn.ELU())
    return nn.Sequential(*layers)


class StudentPolicy(nn.Module):
    """Export boundary: only public sensor/action/command history enters."""
    def __init__(self,model):
        super().__init__()
        self.public_size=model.public_size;self.frame_size=model.frame_size
        self.encoder=copy.deepcopy(model.student_encoder)
        self.actor=copy.deepcopy(model.actor)

    def forward(self,public_history):
        return self.actor(torch.cat((public_history[:,:self.frame_size],self.encoder(public_history)),1))


class CTSActorCritic(ActorCritic):
    def __init__(self,public_size,teacher_size,frame_size,num_actions,latent_size=16,
                 encoder_hidden_dims=(256,128),**kwargs):
        context_size=latent_size+3
        super().__init__(frame_size+context_size,teacher_size+context_size+1,num_actions,**kwargs)
        self.public_size=public_size;self.teacher_size=teacher_size;self.frame_size=frame_size
        self.teacher_encoder=mlp([teacher_size,*encoder_hidden_dims,latent_size])
        self.student_encoder=mlp([public_size,*encoder_hidden_dims,context_size])

    def split(self,obs):
        return obs[:,:self.public_size],obs[:,self.public_size:self.public_size+self.teacher_size],obs[:,-1:]>0.5

    def teacher_context(self,state):
        # Privileged frame begins with the same 51 public fields, then body velocity * 2.
        return torch.cat((state[:,self.frame_size:self.frame_size+3]/2.,self.teacher_encoder(state)),1)

    def context(self,obs):
        public,state,role=self.split(obs)
        return torch.where(role,self.teacher_context(state),self.student_encoder(public).detach())

    def update_distribution(self,observations):
        mean=self.actor(torch.cat((observations[:,:self.frame_size],self.context(observations)),1))
        self.distribution=Normal(mean,self.std.clamp(min=.05,max=.35).expand_as(mean))

    def evaluate(self,critic_observations,**kwargs):
        _,state,role=self.split(critic_observations)
        return self.critic(torch.cat((state,self.context(critic_observations).detach(),role.float()),1))

    def act_inference(self,observations):
        public=observations[:,:self.public_size]
        return self.actor(torch.cat((public[:,:self.frame_size],self.student_encoder(public)),1))

    def student_loss(self,observations):
        public,state,role=self.split(observations)
        student=~role.flatten()
        if not bool(student.any()):raise ValueError('CTS needs student rollout states')
        prediction=self.student_encoder(public[student])
        with torch.no_grad():target=self.teacher_context(state[student])
        velocity=(prediction[:,:3]-target[:,:3]).square().mean()
        latent=(prediction[:,3:]-target[:,3:]).square().mean()
        return velocity+latent,velocity,latent


class OmniStyle(Urdf4AMPReference):
    """Match command *classes*, so the discriminator cannot classify by range.

    The expert is not retimed. Outside its velocity coverage its bonus fades;
    actual velocities remain controlled by the task reward, not discriminator.
    """
    def canonical(self,command):
        scale=command.new_tensor([.3,.15,.5])
        distance=((command[:,None,:]-self.commands[None,:,:])/scale).square().sum(2)
        return distance.min(1)

    def inputs(self,pair,command):
        _,idx=self.canonical(command)
        return super().inputs(pair,self.commands[idx])

    def coverage(self,command):
        distance,_=self.canonical(command)
        return torch.exp(-distance).clamp(min=.1)


class CycleStyle(OmniStyle):
    """Six state/contact samples spanning a complete 0.5 s V2 gait cycle."""
    def __init__(self,path,env,settings,**kwargs):
        super().__init__(path,env,settings=settings,**kwargs)
        self.frames=int(settings['amp_window_frames']);self.stride=int(settings['amp_window_stride'])
        self.span=(self.frames-1)*self.stride
        if abs(self.span*env.dt-env.cfg.gait.period_s)>1e-6:
            raise ValueError('AMP window must span one complete reference cycle')
        manifest=json.loads((Path(path)/'manifest.json').read_text());windows=[]
        for clip in manifest['clips']:
            with np.load(Path(path)/(clip['clip']+'.npz'),allow_pickle=False) as data:
                state=np.concatenate((data['amp_features'],data['contact'].astype(np.float32)),1)
                if state.shape[1]!=50 or not np.isfinite(state).all():raise ValueError('Invalid cycle AMP state')
                ends=np.arange(self.span,len(state))
                # Every intermediate transition must be valid, not just the six sampled frames.
                valid=np.stack([data['transition_valid'][ends-i-1] for i in range(self.span)]).all(0)
                ends=ends[valid]
                ids=ends[:,None]-np.arange(self.frames-1,-1,-1)[None,:]*self.stride
                windows.append(torch.tensor(state[ids].reshape(len(ends),-1),device=env.device))
        self.expert=torch.stack(windows)
        self.mean=torch.cat((self.mean,self.mean.new_zeros(4)))
        self.std=torch.cat((self.std,self.std.new_ones(4)))
        self.feature_weights=torch.cat((self.feature_weights,self.feature_weights.new_ones(4)))
        width=self.frames*50+3
        self.net=mlp([width,256,128,1]).to(env.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-4,weight_decay=1e-4)
        self.buffer=torch.zeros(131072,width,device=env.device)
        self.history=torch.zeros(self.span+1,env.num_envs,50,device=env.device)
        self.history_valid=torch.zeros(self.span+1,env.num_envs,device=env.device,dtype=torch.bool)
        self.history_cursor=0;self.history_count=0

    def inputs(self,pair,command):
        _,idx=self.canonical(command)
        norm=((pair-self.mean.repeat(self.frames))/self.std.repeat(self.frames)).clamp(-10,10)
        return torch.cat((norm*self.feature_weights.repeat(self.frames),
                          self.commands[idx]/command.new_tensor([.3,.15,.5])),1)

    def rollout_reward(self,a):
        state=torch.cat((a['next'].to(self.history.device),a['contact'].to(self.history.device).float()),1)
        slot=self.history_cursor;self.history[slot]=state;self.history_valid[slot]=a['valid'].to(state.device)
        indices=(slot-torch.arange(self.frames-1,-1,-1,device=state.device)*self.stride)%(self.span+1)
        pair=self.history[indices].permute(1,0,2).reshape(len(state),-1)
        transitions=(slot-torch.arange(self.span,device=state.device))%(self.span+1)
        valid=self.history_valid[transitions].all(0)&(self.history_count>=self.span)
        self.history_cursor=(slot+1)%(self.span+1);self.history_count+=1
        x=self.inputs(pair,a['command'].to(state.device))
        score=self.net(x).flatten()
        return (1-.25*(score-1).square()).clamp(0,1),x,valid

    def clear_history(self):
        self.history.zero_();self.history_valid.zero_();self.history_cursor=0;self.history_count=0

    def update(self):
        if self.size<32:return
        stats=[]
        for _ in range(self.disc_steps):
            fake=self.buffer[torch.randint(self.size,(512,),device=self.buffer.device)].clone()
            raw_command=fake[:,-3:]*fake.new_tensor([.3,.15,.5])
            _,ci=self.canonical(raw_command)
            ti=torch.randint(self.expert.shape[1],(len(fake),),device=fake.device)
            real=self.inputs(self.expert[ci,ti],self.commands[ci]).detach().requires_grad_(True)
            dr=self.net(real);df=self.net(fake)
            grad=torch.autograd.grad(dr.sum(),real,create_graph=True)[0][:,:-3]
            gp=grad.square().sum(1).mean()
            loss=.5*((dr-1).square().mean()+(df+1).square().mean())+10.*gp
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite cycle discriminator')
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.optimizer.step()
            stats.append([loss.item(),dr.mean().item(),df.mean().item(),gp.item()])
        self.updates+=1
        self.metrics=dict(zip(('disc_loss','expert_score','policy_score','gradient_penalty'),np.mean(stats,axis=0)))


class Urdf4AMPCTSRunner:
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        if env.cfg.env.test:log_dir=None  # Playback must not create a newer empty training run.
        self.env=env;self.device=torch.device(device);self.train_cfg=train_cfg;self.cfg=train_cfg['runner']
        self.log_dir=log_dir;self.current_learning_iteration=0
        self.num_steps_per_env=self.cfg['num_steps_per_env']
        self.writer=SummaryWriter(log_dir) if log_dir else None
        dims=env.cfg.env
        model=CTSActorCritic(dims.public_obs_size,dims.teacher_state_size,dims.frame_size,
                             env.num_actions,**train_cfg['policy']).to(self.device)
        self.alg=PPO(model,device=self.device,**train_cfg['algorithm'])
        # Keep the student encoder out of the PPO optimizer altogether.
        params=[p for name,p in model.named_parameters() if not name.startswith('student_encoder.')]
        self.alg.optimizer=torch.optim.Adam(params,lr=self.alg.learning_rate)
        self.encoder_optimizer=torch.optim.Adam(model.student_encoder.parameters(),lr=self.cfg['encoder_learning_rate'])
        self.alg.init_storage(env.num_envs,self.num_steps_per_env,[env.num_obs],[env.num_privileged_obs],[env.num_actions])
        self.style=CycleStyle(env.cfg.amp_cts.reference,env,settings=self.cfg,
                             validation_receipt=env.cfg.amp_cts.receipt)
        plant=self.style.reference_settings
        actual={'policy_dt_s':env.dt,'physics_dt_s':env.sim_params.dt,
                'delay_s':env.cfg.rs01_actuator.closed_loop_delay_s,
                'response_tau_s':env.cfg.rs01_actuator.time_constant_s,
                'response_gain':env.cfg.rs01_actuator.response_gain,
                'motor_cap_Nm':env.cfg.rs01_actuator.peak_torque_limit_nm}
        if any(abs(plant[k]-v)>1e-8 for k,v in actual.items()):
            raise ValueError('AMP reference actuator/time contract differs from this task')
        if any(abs(v-plant['kp'])>1e-8 for v in env.cfg.control.stiffness.values()) or any(abs(v-plant['kd'])>1e-8 for v in env.cfg.control.damping.values()):
            raise ValueError('AMP reference PD contract differs from this task')
        self.contract={'schema':env.cfg.amp_cts.reward_revision,'public_size':dims.public_obs_size,
                       'amp_window_frames':self.style.frames,'amp_window_stride':self.style.stride,
                       'teacher_size':dims.teacher_state_size,'frame_size':dims.frame_size,
                       'reference_sha256':self.style.sha,'dt':env.dt,
                       'joint_order':list(env.dof_names),'foot_order':list(env.foot_legs)}
        from legged_gym.utils.helpers import class_to_dict
        self.experiment={'contract':self.contract,'env':class_to_dict(env.cfg),'train':train_cfg}
        sources={}
        for cls in [*type(env).__mro__[:-1],type(self),type(model),PPO,OmniStyle,Urdf4AMPReference]:
            path=inspect.getsourcefile(cls)
            if path and Path(path).is_file():sources[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.experiment['source_sha256']=sources
        if log_dir:
            Path(log_dir).mkdir(parents=True,exist_ok=True)
            (Path(log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2,default=str)+'\n')
        self.last_metrics={}
        env.reset()

    def _balance_advantages(self):
        storage=self.alg.storage;role=storage.observations[:,:,-1]>0.5
        for mask in (role,~role):
            if not bool(mask.any()):raise ValueError('CTS training requires both teacher and student environments')
            a=storage.advantages[mask]
            storage.advantages[mask]=(a-a.mean())/(a.std(unbiased=False)+1e-8)*(.5/float(mask.float().mean()))

    def _update_encoder(self):
        obs=self.alg.storage.observations.flatten(0,1)
        student=obs[obs[:,-1]<.5]
        metrics=[]
        for _ in range(self.cfg['encoder_batches']):
            ids=torch.randint(len(student),(min(len(student),self.cfg['encoder_batch_size']),),device=self.device)
            loss,vel,latent=self.alg.actor_critic.student_loss(student[ids])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTS student loss')
            self.encoder_optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.alg.actor_critic.student_encoder.parameters(),1.)
            self.encoder_optimizer.step();metrics.append([loss.item(),vel.item(),latent.item()])
        return dict(zip(('encoder_loss','velocity_mse','latent_mse'),torch.tensor(metrics).mean(0).tolist()))

    def _tracking_statistics(self,a):
        cmd=a['command'].to(self.device);state=a['next'].to(self.device)
        valid=a['task_valid'].to(self.device);enabled=a['enabled'].to(self.device)
        vx,vy,wz=cmd.unbind(1);planar=cmd[:,:2].norm(dim=1)
        unit=cmd[:,:2]/planar[:,None].clamp(min=.03)
        cross=(unit[:,0]*state[:,37]-unit[:,1]*state[:,36]).abs()
        values=torch.stack(((state[:,36:38]-cmd[:,:2]).norm(dim=1),
                            (state[:,41]-wz).abs(),cross,
                            a['heading_error'].to(self.device).abs()),1)
        pure=wz.abs()<.08;zero_xy=planar<.03
        groups={'teacher':self.env.teacher_mask.to(self.device),'student':~self.env.teacher_mask.to(self.device),
                'forward':(vx>.03)&(vy.abs()<.03)&pure,'backward':(vx<-.03)&(vy.abs()<.03)&pure,
                'left':(vy>.03)&(vx.abs()<.03)&pure,'right':(vy<-.03)&(vx.abs()<.03)&pure,
                'turn_left':zero_xy&(wz>.08),'turn_right':zero_xy&(wz<-.08),
                'diagonal':(vx.abs()>=.03)&(vy.abs()>=.03)&pure,
                'arc':~zero_xy&~pure,'stand':~enabled,'march':zero_xy&pure&enabled}
        # Direction metrics must expose deployable-student behavior, even if
        # the privileged teacher learns a direction more quickly.
        groups={k if k in ('teacher','student') else 'student_'+k:
                mask if k in ('teacher','student') else mask&~self.env.teacher_mask.to(self.device)
                for k,mask in groups.items()}
        return {k:torch.cat(((values*(mask&valid)[:,None]).sum(0),(mask&valid).sum().reshape(1)))
                for k,mask in groups.items()}

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf[:]=torch.randint_like(self.env.episode_length_buf,high=int(self.env.max_episode_length))
        obs=self.env.get_observations().to(self.device)
        critic=self.env.get_privileged_observations().to(self.device)
        self.alg.actor_critic.train()
        for _ in range(num_learning_iterations):
            started=time.monotonic();stats=[];episodes=[];tracking={}
            with torch.no_grad():
                for step in range(self.num_steps_per_env):
                    actions=self.alg.act(obs,critic)
                    obs,critic,rewards,dones,infos=self.env.step(actions.to(self.env.device))
                    obs=obs.to(self.device);critic=critic.to(self.device);rewards=rewards.to(self.device)
                    a=infos['amp'];valid=a['valid'].to(self.device)
                    for k,v in self._tracking_statistics(a).items():tracking[k]=tracking.get(k,0)+v
                    command=a['command'].to(self.device)
                    style,inputs,valid=self.style.rollout_reward(a)
                    if not torch.isfinite(inputs).all():raise RuntimeError('Nonfinite AMP rollout')
                    coverage=torch.ones_like(style)  # This stage samples only validated V2 commands.
                    warm=min(1.,(self.current_learning_iteration+1)/self.cfg['amp_warmup_iterations'])
                    bonus=self.cfg['amp_style_weight']*self.env.dt*warm*style*coverage*valid.float()
                    self.style.add(inputs[valid]);total=rewards+bonus
                    self.alg.process_env_step(total,dones.to(self.device),infos)
                    stats.append(torch.stack((rewards.mean(),bonus.mean(),valid.float().mean(),coverage.mean(),dones.float().mean(),
                                              a['reference_quality'].mean(),a['command_quality'].mean(),
                                              (style*valid).sum()/valid.sum().clamp(min=1))))
                    if 'episode' in infos and bool(dones.any()):episodes.append({k:float(v) for k,v in infos['episode'].items()})
                self.alg.compute_returns(critic)
                self._balance_advantages()
            value,surrogate=self.alg.update()
            encoder=self._update_encoder();self.style.update()
            with torch.no_grad():self.alg.actor_critic.std.clamp_(.05,.35)
            self.current_learning_iteration+=1
            avg=torch.stack(stats).mean(0).tolist()
            metrics=dict(zip(('task_reward_step','amp_bonus_step','amp_valid_fraction','expert_coverage','reset_fraction',
                              'reference_foot_quality','command_quality','amp_raw_style_valid'),avg))
            metrics.update(encoder);metrics.update(self.style.metrics)
            metrics.update(iteration=self.current_learning_iteration,value_loss=value,surrogate_loss=surrogate,
                           teacher_fraction=float(self.env.teacher_mask.float().mean()),
                           seconds=time.monotonic()-started,learning_rate=self.alg.learning_rate)
            for group,values in tracking.items():
                row=values.tolist();count=row[-1]
                metrics['tracking/'+group+'/samples']=count
                if count:
                    for name,value in zip(('linear_error_m_s','yaw_rate_error_rad_s','cross_velocity_m_s','heading_error_rad'),row[:-1]):
                        metrics['tracking/'+group+'/'+name]=value/count
            if episodes:
                for k in episodes[0]:metrics['episode/'+k]=sum(x.get(k,0.) for x in episodes)/len(episodes)
            if not all(torch.isfinite(torch.tensor(v)).item() for v in metrics.values()):
                raise RuntimeError('Nonfinite CTS/PPO metrics')
            self.last_metrics=metrics
            print('CTS '+json.dumps(metrics,sort_keys=True),flush=True)
            if self.writer:
                for k,v in metrics.items():self.writer.add_scalar(k,v,self.current_learning_iteration)
                self.writer.flush()
            if self.log_dir:
                with (Path(self.log_dir)/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                if self.current_learning_iteration%self.cfg['save_interval']==0:self.save(str(Path(self.log_dir)/('model_%d.pt'%self.current_learning_iteration)))
        if self.log_dir:self.save(str(Path(self.log_dir)/('model_%d.pt'%self.current_learning_iteration)))

    def save(self,path,infos=None):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),optimizer_state_dict=self.alg.optimizer.state_dict(),
                        encoder_optimizer_state_dict=self.encoder_optimizer.state_dict(),amp_state=self.style.state_dict(),
                        learning_rate=self.alg.learning_rate,iter=self.current_learning_iteration,infos=infos,
                        contract=self.contract,experiment=self.experiment),path)
        export_dir=Path(path).parent/'exported';export_dir.mkdir(parents=True,exist_ok=True)
        self.export_student(str(export_dir/('student_%d.pt'%self.current_learning_iteration)))

    def load(self,path,load_optimizer=True):
        loaded=torch.load(path,map_location=self.device)
        if loaded.get('contract')!=self.contract:raise ValueError('Checkpoint is not compatible with this CTS/AMP task')
        self.style.load_state_dict(loaded['amp_state'])
        self.style.clear_history()
        self.alg.actor_critic.load_state_dict(loaded['model_state_dict'],strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded['optimizer_state_dict'])
            self.encoder_optimizer.load_state_dict(loaded['encoder_optimizer_state_dict'])
            self.alg.learning_rate=loaded['learning_rate']
        self.current_learning_iteration=loaded['iter']
        return loaded.get('infos')

    def export_student(self,path):
        policy=StudentPolicy(self.alg.actor_critic).cpu().eval()
        torch.jit.script(policy).save(path)
        metadata=dict(self.contract,input='public history only, newest frame first',
                      input_shape=[None,self.contract['public_size']],output_shape=[None,self.env.num_actions],
                      public_frame=['gyro*.25:3','gravity:3','command*[2,2,.25]:3','q-default:12',
                                    'dq*.05:12','previous_clipped_action:12','clock_sin_cos:2',
                                    'stride_mean_yaw_error*4:1','heading_sin_cos_minus1:2','gait_enable:1'],
                      control=self.experiment['env']['control'],actuator=self.experiment['env']['rs01_actuator'],
                      normalization=self.experiment['env']['normalization'],
                      gait=self.experiment['env']['gait'],history_reset='repeat first frame for all 20 slots',
                      heading_target='reset to IMU yaw; integrate commanded wz at 50 Hz; wrap target-yaw to [-pi,pi]',
                      default_joint_angles=self.experiment['env']['init_state']['default_joint_angles'])
        Path(path+'.json').write_text(json.dumps(metadata,indent=2)+'\n')

    def get_inference_policy(self,device=None):
        self.alg.actor_critic.eval()
        if device is not None:self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
