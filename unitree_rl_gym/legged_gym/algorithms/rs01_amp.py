"""Conditional least-squares AMP, separate from actor observations and RS01 control."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner


class StyleLearner:
    def __init__(self,path,device='cpu',allow_kinematic=False,settings=None):
        settings=settings or {}
        p=Path(path);m=json.loads((p/'manifest.json').read_text())
        self.reference_settings=m['settings']
        if not m['amp_training_approved'] and not allow_kinematic:raise ValueError('Kinematic reference needs explicit opt-in')
        if abs(m['frame_dt_s']-.02)>1e-8:raise ValueError('AMP transitions must be 20ms')
        names=[l+'_'+j+'_joint' for l in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
        if m['joint_order']!=names:raise ValueError('Reference joint order mismatch')
        pairs=[];commands=[];cadence=[];digest=hashlib.sha256((p/'manifest.json').read_bytes())
        for c in m['clips']:
            f=p/(c['clip']+'.npz');digest.update(f.read_bytes())
            with np.load(f,allow_pickle=False) as a:
                x=a['amp_features'];idx=np.flatnonzero(a['transition_valid'][:-1])
                if x.ndim!=2 or x.shape[1]!=46 or not np.isfinite(x).all():raise ValueError('Invalid style features')
                if not np.allclose(a['command'],a['command'][0]):raise ValueError('Expected constant-command clips')
                pairs.append(torch.tensor(np.concatenate((x[idx],x[idx+1]),axis=1),device=device,dtype=torch.float32))
                commands.append(a['command'][0])
                if 'cadence_hz' not in c:raise ValueError('Reference clip %s has no measured cadence' % c['clip'])
                cadence.append(float(c['cadence_hz']))
        if len(set(len(x) for x in pairs))!=1:raise ValueError('Reference clips must have equal valid lengths')
        self.expert=torch.stack(pairs);self.commands=torch.tensor(np.array(commands),device=device,dtype=torch.float32)
        self.clip_cadence=torch.tensor(cadence,device=device,dtype=torch.float32)
        states=self.expert[:,:,:46].flatten(0,1)
        floor=states.new_tensor([.10]*12+[1.]*12+[.03]*12+[.1]*3+[.2]*3+[.1]*3+[.03])
        self.mean=states.mean(0);self.std=torch.maximum(states.std(0),floor)
        self.net=nn.Sequential(nn.Linear(95,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(device)
        self.feature_weights=torch.tensor(settings.get('amp_feature_weights',[1.]*46),device=device)
        if self.feature_weights.shape!=(46,) or not torch.isfinite(self.feature_weights).all():raise ValueError('Invalid AMP feature weights')
        self.disc_steps=int(settings.get('amp_disc_steps',4))
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=settings.get('amp_disc_lr',1e-4),weight_decay=1e-4)
        self.buffer=torch.zeros(131072,95,device=device);self.size=0;self.cursor=0;self.updates=0
        self.sha=digest.hexdigest();self.metrics={}

    def inputs(self,pair,command):
        norm=((pair-self.mean.repeat(2))/self.std.repeat(2)).clamp(-10,10)
        return torch.cat((norm*self.feature_weights.repeat(2),command/command.new_tensor([.3,.15,.5])),1)

    def reward(self,previous,next_state,command):
        x=self.inputs(torch.cat((previous,next_state),1),command)
        return (1-.25*(self.net(x).flatten()-1).square()).clamp(0,1),x

    def add(self,x):
        if len(x)==0:return
        x=x[-len(self.buffer):];n=len(x);ids=(torch.arange(n,device=x.device)+self.cursor)%len(self.buffer)
        self.buffer[ids]=x;self.cursor=(self.cursor+n)%len(self.buffer);self.size=min(len(self.buffer),self.size+n)

    def update(self):
        if self.size<32:return
        stats=[]
        for _ in range(self.disc_steps):
            fake=self.buffer[torch.randint(self.size,(512,),device=self.buffer.device)].clone()
            raw_command=fake[:,-3:]*fake.new_tensor([.3,.15,.5])
            ci=((raw_command[:,None,:]-self.commands[None,:,:])/fake.new_tensor([.3,.15,.5])).square().sum(2).argmin(1)
            ti=torch.randint(self.expert.shape[1],(len(fake),),device=fake.device)
            real=self.inputs(self.expert[ci,ti],self.commands[ci]).detach().requires_grad_(True)
            dr=self.net(real);df=self.net(fake)
            grad=torch.autograd.grad(dr.sum(),real,create_graph=True)[0][:,:92]
            gp=grad.square().sum(1).mean()
            loss=.5*((dr-1).square().mean()+(df+1).square().mean())+10.*gp
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite AMP discriminator')
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.optimizer.step()
            stats.append([loss.item(),dr.mean().item(),df.mean().item(),gp.item()])
        self.updates+=1
        self.metrics=dict(zip(('disc_loss','expert_score','policy_score','gradient_penalty'),np.mean(stats,axis=0)))

    def state_dict(self):
        return dict(net=self.net.state_dict(),optimizer=self.optimizer.state_dict(),mean=self.mean,std=self.std,
                    updates=self.updates,reference_sha256=self.sha,feature_weights=self.feature_weights)

    def load_state_dict(self,s,allow_reference_warmstart=False):
        weights=s.get('feature_weights',torch.ones_like(self.feature_weights)).to(self.feature_weights.device)
        if not torch.equal(weights,self.feature_weights):raise ValueError('AMP feature contract changed; start new discriminator')
        if s['reference_sha256']!=self.sha:
            if not allow_reference_warmstart:raise ValueError('AMP reference changed since checkpoint')
            print('Reference changed: retaining actor/critic, using fresh discriminator, normalization and AMP warmup.')
            return
        self.net.load_state_dict(s['net']);self.optimizer.load_state_dict(s['optimizer'])
        self.mean.copy_(s['mean']);self.std.copy_(s['std']);self.updates=s['updates']


class Rs01AMPOnPolicyRunner(OnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        super().__init__(env,train_cfg,log_dir,device)
        self.style=StyleLearner(self.cfg['amp_reference'],device,self.cfg['amp_allow_kinematic'],self.cfg)
        if abs(env.dt-.02)>1e-8:raise ValueError('Actor/reference time-step mismatch')
        settings=self.style.reference_settings
        if abs(settings['duty']-env.cfg.rewards.gait_stance_ratio)>1e-6:
            raise ValueError('AMP reference and reward gait clocks disagree')
        # Same tempo slack the recorder gated the reference with, or a clip it accepted
        # could never pass here and the dataset would be unusable.
        tempo=env.gait_frequencies_for(self.style.commands)
        if bool((tempo-self.style.clip_cadence).abs().gt(self.style.clip_cadence*.05+1e-3).any()):
            raise ValueError('AMP reference steps at %s Hz, this env clock gives %s Hz'
                             % (self.style.clip_cadence.tolist(),tempo.tolist()))
        a=torch.tensor(env.cfg.amp.commands,device=device)
        if not torch.all(torch.cdist(a,self.style.commands).min(1).values<1e-6):raise ValueError('Unsupported training commands')
        if log_dir is not None:
            # Untracked/concurrent edits must not silently change an experiment's identity.
            from legged_gym.utils.helpers import class_to_dict
            sources={}
            for cls in [*type(env).__mro__[:-1],type(self),type(self.alg),type(self.alg.actor_critic)]:
                path=inspect.getsourcefile(cls)
                if path and Path(path).is_file():sources[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            Path(log_dir).mkdir(parents=True,exist_ok=True)
            (Path(log_dir)/'experiment_contract.json').write_text(json.dumps(dict(
                env=class_to_dict(env.cfg),train=train_cfg,source_sha256=sources,
                reference_sha256=self.style.sha),indent=2,default=str)+'\n')
        process=self.alg.process_env_step;update=self.alg.update
        def process_amp(rewards,dones,infos):
            a=infos['amp'];previous=a['previous'].to(device);next_state=a['next'].to(device)
            command=a['command'].to(device);valid=a['valid'].to(device)
            style,x=self.style.reward(previous,next_state,command)
            if not torch.isfinite(x).all():raise RuntimeError('Nonfinite AMP rollout')
            warm=min(1.,(self.style.updates+1)/self.cfg['amp_warmup_iterations'])
            bonus=self.cfg['amp_style_weight']*env.dt*warm*style*valid.float()*a['gate'].to(device)
            rewards.add_(bonus)  # PPO and base runner logging see the same total.
            self.style.add(x[valid])
            self.amp_bonus=float(bonus.mean());self.amp_valid=float(valid.float().mean())
            self.amp_raw_style=float(style.mean());self.amp_gate_mean=float(a['gate'].float().mean())
            return process(rewards,dones,infos)
        def update_amp():
            losses=update();self.style.update();return losses
        self.alg.process_env_step=process_amp;self.alg.update=update_amp

    def log(self,locs,*args,**kwargs):
        super().log(locs,*args,**kwargs)
        for k,v in dict(self.style.metrics,style_reward_step=getattr(self,'amp_bonus',0.),
                        raw_style_reward=getattr(self,'amp_raw_style',0.),
                        style_gate_mean=getattr(self,'amp_gate_mean',0.),
                        valid_transition_fraction=getattr(self,'amp_valid',0.)).items():
            self.writer.add_scalar('AMP/'+k,v,locs['it'])

    def save(self,path,infos=None,iteration=None):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),optimizer_state_dict=self.alg.optimizer.state_dict(),
            iter=self.current_learning_iteration if iteration is None else int(iteration),infos=infos,
            amp_state=self.style.state_dict()),path)

    def load(self,path,load_optimizer=True):
        result=super().load(path,load_optimizer)
        loaded=torch.load(path,map_location=self.device)
        if 'amp_state' in loaded:self.style.load_state_dict(loaded['amp_state'],self.cfg.get('amp_allow_reference_warmstart',False))
        else:print('Warm-start actor/critic only; initializing new AMP discriminator (not a complete AMP resume).')
        return result
