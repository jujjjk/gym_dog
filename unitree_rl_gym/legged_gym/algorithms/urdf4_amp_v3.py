"""Conditional V3 AMP + concurrent teacher/student PPO on the new-mass plant."""
import copy,hashlib,importlib.util,inspect,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rs01_amp import StyleLearner
from .urdf4_amp_cts import Urdf4AMPCTSRunner
from .urdf4_amp_expand import ExpandedActorCritic,ExpandedStudentPolicy
from torch.distributions import Normal
from .urdf4_v3_prior import reference_actions

ROOT=Path(LEGGED_GYM_ROOT_DIR).parent


def stand_blend(frame):
    linear=(frame[:,6:8]/2.).norm(p=2,dim=1)/.04;yaw=(frame[:,8]/.25).abs()/.10
    return torch.maximum(linear,yaw).clamp(0,1)*frame[:,50].clamp(0,1)


class V3ActorCritic(ExpandedActorCritic):
    def __init__(self,*args,v3_reference_count,v3_phase_bins=100,**kwargs):
        super().__init__(*args,**kwargs);self.register_buffer('stand_action',torch.zeros(12))
        self.register_buffer('prior_commands',torch.zeros(v3_reference_count,3));self.register_buffer('prior_table',torch.zeros(v3_reference_count,v3_phase_bins,12))

    def executed_mean(self,public,raw):
        prior=reference_actions(public[:,:self.frame_size],self.prior_commands,self.prior_table,self.stand_action)
        return prior+(.1+.9*public[:,50:51].clamp(0,1))*torch.tanh(raw)

    def update_distribution(self,observations):
        raw=self.actor(torch.cat((self.control_frame(observations),self.context(observations)),1))
        mean=self.executed_mean(observations,raw);self.distribution=Normal(mean,self.std.clamp(min=.05,max=.35).expand_as(mean))

    def act_inference(self,observations):
        raw=super().act_inference(observations);return self.executed_mean(observations,raw)


class V3StudentPolicy(ExpandedStudentPolicy):
    def __init__(self,model):
        super().__init__(model);self.register_buffer('stand_action',model.stand_action.detach().clone())
        self.register_buffer('prior_commands',model.prior_commands.detach().clone());self.register_buffer('prior_table',model.prior_table.detach().clone())

    def forward(self,history):
        public=history[:,:self.public_size];frame=public[:,:self.frame_size]
        command=torch.maximum(torch.minimum(frame[:,6:9],self.command_max),self.command_min)
        direct=torch.cat((frame[:,:6],command,frame[:,9:]),1)*self.frame_gain
        raw=self.actor(torch.cat((direct,self.encoder(public)),1))
        prior=reference_actions(frame,self.prior_commands,self.prior_table,self.stand_action)
        return prior+(.1+.9*frame[:,50:51].clamp(0,1))*torch.tanh(raw)


class V3Style(StyleLearner):
    frames=2;stride=1
    def __init__(self,path,env,settings=None,validation_receipt=None):
        settings=settings or {};path=Path(path);m=json.loads((path/'manifest.json').read_text())
        receipt=json.loads(Path(validation_receipt).read_text())
        if receipt['urdf_sha256']!=m['urdf_sha256']:raise ValueError('V3 validation belongs to another URDF')
        expected=receipt['input_manifests']['gym_verified/manifest.json']
        if hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()!=expected:raise ValueError('V3 manifest receipt mismatch')
        reader=ROOT/'tools/urdf4_amp/reference_v3.py';spec=importlib.util.spec_from_file_location('_urdf4_v3_data',reader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        asset=Path(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR));data=module.V3Reference(path,asset,simulator='isaac_gym')
        dev=env.device;self.reference_settings=m['settings'];self.commands=torch.tensor([c['command'] for c in m['clips']],device=dev)
        self.expert=torch.tensor(np.c_[data.previous,data.following],device=dev,dtype=torch.float32)
        self.expert_commands=torch.tensor(data.commands,device=dev,dtype=torch.float32)
        self.bins=torch.tensor(data.bins,device=dev,dtype=torch.float32);self.counts=torch.tensor([len(x) for x in data.buckets],device=dev)
        self.table=torch.zeros(len(data.buckets),int(self.counts.max()),device=dev,dtype=torch.long)
        for i,ids in enumerate(data.buckets):self.table[i,:len(ids)]=torch.tensor(ids,device=dev)
        self.zero_ids=torch.tensor(data.zero_stand,device=dev,dtype=torch.long)
        states=self._center(self.expert[:,:46],self.expert_commands)
        floor=states.new_tensor([.10]*12+[1.]*12+[.03]*12+[.1]*3+[.2]*3+[.1]*3+[.03])
        self.mean=states.mean(0);self.std=torch.maximum(states.std(0),floor)
        self.feature_weights=torch.tensor(settings.get('amp_feature_weights',[1.]*46),device=dev)
        self.net=nn.Sequential(nn.Linear(95,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(dev)
        self.disc_steps=int(settings.get('amp_disc_steps',2));self.optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-4,weight_decay=1e-4)
        self.buffer=torch.zeros(131072,95,device=dev);self.size=0;self.cursor=0;self.updates=0;self.metrics={}
        digest=hashlib.sha256()
        for folder in [path,path/'transitions']:
            digest.update((folder/'manifest.json').read_bytes())
            for c in json.loads((folder/'manifest.json').read_text())['clips']:digest.update((folder/(c['clip']+'.npz')).read_bytes())
        self.sha=digest.hexdigest();self.reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest();self.pair_count=len(self.expert)

    @staticmethod
    def _center(state,command):
        state=state.clone();state[:,36:38]-=command[:,:2];state[:,41]-=command[:,2];return state

    def _encode(self,pair,velocity_command,condition):
        centered=torch.cat((self._center(pair[:,:46],velocity_command),self._center(pair[:,46:],velocity_command)),1)
        norm=((centered-self.mean.repeat(2))/self.std.repeat(2)).clamp(-10,10)
        return torch.cat((norm*self.feature_weights.repeat(2),condition/condition.new_tensor([.5,.2,.8])),1)

    def inputs(self,pair,command):return self._encode(pair,command,command)

    def nearest(self,command):
        delta=(command[:,None]-self.bins[None])/command.new_tensor([.12,.08,.25])
        return delta.square().sum(2).min(1)

    def coverage(self,command):return torch.exp(-self.nearest(command).values).clamp(min=.15)

    def sample_real(self,command):
        _,bucket=self.nearest(command);offset=(torch.rand(len(bucket),device=bucket.device)*self.counts[bucket]).long()
        ids=self.table[bucket,offset];zero=command.abs().sum(1)<1e-6
        ids[zero]=self.zero_ids[torch.randint(len(self.zero_ids),(int(zero.sum()),),device=ids.device)]
        return self._encode(self.expert[ids],self.expert_commands[ids],command)

    def rollout_reward(self,a):
        bonus,x=self.reward(a['previous'],a['next'],a['style_command']);return bonus,x,a['valid']

    def clear_history(self):pass

    def update(self):
        if self.size<32:return
        metrics=[]
        for _ in range(self.disc_steps):
            fake=self.buffer[torch.randint(self.size,(512,),device=self.buffer.device)].clone()
            command=fake[:,-3:]*fake.new_tensor([.5,.2,.8]);real=self.sample_real(command).detach().requires_grad_(True)
            dr,df=self.net(real),self.net(fake);gradient=torch.autograd.grad(dr.sum(),real,create_graph=True)[0][:,:92];gp=gradient.square().sum(1).mean()
            loss=.5*((dr-1).square().mean()+(df+1).square().mean())+10*gp
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite V3 AMP discriminator')
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.optimizer.step()
            metrics.append([loss.item(),dr.mean().item(),df.mean().item(),gp.item()])
        self.updates+=1;self.metrics=dict(zip(('disc_loss','expert_score','policy_score','gradient_penalty'),np.mean(metrics,axis=0)))


class Urdf4AMPV3Runner(Urdf4AMPCTSRunner):
    style_class=V3Style;actor_class=V3ActorCritic;student_policy_class=V3StudentPolicy
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        train_cfg=copy.deepcopy(train_cfg);train_cfg['policy']['v3_reference_count']=len(env.reference_commands)
        super().__init__(env,train_cfg,log_dir,device)
        meta=json.loads((Path(self.env.cfg.amp_cts.reference)/'manifest.json').read_text());standing=next(c for c in meta['clips'] if c['family']=='stand')
        source=Path(self.env.cfg.amp_cts.reference)/standing['source_folder']/'physical.npz'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=standing['source_npz_sha256']:raise ValueError('Standing control provenance mismatch')
        z=np.load(source);q0=torch.tensor(z['actuator_default'],device=self.device,dtype=torch.float)
        correction=(q0-self.env.default_dof_pos[0])*(1/.928-1)
        target=torch.tensor(z['command_targets'][-200:].mean(0),device=self.device,dtype=torch.float)
        self.alg.actor_critic.stand_action.copy_((target+correction-self.env.default_dof_pos[0])/self.env.action_scale)
        self.alg.actor_critic.prior_commands.copy_(self.env.reference_commands)
        archive=Path(env.cfg.amp_cts.prior_archive)
        if not archive.exists():raise ValueError('Prepare the V3 reference initialization first')
        prior=torch.load(archive,map_location='cpu')
        if prior['urdf_sha256']!=env.urdf_sha256 or prior['reference_manifest_sha256']!=hashlib.sha256((Path(env.cfg.amp_cts.reference)/'manifest.json').read_bytes()).hexdigest():raise ValueError('V3 reference initialization provenance mismatch')
        if not torch.equal(prior['commands'].to(env.device),env.reference_commands):raise ValueError('V3 reference initialization order mismatch')
        self.alg.actor_critic.prior_table.copy_(prior['actions'].to(self.device))
        self._freeze_seed_policy()
        self.alg.reference_action_clip=float(self.env.cfg.normalization.clip_actions)
        self.contract.update(urdf_sha256=self.env.urdf_sha256,reader_sha256=self.style.reader_sha256,
            action_clip=self.env.cfg.normalization.clip_actions,variable_cadence=True,zero_command='stand',
            direction_gains=self.env.cfg.direction.gains)
        self.contract.update(stand_action=self.alg.actor_critic.stand_action.tolist(),standing_residual_action_limit=.1,standing_control_sha256=standing['source_npz_sha256'])
        self.contract.update(reference_prior=True,moving_residual_action_limit=1.)
        self.contract['reference_prior_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
        from legged_gym.utils.helpers import class_to_dict
        self.contract['training_settings']={name:class_to_dict(getattr(self.env.cfg,name)) for name in ['commands','expand','direction','domain_rand','push','imu','joint_sensor','noise','rewards','control','rs01_actuator','normalization']}
        self.stage_updates=0;self.curriculum_windows=0;self.curriculum_metrics={};self.seed_migration=None
        self.alg.actor_critic.teacher_encoder.requires_grad_(False)
        from legged_gym.envs.urdf4.path_tracking import PlanarPathController
        for obj in [type(self.env.path),PlanarPathController,V3Style]:
            file=Path(inspect.getsourcefile(obj));self.experiment['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        file=ROOT/'tools/urdf4_amp/reference_v3.py';self.experiment['source_sha256'][str(file)]=self.style.reader_sha256
        file=Path(inspect.getsourcefile(reference_actions));self.experiment['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        if self.log_dir:(Path(self.log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2,default=str)+'\n')

    def warm_start(self,path):
        saved=torch.load(path,map_location=self.device);old=saved['model_state_dict'];model=self.alg.actor_critic
        if saved['contract']['public_size']!=1020 or saved['contract']['frame_size']!=51:raise ValueError('Expected the selected 1020D URDF4 seed')
        current=model.state_dict()
        # New task: only actor and context encoders migrate. The critic, D,
        # normalization, optimizers and counters belong to this new plant/data.
        for key,value in old.items():
            if key.startswith(('actor.','teacher_encoder.','student_encoder.')):
                if current[key].shape!=value.shape:raise ValueError('Incompatible seed parameter '+key)
                current[key]=value
        model.load_state_dict(current);self._freeze_seed_policy();self.loaded_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.seed_migration=dict(checkpoint=str(path),sha256=self.loaded_checkpoint_sha256,critic_fresh=True,discriminator_fresh=True)

    def _amp_weights(self,a,style,valid):
        coverage=self.style.coverage(a['style_command']);recovery=a['style_factor']
        return coverage*(.25+.75*recovery)*a['legal'],valid&(coverage>=.45)&(recovery>.999)

    def _validate_training_request(self,iterations):
        if iterations>500:
            receipt=json.loads(Path(self.cfg['v3_preflight']).read_text())
            if not receipt.get('approved_for_training') or receipt['checkpoint_sha256']!=self.loaded_checkpoint_sha256 or receipt['contract']!=self.contract or receipt['source_sha256']!=self.experiment['source_sha256']:
                raise ValueError('V3 preflight does not match the selected seed/data/source')

    def _after_update(self,metrics):
        self.stage_updates+=1;env=self.env;cfg=env.cfg.expand
        self.alg.actor_critic.teacher_encoder.requires_grad_(self.current_learning_iteration>=200)
        if self.current_learning_iteration%cfg.curriculum_window==0:
            values=env.v3_meter.clone();env.v3_meter.zero_();count=values[:,-1];means=values[:,:-1]/count[:,None].clamp(min=1)
            stand=env.stand_meter.clone();env.stand_meter.zero_();stand_mean=stand[:2]/stand[2].clamp(min=1)
            ready=bool((count>=cfg.curriculum_min_samples_per_direction).all()) and bool(stand[2]>=100)
            passed=ready and bool((means[:,0]<=.04).all() and (means[:,1]<=.10).all() and (means[:,2]<=.03).all() and (means[:,3]<=.035).all() and (means[:,4:6]<=.001).all() and (means[:,6]<=.25).all() and (means[:,7:]>=.05).all() and stand_mean[0]<=.02 and stand_mean[1]<=.02)
            self.curriculum_windows=self.curriculum_windows+1 if passed and self.stage_updates>=cfg.curriculum_min_updates else 0
            if self.curriculum_windows>=2 and env.curriculum_stage<cfg.max_stage:env.curriculum_stage+=1;self.curriculum_windows=0;self.stage_updates=0
            self.curriculum_metrics=dict(curriculum_ready=float(ready),curriculum_window_passed=float(passed),
                worst_cross_m=float(means[:,2].max()),worst_heading_rad=float(means[:,3].max()),
                minimum_foot_lift_fraction=float(means[:,7:11].min()),minimum_diagonal_fraction=float(means[:,11:].min()),stand_speed_m_s=float(stand_mean[0]))
        return dict(self.curriculum_metrics,curriculum_stage=env.curriculum_stage,nominal_fraction=float(env.nominal_mask.float().mean()),push_count=float(env.push_count.sum()))

    def save(self,path,infos=None):
        if self.current_learning_iteration==0:self._freeze_seed_policy()
        saved=dict(model_state_dict=self.alg.actor_critic.state_dict(),optimizer_state_dict=self.alg.optimizer.state_dict(),
            seed_policy_state_dict=self.alg.reference_actor.state_dict(),encoder_optimizer_state_dict=self.encoder_optimizer.state_dict(),
            amp_state=self.style.state_dict(),learning_rate=self.alg.learning_rate,iter=self.current_learning_iteration,infos=infos,
            contract=self.contract,experiment=self.experiment,
            v3_state=dict(stage=self.env.curriculum_stage,stage_updates=self.stage_updates,pass_windows=self.curriculum_windows,seed_migration=self.seed_migration))
        temporary=str(path)+'.tmp';torch.save(saved,temporary);os.replace(temporary,path)
        export=Path(path).parent/'exported';export.mkdir(exist_ok=True);self.export_student(str(export/('student_%d.pt'%self.current_learning_iteration)))

    def load(self,path,load_optimizer=True):
        result=super().load(path,load_optimizer);saved=torch.load(path,map_location=self.device)
        if 'v3_state' not in saved and saved['iter']!=0:raise ValueError('Missing V3 curriculum resume state')
        state=saved.get('v3_state',dict(stage=0,stage_updates=0,pass_windows=0))
        self.env.curriculum_stage=int(state['stage']);self.stage_updates=int(state['stage_updates']);self.curriculum_windows=int(state['pass_windows']);self.seed_migration=state.get('seed_migration');return result

    def export_student(self,path):
        super().export_student(path);file=Path(path+'.json');meta=json.loads(file.read_text())
        meta.update(training_objective='v3_omni_direction_robust',deployment_requires_path_controller=True,
            path_controller=dict(gains=self.env.cfg.direction.gains,pose='Externally corrected planar localization, including its latency/noise contract'),
            phase_clock=dict(continuous=True,selected_from_user_target=True,frequencies_hz=self.env.reference_frequencies.tolist(),commands=self.env.reference_commands.tolist(),offsets=self.env.reference_phase_offsets.tolist()),
            user_command_curriculum=self.env.cfg.expand.stages,actuator_limits_increased=False)
        file.write_text(json.dumps(meta,indent=2)+'\n')
