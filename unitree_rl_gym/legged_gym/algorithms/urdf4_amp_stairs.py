"""AMP+CTS continuation with a local-terrain action residual."""
import copy
import hashlib
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal

from legged_gym.envs.urdf4.amp_stairs import BASE_PUBLIC, SCAN_SIZE, SCAN_X, SCAN_Y
from legged_gym.utils.helpers import class_to_dict
from .urdf4_amp_transition_training import Urdf4AMPTransitionTrainingRunner
from .urdf4_amp_v4 import V4ActorCritic,V4StudentPolicy


def stair_action(net, public, scan):
    strength = (scan.abs().amax(1,keepdim=True)*12.).clamp(0.,1.)
    return 1.25 * strength * torch.tanh(net(torch.cat((public[:,:102], public[:,1026:1029],scan),1)))


def stair_foothold_prior(public, scan, settings: Tuple[float,float,float]):
    lift_gain, center, leveling_gain = settings
    if lift_gain == 0. and leveling_gain == 0.:
        return torch.zeros_like(public[:,:12])
    heights = scan.reshape(-1,6,2)*.14
    front = (heights[:,5]-heights[:,3]).clamp(min=0.)
    rear = (heights[:,3]-heights[:,0]).clamp(min=0.)
    rise = torch.cat((front,rear),1)
    stance = torch.cat((heights[:,4],heights[:,1]),1)
    local = stance-stance.mean(1,keepdim=True)
    phase = torch.atan2(public[:,45:46],public[:,46:47])/(2.*torch.pi)
    offsets = torch.tensor([0.,.5,.5,0.],device=public.device,dtype=public.dtype)
    distance = torch.remainder(phase+offsets-center+.5,1.)-.5
    pulse = (1.-distance.abs()/.25).clamp(min=0.)
    lift = (lift_gain*pulse*rise+leveling_gain*local)*public[:,50:51]
    # Neutral-pose Jacobian in canonical FL, FR, RL, RR order.
    sign = torch.tensor([-1.,1.,-1.,1.],device=public.device,dtype=public.dtype)
    knee = lift*sign/(.1235*.25)
    return torch.stack((torch.zeros_like(knee),-.515*knee,knee),2).flatten(1)


class StairActor(V4ActorCritic):
    def __init__(self,*args,stair_prior=(0.,.76,0.),**kwargs):
        super().__init__(*args,**kwargs)
        self.stair_prior = tuple(stair_prior)
        self.stair_residual=nn.Sequential(nn.Linear(105+SCAN_SIZE,128),nn.ELU(),nn.Linear(128,64),
                                          nn.ELU(),nn.Linear(64,12))
        self.stair_value=nn.Sequential(nn.Linear(51+SCAN_SIZE,64),nn.ELU(),nn.Linear(64,1))
        for net in (self.stair_residual,self.stair_value):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def split(self,obs):
        start=self.public_size+SCAN_SIZE
        return obs[:,:self.public_size],obs[:,start:start+self.teacher_size],obs[:,-1:]>0.5

    def update_distribution(self,obs):
        raw=self.actor(torch.cat((self.control_frame(obs),self.context(obs)),1))
        base=self.executed_mean(obs,raw)
        measured=obs[:,BASE_PUBLIC:BASE_PUBLIC+SCAN_SIZE]
        true=obs[:,-SCAN_SIZE-1:-1]
        scan=torch.where(obs[:,-1:]>0.5,true,measured)
        mean=base+stair_foothold_prior(obs,scan,self.stair_prior)+stair_action(self.stair_residual,obs,scan)
        self.distribution=Normal(mean,self.std.clamp(.025,.10).expand_as(mean))

    def evaluate(self,obs,**kwargs):
        scan=obs[:,-SCAN_SIZE-1:-1]
        return super().evaluate(obs,**kwargs)+self.stair_value(torch.cat((obs[:,:51],scan),1))

    def act_inference(self,obs):
        public=obs[:,:BASE_PUBLIC]
        scan=obs[:,BASE_PUBLIC:BASE_PUBLIC+SCAN_SIZE]
        return self.public_mean(public)+stair_foothold_prior(public,scan,self.stair_prior)+stair_action(self.stair_residual,public,scan)


class StairStudent(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.base=V4StudentPolicy(model)
        self.residual=copy.deepcopy(model.stair_residual)
        self.stair_prior=model.stair_prior

    def forward(self,history):
        public=history[:,:1029]
        scan=history[:,1029:1041]
        strength=(scan.abs().amax(1,keepdim=True)*12.).clamp(0.,1.)
        extra=self.residual(torch.cat((public[:,:102], public[:,1026:1029],scan),1))
        return self.base(public)+stair_foothold_prior(public,scan,self.stair_prior)+1.25*strength*torch.tanh(extra)


class Urdf4AMPStairsRunner(Urdf4AMPTransitionTrainingRunner):
    actor_class=StairActor
    student_policy_class=StairStudent

    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        train_cfg = copy.deepcopy(train_cfg)
        stairs = env.cfg.stairs
        train_cfg['policy']['stair_prior'] = (stairs.lift_gain,stairs.lift_phase,stairs.leveling_gain)
        super().__init__(env,train_cfg,log_dir,device)
        self.contract['stairs']=class_to_dict(env.cfg.stairs)
        if self.log_dir:
            (Path(self.log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2)+'\n')

    def warm_start(self,path):
        seed=torch.load(path,map_location=self.device)
        for key in ('urdf_sha256','public_size','native_prior_sha256','balance_archive_sha256','reference_sha256'):
            if seed['contract'][key]!=self.contract[key]:
                raise ValueError(f'Incompatible staircase seed {key}')
        state=self.alg.actor_critic.load_state_dict(seed['model_state_dict'],strict=False)
        expected={k for k in self.alg.actor_critic.state_dict() if k.startswith(('stair_residual.','stair_value.'))}
        if 'stairs' in seed['contract']:
            expected=set()
        if state.unexpected_keys or set(state.missing_keys)!=expected:
            raise ValueError(f'Unexpected staircase seed migration: {state}')
        self.style.load_state_dict(seed['amp_state'])
        self.style.clear_history()
        self._freeze_seed_policy()
        self.seed_migration=dict(checkpoint=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                 zero_initialized_stair_residual=bool(expected))

    def _amp_weights(self,amp,style,valid):
        weights,replay=super()._amp_weights(amp,style,valid)
        edge = amp['stair_edge']
        return weights * ~edge,replay & ~edge

    def _after_update(self,metrics):
        result=super()._after_update(metrics)
        upper,exit,fell,timeout,episodes,down,down_episodes=self.env.stair_events.tolist()
        self.env.stair_events.zero_()
        result.update(stairs_up=upper,stairs_through=exit,stairs_falls=fell,
                      stairs_incomplete=timeout,stairs_episodes=episodes,stairs_down=down,
                      stairs_down_episodes=down_episodes)
        return result

    def export_student(self,path):
        super().export_student(path)
        file=Path(path+'.json')
        data=json.loads(file.read_text())
        data.update(public_size=BASE_PUBLIC+SCAN_SIZE,input_shape=[None,BASE_PUBLIC+SCAN_SIZE],
                    stair_scan=dict(x_m=SCAN_X,y_m=SCAN_Y,relative_height_divisor=.14))
        file.write_text(json.dumps(data,indent=2)+'\n')
