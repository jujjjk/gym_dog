"""Infer feedback phase from measured joints, separately from trajectory phase.

This compensates bounded actuator/contact lag. The commanded foot trajectory
keeps its original clock. Inference uses public q/dq/history only; no simulator
contact, root-height, velocity truth or teacher is consulted.
"""
import math
import torch
from torch import nn
from .urdf4_amp_v4 import V4StudentPolicy
from .urdf4_phase_feedback import balance_actions


def feedback_history(public,reference,stand,offset,dq_weight:float,phase_gain:float,deadzone:float,limit:float):
    n=public.shape[0];history=public[:,:1020].reshape(n,20,51)
    amplitude=public[:,1028:1029].clamp(0,1)
    current=torch.cat((history[:,:,9:21]/.05,history[:,:,21:33]*dq_weight/.10),2).reshape(-1,24)
    baseline=torch.cat((stand[5:17]/.05,stand[17:29]*dq_weight/.10))
    table=torch.cat((reference[:,5:17]/.05,reference[:,17:29]*dq_weight/.10),1)-baseline
    measured=current-baseline;amp=amplitude[:,None,:].expand(-1,20,-1).reshape(-1,1)
    distances=measured.square().sum(1,keepdim=True)+amp.square()*table.square().sum(1)[None,:]-2*amp*(measured@table.T)
    planned=torch.remainder(torch.atan2(history[:,:,45],history[:,:,46])/(2*math.pi)-offset,1.).flatten()
    phases=torch.arange(reference.shape[0],device=public.device,dtype=public.dtype)/reference.shape[0]
    delta=torch.remainder(phases[None,:]-planned[:,None]+.5,1.)-.5
    scores=-.5*distances-2*(delta/.125).square()
    scores=torch.where(delta.abs()<=.15,scores,torch.ones_like(scores)*-1.e6)
    weights=torch.softmax(scores,1)
    sine=(weights*torch.sin(2*math.pi*phases)[None,:]).sum(1)
    cosine=(weights*torch.cos(2*math.pi*phases)[None,:]).sum(1)
    estimated=torch.atan2(sine,cosine)/(2*math.pi)
    error=torch.remainder(estimated-planned+.5,1.)-.5
    gate=((error.abs()-deadzone)/.02).clamp(0,1)
    correction=(phase_gain*error*gate).clamp(-limit,limit)*((amp[:,0]-.3)/.4).clamp(0,1)
    phase=(planned+correction+offset).reshape(n,20)
    changed=history.clone();changed[:,:,45]=torch.sin(2*math.pi*phase);changed[:,:,46]=torch.cos(2*math.pi*phase)
    return torch.cat((changed.flatten(1),public[:,1020:1029]),1)


class AdaptiveFastStudent(nn.Module):
    def __init__(self,model,dq_weight=1.,phase_gain=1.,deadzone=.02,limit=.08):
        super().__init__();self.base=V4StudentPolicy(model)
        self.dq_weight=dq_weight;self.phase_gain=phase_gain;self.deadzone=deadzone;self.limit=limit

    def forward(self,public):
        original=self.base(public);extra=public[:,1020:1029];goal=extra[:,0];amp=extra[:,8].clamp(0,1)
        active=(goal>.35)&(goal<.495)&(extra[:,1:3].abs().sum(1)<.001)
        canonical=public.clone();canonical[:,1020]=.5;canonical[:,1023]=.5*amp
        history=canonical[:,:1020].reshape(-1,20,51)
        direction=torch.stack((history[:,:,49]+1,history[:,:,48]),2);delta=(.5-goal)*amp
        history[:,:,6:8]+=2*direction*delta[:,None,None];canonical[:,1026:1028]+=direction[:,0,:]*delta[:,None]
        changed=feedback_history(canonical,self.base.balance_reference,self.base.balance_stand,self.base.balance_offset,
            self.dq_weight,self.phase_gain,self.deadzone,self.limit)
        current=balance_actions(canonical,self.base.balance_reference,self.base.balance_stand,self.base.balance_scales,self.base.balance_offset,self.base.balance_gain)
        corrected=balance_actions(changed,self.base.balance_reference,self.base.balance_stand,self.base.balance_scales,self.base.balance_offset,self.base.balance_gain)
        candidate=self.base(canonical)+corrected-current
        return torch.where(active[:,None],candidate,original)
