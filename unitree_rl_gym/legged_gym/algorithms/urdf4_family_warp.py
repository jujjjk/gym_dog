"""Public-state coordinates around physically executed fast gait families.

The frozen, verified fast feedback is shared in local deviation coordinates.
Family feedforward comes from real Gym command records. No teacher or private
state is accessed during forward(). The original policy is exact at 0.50 and
at all low-speed/omnidirectional commands outside this experimental branch.
"""
import math
import torch
from torch import nn


def phase_samples(table,phase):
    index=phase*table.shape[0];lower=index.floor().long()
    lower=torch.where(lower>=table.shape[0],torch.zeros_like(lower),lower)
    upper=lower+1;upper=torch.where(upper>=table.shape[0],torch.zeros_like(upper),upper)
    fraction=(index-index.floor())[:,:,None]
    return table[lower]*(1-fraction)+table[upper]*fraction


def family_coordinates(public,speeds,frames,action_offsets,phase_offset,feature_mask,phase_shifts):
    count=public.shape[0];history=public[:,:1020].reshape(count,20,51);extra=public[:,1020:1029]
    goal=extra[:,0];amplitude=extra[:,8].clamp(0,1)
    active=(goal>.35)&(goal<.495)&(extra[:,1:3].abs().sum(1)<.001)
    query=goal.clamp(float(speeds[0]),float(speeds[-1]))
    upper=torch.searchsorted(speeds,query.contiguous()).clamp(1,len(speeds)-1);lower=upper-1
    ratio=(query-speeds[lower])/(speeds[upper]-speeds[lower])
    phase=torch.remainder(torch.atan2(history[:,:,45],history[:,:,46])/(2*math.pi)-phase_offset,1.)
    index=phase*frames.shape[1];lo=index.floor().long();lo=torch.where(lo>=frames.shape[1],torch.zeros_like(lo),lo)
    hi=lo+1;hi=torch.where(hi>=frames.shape[1],torch.zeros_like(hi),hi);weight=(index-index.floor())[:,:,None]
    first=frames[lower[:,None],lo]*(1-weight)+frames[lower[:,None],hi]*weight
    second=frames[upper[:,None],lo]*(1-weight)+frames[upper[:,None],hi]*weight
    expected=first+(second-first)*ratio[:,None,None]
    shift=phase_shifts[lower]+(phase_shifts[upper]-phase_shifts[lower])*ratio
    canonical_phase=torch.remainder(phase+shift[:,None],1.)
    baseline=phase_samples(frames[-1],canonical_phase)
    warped=history+(baseline-expected)*feature_mask[None,None,:]*amplitude[:,None,None]
    warped[:,:,45]=torch.sin(2*math.pi*(canonical_phase+phase_offset))
    warped[:,:,46]=torch.cos(2*math.pi*(canonical_phase+phase_offset))
    direction=torch.stack((history[:,:,49]+1,history[:,:,48]),2)
    delta=(.5-goal)*amplitude
    warped[:,:,6:8]+=2*direction*delta[:,None,None]
    extra_warped=extra.clone();extra_warped[:,0]=.5;extra_warped[:,3]=.5*amplitude
    extra_warped[:,6:8]+=direction[:,0,:]*delta[:,None]
    result=torch.cat((warped.flatten(1),extra_warped),1)
    # Offsets are actions issued for the current measured state, before the
    # unchanged rate limiter/delay/PD chain, never post-physics joint writes.
    a=action_offsets[lower,lo[:,0]]*(1-weight[:,0])+action_offsets[lower,hi[:,0]]*weight[:,0]
    b=action_offsets[upper,lo[:,0]]*(1-weight[:,0])+action_offsets[upper,hi[:,0]]*weight[:,0]
    correction=(a+(b-a)*ratio[:,None])*amplitude[:,None]
    return torch.where(active[:,None],result,public),correction*active[:,None].to(public.dtype)


class FamilyStudent(nn.Module):
    def __init__(self,policy,archive):
        super().__init__();self.policy=policy
        self.register_buffer('phase_shifts',archive.get('phase_shifts',torch.zeros_like(archive['speeds'])).detach().clone())
        for key in ('speeds','frames','action_offsets','phase_offset','feature_mask'):
            self.register_buffer(key,archive[key].detach().clone())

    def forward(self,public):
        canonical,offset=family_coordinates(public,self.speeds,self.frames,self.action_offsets,self.phase_offset,self.feature_mask,self.phase_shifts)
        return self.policy(canonical)+offset


class FixedFastStudent(nn.Module):
    def __init__(self,policy,canonical_velocity=False):
        super().__init__();self.policy=policy;self.canonical_velocity=canonical_velocity

    def forward(self,public):
        extra=public[:,1020:1029];goal=extra[:,0];amplitude=extra[:,8].clamp(0,1)
        active=(goal>.35)&(goal<.495)&(extra[:,1:3].abs().sum(1)<.001)
        changed=public.clone();changed[:,1020]=.5
        if self.canonical_velocity:
            history=changed[:,:1020].reshape(-1,20,51)
            direction=torch.stack((history[:,:,49]+1,history[:,:,48]),2)
            delta=(.5-goal)*amplitude
            history[:,:,6:8]+=2*direction*delta[:,None,None]
            changed[:,1023]=.5*amplitude
            changed[:,1026:1028]+=direction[:,0,:]*delta[:,None]
        return self.policy(torch.where(active[:,None],changed,public))
