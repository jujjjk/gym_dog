"""Tensor helpers shared by the speed/robustness environment and AMP learner."""
import torch

def reference_coverage(command,references,inner=.10,outer=.80):
    scale=command.new_tensor([.3,.15,.5])
    distance=((command[:,None]-references[None])/scale).square().sum(2).min(1).values.sqrt()
    u=((outer-distance)/(outer-inner)).clamp(0,1)
    return u.square()*(3-2*u)

def command_classes(command):
    x,y,w=command.unbind(1);turn=w.abs()>.08;xy=command[:,:2].norm(dim=1)>.03
    group=torch.where(x>=0,0,1)
    group=torch.where(y.abs()>x.abs(),torch.where(y>=0,2,3),group)
    group=torch.where((x.abs()>.03)&(y.abs()>.03),6,group)
    group=torch.where(turn&~xy,torch.where(w>=0,4,5),group)
    return torch.where(turn&xy,7,group)
