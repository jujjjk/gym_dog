"""Native temporal samples, with the same command-space fit as the V3 baseline."""
import math
import torch


def native_actions(frame,commands,table,counts,offsets,standing):
    query=frame[:,6:9]/torch.tensor([2.,2.,.25],device=frame.device,dtype=frame.dtype)
    delta=(commands[None]-query[:,None])/torch.tensor([.5,.2,.8],device=frame.device,dtype=frame.dtype)
    ds,ids=torch.topk(delta.square().sum(2),8,dim=1,largest=False)
    dx=delta.gather(1,ids[:,:,None].expand(-1,-1,3));x=torch.cat((torch.ones_like(ds[:,:,None]),dx),2)
    weights=1/(ds+.005).square();weights/=weights.amax(1,keepdim=True)
    phase=torch.remainder(torch.atan2(frame[:,45],frame[:,46])/(2*math.pi),1.)
    index=torch.remainder(phase[:,None]-offsets[ids],1.)*counts[ids]
    # The phase is already in [0,1). Avoid integer tensor remainder: the
    # installed CUDA TorchScript fuser lowers it to an ambiguous fmod call.
    lengths=counts[ids];lo=index.floor().long()
    lo=torch.where(lo>=lengths,torch.zeros_like(lo),lo)
    hi=lo+1;hi=torch.where(hi>=lengths,torch.zeros_like(hi),hi)
    blend=(index-index.floor())[:,:,None]
    y=(1-blend)*table[ids,lo]+blend*table[ids,hi];wx=x*weights[:,:,None]
    ridge=torch.diag(torch.tensor([1e-8,1e-4,1e-4,1e-4],device=frame.device,dtype=frame.dtype))
    result=torch.linalg.solve(x.transpose(1,2)@wx+ridge,wx.transpose(1,2)@y)[:,0]
    moving=frame[:,50:51].clamp(0,1)
    return moving*result+(1-moving)*standing


def conditioned_native_actions(public,commands,table,counts,offsets,stands,standing):
    frame=public[:,:51].clone();extra=public[:,1020:1029]
    goal=extra[:,:3];external=extra[:,3:6];amplitude=extra[:,8:9].clamp(0,1)
    # Unsupported goals remain a learning objective; do not extrapolate the
    # feedforward gait beyond its measured speed envelope.
    base=torch.maximum(torch.minimum(goal,torch.tensor([.5,.2,.6],device=goal.device,dtype=goal.dtype)),torch.tensor([-.35,-.2,-.4],device=goal.device,dtype=goal.dtype))
    # High-speed feedback is learned in a common local state/velocity frame.
    # Cross-speed joint-space slopes are not valid steering derivatives when
    # source height, duty and timing differ.
    query=base
    frame[:,6:9]=query*torch.tensor([2.,2.,.25],device=frame.device);frame[:,50]=1.
    walking=native_actions(frame,commands,table,counts,offsets,standing)
    distance=((base[:,None]-commands[None])/torch.tensor([.5,.2,.8],device=goal.device)).square().sum(2)
    initial=stands[distance.argmin(1)]
    return initial+amplitude*(walking-initial)
