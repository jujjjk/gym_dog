"""Public command/phase joint-target prior with local affine interpolation.

Convex nearest-clip blending limits heading corrections at the expert envelope
and mixes unwanted translation into turns. A regularized local affine fit uses
8 nearby validated trajectories, preserving the command's local derivatives.
Only the correction network adapts online; no full-state teacher runs here.
"""
import math
import torch


def reference_actions(frame,commands,table,standing):
 query=frame[:,6:9]/torch.tensor([2.,2.,.25],device=frame.device,dtype=frame.dtype);scale=torch.tensor([.5,.2,.8],device=frame.device,dtype=frame.dtype)
 delta=(commands[None]-query[:,None])/scale;distance=delta.square().sum(2)
 ds,ids=torch.topk(distance,8,dim=1,largest=False)
 dx=delta.gather(1,ids[:,:,None].expand(-1,-1,3));X=torch.cat((torch.ones_like(ds[:,:,None]),dx),2)
 w=1/(ds+.005).square();w/=w.amax(1,keepdim=True)
 phase=torch.remainder(torch.atan2(frame[:,45],frame[:,46])/(2*math.pi),1.)*table.shape[1]
 lo=phase.floor().long()%table.shape[1];hi=(lo+1)%table.shape[1];blend=(phase-phase.floor())[:,None,None]
 y=(1-blend)*table[ids,lo[:,None]]+blend*table[ids,hi[:,None]]
 wx=X*w[:,:,None];matrix=X.transpose(1,2)@wx
 ridge=torch.diag(torch.tensor([1e-8,1e-4,1e-4,1e-4],device=frame.device,dtype=frame.dtype))
 fit=torch.linalg.solve(matrix+ridge,wx.transpose(1,2)@y);result=fit[:,0]
 moving=frame[:,50:51].clamp(0,1);return moving*result+(1-moving)*standing
