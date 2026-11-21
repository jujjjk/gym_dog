"""Sensor-only, phase-conditioned linear balance feedback, TorchScript portable."""
import math
import torch


def frame_balance_features(frame,extra,reference,stand,scales,offset):
    amplitude=extra[:,8:9].clamp(0,1)
    phase=torch.remainder(torch.atan2(frame[:,45],frame[:,46])/(2*math.pi)-offset,1.)
    index=phase*reference.shape[0];lo=index.floor().long()%reference.shape[0];hi=(lo+1)%reference.shape[0];weight=(index-index.floor())[:,None]
    expected=(1-weight)*reference[lo]+weight*reference[hi]
    expected=stand+amplitude*(expected-stand)
    gyro=frame[:,:3].clone();gyro[:,2]-=frame[:,8]  # both are scaled by .25
    measured=torch.cat((gyro,frame[:,3:5],frame[:,9:45],extra[:,6:8]-frame[:,6:8]/2.),1)
    if reference.shape[1]==45:
        measured=torch.cat((measured,frame[:,48:50]),1)
    error=((measured-expected)/scales).clamp(-10,10)
    error=torch.cat((torch.ones_like(amplitude),error),1)
    basis=[torch.ones_like(phase)]
    for harmonic in range(1,5):
        basis.append(torch.sin(phase*2*math.pi*harmonic));basis.append(torch.cos(phase*2*math.pi*harmonic))
    waves=torch.stack(basis,1)
    return (waves[:,:,None]*error[:,None,:]).flatten(1)


def balance_features(public,reference,stand,scales,offset):
    return frame_balance_features(public[:,:51],public[:,1020:1029],reference,stand,scales,offset)


def filtered_balance_features(public,reference,stand,scales,offset):
    count=public.shape[0]
    frames=public[:,:1020].reshape(count,20,51)
    extra=public[:,None,1020:1029].expand(-1,20,-1)
    features=frame_balance_features(frames.reshape(-1,51),extra.reshape(-1,9),reference,stand,scales,offset).reshape(count,20,-1)
    ages=torch.arange(20,device=public.device,dtype=public.dtype)
    weights=.35*torch.pow(torch.ones_like(ages)*.65,ages);weights/=weights.sum()
    return (features*weights[None,:,None]).sum(1)


def balance_actions(public,reference,stand,scales,offset,gain):
    features=filtered_balance_features(public,reference,stand,scales,offset)
    return public[:,1028:1029].clamp(0,1)*torch.tanh(features@gain)
