"""URDF FK and inverse-dynamics bias from public joints and IMU only.

No simulator handles, root pose/velocity or contact truth are accepted. The
floating-base gauge has zero yaw; gravity and angular velocity are body-frame.
"""
import xml.etree.ElementTree as ET
import torch
from torch import nn
from .urdf4_ground_observer import JointGroundObserver


class PublicLegDynamics(nn.Module):
    def __init__(self, urdf, joint_names, default_q):
        super().__init__()
        self.observer = JointGroundObserver(urdf, joint_names, default_q)
        root = ET.parse(urdf).getroot()
        links = {x.attrib['name']: x for x in root.findall('link')}
        joints = {x.attrib['name']: x for x in root.findall('joint')}

        def inertial(name):
            node = links[name].find('inertial')
            origin = node.find('origin')
            assert max(abs(float(x)) for x in origin.get('rpy', '0 0 0').split()) < 1.e-12
            com = list(map(float, origin.get('xyz').split()))
            mass = float(node.find('mass').get('value'))
            a = node.find('inertia').attrib
            xx, xy, xz, yy, yz, zz = [float(a[k]) for k in ('ixx','ixy','ixz','iyy','iyz','izz')]
            return mass, com, [[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]

        masses, coms, inertias = [], [], []
        for leg in ('FL', 'FR', 'RL', 'RR'):
            values = [inertial(leg+'_'+kind) for kind in ('hip_joint','thigh_joint','calf_joint','foot')]
            masses.append([v[0] for v in values]); coms.append([v[1] for v in values]); inertias.append([v[2] for v in values])
        trunk_mass, trunk_com, _ = inertial('trunk')
        root_moment = torch.tensor(trunk_com)*trunk_mass
        for name in links:
            if name == 'trunk' or name.startswith(('FL_', 'FR_', 'RL_', 'RR_')): continue
            joint = next(j for j in joints.values() if j.find('child').get('link') == name)
            assert joint.attrib['type'] == 'fixed' and joint.find('parent').get('link') == 'trunk'
            origin = joint.find('origin')
            assert max(abs(float(x)) for x in origin.get('rpy','0 0 0').split()) < 1.e-12
            mass, com, _ = inertial(name)
            root_moment += mass*(torch.tensor(com)+torch.tensor(list(map(float,origin.get('xyz').split()))))
            trunk_mass += mass
        self.register_buffer('mass', torch.tensor(masses))
        self.register_buffer('com_local', torch.tensor(coms))
        self.register_buffer('inertia_local', torch.tensor(inertias))
        self.register_buffer('root_moment', root_moment)
        self.register_buffer('total_mass', self.mass.sum()+trunk_mass)

    def forward(self, frame):
        """Return feet, per-leg 3x3 Jacobians, whole-body CoM and joint bias."""
        count = frame.shape[0]
        q = (frame[:,9:21]+self.observer.default_q).reshape(count,4,3)
        dq = (frame[:,21:33]/.05).reshape(count,4,3)
        gravity = frame[:,3:6]
        gravity = gravity/gravity.norm(p=2,dim=1,keepdim=True).clamp(min=1.e-6)
        omega = (frame[:,:3]/.25)[:,None,:].expand(-1,4,-1)
        alpha = torch.zeros_like(omega)
        acceleration = (-9.81*gravity)[:,None,:].expand(-1,4,-1)
        position = torch.zeros_like(omega)
        rotation = torch.eye(3,device=frame.device,dtype=frame.dtype).expand(count,4,3,3)
        origins = []; directions = []; centers = []; forces = []; moments = []
        for j in range(4):
            r = (rotation@self.observer.offsets[:,j,:,None]).squeeze(-1)
            position = position+r
            acceleration = acceleration+torch.cross(alpha,r,dim=-1)+torch.cross(omega,torch.cross(omega,r,dim=-1),dim=-1)
            if j < 3:
                axis = (rotation@self.observer.axes[:,j,:,None]).squeeze(-1)
                origins.append(position); directions.append(axis)
                rate = axis*dq[:,:,j,None]
                alpha = alpha+torch.cross(omega,rate,dim=-1)
                omega = omega+rate
                skew = self.observer.skews[:,j]
                angle = q[:,:,j,None,None]
                local = torch.eye(3,device=frame.device,dtype=frame.dtype)+torch.sin(angle)*skew+(1-torch.cos(angle))*(skew@skew)
                rotation = rotation@local
            center_offset = (rotation@self.com_local[:,j,:,None]).squeeze(-1)
            center = position+center_offset
            com_acc = acceleration+torch.cross(alpha,center_offset,dim=-1)+torch.cross(omega,torch.cross(omega,center_offset,dim=-1),dim=-1)
            inertia = rotation@self.inertia_local[:,j]@rotation.transpose(-1,-2)
            force = self.mass[:,j,None]*com_acc
            moment = (inertia@alpha.unsqueeze(-1)).squeeze(-1)+torch.cross(omega,(inertia@omega.unsqueeze(-1)).squeeze(-1),dim=-1)
            centers.append(center); forces.append(force); moments.append(moment)
        centers_tensor = torch.stack(centers,2)
        com = (centers_tensor*self.mass[None,:,:,None]).sum((1,2))+self.root_moment
        com = com/self.total_mass
        jacobian = torch.stack([torch.cross(directions[j],position-origins[j],dim=-1) for j in range(3)],-1)
        bias = []
        for j in range(3):
            net = torch.zeros_like(position)
            for k in range(j,4):
                net = net+moments[k]+torch.cross(centers[k]-origins[j],forces[k],dim=-1)
            bias.append((directions[j]*net).sum(-1))
        return position, jacobian, com, torch.stack(bias,-1).flatten(1)
