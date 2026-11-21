"""URDF-derived public joint/IMU ground-height and vertical-speed observer."""
import hashlib,math,xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch import nn


class JointGroundObserver(nn.Module):
    def __init__(self,urdf,joint_names,default_positions):
        super().__init__();root=ET.parse(urdf).getroot();joints={j.attrib['name']:j for j in root.findall('joint')}
        names=[leg+'_'+kind+'_joint' for leg in ('FL','FR','RL','RR') for kind in ('hip','thigh','calf')]
        assert list(joint_names)==names
        offsets=[];axes=[];skews=[]
        for leg in ('FL','FR','RL','RR'):
            translations=[];directions=[];crosses=[]
            for kind in ('hip','thigh','calf','foot'):
                name=leg+'_'+kind+('_fixed' if kind=='foot' else '_joint');joint=joints[name]
                origin=joint.find('origin');assert max(abs(float(x)) for x in origin.get('rpy','0 0 0').split())<1.e-12
                translations.append(list(map(float,origin.get('xyz').split())))
                if kind!='foot':
                    axis=list(map(float,joint.find('axis').get('xyz').split()));directions.append(axis)
                    x,y,z=axis;crosses.append([[0,-z,y],[z,0,-x],[-y,x,0]])
            offsets.append(translations);axes.append(directions);skews.append(crosses)
            link=next(l for l in root.findall('link') if l.attrib['name']==leg+'_foot')
            assert abs(float(link.find('collision/geometry/sphere').get('radius'))-.02)<1.e-12
        self.register_buffer('offsets',torch.tensor(offsets));self.register_buffer('axes',torch.tensor(axes))
        self.register_buffer('skews',torch.tensor(skews,dtype=torch.float));self.register_buffer('default_q',torch.tensor(default_positions,dtype=torch.float))
        self.register_buffer('reference4',torch.zeros(24,2));self.register_buffer('reference5',torch.zeros(24,2))
        self.register_buffer('stand4',torch.zeros(2));self.register_buffer('stand5',torch.zeros(2))
        self.register_buffer('scales',torch.tensor([.01,.1]));self.urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest()

    @torch.jit.export
    def feet(self,q,dq):
        count=q.shape[0];angle=q.reshape(count,4,3);velocity=dq.reshape(count,4,3)
        rotation=torch.eye(3,device=q.device,dtype=q.dtype).expand(count,4,3,3)
        position=torch.zeros(count,4,3,device=q.device,dtype=q.dtype);origins=[];directions=[]
        for j in range(3):
            position=position+(rotation@self.offsets[:,j,:,None]).squeeze(-1)
            origins.append(position);directions.append((rotation@self.axes[:,j,:,None]).squeeze(-1))
            cross=self.skews[:,j];a=angle[:,:,j,None,None]
            local=torch.eye(3,device=q.device,dtype=q.dtype)+torch.sin(a)*cross+(1-torch.cos(a))*(cross@cross)
            rotation=rotation@local
        position=position+(rotation@self.offsets[:,3,:,None]).squeeze(-1)
        axis=torch.stack(directions,2);origin=torch.stack(origins,2)
        linear=(torch.cross(axis,position[:,:,None,:]-origin,dim=-1)*velocity[:,:,:,None]).sum(2)
        angular=(axis*velocity[:,:,:,None]).sum(2)
        return position,linear,angular

    def forward(self,frame):
        q=frame[:,9:21]+self.default_q;dq=frame[:,21:33]/.05
        position,linear,angular=self.feet(q,dq)
        gravity=frame[:,3:6];gravity=gravity/gravity.norm(p=2,dim=1,keepdim=True).clamp(min=1.e-6)
        gyro=frame[:,:3]/.25
        heights=.02+(position*gravity[:,None,:]).sum(2)
        weights=torch.softmax((heights-heights.amax(1,keepdim=True))/.003,1)
        moving_gyro=gyro[:,None,:].expand(-1,4,-1)
        rolling=torch.cross(angular+moving_gyro,.02*gravity[:,None,:].expand(-1,4,-1),dim=-1)
        relative=linear+torch.cross(moving_gyro,position,dim=-1)+rolling
        body_velocity=-(relative*weights[:,:,None]).sum(1)
        height=(heights*weights).sum(1);vertical=-(body_velocity*gravity).sum(1)
        return torch.stack((height,vertical),1)

    @torch.jit.ignore
    def configure(self,reference4,stand4,reference5,stand5):
        def frame(values):
            values=values.reshape(-1,43);f=torch.zeros(len(values),51,device=values.device)
            f[:,:3]=values[:,:3];f[:,3:5]=values[:,3:5];f[:,5]=-torch.sqrt((1-f[:,3:5].square().sum(1)).clamp(min=0))
            f[:,9:45]=values[:,5:41]
            return f
        with torch.no_grad():
            self.reference4.copy_(self(frame(reference4)));self.reference5.copy_(self(frame(reference5)))
            self.stand4.copy_(self(frame(stand4))[0]);self.stand5.copy_(self(frame(stand5))[0])

    @torch.jit.export
    def feedback_features(self,public,offset):
        count=public.shape[0];history=public[:,:1020].reshape(count,20,51)
        observed=self(history.reshape(-1,51)).reshape(count,20,2)
        phase=torch.remainder(torch.atan2(history[:,:,45],history[:,:,46])/(2*math.pi)-offset,1.)
        index=phase*24;lo=index.floor().long();lo=torch.where(lo>=24,torch.zeros_like(lo),lo)
        hi=lo+1;hi=torch.where(hi>=24,torch.zeros_like(hi),hi);weight=(index-index.floor())[:,:,None]
        ref4=(1-weight)*self.reference4[lo]+weight*self.reference4[hi]
        ref5=(1-weight)*self.reference5[lo]+weight*self.reference5[hi]
        coefficient=((.5-public[:,1020:1021])/.1).clamp(0.,1.5)[:,None,:]
        target=ref5+coefficient*(ref4-ref5);static=self.stand5+coefficient*(self.stand4-self.stand5)
        amplitude=public[:,1028:1029].clamp(0,1)[:,None,:];target=static+amplitude*(target-static)
        error=((observed-target)/self.scales).clamp(-10,10);basis=[torch.ones_like(phase)]
        for k in range(1,5):basis.extend([torch.sin(2*math.pi*k*phase),torch.cos(2*math.pi*k*phase)])
        features=(torch.stack(basis,2)[:,:,:,None]*error[:,:,None,:]).flatten(2)
        ages=torch.arange(20,device=public.device,dtype=public.dtype);weights=.35*torch.pow(torch.ones_like(ages)*.65,ages);weights/=weights.sum()
        return (features*weights[None,:,None]).sum(1)
