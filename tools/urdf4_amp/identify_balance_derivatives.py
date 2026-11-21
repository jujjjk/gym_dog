"""Local teacher-function derivatives on measured states, NOT AMP trajectories.

Joint/IMU/velocity observations are perturbed consistently. Body-pose changes
keep world foot points fixed to first order, verified by actual URDF FK. These
queries identify feedback signs independently of autocorrelated action labels.
"""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/urdf4_amp'))
import numpy as np,torch
import mujoco as mj
from teacher_v3 import Teacher,rot,rpy


def main():
    base=ROOT/'artifacts/urdf4_speed_fix_20261117';output=base/'counterfactual_feedback.pt';assert not output.exists()
    old=torch.load(base/'corrective1/model_0.pt',map_location='cpu');cfg=old['experiment']['env'];path=Path(cfg['amp_cts']['reference'])
    meta=json.load(open(path/'manifest.json'));spec=copy.deepcopy(next(c for c in meta['clips'] if c['command']==[.5,0,0]))
    rp=Path(spec['isaac_gym_validation']['receipt']);receipt=json.load(open(rp));file=rp.parent/'live_rollout.npz';z=np.load(file)
    ci=next(i for i,c in enumerate(receipt['cases']) if c['clip']==spec['isaac_gym_validation']['case']['clip'])
    teacher=Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec);m,d=teacher.m,teacher.d
    default=np.array([cfg['init_state']['default_joint_angles'][n] for n in meta['joint_order']]);scale=np.array([.15,.25,.25]*4);correction=(teacher.q0-default)*(1/.928-1)
    alpha=spec['design']['feedforward_alpha'];times=z['time_s']-.02;indices=np.flatnonzero((times>=10)&(times<10+spec['design']['period_s']))
    before=[];after=[];deltas=[];labels=[];maxfooterror=0.;skipped=0

    def observe(qp,qv,t,previous,heading,command):
        frame=np.zeros(51);R=rot(qp[3:7]);gravity=R.T@np.array([0,0,-1.]);phase=(t/spec['design']['period_s']+spec['phase_alignment_offset_cycles'])%1
        frame[:3]=qv[3:6]*.25;frame[3:6]=gravity;frame[6:9]=command*np.array([2,2,.25]);frame[9:21]=qp[teacher.qa]-default
        frame[21:33]=qv[teacher.va]*.05;frame[33:45]=previous;frame[45:47]=[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)]
        frame[48:51]=[np.sin(heading),np.cos(heading)-1,1]
        yaw=rpy(qp[3:7])[2];c,s=np.cos(yaw),np.sin(yaw);vx,vy=qv[:2];measured=np.array([c*vx+s*vy,-s*vx+c*vy])
        return np.r_[np.tile(frame,20),spec['command'],spec['command'],measured,1.].astype(np.float32)

    def command(qp,qv,t,heading,body_command,derivatives):
        teacher.ff[:]=0.;teacher.yaw_mean=0.;teacher.yaw_integral=0.;teacher.heading_integral=0.;teacher.velocity_mean[:]=0.;teacher.velocity_integral[:]=0.
        teacher.reference_xy=qp[:2].copy();teacher.reference_yaw=rpy(qp[3:7])[2]+heading
        c,s=np.cos(heading),np.sin(heading);planned=body_command.copy();planned[:2]=[c*body_command[0]+s*body_command[1],-s*body_command[0]+c*body_command[1]]
        original=teacher.derivatives
        if not derivatives:teacher.derivatives={}
        value=teacher.command(t,qp,qv,phase_override=t/spec['design']['period_s']%1,planned_override=planned,amplitude_override=1.)
        teacher.derivatives=original
        return (value+correction-default)/scale

    for idx in indices:
        f=z['amp_features'][idx-1,ci];root=z['root_state_xyzw_world'][idx-1,ci];t=times[idx]
        qp=np.zeros(m.nq);qv=np.zeros(m.nv);qp[:3]=root[:3];qp[3:7]=root[[6,3,4,5]];qp[teacher.qa]=f[:12]
        qv[:3]=root[7:10];qv[3:6]=f[39:42];qv[teacher.va]=f[12:24]
        yaw=rpy(qp[3:7])[2];heading=-yaw;body_command=np.array([.5,0.,0.]);previous=(z['issued_joint_targets'][idx-1,ci]+correction-default)/scale
        pub=observe(qp,qv,t,previous,heading,body_command);base_full=command(qp,qv,t,heading,body_command,True);base_force=command(qp,qv,t,heading,body_command,False)
        d.qpos[:]=qp;d.qvel[:]=qv;mj.mj_forward(m,d);feet0=d.xpos[teacher.fb].copy();R0=rot(qp[3:7]);rel=(R0.T@(feet0-qp[:3]).T).T;jac=[]
        for leg,foot in enumerate(teacher.fb):
            J=np.zeros((3,m.nv));mj.mj_jacBody(m,d,J,None,foot);jac.append(R0.T@J[:,teacher.va[leg*3:leg*3+3]])
        tests=[]
        for axis in [0,1]:
            for value in [-.08,.08]:tests.append(('velocity',axis,value))
        for axis in range(3):
            for value in [-.10,.10]:tests.append(('gyro',axis,value))
            for value in [-.002,.002]:tests.append(('position',axis,value))
            for value in [-.004,.004]:tests.append(('attitude',axis,value))
        for value in [-.015,.015]:tests.append(('heading',0,value))
        for axis in range(12):
            for value in [-.25,.25]:tests.append(('joint_velocity',axis,value))
        for kind,axis,value in tests:
            qpc=qp.copy();qvc=qv.copy();hc=heading;delta_foot=np.zeros((4,3));pose=False
            if kind=='velocity':
                shift=np.zeros(3);shift[axis]=value;c,s=np.cos(yaw),np.sin(yaw);qvc[:2]+=np.array([[c,-s],[s,c]])@shift[:2]
            elif kind=='gyro':qvc[3+axis]+=value
            elif kind=='joint_velocity':qvc[teacher.va[axis]]+=value
            elif kind=='heading':hc+=value
            elif kind=='position':
                shift=np.zeros(3);shift[axis]=value;qpc[:3]+=shift;delta_foot[:]=-R0.T@shift;pose=True
            elif kind=='attitude':
                vector=np.zeros(3);vector[axis]=value;rotation=np.r_[np.cos(value/2),np.eye(3)[axis]*np.sin(value/2)]
                mj.mju_mulQuat(qpc[3:7],qp[3:7],rotation);delta_foot=-np.cross(np.tile(vector,(4,1)),rel);pose=True
                hc=heading-(rpy(qpc[3:7])[2]-yaw)
            if pose:
                for leg in range(4):qpc[teacher.qa[leg*3:leg*3+3]]+=np.linalg.solve(jac[leg],delta_foot[leg])
                d.qpos[:]=qpc;d.qvel[:]=qvc;mj.mj_forward(m,d);error=float(abs(d.xpos[teacher.fb]-feet0).max());maxfooterror=max(maxfooterror,error)
                if error>.0005 or np.any(qpc[teacher.qa]<teacher.limits[:,0]) or np.any(qpc[teacher.qa]>teacher.limits[:,1]):skipped+=1;continue
            full=command(qpc,qvc,t,hc,body_command,True);force=command(qpc,qvc,t,hc,body_command,False)
            # Filter force feedback once at runtime; kinematic heading trim is
            # not multiplied by the force filter a second time.
            delta=(force-base_force)/alpha+(full-base_full)-(force-base_force)
            before.append(pub);after.append(observe(qpc,qvc,t,previous,hc,body_command));deltas.append(delta);labels.append(kind)
    report=dict(samples=len(before),source_record_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),urdf_sha256=meta['urdf_sha256'],
        maximum_pose_perturbation_FK_foot_error_m=maxfooterror,skipped=skipped,not_AMP_expert_data=True,not_physical_trajectory_validation=True,
        meaning='kinematically consistent local teacher-function derivatives; no runtime teacher and no plant parameter changes')
    torch.save(dict(previous_public=torch.tensor(np.stack(before)),following_public=torch.tensor(np.stack(after)),delta_action=torch.tensor(np.stack(deltas),dtype=torch.float),labels=labels,report=report),output)
    (base/'counterfactual_feedback.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
