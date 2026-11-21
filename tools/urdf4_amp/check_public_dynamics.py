"""Independent MuJoCo oracle check; not a locomotion acceptance test."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch,mujoco as mj
from legged_gym.algorithms.urdf4_public_dynamics import PublicLegDynamics
from teacher_v3 import Teacher
from probe_public_balance import public_state


def main():
    base=ROOT/'artifacts/urdf4_speed_fix_20261117'
    source=np.load(base/'public045_confirm60/rollout.npz')
    manifest=json.load(open(ROOT/'artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/gym_verified/manifest.json'))
    spec=next(c for c in manifest['clips'] if c['command']==[.5,0,0])
    teacher=Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec)
    config=json.load(open(base/'selected/provenance.json'))
    # Runtime defaults, not the different native-template stand posture.
    from legged_gym.envs.urdf4.amp_v5_config import Urdf4AMPV5Cfg
    names=[l+'_'+j+'_joint' for l in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
    default=np.array([Urdf4AMPV5Cfg.init_state.default_joint_angles[n] for n in names])
    model=PublicLegDynamics(ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf',names,default).double()
    frames=source['observations'][::31,:,:51].reshape(-1,51).astype(np.float64)
    actual=model(torch.tensor(frames));errors=np.zeros(4)
    for i,frame in enumerate(frames):
        qp,qv,_,_=public_state(teacher,frame.copy(),default)
        qp[:3]=0.;teacher.d.qpos[:]=qp;teacher.d.qvel[:]=qv;mj.mj_forward(teacher.m,teacher.d)
        rotation=teacher.d.xmat[1].reshape(3,3)
        feet=(teacher.d.xpos[teacher.fb]-teacher.d.xpos[1])@rotation
        com=(teacher.d.subtree_com[1]-teacher.d.xpos[1])@rotation
        jac=[]
        for j,body in enumerate(teacher.fb):
            mj.mj_jacBody(teacher.m,teacher.d,teacher.J,None,body)
            jac.append(rotation.T@teacher.J[:,teacher.va[j*3:j*3+3]])
        expected=[feet,np.array(jac),com,teacher.d.qfrc_bias[teacher.va]]
        for k in range(4):errors[k]=max(errors[k],np.max(np.abs(actual[k][i].detach().numpy()-expected[k])))
    scripted=torch.jit.script(model)
    jit_errors=[float((a-b).abs().max()) for a,b in zip(actual,scripted(torch.tensor(frames)))]
    result=dict(frames=len(frames),feet_max_error_m=errors[0],jacobian_max_error_m_rad=errors[1],com_max_error_m=errors[2],bias_max_error_Nm=errors[3],jit_max_errors=jit_errors,passed=bool(max(errors)<2.e-5))
    print(json.dumps(result,indent=2));(base/'public_dynamics_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed']


if __name__=='__main__':main()
