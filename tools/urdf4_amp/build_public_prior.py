"""Export a sensor-only model prior; behavior still needs independent tests."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch
from legged_gym.algorithms.urdf4_public_prior import PublicCentroidalPrior,ProtectedPublicPrior
from legged_gym.envs.urdf4.amp_v5_config import Urdf4AMPV5Cfg
from teacher_v3 import Teacher


def main():
    base=ROOT/'artifacts/urdf4_speed_fix_20261117';out=base/'public_prior1';out.mkdir(exist_ok=False)
    receipt=json.load(open(base/'public_noI60/result.json'))
    case=next(c for c in receipt['cases'] if c['passed'] and c['spec']['command']==[.45,0.,0.])
    spec=case['spec'].copy();spec['command']=[.5,0.,0.]
    teacher=Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec)
    s=teacher.s;names=[l+'_'+j+'_joint' for l in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
    default=[Urdf4AMPV5Cfg.init_state.default_joint_angles[n] for n in names]
    leads=np.array([teacher.lead+s.get('front_lead_offset_s',0)]*2+[teacher.lead+s.get('rear_lead_offset_s',0)]*2)+np.array(s.get('leg_lead_offsets_s',[0.]*4))
    archive=dict(urdf_sha256=receipt['urdf_sha256'],geometry=torch.tensor(teacher.q),yaw_derivative=torch.tensor(teacher.derivatives[2]),
        stand=torch.tensor(teacher.q0),leads=torch.tensor(leads),duties=torch.tensor(s.get('leg_stance_ratio',[teacher.duty]*4)),
        action_scale=torch.tensor([.15,.25,.25]*4),height=teacher.h,period=teacher.T,offset=spec['phase_alignment_offset_cycles'],
        feed=teacher.feed,velocity_gain=s['velocity_gain'],yaw_gain=s.get('yaw_velocity_gain',1.),yaw_step=s['yaw_step_feedback'],
        yaw_step_integral=s['yaw_step_integral'],alpha=s['feedforward_alpha'],support_ramp=s.get('support_ramp_phase',.08))
    path=out/'prior.pt';torch.save(archive,path)
    prior=PublicCentroidalPrior(ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf',names,default,path)
    data=np.load(base/'public_noI60/rollout.npz');idx=receipt['cases'].index(case)
    observations=torch.tensor(data['observations'][::7,idx,:1029])
    with torch.no_grad():actions=prior(observations);script=torch.jit.script(prior);jit=script(observations)
    ref=torch.tensor(data['actions'][::7,idx]);steady=torch.arange(len(observations))*7*.02>8
    # Windowed filters approximate the reference's unbounded sensor memory.
    difference=actions[steady]-ref[steady]
    result=dict(action_rms_error=float(difference.square().mean().sqrt()),action_max_error=float(difference.abs().max()),
        jit_max_error=float((jit-actions).abs().max()),policy_accepted=False,private_inputs=False)
    protected=base/'selected/exported/student_50.pt';policy=ProtectedPublicPrior(protected,prior)
    script=torch.jit.script(policy);script.save(str(out/'student.pt'))
    metadata=json.load(open(base/'selected/exported/student_50.pt.json'))
    metadata.update(fast_straight_reference=[.5,0.,0.],fast_straight_threshold_m_s=.35,public_physics_prior=True,policy_accepted=False)
    (out/'student.pt.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
