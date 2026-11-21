"""Compare action reconstruction with actual and repeated velocity history."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=ROOT/'artifacts/urdf4_speed_fix_20261117/public_yawstep_i25_60'
    receipt=json.load(open(base/'result.json'));data=np.load(base/'rollout.npz');policy=torch.jit.load(str(args.policy)).eval();rows=[]
    for clip in ('public_400_step0.7','public_450_step0.3'):
        i=next(i for i,c in enumerate(receipt['cases']) if c['spec']['clip']==clip)
        ids=np.arange(401,len(data['time_s']),7);frame=data['observations'][:,i,:1029]
        ages=np.arange(20);actual=frame[np.maximum(ids[:,None]-ages[None,:],0),1026:1029]
        observation=torch.tensor(np.c_[frame[ids],actual.reshape(len(ids),-1)])
        repeated=observation.clone();repeated[:,1029:]=observation[:,None,1026:1029].expand(-1,20,-1).flatten(1)
        with torch.no_grad():
            a=torch.cat([policy(batch) for batch in observation.split(128)])
            b=torch.cat([policy(batch) for batch in repeated.split(128)])
        target=torch.tensor(data['actions'][ids,i]);error=a-target;old=b-target
        rows.append(dict(clip=clip,frames=len(ids),actual_velocity_history_action_rms=float(error.square().mean().sqrt()),
            repeated_current_velocity_action_rms=float(old.square().mean().sqrt()),actual_history_max_error=float(error.abs().max()),behavior_accepted=False))
    result=dict(pure_public_reconstruction=True,cases=rows,root_states_used=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
