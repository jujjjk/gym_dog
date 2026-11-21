"""Package physically qualified Gym states, with an independent URDF FK check."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import numpy as np,torch,mujoco as mj
from design_v2 import setup


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--clip',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    receipt=json.loads((args.source/'result.json').read_text());i=next(j for j,c in enumerate(receipt['cases']) if c['spec']['clip']==args.clip);row=receipt['cases'][i]
    assert receipt['seconds']>=60 and row['passed'] and not receipt['automatic_resets'] and not receipt['root_or_joint_overwrite_during_rollout']
    assert receipt['plant_settings']['actuator']['peak_torque_limit_nm']==14.
    assert all(v==40. for v in receipt['plant_settings']['control']['stiffness'].values())
    z=np.load(args.source/'rollout.npz');time=z['time_s']-.02;indices=np.flatnonzero((time>=8)&(time<receipt['seconds']-2))
    assert np.all(z['valid'][indices,i]) and len(indices)>2000
    f=z['features'][indices,i];model=mj.MjModel.from_xml_path(str(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant/scene.xml'));data=mj.MjData(model);qa,va,feet,limits=setup(model)
    maximum=0.
    for frame in f:
        data.qpos[:]=0.;data.qpos[2]=.3;data.qpos[3]=1.;data.qpos[qa]=frame[:12];mj.mj_forward(model,data)
        error=abs((data.xpos[feet]-data.qpos[:3])-frame[24:36].reshape(4,3)).max();maximum=max(maximum,float(error))
    assert maximum<5.e-5,'Actual Gym feature feet must match this URDF FK'
    previous=z['features'][indices-1,i];following=z['features'][indices,i]
    command=np.tile(row['spec']['command'],(len(indices),1));observations=z['observations'][indices,i]
    actions=np.clip(z['actions'][indices,i],-4,4)
    def public_features(obs,nominal):
        return np.c_[obs[:,:3],obs[:,3:5],obs[:,9:45],obs[:,1026:1028]-nominal[:2]]
    offset=row['spec']['phase_alignment_offset_cycles']
    phase=(np.arctan2(observations[:,45],observations[:,46])/(2*np.pi)-offset)%1
    bucket=np.rint(phase*24).astype(int)%24
    values=public_features(observations,np.array(row['spec']['command']))
    nominal_reference=np.stack([values[bucket==j].mean(0) for j in range(24)])
    cold=(time>1.5)&(time<1.95);nominal_stand=public_features(z['observations'][cold,i],np.zeros(3)).mean(0)
    manifest=dict(kind='actual_free_base_gym_reference',urdf_sha256=receipt['urdf_sha256'],seconds=receipt['seconds'],
        source_receipt=str((args.source/'result.json').resolve()),source_receipt_sha256=hashlib.sha256((args.source/'result.json').read_bytes()).hexdigest(),
        source_rollout_sha256=hashlib.sha256((args.source/'rollout.npz').read_bytes()).hexdigest(),case=row,
        maximum_independent_FK_error_m=maximum,frames=len(indices),policy_dt_s=.02,action_clip=4.,
        plant_settings=receipt['plant_settings'],
        raw_states_unchanged=True,teacher_used_for_reference_only=True,policy_accepted=False)
    artifact=dict(pairs=torch.tensor(np.c_[previous,following]),commands=torch.tensor(command,dtype=torch.float),
        observations=torch.tensor(observations),actions=torch.tensor(actions),holdout=torch.arange(len(indices))%5==4,
        nominal_reference=torch.tensor(nominal_reference,dtype=torch.float),nominal_stand=torch.tensor(nominal_stand,dtype=torch.float),
        manifest=manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(artifact,args.output)
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Packaged',len(indices),'FK max',maximum,flush=True)


if __name__=='__main__':main()
