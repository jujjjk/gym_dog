"""Preserve the measured 50 Hz phase knots; never round 24/25/26 ticks to 100."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def build(env,output):
    path=Path(env.cfg.amp_cts.reference);meta=json.loads((path/'manifest.json').read_text());cache={};tables=[];counts=[];offsets=[];stands=[];receipts=[]
    default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy()
    for c in meta['clips']:
        rp=Path(c['isaac_gym_validation']['receipt']);r=json.loads(rp.read_text());source=rp.parent/'live_rollout.npz'
        if source not in cache:
            assert hashlib.sha256(source.read_bytes()).hexdigest()==r['live_rollout_sha256'];cache[source]=np.load(source)
        z=cache[source];i=next(j for j,v in enumerate(r['cases']) if v['clip']==c['isaac_gym_validation']['case']['clip'])
        physical=path/c['source_folder']/'physical.npz';assert hashlib.sha256(physical.read_bytes()).hexdigest()==c['source_npz_sha256']
        q0=np.load(physical)['actuator_default'];correction=(q0-default)*(1/.928-1)
        time=z['time_s']-.02;take=(time>=5)&(time<55);n=round(c['design']['period_s']/.02)
        assert abs(n*.02-c['design']['period_s'])<1e-6
        bucket=np.rint(time[take]/.02).astype(int)%n;actions=(z['issued_joint_targets'][take,i]+correction-default)/scale
        table=np.stack([actions[bucket==j].mean(0) for j in range(n)])
        tables.append(table);counts.append(n);offsets.append(c['phase_alignment_offset_cycles'])
        settle=(time>-.5)&(time<-.05);stands.append((z['issued_joint_targets'][settle,i].mean(0)+correction-default)/scale)
        receipts.append(dict(clip=c['clip'],live_sha256=r['live_rollout_sha256'],source_npz_sha256=c['source_npz_sha256']))
    length=max(counts);padded=np.stack([np.r_[a,np.tile(a[-1:],(length-len(a),1))] for a in tables])
    result=dict(actions=torch.tensor(padded,dtype=torch.float),commands=env.reference_commands.cpu(),counts=torch.tensor(counts),
        offsets=torch.tensor(offsets),stand_actions=torch.tensor(np.stack(stands),dtype=torch.float),urdf_sha256=env.urdf_sha256,
        reference_manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),provenance=receipts,
        phase_definition='(control tick dt / period + contact alignment) modulo 1; knots retain native ticks')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Preserve existing native prior')
    torch.save(result,output);return result
