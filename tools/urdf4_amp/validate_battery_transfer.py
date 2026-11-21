"""Frozen-policy diagnostic across a mass change; never approves AMP data.

Uses the old manifest only for command labels and geometry diagnostics. No AMP
learner is instantiated, no reference-state bank is loaded or restored, and no
checkpoint/data/validation contract is relabeled to the new URDF.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from evaluate_path_tracking import ROOT,run
import torch,numpy as np
from legged_gym.utils import get_args,task_registry
from select_cts_direction import annotate_stops,summarize


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--urdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60.);p.add_argument('--seeds',default='1,2,3')
    p.add_argument('--model',type=Path,default=ROOT/'artifacts/urdf4_direction_selection_20261116/best/student.pt')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_direction_v1';args.headless=True
    cfg,_=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg)
    manifest=Path(cfg.amp_cts.reference)/'manifest.json';m=json.loads(manifest.read_text())
    cases=[dict(name=c['clip'],mode=2,command=c['command'],moving=True) for c in m['clips'] if c['clip']!='march']
    for c in cases:
        if c['name']=='forward':c['command']=[.25,0.,0.]
    args.num_envs=len(cases)*4;cfg.env.test=False;cfg.noise.add_noise=True;cfg.push.enabled=True
    cfg.asset.file=str(own.urdf.resolve());cfg.amp_cts.reference_reset_probability=0.
    cfg.amp_cts.reference_bank=str(own.output/'NO_REFERENCE_STATE_BANK.pt')
    cfg.rewards.heading_gain=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);env._push_strength=lambda:1.
    assert env.reference_bank is None
    masses=[sum(p.mass for p in env.gym.get_actor_rigid_body_properties(e,a)) for e,a in zip(env.envs,env.actor_handles)]
    policy=torch.jit.load(str(own.model),map_location=env.device).eval()
    for value in policy.parameters():value.requires_grad_(False)
    digest=lambda:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in policy.state_dict().values())).hexdigest()
    initial=digest();file_sha=hashlib.sha256(own.model.read_bytes()).hexdigest();reference_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    own.output.mkdir(parents=True,exist_ok=False);reports=[]
    for seed in map(int,own.seeds.split(',')):
        peaks=torch.zeros(env.num_envs,3,device=env.device);over17=torch.zeros(env.num_envs,12,device=env.device);ticks=[0]
        original=env._compute_torques
        def actuator(action):
            applied=original(action)
            peaks[:,0]=torch.maximum(peaks[:,0],env.raw_pd_torques.abs().amax(1))
            peaks[:,1]=torch.maximum(peaks[:,1],env.motor_torques.abs().amax(1))
            peaks[:,2]=torch.maximum(peaks[:,2],applied.abs().amax(1))
            over17.add_((env.raw_pd_torques.abs()>17).float());ticks[0]+=1
            return applied
        env._compute_torques=actuator
        try:report,arrays=run(env,policy,cases,own.seconds,seed,'localization_sim',env.path.gains,stop_s=5.)
        finally:env._compute_torques=original
        annotate_stops(report['results'],arrays,0,len(cases),own.seconds,env.dt)
        for j,row in enumerate(report['results']):
            i=j//2;ids=[4*i] if row['kind']=='nominal' else list(range(4*i+1,4*i+4))
            row.update(motor_peak_Nm=float(peaks[ids,1].max()),applied_peak_Nm=float(peaks[ids,2].max()),
                raw_over17_fraction=float((over17[ids]/ticks[0]).mean()))
        reports.append(report)
        np.savez_compressed(str(own.output/('seed%d.npz'%seed)),**arrays)
        result=dict(urdf=str(own.urdf.resolve()),urdf_sha256=hashlib.sha256(own.urdf.read_bytes()).hexdigest(),
            policy=str(own.model.resolve()),policy_sha256=file_sha,compiled_actor_masses_kg=masses,
            reports=reports,summary=summarize(reports),frozen_weights_unchanged=digest()==initial,
            diagnostic_only=True,amp_training_approved=False,reference_bank_used=False,
            reference_manifest_sha256=reference_hash,reference_urdf_sha256=m['urdf_sha256'],
            reference_manifest_unchanged=hashlib.sha256(manifest.read_bytes()).hexdigest()==reference_hash,
            note='Frozen policy transfer test; old reference clips supply command labels only, not validation for the changed mass. Nominal truth pose / robust noisy globally corrected pose assumption retained.',long_training_started=False)
        (own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BATTERY TRANSFER',seed,result['summary'],flush=True)
    assert digest()==initial and hashlib.sha256(own.model.read_bytes()).hexdigest()==file_sha


if __name__=='__main__':main()
