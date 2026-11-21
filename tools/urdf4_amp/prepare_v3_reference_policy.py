"""Build a public phase/command joint-target prior and zero the learned correction.

Uses validated issued actuator targets, not q poses. No root trajectory is stored
inside the policy. This is initialization, not evidence of PPO success.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,default=ROOT/'artifacts/urdf4_amp_v3_training/reference_init');p.add_argument('--encoder-steps',type=int,default=500)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    if not 0<=own.encoder_steps<=1000:raise ValueError('Finite context initialization only')
    own.output.mkdir(parents=True,exist_ok=False);args=get_args();args.task='urdf4_amp_cts_v3';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    path=Path(cfg.amp_cts.reference);m=json.loads((path/'manifest.json').read_text());cache={};tables=[];provenance=[];bins=100
    default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy()
    for c in m['clips']:
        receipt_path=Path(c['isaac_gym_validation']['receipt']);r=json.loads(receipt_path.read_text());source=receipt_path.parent/'live_rollout.npz'
        if source not in cache:
            if hashlib.sha256(source.read_bytes()).hexdigest()!=r['live_rollout_sha256']:raise ValueError('Issued target recording changed')
            z=np.load(source);cache[source]=dict(targets=z['issued_joint_targets'],time=z['time_s'])
        record=cache[source];case=c['isaac_gym_validation']['case']['clip'];index=next(i for i,row in enumerate(r['cases']) if row['clip']==case)
        raw=path/c['source_folder']/'physical.npz'
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=c['source_npz_sha256']:raise ValueError('Actuator initialization changed')
        q0=np.load(raw)['actuator_default'];correction=(q0-default)*(1/.928-1)
        t=record['time']-.02;take=(t>=5)&(t<55);actions=(record['targets'][take,index]+correction-default)/scale
        phase=(t[take]/c['design']['period_s']+c['phase_alignment_offset_cycles'])%1;bucket=np.rint(phase*bins).astype(int)%bins
        occupied=np.unique(bucket);means=np.stack([actions[bucket==i].mean(0) for i in occupied])
        table=np.stack([np.interp(np.arange(bins),np.r_[occupied-bins,occupied,occupied+bins],np.tile(means[:,j],3)) for j in range(12)],1)
        tables.append(table);provenance.append(dict(clip=c['clip'],live_sha256=r['live_rollout_sha256'],source_case=case))
    archive=Path(cfg.amp_cts.prior_archive);archive.parent.mkdir(parents=True,exist_ok=True)
    content=dict(actions=torch.tensor(np.stack(tables),dtype=torch.float32),commands=env.reference_commands.cpu(),urdf_sha256=env.urdf_sha256,
        reference_manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),provenance=provenance,
        meaning='Public command/phase joint-target prior with a learned 12-action bounded correction',bins=bins)
    if archive.exists():
        existing=torch.load(archive,map_location='cpu')
        for key in ['urdf_sha256','reference_manifest_sha256','provenance','bins']:
            if existing[key]!=content[key]:raise ValueError('Existing prior has another provenance; preserve it')
        if not torch.equal(existing['commands'],content['commands']) or not torch.allclose(existing['actions'],content['actions'],atol=1e-6,rtol=0):
            raise ValueError('Existing prior has different targets; preserve it')
    else:torch.save(content,archive)
    runner=Urdf4AMPV3Runner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
    runner.warm_start(ROOT/'artifacts/urdf4_direction_selection_20261116/best/model_100.pt')
    model=runner.alg.actor_critic
    with torch.no_grad():model.actor[-1].weight.zero_();model.actor[-1].bias.zero_()
    samples=ROOT/'artifacts/urdf4_amp_v3_training/bootstrap/samples.pt'
    if samples.exists():data=torch.load(samples,map_location='cpu')
    else:
        from prepare_v3_seed import collect_samples
        data=collect_samples(env,own.output)
    obs=data['observations'].to(args.rl_device)
    optimizer=torch.optim.Adam(model.student_encoder.parameters(),lr=2e-4)
    for k in range(own.encoder_steps):
        ids=torch.randint(len(obs),(2048,),device=args.rl_device);public,state,_=model.split(obs[ids]);prediction=model.student_encoder(public)
        with torch.no_grad():target=model.teacher_context(state)
        loss=(prediction-target).square().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.student_encoder.parameters(),1.);optimizer.step()
    runner.seed_migration.update(reference_initialization=True,actor_correction_zeroed=True)
    runner.save(str(own.output/'run/model_0.pt'))
    with torch.no_grad():
        public=obs[:256,:1020];expected=model.act_inference(public);jit=torch.jit.load(str(own.output/'run/exported/student_0.pt'),map_location=args.rl_device);error=float((jit(public)-expected).abs().max());assert error<1e-5
    (own.output/'result.json').write_text(json.dumps(dict(checkpoint=str(own.output/'run/model_0.pt'),prior_archive=str(archive),
        clips=len(tables),encoder_updates=own.encoder_steps,actor_correction_initially_zero=True,JIT_error=error,
        no_full_state_teacher_at_runtime=True,behavior_accepted=False),indent=2)+'\n')


if __name__=='__main__':main()
