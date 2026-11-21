"""Three-seed finalist checks with independent, identically distributed DR."""
import argparse,copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from select_cts_expand import ROOT,evaluate,summary,rank,PolicyBatch
import torch,numpy as np
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_expand import Urdf4AMPExpandRunner,ExpandedStudentPolicy

p=argparse.ArgumentParser(add_help=False);p.add_argument('--selection',type=Path,required=True)
p.add_argument('--stationary_safe',action='store_true')
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
args=get_args();args.task='urdf4_amp_cts_expand_x_v1';args.headless=True
root=own.selection;out=root/('safe_batch_final' if own.stationary_safe else 'batch_final');out.mkdir(exist_ok=True)
first=json.loads((root/'coarse.json').read_text());second=json.loads((root/'robust_pipeline/coarse.json').read_text())
assert len(first)==len(second)==301
paths={}
for pool in [first,second]:
    if own.stationary_safe:
        for r in pool:r['summary']=summary([r['report']])
        chosen=[r for r in pool if r['summary']['core_failures']==0
            and r['summary']['expanded_failures']==0 and r['summary']['stationary_safety_failures']==0]
        chosen+=[next(r for r in pool if r['baseline'])]
    else:
        chosen=sorted(pool,key=rank)[:5]+[next(r for r in pool if r['baseline']),max(pool,key=lambda r:r['iteration'])]
    for r in chosen:paths[r['checkpoint']]=r
items=sorted(paths.values(),key=lambda r:r['iteration']);count=len(items)
cfg,train=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg);args.num_envs=count*68
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner=Urdf4AMPExpandRunner(env,class_to_dict(train),None,device=args.rl_device)
policies=[]
for item in items:
    state=torch.load(item['checkpoint'],map_location=runner.device);assert state['contract']==runner.contract
    runner.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
    policies.append(ExpandedStudentPolicy(runner.alg.actor_critic).to(runner.device).eval().requires_grad_(False))
batch=PolicyBatch(policies,runner.device);wrapper=SimpleNamespace(alg=SimpleNamespace(actor_critic=batch))
records=[dict(checkpoint=x['checkpoint'],iteration=x['iteration'],baseline=x['baseline'],reports=[]) for x in items]
start=time.time()
for seed in [1,2,3]:
    report,arrays=evaluate(env,wrapper,30.,seed,replicas=count)
    for i,record in enumerate(records):
        part={k:v for k,v in report.items() if k!='results'}
        part['results']=[r for r in report['results'] if r['replica']==i]
        part['domain_randomization']='Independent per-policy physical draws from the same ranges; fixed across seeds, sensor/push RNG reseeded'
        record['reports'].append(part);record['summary']=summary(record['reports'])
        if seed==1:
            np.savez_compressed(str(out/('model_%d_seed1.npz'%record['iteration'])),
                **{k:v[:,i*68:(i+1)*68] for k,v in arrays.items()})
    (out/'final.json').write_text(json.dumps(records,indent=2)+'\n')
    print('BATCH FINAL',seed,'policies',count,'seconds',round(time.time()-start,1),
        [(r['iteration'],r['summary']['core_failures'],r['summary']['stationary_safety_failures'],round(r['summary']['score'],3)) for r in sorted(records,key=rank)],flush=True)
(out/'protocol.json').write_text(json.dumps(dict(candidate_iterations=[r['iteration'] for r in records],
    public_size=1020,vectorization_max_error=batch.parity_error,seconds=30,seeds=[1,2,3],
    environments=count*68,independent_domain_draws=True,per_case_nominal=1,per_case_robust=3),indent=2)+'\n')
print('BATCH FINAL COMPLETE',flush=True)
