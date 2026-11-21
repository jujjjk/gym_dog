"""Publish a frozen V3 selection after screening and finalist evaluations."""
import argparse,csv,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner
from select_v3_models import summary,rank


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',default='600,700')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    screens=[json.loads((own.output/('screen%d/screen.json'%i)).read_text()) for i in [0,1]]
    screened=[x for r in screens for x in r['candidates']]
    assert len(screened)==301 and len({x['checkpoint'] for x in screened})==301
    final=[]
    for iteration in map(int,own.iterations.split(',')):
        nominal=own.output/('final_%d_nominal_s1/result.json'%iteration)
        paths=[nominal]+[own.output/('final_%d_robust_s%d/result.json'%(iteration,seed)) for seed in [1,2,3]]
        reports=[json.loads(path.read_text()) for path in paths]
        seed=next(x for x in screened if x['iteration']==iteration)
        assert all(r['checkpoint_sha256']==seed['checkpoint_sha256'] and r['seconds']==60 and r['pure_student'] and not r['automatic_resets'] for r in reports)
        assert all(r['full_strength_push'] and r['nominal_fraction']==.25 and r['seed']==i+1 for i,r in enumerate(reports[1:]))
        ns=summary(reports[0]['cases']);rs=summary([c for r in reports[1:] for c in r['cases']])
        finals=dict(iteration=iteration,checkpoint=seed['checkpoint'],checkpoint_sha256=seed['checkpoint_sha256'],
            baseline=iteration==40,nominal=ns,robust=rs,nominal_qualified=reports[0]['core_passed'],
            robust_qualified=all(r['core_passed'] for r in reports[1:]),reports=[str(x.resolve()) for x in paths])
        # Hard nominal rejection, then robustness safety/gait/stop/direction.
        finals['rank']=[not finals['nominal_qualified'],rs['core_safety_failures'],rs['core_gait_failures'],
            rs['core_stop_failures'],rs['core_direction_failures'],-rs['core_passed'],-rs['frontier_passed'],
            -ns['frontier_passed'],rs['core_tracking_score']]
        final.append(finals)
    best=min(final,key=lambda x:x['rank']);assert best['nominal_qualified'],'No nominal-qualified finalist'
    args=get_args();args.task='urdf4_amp_cts_v3';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device)
    assert all(r['source_sha256']==runner.experiment['source_sha256'] for r in screens)
    for item in final:
        for path in item['reports']:
            report=json.loads(Path(path).read_text())
            assert report['contract']==runner.contract and report['source_sha256']==runner.experiment['source_sha256']
    runner.load(best['checkpoint'],load_optimizer=False)
    source=Path(best['checkpoint']);destination=own.output/'best';destination.mkdir(exist_ok=False)
    target=destination/source.name;shutil.copy2(source,target)
    original_jit=source.parent/'exported'/('student_%d.pt'%best['iteration'])
    shutil.copy2(original_jit,destination/'student.pt');shutil.copy2(Path(str(original_jit)+'.json'),destination/'student.pt.json')
    with torch.no_grad():
        probe=torch.randn(32,1020,device=env.device)*.05;probe[:,50]=1.
        jit=torch.jit.load(str(destination/'student.pt'),map_location=env.device)
        error=float((runner.alg.actor_critic.act_inference(probe)-jit(probe)).abs().max());assert error<1e-5,error
        public=env.obs_buf.clone();expected=runner.alg.actor_critic.act_inference(public);public[:,1020:]=torch.randn_like(public[:,1020:])
        assert torch.equal(expected,runner.alg.actor_critic.act_inference(public))
    receipt=dict(selected=best,checkpoint=str(target.resolve()),checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        student=str((destination/'student.pt').resolve()),JIT_parity_max_error=error,scanned_training_checkpoints=300,
        baseline_checkpoint_count=1,finalists=sorted(final,key=lambda x:x['rank']),snapshot_latest_iteration=30000,
        nominal_qualified=best['nominal_qualified'],robust_qualified=best['robust_qualified'],hardware_deployment_approved=False,
        ranking='Hard reject nominal failures, then robust safety, diagonal gait, stop, direction, full passes, expanded passes, tracking score',
        selection_only=True,training_started_during_selection=False,source_sha256=runner.experiment['source_sha256'],contract=runner.contract)
    (own.output/'selected.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (own.output/'coarse_ranking.csv').open('w') as f:
        fields=['iteration','baseline','stage',*screened[0]['summary']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in sorted(screened,key=rank):writer.writerow(dict(iteration=x['iteration'],baseline=x['baseline'],stage=x['stage'],**x['summary']))
    (own.output/'final.json').write_text(json.dumps(sorted(final,key=lambda x:x['rank']),indent=2)+'\n')
    print('SELECTED',best['iteration'],json.dumps(best),flush=True)


if __name__=='__main__':main()
