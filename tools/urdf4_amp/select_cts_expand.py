"""Evaluate saved X-expansion policies in parallel, then recheck finalists."""
import argparse,copy,csv,hashlib,json,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace

from evaluate_cts_expand import ROOT,evaluate
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_expand import Urdf4AMPExpandRunner,ExpandedStudentPolicy


def summary(reports):
    rows=[r for report in reports for r in report['results']]
    probes={'stand','forward_035','forward_040','forward_050','forward_060'}
    core=[r for r in rows if r['name'] not in probes]
    # "forward" belongs to the original manifest; forward_035..060 are probes.
    expanded=[r for r in rows if r['name'] in ('forward_035','forward_040')]
    fast=[r for r in rows if r['name'] in ('forward_050','forward_060')]
    stand=[r for r in rows if r['name']=='stand']
    assert len(core)==24*len(reports) or len(core)==12*len(reports)
    mean=lambda data,key:sum(r[key] for r in data)/max(len(data),1)
    return dict(core_failures=sum(not r['gait_passed'] for r in core),core_groups=len(core),
        expanded_failures=sum(not r['gait_passed'] for r in expanded),expanded_groups=len(expanded),
        core_linear_error_m_s=mean(core,'linear_error_m_s'),core_yaw_error_rad_s=mean(core,'yaw_error_rad_s'),
        expanded_linear_error_m_s=mean(expanded,'linear_error_m_s'),fast_linear_error_m_s=mean(fast,'linear_error_m_s'),
        core_raw_peak_Nm=max(r['raw_peak_Nm'] for r in core),core_resets=sum(r['resets'] for r in core),
        core_motor_rms_Nm=mean(core,'motor_rms_Nm'),
        stationary_safety_failures=sum(r['resets']>0 or r['flight_fraction']>0 or r['raw_peak_Nm']>17
            or r['illegal_support_fraction']>(.001 if r['kind']=='nominal' else .01)
            or r['linear_error_m_s']>.05 or r['yaw_error_rad_s']>.10 for r in stand),
        stationary_groups=len(stand),stationary_static_groups_passed=sum(r['gait_passed'] for r in stand),
        stationary_raw_peak_Nm=max(r['raw_peak_Nm'] for r in stand),
        score=.55*(mean(core,'linear_error_m_s')/.05+mean(core,'yaw_error_rad_s')/.1)
            +.35*mean(expanded,'linear_error_m_s')/.1+.1*mean(core,'motor_rms_Nm')/6.)


def rank(item):
    s=item['summary']
    failures=s['core_failures']+s.get('stationary_safety_failures',0)
    return (failures>0,failures,s['expanded_failures'],s['score'])


class PolicyBatch:
    def __init__(self,policies,device):
        self.count=len(policies);self.device=device
        self.params,self.buffers=torch.func.stack_module_state(policies)
        base=copy.deepcopy(policies[0]).to('meta')
        def call(params,buffers,history):
            return torch.func.functional_call(base,(params,buffers),(history,),strict=True)
        self.forward=torch.vmap(call,in_dims=(0,0,0))
        with torch.no_grad():
            test=torch.randn(self.count,68,1020,device=device)*.1
            expected=torch.stack([p(test[i]) for i,p in enumerate(policies)])
            self.parity_error=float((expected-self.forward(self.params,self.buffers,test)).abs().max())
            assert self.parity_error<1.e-5,self.parity_error

    def act_inference(self,obs):
        history=obs[:,:1020].reshape(self.count,68,1020)
        return self.forward(self.params,self.buffers,history).reshape(-1,12)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=20)
    p.add_argument('--coarse_seconds',type=float,default=10.)
    p.add_argument('--final_seconds',type=float,default=30.);p.add_argument('--final_seeds',default='1,2,3')
    p.add_argument('--shortlist',type=int,default=5)
    p.add_argument('--coarse_robust',action='store_true')
    p.add_argument('--coarse_only',action='store_true')
    p.add_argument('--reuse_final',type=Path)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_expand_x_v1';args.headless=True
    own.output.mkdir(parents=True,exist_ok=True)
    checkpoints=[own.baseline]+sorted(own.run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    assert len(set(map(str,checkpoints)))==len(checkpoints)
    assert own.coarse_seconds>=8 and own.final_seconds>=15 and 1<=own.batch<=40
    cfg,train=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg)
    if not own.coarse_robust:
        cfg.env.test=True;cfg.noise.add_noise=False;cfg.push.enabled=False
    args.num_envs=68*own.batch
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPExpandRunner(env,class_to_dict(train),None,device=args.rl_device)
    results=[];start=time.time()
    for offset in range(0,len(checkpoints),own.batch):
        paths=checkpoints[offset:offset+own.batch];padded=paths+[paths[0]]*(own.batch-len(paths))
        policies=[]
        for path in padded:
            state=torch.load(str(path),map_location=runner.device)
            assert state['contract']==runner.contract
            runner.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
            policy=ExpandedStudentPolicy(runner.alg.actor_critic).to(runner.device).eval().requires_grad_(False)
            policies.append(policy)
        batch=PolicyBatch(policies,runner.device)
        wrapper=SimpleNamespace(alg=SimpleNamespace(actor_critic=batch))
        report,_=evaluate(env,wrapper,own.coarse_seconds,1,replicas=own.batch)
        for i,path in enumerate(paths):
            rows=[r for r in report['results'] if r['replica']==i and (own.coarse_robust or r['kind']=='nominal')]
            item=dict(checkpoint=str(path.resolve()),iteration=int(path.stem.split('_')[-1]),baseline=path==own.baseline,
                report=dict(seed=1,seconds=own.coarse_seconds,nominal_only=not own.coarse_robust,results=rows),
                vectorization_max_error=batch.parity_error)
            item['summary']=summary([item['report']]);results.append(item)
        (own.output/'coarse.json').write_text(json.dumps(results,indent=2)+'\n')
        top=sorted(results,key=rank)[:3]
        print('COARSE',len(results),'/',len(checkpoints),'seconds',round(time.time()-start,1),
            'top',[(r['iteration'],r['summary']['core_failures'],round(r['summary']['score'],3)) for r in top],flush=True)
        del batch,wrapper,policies,report
    env.gym.destroy_sim(env.sim);del runner,env
    torch.cuda.empty_cache()
    if own.coarse_only:return
    shortlist=sorted(results,key=rank)[:own.shortlist]
    baseline=next(r for r in results if r['baseline'])
    if baseline not in shortlist:shortlist.append(baseline)
    # Always retain the final checkpoint as a measured comparison.
    last=next(r for r in results if r['iteration']==max(x['iteration'] for x in results))
    if last not in shortlist:shortlist.append(last)
    cfg,train=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg);args.num_envs=128
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPExpandRunner(env,class_to_dict(train),None,device=args.rl_device)
    final=[]
    for item in shortlist:
        cached=[]
        if own.reuse_final and own.reuse_final.exists():cached=json.loads(own.reuse_final.read_text())
        previous=next((x for x in cached if x['checkpoint']==item['checkpoint']
            and [r['seed'] for r in x['reports']]==list(map(int,own.final_seeds.split(',')))
            and all(r['seconds']==own.final_seconds for r in x['reports'])),None)
        if previous is not None:
            final.append(previous)
            (own.output/'final.json').write_text(json.dumps(final,indent=2)+'\n')
            print('REUSED FINAL',item['iteration'],flush=True)
            continue
        runner.load(item['checkpoint'],False);reports=[]
        for seed in map(int,own.final_seeds.split(',')):
            report,arrays=evaluate(env,runner,own.final_seconds,seed);reports.append(report)
            # Keep complete metrics for every seed; raw trajectories for seed 1.
            if seed==1:
                import numpy as np
                np.savez_compressed(str(own.output/('model_%d_seed%d.npz'%(item['iteration'],seed))),**arrays)
        record=dict(checkpoint=item['checkpoint'],iteration=item['iteration'],baseline=item['baseline'],
            reports=reports,summary=summary(reports),curriculum_stage=env.curriculum_stage)
        final.append(record);(own.output/'final.json').write_text(json.dumps(final,indent=2)+'\n')
        print('FINAL',record['iteration'],record['summary'],flush=True)
    best=min(final,key=rank)
    eligible=best['summary']['core_failures']==0 and best['summary']['stationary_safety_failures']==0
    if not eligible:
        (own.output/'no_qualified_candidate.json').write_text(json.dumps(dict(best=best,reason='All finalists failed old-motion gait gates'),indent=2)+'\n')
        raise RuntimeError('No finalist passed both the old motions and stationary safety')
    source=Path(best['checkpoint']);destination=own.output/source.name
    shutil.copy2(str(source),str(destination))
    export=source.parent/'exported'/('student_%d.pt'%best['iteration'])
    if export.exists():
        shutil.copy2(str(export),str(own.output/'best_student.pt'))
        if Path(str(export)+'.json').exists():shutil.copy2(str(export)+'.json',str(own.output/'best_student.pt.json'))
    result=dict(selected=best,checkpoint=str(destination.resolve()),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        scanned_checkpoints=len(checkpoints),coarse_seconds=own.coarse_seconds,coarse_nominal_only=not own.coarse_robust,
        final_candidates=len(final),final_seconds=own.final_seconds,final_seeds=own.final_seeds,
        ranking='Reject old-motion or stationary-safety failures, then expansion gait failures, then tracking/torque score',
        zero_six_speed_achieved=all(r['gait_passed'] and abs(r['mean_velocity'][0]-.6)<=.03
            and r['linear_error_m_s']<=.04 for report in best['reports'] for r in report['results'] if r['name']=='forward_060'),
        long_training_started=False)
    (own.output/'selected.json').write_text(json.dumps(result,indent=2)+'\n')
    with (own.output/'ranking.csv').open('w') as f:
        fields=['iteration','baseline',*best['summary'].keys()];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in sorted(final,key=rank):writer.writerow(dict(iteration=row['iteration'],baseline=row['baseline'],**row['summary']))
    print('SELECTED',best['iteration'],'baseline',best['baseline'],best['summary'],flush=True)


if __name__=='__main__':main()
