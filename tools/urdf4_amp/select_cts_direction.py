"""Frozen direction-checkpoint selection: batch screening, native finalist tests."""
import argparse,copy,csv,hashlib,json,shutil,sys,time
from pathlib import Path
from evaluate_path_tracking import ROOT,run
import torch,numpy as np
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_direction import Urdf4AMPDirectionRunner
from legged_gym.algorithms.urdf4_amp_expand import ExpandedStudentPolicy


class PolicyBatch:
    def __init__(self,policies,per_policy):
        self.count=len(policies);self.per_policy=per_policy
        self.params,self.buffers=torch.func.stack_module_state(policies)
        base=copy.deepcopy(policies[0]).to('meta')
        def call(params,buffers,history):
            return torch.func.functional_call(base,(params,buffers),(history,),strict=True)
        self.vectorized=torch.vmap(call,in_dims=(0,0,0))
        device=next(policies[0].parameters()).device
        with torch.no_grad():
            probe=torch.randn(self.count,per_policy,1020,device=device)*.1
            expected=torch.stack([p(probe[i]) for i,p in enumerate(policies)])
            actual=self.vectorized(self.params,self.buffers,probe)
            self.parity_error=float((expected-actual).abs().max())
            assert self.parity_error<1.e-5,self.parity_error

    def __call__(self,history):
        return self.vectorized(self.params,self.buffers,history.reshape(self.count,self.per_policy,1020)).reshape(-1,12)


def annotate_stops(rows,arrays,first_case,case_count,seconds,dt):
    start=round((seconds+1.)/dt)  # static assessment after the first second
    for j,row in enumerate(rows):
        i=first_case+j//2
        ids=[4*i] if row['kind']=='nominal' else list(range(4*i+1,4*i+4))
        c=arrays['contact'][start:,ids];n=c.sum(2)
        diagonal=(c[:,:,0]&c[:,:,3])|(c[:,:,1]&c[:,:,2])
        illegal=((n==0)|(n==1)|((n==2)&~diagonal)).mean()
        row['stop'].update(flight_fraction=float((n==0).mean()),illegal_support_fraction=float(illegal),
            four_contact_fraction=float((n==4).mean()),foot_clearance_p95_mm=float(np.percentile(arrays['feet'][start:,ids],95)*1000))


def summarize(reports):
    rows=[r for rep in reports for r in rep['results']]
    unsafe=lambda r:r['resets']>0 or r['flight_fraction']>0 or r['raw_peak_Nm']>17 or r['illegal_support_fraction']>(.01 if r['kind']=='robust' else .001)
    stop_bad=lambda r:r['stop']['resets']>0 or r['stop']['flight_fraction']>0 or r['stop']['illegal_support_fraction']>.001 or r['stop']['cycle_linear_error_m_s']>.05 or r['stop']['cycle_yaw_error_rad_s']>.1
    return dict(groups=len(rows),unsafe_groups=sum(unsafe(r) for r in rows),
        gait_failures=sum(not r['gait_passed'] for r in rows),stop_safety_failures=sum(stop_bad(r) for r in rows),
        nonstatic_stops=sum(r['stop']['four_contact_fraction']<.95 or r['stop']['foot_clearance_p95_mm']>5 for r in rows),
        path_failures=sum(not r['path_passed'] for r in rows),
        full_passed=sum(r['gait_passed'] and r['path_passed'] and not stop_bad(r) for r in rows),
        mean_cross_rms_m=float(np.mean([r['cross_rms_m'] for r in rows])),
        mean_heading_deg=float(np.mean([r['heading_cycle_mean_abs_deg'] for r in rows])),
        max_raw_Nm=max(r['raw_peak_Nm'] for r in rows),
        score=float(np.mean([r['cross_rms_m']/.03+r['heading_cycle_mean_abs_deg']/2 for r in rows])))


def rank(item):
    s=item['summary']
    return (s['unsafe_groups']+s['stop_safety_failures'],s['gait_failures'],s['path_failures'],s['nonstatic_stops'],s['score'])


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=24);p.add_argument('--shortlist',type=int,default=4)
    p.add_argument('--coarse_seconds',type=float,default=10.)
    p.add_argument('--final_seconds',type=float,default=60.);p.add_argument('--coarse_only',action='store_true')
    p.add_argument('--final_only',action='store_true');p.add_argument('--final_iterations')
    p.add_argument('--batch_final',action='store_true')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_direction_v1';args.headless=True
    own.output.mkdir(parents=True,exist_ok=True)
    cfg,train=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg)
    manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text())
    cases=[dict(name=c['clip'],command=c['command'],mode=2,moving=True) for c in manifest['clips'] if c['clip']!='march']
    for case in cases:
        if case['name']=='forward':case['command']=[.25,0.,0.]
    count=len(cases);per=count*4
    checkpoints=sorted([x for x in own.run.glob('model_*.pt') if time.time()-x.stat().st_mtime>5],key=lambda x:int(x.stem.split('_')[-1]))
    initial=ROOT/'artifacts/urdf4_direction_20261115/preflight_v3/run/model_100.pt'
    # The run's model_100 is an exact copy of the selected preflight weights.
    assert checkpoints and initial.exists()
    if not own.final_only:
        cfg.env.test=True;cfg.noise.add_noise=False;cfg.push.enabled=False
        args.num_envs=own.batch*per
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        runner=Urdf4AMPDirectionRunner(env,class_to_dict(train),None,device=args.rl_device)
        coarse=[];started=time.time()
        for offset in range(0,len(checkpoints),own.batch):
            paths=checkpoints[offset:offset+own.batch];padded=paths+[paths[0]]*(own.batch-len(paths));policies=[]
            for path in padded:
                saved=torch.load(str(path),map_location=runner.device)
                assert saved['contract']==runner.contract
                runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
                policies.append(ExpandedStudentPolicy(runner.alg.actor_critic).to(runner.device).eval().requires_grad_(False))
            batch=PolicyBatch(policies,per)
            report,arrays=run(env,batch,cases*own.batch,own.coarse_seconds,1,'oracle',env.path.gains,stop_s=3.)
            for i,path in enumerate(paths):
                part={k:v for k,v in report.items() if k!='results'}
                part['results']=report['results'][i*count*2:(i+1)*count*2]
                annotate_stops(part['results'],arrays,i*count,count,own.coarse_seconds,env.dt)
                item=dict(checkpoint=str(path.resolve()),iteration=int(path.stem.split('_')[-1]),reports=[part],vectorization_max_error=batch.parity_error)
                item['summary']=summarize(item['reports']);coarse.append(item)
            (own.output/'coarse.json').write_text(json.dumps(coarse,indent=2)+'\n')
            print('DIRECTION COARSE',len(coarse),'/',len(checkpoints),'seconds',round(time.time()-started,1),
                [(x['iteration'],x['summary']) for x in sorted(coarse,key=rank)[:3]],flush=True)
            del batch,policies,report,arrays
        env.gym.destroy_sim(env.sim);del runner,env;torch.cuda.empty_cache()
    else:coarse=json.loads((own.output/'coarse.json').read_text())
    if own.coarse_only:return
    shortlist=sorted(coarse,key=rank)[:own.shortlist]
    for item in [coarse[0],coarse[-1]]:
        if item not in shortlist:shortlist.append(item)
    if own.final_iterations:
        wanted=set(map(int,own.final_iterations.split(',')));shortlist=[x for x in coarse if x['iteration'] in wanted]
        assert {x['iteration'] for x in shortlist}==wanted
    cfg,train=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg);args.num_envs=per*(len(shortlist) if own.batch_final else 1)
    cfg.env.test=False;cfg.noise.add_noise=True;cfg.push.enabled=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPDirectionRunner(env,class_to_dict(train),None,device=args.rl_device)
    env._push_strength=lambda:1.
    finalists=[]
    if own.batch_final:
        policies=[]
        for item in shortlist:
            saved=torch.load(item['checkpoint'],map_location=runner.device)
            assert saved['contract']==runner.contract
            runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
            policies.append(ExpandedStudentPolicy(runner.alg.actor_critic).to(runner.device).eval().requires_grad_(False))
        batch=PolicyBatch(policies,per)
        finalists=[dict(checkpoint=x['checkpoint'],iteration=x['iteration'],reports=[],vectorization_max_error=batch.parity_error) for x in shortlist]
        for seed in [1,2,3]:
            report,arrays=run(env,batch,cases*len(shortlist),own.final_seconds,seed,'localization_sim',env.path.gains,stop_s=5.)
            for i,record in enumerate(finalists):
                part={k:v for k,v in report.items() if k!='results'}
                part['results']=report['results'][i*count*2:(i+1)*count*2]
                annotate_stops(part['results'],arrays,i*count,count,own.final_seconds,env.dt)
                record['reports'].append(part);record['summary']=summarize(record['reports'])
                np.savez_compressed(str(own.output/('model_%d_seed%d.npz'%(record['iteration'],seed))),
                    **{k:(v[:,i*per:(i+1)*per] if v.ndim>=2 else v) for k,v in arrays.items()})
                (own.output/('candidate_%d.json'%record['iteration'])).write_text(json.dumps(record,indent=2)+'\n')
            (own.output/'final.json').write_text(json.dumps(finalists,indent=2)+'\n')
            print('DIRECTION BATCH FINAL',seed,[(x['iteration'],x['summary']) for x in sorted(finalists,key=rank)],flush=True)
    for item in ([] if own.batch_final else shortlist):
        runner.load(item['checkpoint'],False)
        policy=runner.alg.actor_critic.act_inference
        reports=[]
        for seed in [1,2,3]:
            report,arrays=run(env,policy,cases,own.final_seconds,seed,'localization_sim',env.path.gains,stop_s=5.)
            annotate_stops(report['results'],arrays,0,count,own.final_seconds,env.dt)
            reports.append(report)
            np.savez_compressed(str(own.output/('model_%d_seed%d.npz'%(item['iteration'],seed))),**arrays)
            partial=dict(checkpoint=item['checkpoint'],iteration=item['iteration'],reports=reports,summary=summarize(reports))
            (own.output/('candidate_%d.json'%item['iteration'])).write_text(json.dumps(partial,indent=2)+'\n')
            print('DIRECTION FINAL',item['iteration'],seed,partial['summary'],flush=True)
        finalists.append(partial);(own.output/'final.json').write_text(json.dumps(finalists,indent=2)+'\n')
    best=min(finalists,key=rank);source=Path(best['checkpoint']);destination=own.output/'best';destination.mkdir(exist_ok=True)
    target=destination/source.name;shutil.copy2(str(source),str(target))
    jit=source.parent/'exported'/('student_%d.pt'%best['iteration'])
    shutil.copy2(str(jit),str(destination/'student.pt'));shutil.copy2(str(jit)+'.json',str(destination/'student.pt.json'))
    result=dict(selected=best,checkpoint=str(target.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        scanned_checkpoints=len(coarse),final_candidates=len(finalists),qualified=best['summary']['full_passed']==best['summary']['groups'],
        snapshot_latest_iteration=max(x['iteration'] for x in coarse),training_left_running=True,
        ranking='unsafe/stop safety failures, gait failures, path failures, nonstatic stops, tracking score',
        protocol=dict(coarse_seconds=own.coarse_seconds,final_seconds=own.final_seconds,seeds=[1,2,3],full_strength_push_N=[5,10],localization='diagnostic nominal truth; robust simulated globally corrected localization'))
    (own.output/'selected.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED DIRECTION',best['iteration'],best['summary'],'qualified',result['qualified'],flush=True)


if __name__=='__main__':main()
