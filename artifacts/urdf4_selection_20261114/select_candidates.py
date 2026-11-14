"""Read-only gait-first checkpoint selection; no training or model writes."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/urdf4_amp'))
import evaluate_cts_behavior as e

sys.argv=[sys.argv[0],'--headless','--num_envs','48','--sim_device','cuda:0','--rl_device','cuda:0']
args=e.get_args();args.task='urdf4_amp_cts_v1'
cfg,train=e.task_registry.get_cfgs(args.task);cfg.rewards.scales.command_tracking=0.
env,_=e.task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner=e.Urdf4AMPCTSRunner(env,e.class_to_dict(train),None,device=args.rl_device)
run=ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000'
out=Path(__file__).resolve().parent
fixed_style=e.torch.load(str(run/'model_30000.pt'),map_location=env.device)['amp_state']
candidates=[('seed100',ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/trust300/run/model_100.pt')]
candidates += [(str(k),run/('model_%d.pt'%k)) for k in [200,1000,5000,8400,13000,13100,17800,20000,30000]]
selection=[]
for label,checkpoint in candidates:
    runner.load(str(checkpoint),load_optimizer=False)
    runner.style.load_state_dict(fixed_style)
    report,arrays=e.evaluate(env,runner,10.,1,'cold')
    rows=report['results']
    gait_reasons=lambda r:[x for x in r['rejection_reasons'] if x not in ('linear_tracking','yaw_tracking')]
    gait_passes=sum(not gait_reasons(r) for r in rows)
    phase=float(e.np.mean([r['phase_quality'] for r in rows]))
    foot_error=float(e.np.mean([r['clearance_error_mm'] for r in rows]))
    summary=dict(label=label,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        gait_passes=gait_passes,phase_quality=phase,mean_clearance_error_mm=foot_error,
        raw_peak_Nm=max(r['raw_peak_Nm'] for r in rows),resets=sum(r['resets'] for r in rows),
        max_flight_fraction=max(r['flight_fraction'] for r in rows),
        max_illegal_support_fraction=max(r['illegal_support_fraction'] for r in rows),
        fixed_discriminator_style=float(e.np.mean([r['AMP_style'] for r in rows])),
        score=phase-.01*foot_error,report=report)
    selection.append(summary)
    e.np.savez_compressed(str(out/('screen_'+label+'.npz')),**arrays)
    (out/'candidate_comparison.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('CANDIDATE',label,'gait',gait_passes,'phase',round(phase,4),'foot_mm',round(foot_error,2),
          'raw_Nm',round(summary['raw_peak_Nm'],2),flush=True)
ranked=sorted(selection,key=lambda r:(r['gait_passes'],r['score']),reverse=True)
(out/'shortlist.json').write_text(json.dumps(ranked,indent=2)+'\n')
print('RANKING',[(r['label'],r['gait_passes'],round(r['score'],4)) for r in ranked],flush=True)
