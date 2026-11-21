"""Validate TRAINING readiness, separately from final omni acceptance.

The final 5/3 cm and 2 degree path gate remains in evaluate_path_tracking.py.
This receipt permits learning a policy that still has measured direction errors;
it never labels that policy as a qualified deployment model.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_direction import Urdf4AMPDirectionRunner


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--behavior',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--checkpoint_path',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    data=json.loads(own.behavior.read_text());baseline=json.loads(own.baseline.read_text())
    assert data['frozen_weights_unchanged'] and len(data['reports'])==3
    assert {r['seed'] for r in data['reports']}=={1,2,3}
    assert all(r['seconds']>=30 and r['stop_s']>=5 for r in data['reports'])
    rows=[r for report in data['reports'] for r in report['results'] if r['name'].startswith('path_')]
    off=[r for report in data['reports'] for r in report['results'] if r['name'].startswith('off_')]
    nominal=[r for r in rows if r['kind']=='nominal'];robust=[r for r in rows if r['kind']=='robust']
    assert len(nominal)==33 and len(robust)==33
    # Preserve strict nominal diagonal/support/torque gates before a long run.
    assert all(r['gait_passed'] for r in nominal),[(r['name'],r['rejection_reasons']) for r in nominal if not r['gait_passed']]
    assert all(r['resets']==0 and r['flight_fraction']==0 and r['raw_peak_Nm']<=17
        and r['illegal_support_fraction']<=.01 for r in robust)
    # Readiness thresholds are intentionally named here; they are NOT final
    # path acceptance. Expose every final failure in the receipt and report.
    assert max(r['cross_max_m'] for r in nominal)<=.08
    assert max(r['heading_cycle_mean_abs_deg'] for r in nominal)<=5.
    cross_ratio=np.mean([r['cross_rms_m'] for r in rows])/np.mean([r['cross_rms_m'] for r in off])
    heading_ratio=np.mean([r['heading_cycle_mean_abs_deg'] for r in rows])/np.mean([r['heading_cycle_mean_abs_deg'] for r in off])
    assert cross_ratio<.20 and heading_ratio<.20,(cross_ratio,heading_ratio)
    assert all(r['stop']['resets']==0 for r in rows)
    policy=own.checkpoint_path.parent/'exported'/('student_%d.pt'%int(own.checkpoint_path.stem.split('_')[-1]))
    learning_path=own.checkpoint_path.parent/'learning_checks.json'
    if not learning_path.exists():
        initial=torch.load(str(own.checkpoint_path.parent/'model_0.pt'),map_location='cpu')['model_state_dict']
        trained=torch.load(str(own.checkpoint_path),map_location='cpu')['model_state_dict']
        changed=any(not torch.equal(initial[k],v) for k,v in trained.items() if k.startswith('actor.'))
        metrics=json.loads((own.checkpoint_path.parent/'metrics.jsonl').read_text().splitlines()[-1])
        learning_path.write_text(json.dumps(dict(actor_updated=changed,iterations=metrics['iteration'],
            amp_valid_fraction=metrics['amp_valid_fraction'],amp_bonus_step=metrics['amp_bonus_step'],metrics=metrics),indent=2)+'\n')
    learning=json.loads(learning_path.read_text())
    assert learning['actor_updated'] and learning['amp_valid_fraction']>.5 and learning['amp_bonus_step']>0
    assert hashlib.sha256(policy.read_bytes()).hexdigest()==data['policy_sha256']
    args=get_args();args.task='urdf4_amp_cts_direction_v1';args.headless=True;args.num_envs=64
    cfg,train=task_registry.get_cfgs(args.task)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPDirectionRunner(env,class_to_dict(train),None,device=args.rl_device)
    runner.load(str(own.checkpoint_path),True)
    with torch.no_grad():
        exported=torch.jit.load(str(policy),map_location=env.device)
        error=float((exported(env.obs_buf[:,:1020])-runner.alg.actor_critic.act_inference(env.obs_buf)).abs().max())
        assert error<1.e-6,error
    failures=[dict(seed=report['seed'],**r) for report in data['reports'] for r in report['results']
        if r['name'].startswith('path_') and not(r['gait_passed'] and r['path_passed'])]
    summary=dict(approved_for_training=True,final_omni_accepted=not failures,
        checkpoint=str(own.checkpoint_path.resolve()),checkpoint_sha256=hashlib.sha256(own.checkpoint_path.read_bytes()).hexdigest(),
        contract=runner.contract,source_sha256=runner.experiment['source_sha256'],
        tested_behavior=str(own.behavior.resolve()),baseline=str(own.baseline.resolve()),
        seeds=[1,2,3],seconds=30,stop_s=5,nominal_gait_passed=sum(r['gait_passed'] for r in nominal),
        robust_gait_passed=sum(r['gait_passed'] for r in robust),groups=len(rows),
        cross_rms_ratio_to_off=float(cross_ratio),heading_ratio_to_off=float(heading_ratio),
        export_reload_error=error,remaining_final_gate_failures=failures,
        amp_learning_checks=learning,
        training_gate='nominal gait strict; robust no fall/flight/raw>17; nominal cross<=8cm, heading<=5deg; both error ratios<20%; final gates unchanged')
    own.receipt.parent.mkdir(parents=True,exist_ok=True)
    own.receipt.write_text(json.dumps(summary,indent=2)+'\n')
    print('DIRECTION TRAINING PREFLIGHT PASSED',json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','remaining_final_gate_failures')}),flush=True)


if __name__=='__main__':main()
