"""Select a behavior-checked seed and bind a manual long-training launch to it.

Training readiness is distinct from hardware or full robustness acceptance.
Wide-speed frontier failures are retained in the receipt, never hidden.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_v3_training'
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v3 import Urdf4AMPV3Runner


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--nominal',type=Path,required=True)
    p.add_argument('--robust',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    nominal=json.loads(own.nominal.read_text());robust=json.loads(own.robust.read_text());smoke=json.loads(own.smoke.read_text())
    original_sha=hashlib.sha256(own.model.read_bytes()).hexdigest()
    for r in [nominal,robust]:
        assert r['checkpoint_sha256']==original_sha and r['pure_student'] and not r['teacher_controller_used']
        assert not r['automatic_resets'] and not r['root_or_joint_overwrite_during_rollout']
        assert r['seconds']>=60 and r['copies']>=3
    assert nominal['core_passed'] and not nominal['randomized'] and robust['randomized']
    core=[c for c in robust['cases'] if not c['frontier']]
    # Initial robustness safety only. Timing/direction misses remain explicit
    # training targets and do not count as completed deployment acceptance.
    assert robust['push_count']>0
    assert all(c['fallen_count']==0 and c['raw_peak_Nm']<=17 and c['flight_fraction']==0
               and c['illegal_support_fraction']<=.01 and 'stop' not in c['rejection_reasons'] for c in core)
    assert smoke['passed'] and smoke['actor_updated'] and smoke['student_private_input_independent']
    assert smoke['metrics']['amp_valid_fraction']>.5 and smoke['metrics']['reset_fraction']==0
    checks=json.loads((BASE/'contract_checks.json').read_text());assert checks['passed']
    args=get_args();args.task='urdf4_amp_cts_v3';args.num_envs=16;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV3Runner(env,class_to_dict(train),None,device=args.rl_device)
    assert nominal['contract']==runner.contract==robust['contract']
    assert nominal['source_sha256']==runner.experiment['source_sha256']==robust['source_sha256']
    smoke_experiment=json.loads((own.smoke.parent/'experiment_contract.json').read_text())
    assert smoke_experiment['source_sha256']==runner.experiment['source_sha256']
    runner.load(own.model)
    output=BASE/'selected';output.mkdir(exist_ok=False)
    selected=output/('model_%d.pt'%runner.current_learning_iteration)
    runner.save(str(selected))
    saved=torch.load(selected,map_location='cpu');original=torch.load(own.model,map_location='cpu')
    assert all(torch.equal(value,original['model_state_dict'][key]) for key,value in saved['model_state_dict'].items())
    receipt=dict(approved_for_training=True,deployment_accepted=False,
        scope='New-mass V3 all-direction initialization and finite PPO readiness; wide-speed/robust mastery remains to be trained',
        checkpoint=str(selected),checkpoint_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        evaluated_checkpoint=str(own.model),evaluated_checkpoint_sha256=original_sha,
        selected_weights_identical_to_evaluated=True,contract=runner.contract,source_sha256=runner.experiment['source_sha256'],
        nominal_result=str(own.nominal),robust_result=str(own.robust),smoke_result=str(own.smoke),
        nominal_core_passed=True,robust_core_passed=robust['core_passed'],
        remaining_robust_cases=[dict(motion=c['motion'],reasons=c['rejection_reasons']) for c in core if not c['passed']],
        frontier_results=[dict(motion=c['motion'],passed=c['passed'],reasons=c['rejection_reasons']) for c in robust['cases'] if c['frontier']],
        long_training_started=False)
    (BASE/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['approved_for_training','checkpoint','nominal_core_passed','robust_core_passed','long_training_started']}),flush=True)


if __name__=='__main__':main()
