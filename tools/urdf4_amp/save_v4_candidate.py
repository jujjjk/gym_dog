"""Re-export evaluated weights with the current runtime and explicit provenance."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_v4 import Urdf4AMPV4Runner


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_v4';args.num_envs=16;args.headless=True
    saved=torch.load(own.model,map_location='cpu');cfg,train=task_registry.get_cfgs(args.task);cfg.env.test=True
    for key in ('balance_archive','balance_variant'):setattr(cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
    cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner=Urdf4AMPV4Runner(env,class_to_dict(train),None,device=args.rl_device);runner.load(own.model)
    own.output.mkdir(parents=True,exist_ok=False);selected=own.output/('model_%d.pt'%runner.current_learning_iteration)
    runner.save(str(selected));exported=torch.load(selected,map_location='cpu')
    assert all(torch.equal(v,saved['model_state_dict'][k]) for k,v in exported['model_state_dict'].items())
    receipt=dict(source_checkpoint=str(own.model.resolve()),source_sha256=hashlib.sha256(own.model.read_bytes()).hexdigest(),
        checkpoint=str(selected.resolve()),checkpoint_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),weights_identical=True,
        runtime_source_sha256=runner.experiment['source_sha256'],approved_for_training=False,deployment_accepted=False,
        note='Candidate export only. Independent evaluations must establish accepted commands and robustness.')
    (own.output/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
