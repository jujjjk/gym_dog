"""Bounded style acquisition check; never substitutes for full gait acceptance."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner
p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=150)
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
if not 1<=own.updates<=200:raise ValueError('Short verification capped at 200 updates')
args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True
if args.num_envs is None:args.num_envs=1024
if args.num_envs<48:raise ValueError('Need at least four environments per reference clip')
env,cfg=task_registry.make_env(args.task,args=args);_,train=task_registry.get_cfgs(args.task)
own.output.mkdir(parents=True,exist_ok=True)
runner=Urdf4AMPCTSRunner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
model=runner.alg.actor_critic;style=runner.style
ids=torch.arange(env.num_envs,device=env.device)%len(env.reference_commands)
commands=env.reference_commands[ids]

def evaluate(label):
 env.set_command_override(commands);obs,_=env.reset();style.clear_history();record={k:[] for k in ['state','clearance','contact','phase_quality','inputs','valid','done']}
 model.eval()
 with torch.no_grad():
  for step in range(400):
   obs,_,_,done,info=env.step(model.act_inference(obs));a=info['amp'];_,inputs,valid=style.rollout_reward(a)
   if step>=150:
    values=[a['next'],env.feet_pos[:,:,2]-.02,a['contact'],a['reference_quality'],inputs,valid,done]
    for key,value in zip(record,values):record[key].append(value[:48].cpu().numpy().copy())
 arrays={k:np.stack(v) for k,v in record.items()};np.savez_compressed(str(own.output/(label+'.npz')),**arrays)
 env.command_override=None;env.reset();style.clear_history();model.train()
 return arrays

before=evaluate('before')
runner.learn(own.updates,init_at_random_ep_len=True)
after=evaluate('after');results={}
manifest=json.loads((Path(cfg.amp_cts.reference)/'manifest.json').read_text())
with torch.no_grad():
 for label,arrays in [('before',before),('after',after)]:
  inputs=torch.tensor(arrays['inputs'].reshape(-1,303),device=env.device)
  scores=torch.cat([style.net(x).flatten() for x in inputs.split(512)])
  style_scores=(1-.25*(scores-1).square()).clamp(0,1).cpu().numpy().reshape(arrays['valid'].shape)
  rows=[]
  for i,clip in enumerate(manifest['clips']):
   mask=(np.arange(48)%12==i)[None,:]&arrays['valid'];x=arrays['state'][mask];c=arrays['contact'][mask];n=c.sum(1)
   diag=(c[:,0]&c[:,3])|(c[:,1]&c[:,2]);cmd=np.array(clip['command'])
   rows.append(dict(clip=clip['clip'],command=clip['command'],valid_samples=len(x),
                    reference_foot_quality=float(arrays['phase_quality'][mask].mean()) if len(x) else 0,
                    AMP_style_same_final_discriminator=float(style_scores[mask].mean()) if len(x) else 0,
                    mean_velocity=x[:,[36,37,41]].mean(0).tolist() if len(x) else [],
                    velocity_error=float(np.linalg.norm(x[:,36:38]-cmd[:2],axis=1).mean()) if len(x) else 0,
                    yaw_rate_error=float(np.abs(x[:,41]-cmd[2]).mean()) if len(x) else 0,
                    clearance_p95_mm=(np.percentile(arrays['clearance'][mask],95,axis=0)*1000).tolist() if len(x) else [],
                    flight_fraction=float((n==0).mean()) if len(x) else 0,
                    illegal_support_fraction=float(((n==0)|(n==1)|((n==2)&~diag)).mean()) if len(x) else 0,
                    resets=int(arrays['done'][:,np.arange(48)%12==i].sum())))
  results[label]=rows
summary=dict(task=args.task,revision=cfg.amp_cts.reward_revision,updates=own.updates,seed=train.seed,
             num_envs=env.num_envs,evaluation_seconds=8.,warmup_seconds=3.,eval_envs_per_clip=4,
             checkpoint=str(own.output/'run'/('model_%d.pt'%own.updates)),results=results,
             finite=all(np.isfinite(x).all() for group in (before,after) for x in group.values()),
             long_training_started=False,gait_acceptance_proven=False)
summary['short_learning_signal_passed']=all(b['reference_foot_quality']>a['reference_foot_quality']
                                          for a,b in zip(results['before'],results['after']))
summary['speed_expansion_approved']=False  # 8 s checks do not replace 30-60 s acceptance.
(own.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SHORT STYLE CHECK COMPLETE',flush=True)
