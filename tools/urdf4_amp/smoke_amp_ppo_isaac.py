"""One bounded AMP+PPO integration update, not a training run or gait claim."""
import sys,json,argparse,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import play_urdf4_omni as play
import torch
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.urdf4.amp_reference import Urdf4AMPReference,amp_features,valid_transition
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser(add_help=False);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);own,remaining=p.parse_known_args();sys.argv=sys.argv[:1]+remaining;args=get_args();args.headless=True;args.task=play.TASK;env,cfg=play.make_env(args,type('Flags',(),{'disturbed':False})(),64)
m=json.loads((own.dataset/'manifest.json').read_text());commands=torch.tensor([c['command'] for c in m['clips']],device=env.device).repeat(6,1)[:64];env.set_command_override(commands)
# No existing checkpoint is loaded; the old 5400 policy remains untouched.
train=class_to_dict(cfg);train['runner']['num_steps_per_env']=24;train['algorithm']['num_learning_epochs']=1;train['algorithm']['num_mini_batches']=2;runner=OnPolicyRunner(env,train,log_dir=None,device=env.device)
style=Urdf4AMPReference(own.dataset,env,validation_only=True,settings={'amp_disc_steps':1})
guard=False
try:Urdf4AMPReference(own.dataset,env)
except ValueError:guard=True
with tempfile.TemporaryDirectory() as tmp:
 bad=dict(m);bad['urdf_sha256']='0'*64;(Path(tmp)/'manifest.json').write_text(json.dumps(bad));wrong_urdf_guard=False
 try:Urdf4AMPReference(tmp,env,validation_only=True)
 except ValueError:wrong_urdf_guard=True
terminal={};original=env.compute_reward

def reward_hook():
 original();terminal['features']=amp_features(env).clone();terminal['command']=env.commands[:,:3].clone()
env.compute_reward=reward_hook;obs,critic=env.reset();valid_count=0;bonus_sum=0.
with torch.no_grad():
 for _ in range(24):
  prev=amp_features(env).clone();command=env.commands[:,:3].clone();previous_reset=env.reset_buf.bool().clone();actions=runner.alg.act(obs,critic);obs,critic,rewards,dones,infos=env.step(actions);nxt=terminal['features'];valid=valid_transition(previous_reset,dones,command,terminal['command']);score,inputs=style.reward(prev,nxt,command);style.add(inputs[valid]);bonus=.25*env.dt*score*valid.float();runner.alg.process_env_step(rewards+bonus,dones,infos);valid_count+=int(valid.sum());bonus_sum+=float(bonus.mean())
 runner.alg.compute_returns(critic)
losses=runner.alg.update();style.update();vals=[float(x) for x in losses]
r={'task':play.TASK,'num_envs':64,'policy_steps_per_env':24,'PPO_updates':1,'discriminator_updates':1,'policy_loaded':False,'existing_checkpoints_modified':False,'validated_transition_count':valid_count,'ppo_losses':vals,'amp_metrics':style.metrics,'amp_bonus_mean_per_step':bonus_sum/24,'finite':bool(torch.isfinite(torch.tensor(vals+list(style.metrics.values()))).all()),'unapproved_reference_guard':guard,'wrong_urdf_guard':wrong_urdf_guard,'numerical_smoke_only':True,'gait_learning_proven':False,'production_amp_task_registered':False}
own.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
