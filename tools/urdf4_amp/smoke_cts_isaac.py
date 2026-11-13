"""Bounded full-task CTS/AMP smoke, including PPO branch and export contracts."""
import argparse
import json
import sys
import numpy as np
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym  # must precede torch
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_cts import Urdf4AMPCTSRunner
from legged_gym.envs.urdf4.amp_reference import valid_transition,Urdf4AMPReference
from legged_gym.envs.urdf4.amp_reference import amp_features

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--updates',type=int,default=3)
own,remaining=p.parse_known_args();sys.argv=sys.argv[:1]+remaining
if not 1<=own.updates<=5:raise ValueError('Smoke limited to 1-5 updates')
args=get_args();args.task='urdf4_amp_cts_v1';args.headless=True
if args.num_envs is None:args.num_envs=128
if args.num_envs<4:raise ValueError('Need both CTS roles')
env,cfg=task_registry.make_env(args.task,args=args)
_,train=task_registry.get_cfgs(args.task)
train=class_to_dict(train)
train['runner']['save_interval']=100
# Use the real four-epoch PPO and discriminator configuration.
own.output.mkdir(parents=True,exist_ok=True)
runner=Urdf4AMPCTSRunner(env,train,str(own.output/'run'),device=args.rl_device)
with tempfile.TemporaryDirectory() as temp:
    receipt=json.loads(Path(cfg.amp_cts.receipt).read_text());receipt['dataset_manifest_sha256']='0'*64
    receipt_path=Path(temp)/'invalid_receipt.json';receipt_path.write_text(json.dumps(receipt))
    try:Urdf4AMPReference(cfg.amp_cts.reference,env,validation_receipt=receipt_path)
    except ValueError:pass
    else:raise AssertionError('Mismatched validation receipt was accepted')
model=runner.alg.actor_critic
obs=env.get_observations().clone()
assert obs.shape==(env.num_envs,1120)
assert torch.isfinite(obs).all()
assert int(env.teacher_mask.sum())==int(env.num_envs*.75)
assert not cfg.domain_rand.randomize_rs01_actuator and not cfg.noise.add_noise and not cfg.push.enabled

with torch.no_grad():
    student=model.act_inference(obs)
    changed=obs.clone();changed[:,1020:]=torch.randn_like(changed[:,1020:])*5
    assert torch.equal(student,model.act_inference(changed)),'Student leaked privileged information'
    teacher=obs.clone();teacher[:,-1]=1
    model.update_distribution(teacher);before=model.action_mean.clone()
    teacher[:,1020:1119]+=.3
    model.update_distribution(teacher)
    assert (before-model.action_mean).abs().max()>1e-5,'Teacher ignores privileged state'

# Differential student estimation on student states must not backpropagate into teacher.
model.zero_grad(set_to_none=True)
loss,_,_=model.student_loss(obs);loss.backward()
assert all(p.grad is None for p in model.teacher_encoder.parameters())
assert all(p.grad is None for p in model.actor.parameters())
assert any(p.grad is not None for p in model.student_encoder.parameters())
model.zero_grad(set_to_none=True)

# Style acquisition samples all V2 clips, plus true standing.
samples=[];modes=[]
for _ in range(40):
    env._resample_commands(torch.arange(env.num_envs,device=env.device))
    samples.append(env.commands[:,:3].clone());modes.append(env.gait_enabled.clone())
samples=torch.cat(samples);modes=torch.cat(modes)
assert torch.cdist(samples,env.reference_commands).min(1).values.max()<1e-6
assert torch.cdist(env.reference_commands,samples).min(1).values.max()<1e-6
assert samples[:,0].max()<=.280001 and samples[:,0].min()>=-.180001
assert samples[:,1].abs().max()<=.100001 and samples[:,2].abs().max()<=.400001
assert set(env.reward_scales)=={'command_tracking','phase_reference','illegal_support','raw_torque_over_peak','collision','action_rate','termination'}
zero=samples.abs().sum(1)<1e-6
assert bool((zero&~modes).any()) and bool((zero&modes).any())
# All original expert commands keep their per-clip duty, while fast commands
# get swing return time without changing the RS01 target rate/torque limits.
for i,command in enumerate(env.reference_commands):
    env.set_command_override(command.repeat(env.num_envs,1))
    assert torch.allclose(env._stance_ratio(),env.reference_duties[i].expand(env.num_envs))
env.set_command_override(torch.tensor([.6,0.,0.],device=env.device).repeat(env.num_envs,1))
assert bool((env._stance_ratio()<.56).all()) and bool((env._stance_ratio()>.5).all())
# Positive and negative commanded heading targets, wrap and reset semantics.
ids=torch.arange(env.num_envs,device=env.device)
env.set_command_override(torch.tensor([0.,0.,.8],device=env.device).repeat(env.num_envs,1))
env.heading_target[:]=0
with torch.no_grad():env._post_physics_step_callback()
assert torch.allclose(env.heading_target,torch.full_like(env.heading_target,.8*env.dt),atol=1e-6)
env.set_command_override(torch.tensor([0.,0.,-.8],device=env.device).repeat(env.num_envs,1))
with torch.no_grad():env._post_physics_step_callback()
assert torch.allclose(env.heading_target,torch.zeros_like(env.heading_target),atol=1e-6)
env.command_override=None
env.reset_idx(ids);env.compute_observations()
assert torch.equal(env.heading_error,torch.zeros_like(env.heading_error))
env.set_command_override(torch.zeros(env.num_envs,3,device=env.device),gait_enabled=False)
env.reset_idx(ids)
assert not bool(env.gait_enabled.any()),'Standing override changed to march on reset'
env.command_override=None
env.reset_idx(ids);env.compute_observations()
# Explicitly reject reset transitions and command changes.
x=torch.zeros(4,3,device=env.device);y=x.clone();y[2,0]=.2
mask=valid_transition(torch.tensor([1,0,0,0],device=env.device),torch.tensor([0,1,0,0],device=env.device),x,y)
assert mask.tolist()==[False,False,False,True]
# Exact reference geometry is the unique maximum of the guidance term;
# lifting or lowering the front pair by 1 cm must both reduce it.
env.gait_enabled[:]=True
env.amp_terminal=amp_features(env).clone()
env.amp_terminal[:,24:36]=env.reference_foot_target()
assert torch.allclose(env._reward_phase_reference(),torch.ones(env.num_envs,device=env.device))
for sign in (-1,1):
    env.amp_terminal[:,24:36]=env.reference_foot_target()
    env.amp_terminal[:,[26,29]]+=sign*.01
    assert bool((env._reward_phase_reference()<1).all())
# Heading correction and yaw tracking share one target: restoring yaw must
# score better than yawing away from that target.
env.commands[:,:3]=0;env.base_lin_vel[:,:2]=0;env.heading_error[:]=.1;env.velocity_drift[:,:2]=0
expected_yaw=cfg.rewards.heading_gain*.1
env.velocity_drift[:,2]=-expected_yaw
correct=env._reward_command_tracking().clone()
env.velocity_drift[:,2]=expected_yaw
assert bool((correct>env._reward_command_tracking()).all())
# Temporal discriminator: complete-cycle history, with every intermediate
# transition checked; a reset/command boundary invalidates the next .5 s.
style=runner.style;style.clear_history()
fake={'next':torch.zeros(env.num_envs,46,device=env.device),
      'contact':torch.zeros(env.num_envs,4,device=env.device),
      'valid':torch.ones(env.num_envs,device=env.device,dtype=torch.bool),
      'command':torch.zeros(env.num_envs,3,device=env.device)}
for i in range(style.span+1):
    fake['next'][:]=i
    _,inputs,valid=style.rollout_reward(fake)
    assert bool(valid.all())==(i==style.span)
expected=torch.cat([torch.cat((torch.full((env.num_envs,46),float(i),device=env.device),fake['contact']),1)
                    for i in range(0,style.span+1,style.stride)],1)
assert inputs.shape[1]==303 and torch.equal(inputs,style.inputs(expected,fake['command']))
fake['valid'][:]=False
assert not bool(style.rollout_reward(fake)[2].any())
fake['valid'][:]=True
for i in range(style.span):
    _,_,valid=style.rollout_reward(fake)
    assert bool(valid.all())==(i==style.span-1)
style.clear_history();env.command_override=None;env.reset()

ratios=[];ppo_deltas=[]
original=runner.alg.update

def checked_update():
    storage=runner.alg.storage
    flat=storage.observations.flatten(0,1)
    with torch.no_grad():
        model.update_distribution(flat)
        lp=model.get_actions_log_prob(storage.actions.flatten(0,1))
        delta=(lp-storage.actions_log_prob.flatten()).abs().max().item()
        assert delta<2e-5,delta
        ratios.append(delta)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    result=original()
    delta={prefix:sum(float((p-before[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix))
           for prefix in ('actor.','teacher_encoder.','student_encoder.')}
    assert delta['actor.']>0 and delta['teacher_encoder.']>0
    assert delta['student_encoder.']==0,'PPO updated detached student encoder'
    ppo_deltas.append(delta)
    return result
runner.alg.update=checked_update
before_encoder={n:p.detach().clone() for n,p in model.student_encoder.named_parameters()}
runner.learn(own.updates,init_at_random_ep_len=True)
assert any(not torch.equal(p,before_encoder[n]) for n,p in model.student_encoder.named_parameters())
checkpoint=own.output/'run'/('model_%d.pt'%own.updates)
public=env.get_observations()[:,:1020].clone()
with torch.no_grad():
    expected=model.act_inference(public).clone()
    exported=torch.jit.load(str(checkpoint.parent/'exported'/('student_%d.pt'%own.updates)),map_location=env.device)
    export_error=(exported(public)-expected).abs().max().item()
    assert export_error<1e-6
# Reload all three optimizers and both policy encoders in the same registered runner.
with torch.no_grad():
    for p in model.parameters():p.add_(.1)
runner.load(str(checkpoint))
with torch.no_grad():reload_error=(model.act_inference(public)-expected).abs().max().item()
assert reload_error==0 and runner.current_learning_iteration==own.updates
# A bounded recorder pass using only the student policy, fixed command and seed.
env.set_command_override(torch.tensor([.28,0.,0.],device=env.device).repeat(env.num_envs,1))
obs,_=env.reset();record={k:[] for k in ('state46','q','dq','action','contact','raw_pd_Nm','motor_Nm','applied_Nm','done')}
record_resets=0
with torch.no_grad():
    for _ in range(50):
        action=model.act_inference(obs)
        obs,_,_,done,_=env.step(action)
        values=(amp_features(env),env.dof_pos,env.dof_vel,action,env.get_foot_contact_mask(),
                env.raw_pd_torques,env.motor_torques,env.torques,done[:,None])
        for name,value in zip(record,values):record[name].append(value[0].cpu().numpy().copy())
        record_resets+=int(done[0])
        assert torch.isfinite(obs).all() and env.motor_torques.abs().max()<=14.00001
np.savez_compressed(str(own.output/'student_recorder.npz'),**{k:np.stack(v) for k,v in record.items()},
                    command=np.array([.28,0,0]),dt_s=env.dt,seed=train['seed'],checkpoint=str(checkpoint))
assert torch.isfinite(env.raw_pd_torques).all() and torch.isfinite(env.torques).all()
assert env.motor_torques.abs().max()<=14.00001
result=dict(task=args.task,num_envs=env.num_envs,updates=own.updates,steps_per_update=24,
            seed=train['seed'],checkpoint=str(checkpoint),finite=True,
            public_size=1020,packed_size=1120,teacher_student_ratio='3:1',
            no_privileged_student_leak=True,teacher_uses_privileged_state=True,
            teacher_target_stop_gradient=True,PPO_student_encoder_frozen=True,
            student_encoder_updated_by_supervision=True,PPO_logprob_max_errors=ratios,
            PPO_parameter_changes=ppo_deltas,student_export_max_error=export_error,
            checkpoint_reload_max_error=reload_error,
            command_min=samples.min(0).values.tolist(),command_max=samples.max(0).values.tolist(),
            stand_march_separated=True,command_change_and_reset_transition_guards=True,
            heading_target_integrates_and_resets=True,complete_cycle_and_contact_AMP=True,
            mismatched_validation_receipt_rejected=True,
            phase_reference_reward_has_no_height_conflict=True,unified_heading_and_yaw_target=True,
            original_expert_duty_preserved_and_fast_swing_time_enabled=True,
            final_raw_peak_Nm=float(env.raw_pd_torques.abs().max()),
            final_motor_peak_Nm=float(env.motor_torques.abs().max()),
            peak_torch_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
            student_recorder_seconds=1.,student_recorder_command=[.28,0,0],student_recorder_resets=record_resets,
            metrics=runner.last_metrics,smoke_only=True,learned_gait_validated=False)
(own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('SMOKE PASSED '+json.dumps(result,sort_keys=True),flush=True)
