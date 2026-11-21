"""Bounded expansion checks: real PPO, sensor boundary, RSI and curriculum."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.algorithms.urdf4_amp_expand import Urdf4AMPExpandRunner
from legged_gym.envs.urdf4.disturbances import quat_mul,quat_from_rpy

p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=3)
p.add_argument('--x_only',action='store_true')
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
assert 1<=own.updates<=5
args=get_args();args.task='urdf4_amp_cts_expand_x_v1' if own.x_only else 'urdf4_amp_cts_expand_v1';args.headless=True
if args.num_envs is None:args.num_envs=128
assert args.num_envs>=64
cfg,train=task_registry.get_cfgs(args.task)
if own.x_only:
    base,_=task_registry.get_cfgs('urdf4_amp_cts_expand_v1')
    for group in ['domain_rand','push','imu','joint_sensor','noise','rs01_actuator','control','rewards']:
        assert class_to_dict(getattr(cfg,group))==class_to_dict(getattr(base,group)),group
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
own.output.mkdir(parents=True,exist_ok=True)
runner=Urdf4AMPExpandRunner(env,class_to_dict(train),str(own.output/'run'),device=args.rl_device)
seed=ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000/model_17800.pt'
runner.load(str(seed),False);model=runner.alg.actor_critic
ids=torch.arange(env.num_envs,device=env.device);nominal=env.nominal_mask
assert env.obs_buf.shape==(env.num_envs,1121)
assert env.cfg.rewards.scales.command_tracking>0 and env.cfg.commands.reference_probability<1
assert not cfg.domain_rand.randomize_gains and not cfg.domain_rand.randomize_armature
assert torch.all(env.response_gain[nominal]==cfg.rs01_actuator.response_gain)
assert torch.all(env.time_constant[nominal]==cfg.rs01_actuator.time_constant_s)
assert torch.all(env.delay_steps[nominal]==env.nominal_delay_steps)
assert torch.all(env.encoder_offset[nominal]==0) and bool(env.imu.clean[nominal].all())
assert torch.all(env.payload_kg[nominal]==0) and torch.all(env.com_offset_m[nominal]==0)
assert torch.all(env.friction_coeff[nominal]==.8)
assert float(env.response_gain[~nominal].std())>0 and env.delay_steps.min()>=7 and env.delay_steps.max()<=9
with torch.no_grad():
    public=env.obs_buf[:,:1020].clone();legacy=torch.jit.load(str(seed.parent/'exported/student_17800.pt'),map_location=env.device)
    warmstart_error=float((legacy(public)-model.act_inference(public)).abs().max());assert warmstart_error<1.e-6
    altered=env.obs_buf.clone();altered[:,1020:]=torch.randn_like(altered[:,1020:])*3
    assert torch.equal(model.act_inference(altered),model.act_inference(env.obs_buf))
    probe=public.clone();probe[:,6]=.8
    seen=[];hook=model.student_encoder.register_forward_pre_hook(lambda module,inputs:seen.append(inputs[0].clone()))
    model.act_inference(probe);hook.remove()
    assert torch.equal(seen[0],probe),'Expanded target was hidden from the history encoder'
    assert torch.all(model.control_frame(probe)[:,6]<=1.000001)

# Public heading follows the sensor, while the teacher may read truth.
root=env.root_states.clone();heading=env.heading_error.clone();window=env.measured_yaw_error.window.clone();head=env.measured_yaw_error.head
torch.manual_seed(77);env.compute_observations();before=env.obs_buf[:,:51].clone()
env.measured_yaw_error.window.copy_(window);env.measured_yaw_error.head=head
zeros=torch.zeros(env.num_envs,device=env.device)
env.root_states[:,3:7]=quat_mul(root[:,3:7],quat_from_rpy(zeros,zeros,zeros+.3));env.heading_error+=.3
torch.manual_seed(77);env.compute_observations();assert torch.equal(before,env.obs_buf[:,:51]),'Simulator heading leaked into public frame'
env.root_states.copy_(root);env.heading_error.copy_(heading)

# Full RSI only in nominal environments with exactly covered commands.
command=torch.tensor([.28,0,0],device=env.device).repeat(env.num_envs,1)
env.reference_reset_probability=1.;env.set_command_override(command);env.reset()
assert bool((env.last_reference_bank_ids[nominal]>=0).all()) and bool((env.last_reference_bank_ids[~nominal]==-1).all())
bank=env.reference_bank['snapshots'];selected=env.last_reference_bank_ids[nominal]
assert torch.equal(env.delay_buffer[:,nominal].transpose(0,1),bank['delay_buffer'][selected])
assert torch.equal(env.obs_history[nominal],bank['obs_history'][selected])
command[:,0]=.4;env.set_command_override(command);env.reset();assert bool((env.last_reference_bank_ids==-1).all())
env.cold_start[:]=False;before_command=env.commands[:,:3].clone()
with torch.no_grad():env.step(model.act_inference(env.obs_buf))
delta=(env.commands[:,:3]-before_command).abs()
assert bool((delta<=env.commands.new_tensor(cfg.expand.command_acceleration)*env.dt+1.e-6).all()) and bool((delta[:,0]>0).all())
env.reference_reset_probability=cfg.amp_cts.reference_reset_probability

commands=torch.tensor([[.28,0,0],[.4,0,0],[.6,0,0]],device=env.device)
coverage=runner.style.coverage(commands)
assert abs(float(coverage[0])-1)<1.e-6 and 0<float(coverage[1])<1 and float(coverage[2])==0
weight,replay=runner._amp_weights(dict(command=commands,style_factor=torch.tensor([1.,0.,1.],device=env.device)),torch.ones(3,device=env.device),torch.ones(3,device=env.device,dtype=torch.bool))
assert torch.equal(replay,torch.tensor([True,False,False],device=env.device)) and float(weight[1:].sum())==0

# Real force insertion changes no velocity instantaneously and opens grace.
env.reference_reset_probability=0.;env.reset();env.push_wait[:]=1.e9;env.push_wait[1]=0.
oldroot=env.root_states.clone();env._apply_push()
assert torch.equal(oldroot,env.root_states) and env.push_count[1]==1 and env.recovery_left[1]>0
assert env.recovery_factor()[1]==0
env.recovery_left[1]=cfg.expand.recovery_grace_s;assert env.recovery_factor()[1]==0
env.recovery_left[1]=0;assert env.recovery_factor()[1]==1

# Performance, not elapsed training time, advances the speed curriculum.
env.curriculum_stage=0
for iteration in [200,250]:
    env.curriculum_meter[:]=0;env.curriculum_meter[:,4]=400
    env.curriculum_meter[:,0]=400*.02;env.curriculum_meter[:,1]=400*.05
    env.curriculum_meter[:,5]=400*.15;env.curriculum_meter[:,6:]=400*.2
    runner.current_learning_iteration=iteration;runner._after_update({})
assert env.curriculum_stage==1
env.curriculum_meter[:]=0;runner.current_learning_iteration=300;runner._after_update({})
assert env.curriculum_stage==1 and env.curriculum_pass_windows==0
# Velocity alone cannot promote a policy with no alternating diagonal support.
env.curriculum_stage=0;env.curriculum_meter[:]=0;env.curriculum_meter[:,4]=400
runner.current_learning_iteration=350;runner._after_update({});assert env.curriculum_stage==0
env.curriculum_stage=1
env.command_override=None;env._resample_commands(ids)
assert len(torch.unique(env.target_commands,dim=0))>20
assert bool((env.target_commands[:,0]>.28).any()|(env.target_commands[:,1].abs()>.1).any()|(env.target_commands[:,2].abs()>.4).any())
if own.x_only:
    assert runner.alg.reference_action_hinge_coef==1000.
    for stage in range(3):
        env.curriculum_stage=stage
        for _ in range(16):
            env._resample_commands(ids);sample=env.target_commands
            assert sample[:,0].min()>=-.180001 and sample[:,0].max()<=cfg.expand.stages[stage][1]+1.e-6
            assert sample[:,1].abs().max()<=.100001 and sample[:,2].abs().max()<=.400001
            expanded=sample[:,0]>.280001
            assert bool((sample[expanded,1:]==0).all()),'High-speed mixed command escaped the old bounds'
            nearest=(sample[~expanded,None]-env.reference_commands[None]).abs().amax(2).min(1).values
            assert bool((nearest<2.e-6).all()),'Old commands were replaced with new mixtures'
    assert bool((sample[:,0]>.4).any()),'The 0.6 ceiling was not sampled'
env.curriculum_stage=min(getattr(cfg.expand,'initial_stage',0),cfg.expand.max_stage)
env.curriculum_pass_windows=0;env.curriculum_meter.zero_();runner.current_learning_iteration=0
env.reference_reset_probability=cfg.amp_cts.reference_reset_probability;obs,_=env.reset()
with torch.no_grad():
    _,_,_,first_done,_=env.step(model.act_inference(obs))
assert not bool(first_done.any()) and float(env.root_states[:,2].min())>.20,'Mixed RSI/cold reset was not committed for every actor'
runner.save(str(own.output/'run/model_0.pt'))

errors=[];same_batch_errors=[];original=runner.alg.update
def checked_update():
    storage=runner.alg.storage;flat=storage.observations.flatten(0,1)
    with torch.no_grad():
        model.update_distribution(flat);lp=model.get_actions_log_prob(storage.actions.flatten(0,1))
        error=float((lp-storage.actions_log_prob.flatten()).abs().max());errors.append(error)
        chunks=[]
        for k in range(storage.observations.shape[0]):
            model.update_distribution(storage.observations[k])
            chunks.append(model.get_actions_log_prob(storage.actions[k]))
        exact=float((torch.stack(chunks).flatten()-storage.actions_log_prob.flatten()).abs().max())
        assert exact<2.e-5,exact;same_batch_errors.append(exact)
        # Flattening changes GEMM batch shapes for pretrained float32 networks.
        # Keep its numerical ratio error below 1% of the actual PPO clip width.
        assert error<runner.alg.clip_param*.01,error
    old=[p.clone() for p in model.student_encoder.parameters()]
    result=original()
    assert all(torch.equal(p,q) for p,q in zip(model.student_encoder.parameters(),old))
    return result
runner.alg.update=checked_update
before_encoder=[p.clone() for p in model.student_encoder.parameters()]
runner.learn(own.updates,init_at_random_ep_len=True)
assert any(not torch.equal(p,q) for p,q in zip(model.student_encoder.parameters(),before_encoder))
checkpoint=own.output/('run/model_%d.pt'%own.updates)
with torch.no_grad():
    public=env.obs_buf[:,:1020].clone();expected=model.act_inference(public)
    exported=torch.jit.load(str(own.output/('run/exported/student_%d.pt'%own.updates)),map_location=env.device)
    export_error=float((expected-exported(public)).abs().max());assert export_error<1.e-6
    runner.load(str(checkpoint));reload_error=float((expected-model.act_inference(public)).abs().max());assert reload_error==0
result=dict(task=args.task,envs=env.num_envs,updates=own.updates,finite=True,warmstart_max_error=warmstart_error,
    public_size=1020,packed_size=1121,private_weight_not_visible_to_student=True,student_heading_has_no_truth_leak=True,
    nominal_randomization_excluded=True,RSI_only_nominal_exact_reference=True,mixed_reset_first_step_no_fall=True,off_support_AMP_and_replay_disabled=True,
    push_changes_forces_not_velocities=True,recovery_grace_verified=True,performance_curriculum_verified=True,
    continuous_and_expanded_commands_verified=True,PPO_logprob_max_errors=errors,PPO_same_batch_logprob_errors=same_batch_errors,PPO_student_encoder_frozen=True,
    encoder_supervision_updates=True,export_max_error=export_error,reload_max_error=reload_error,
    checkpoint=str(checkpoint),metrics=runner.last_metrics,behavior_validated=False,long_training_started=False)
(own.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('EXPAND SMOKE PASSED',json.dumps(result),flush=True)
