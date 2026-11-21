import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np, torch


def play_urdf4(args):
    """Native viewer: leave the camera under mouse control throughout playback."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 1
    env_cfg.env.test = True
    cts = args.task in ("urdf4_amp_cts_v1","urdf4_amp_cts_expand_v1","urdf4_amp_cts_expand_x_v1","urdf4_amp_cts_direction_v1")
    if cts:
        env_cfg.amp_cts.student_only = True
    env_cfg.env.episode_length_s = args.duration_s + 10.0
    env_cfg.noise.add_noise = False
    env_cfg.push.enabled = False
    env_cfg.domain_rand.push_robots = False
    for flag in (
        "randomize_friction", "randomize_restitution", "randomize_base_mass",
        "randomize_base_com", "randomize_link_mass", "randomize_armature",
    ):
        setattr(env_cfg.domain_rand, flag, False)
    commands = (
        ("forward", 0.6, 0.0, 0.0),
        ("backward", -0.4, 0.0, 0.0),
        ("left", 0.0, 0.4 if cts else 0.3, 0.0),
        ("right", 0.0, -0.4 if cts else -0.3, 0.0),
        ("turn left", 0.0, 0.0, 1.2 if cts else 1.0),
        ("turn right", 0.0, 0.0, -1.2 if cts else -1.0),
        ("diagonal", 0.4, 0.2, 0.0),
        ("arc", 0.3, 0.0, 0.6),
        ("mixed", 0.3, -0.2, -0.5),
        ("stand", 0.0, 0.0, 0.0),
    ) if args.sweep else (("fixed", args.vx, args.vy, args.wz),)
    if cts and args.sweep and env_cfg.amp_cts.stage == 'v2_style':
        import json
        from pathlib import Path
        manifest=json.loads((Path(env_cfg.amp_cts.reference)/'manifest.json').read_text())
        commands=tuple((c['clip'],*c['command']) for c in manifest['clips'])+(("stand",0.,0.,0.),)
    env_cfg.commands.fixed_command = list(commands[0][1:])
    if args.task=='urdf4_amp_cts_direction_v1' and args.sweep:
        # Frame the full nominal route once; subsequent mouse motion is free.
        env_cfg.viewer.pos=[3.,-5.,2.5]
        env_cfg.viewer.lookat=[2.5,0.,.25]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    if args.task in ('urdf4_amp_cts_expand_v1','urdf4_amp_cts_expand_x_v1','urdf4_amp_cts_direction_v1') and args.sweep:
        low,high,side,yaw=env_cfg.expand.stages[env.curriculum_stage]
        mixed_high=env_cfg.expand.stages[0][1] if getattr(env_cfg.expand,'x_only',False) else high
        commands=(("forward",high,0.,0.),("backward",low,0.,0.),
            ("left",0.,side,0.),("right",0.,-side,0.),("turn_left",0.,0.,yaw),
            ("turn_right",0.,0.,-yaw),("forward_left",.7*mixed_high,.7*side,0.),
            ("forward_right",.7*mixed_high,-.7*side,0.),("arc_left",.7*mixed_high,0.,.7*yaw),
            ("arc_right",.7*mixed_high,0.,-.7*yaw),("stand",0.,0.,0.))
        if args.task=='urdf4_amp_cts_direction_v1':
            import json
            from pathlib import Path
            manifest=json.loads((Path(env_cfg.amp_cts.reference)/'manifest.json').read_text())
            commands=tuple((c['clip'],*( [high,0.,0.] if c['clip']=='forward' else c['command']))
                for c in manifest['clips'] if c['clip']!='march')+(("stand",0.,0.,0.),)
    if cts:
        env.set_command_override(torch.zeros(env.num_envs,3,device=env.device),gait_enabled=False)
    obs, _ = env.reset()
    last_case = None
    with torch.no_grad():
        if cts:
            # Match behavior evaluation: settle the real plant before the first
            # network action; preserve native mouse/camera interaction.
            for _ in range(round(2./env.dt)):
                obs,_,_,_,_=env.step(torch.zeros(env.num_envs,12,device=env.device))
            if hasattr(env,'path'):
                env.reset_path(torch.arange(env.num_envs,device=env.device))
                print('[native play] diagnostic simulator localization; real deployment requires corrected external odometry.',flush=True)
        for step in range(int(round(args.duration_s / env.dt))):
            case = int(step * env.dt / 10.0) % len(commands)
            if case != last_case:
                label, vx, vy, wz = commands[case]
                env.set_command_override(
                    torch.tensor([vx, vy, wz], device=env.device).repeat(env.num_envs, 1),
                    **({"gait_enabled": abs(vx)+abs(vy)+abs(wz)>0 or label=='march' or args.march} if cts else {})
                )
                if cts:
                    if hasattr(env,'path'):
                        scaled=env.commands[:,:3]*env.commands_scale
                        env.obs_history[:,0,6:9]=scaled;env.obs_buf[:,6:9]=scaled;env.obs_buf[:,1026:1029]=scaled
                        enabled=env.gait_enabled.float()
                        env.obs_history[:,0,50]=enabled;env.obs_buf[:,50]=enabled;env.obs_buf[:,1070]=enabled
                    else:env.compute_observations()
                    obs=env.get_observations()
                print(f"[native play] {label}: vx={vx}, vy={vy}, wz={wz}", flush=True)
                last_case = case
            obs, _, _, _, _ = env.step(policy(obs))


def play_urdf4_v3(args):
    """Native mouse-controlled viewer using the saved V3/V4 student runtime."""
    env_cfg,train_cfg=task_registry.get_cfgs(name=args.task)
    if args.task in ('urdf4_amp_cts_v4','urdf4_amp_cts_v5','urdf4_amp_cts_rough','urdf4_amp_cts_transition','urdf4_amp_cts_transition_training'):
        from legged_gym.utils.helpers import get_load_path
        root=os.path.join(LEGGED_GYM_ROOT_DIR,'logs',train_cfg.runner.experiment_name)
        checkpoint=get_load_path(root,args.load_run,args.checkpoint)
        saved=torch.load(checkpoint,map_location='cpu')
        if args.task in ('urdf4_amp_cts_rough','urdf4_amp_cts_transition','urdf4_amp_cts_transition_training'):
            for key,value in saved['contract']['rough_ground'].items():setattr(env_cfg.rough,key,value)
        if args.task=='urdf4_amp_cts_transition_training':
            from legged_gym.envs.urdf4.amp_transition_training import restore_transition_contract
            restore_transition_contract(env_cfg, train_cfg, saved)
        # Restore the capped robustness continuation's recorded task envelope.
        if saved['experiment']['env']['expand']['max_stage']==1 and saved['experiment']['env']['expand']['stages'][-1]==[-.30,.50,.20,.60]:
            for section in ('commands','expand'):
                for key,value in saved['experiment']['env'][section].items():
                    if isinstance(value,dict):
                        for child,item in value.items():setattr(getattr(getattr(env_cfg,section),key),child,item)
                    else:setattr(getattr(env_cfg,section),key,value)
        for key in ('balance_archive','balance_variant'):
            setattr(env_cfg.amp_cts,key,saved['experiment']['env']['amp_cts'][key])
        env_cfg.direction.speed_feedback_enabled=saved['experiment']['env']['direction']['speed_feedback_enabled']
        if args.task=='urdf4_amp_cts_v5':
            env_cfg.amp_cts.extra_reference=saved['experiment']['env']['amp_cts']['extra_reference']
            env_cfg.amp_cts.adaptive_feedback_phase=saved['experiment']['env']['amp_cts']['adaptive_feedback_phase']
            env_cfg.amp_cts.use_public_prior=saved['experiment']['env']['amp_cts'].get('use_public_prior',False)
            env_cfg.amp_cts.reward_revision=saved['experiment']['env']['amp_cts']['reward_revision']
            env_cfg.amp_cts.public_prior_archive=saved['experiment']['env']['amp_cts'].get('public_prior_archive',env_cfg.amp_cts.public_prior_archive)
            env_cfg.direction.mid_cross_integral_gain=saved['experiment']['env']['direction'].get('mid_cross_integral_gain',0.)
            env_cfg.amp_cts.public_feedback_history=saved['experiment']['env']['amp_cts'].get('public_feedback_history',False)
            for key in ('public_obs_size','num_observations','num_privileged_obs'):
                setattr(env_cfg.env,key,saved['experiment']['env']['env'][key])
    env_cfg.env.num_envs=args.num_envs or 1
    env_cfg.env.test=True
    env_cfg.env.episode_length_s=args.duration_s+10.
    # test=True supplies nominal physics/sensors. Keep the saved training
    # settings intact so the checkpoint's actuator/DR contract remains checked.
    env,_=task_registry.make_env(name=args.task,args=args,env_cfg=env_cfg)
    train_cfg.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train_cfg)
    policy=runner.get_inference_policy(device=env.device)
    segment_seconds = 10.
    if args.variable_sweep:
        segment_seconds = 6.
        commands = (
            ('forward_slow', .18, 0., 0.), ('forward_fast', .50, 0., 0.), ('forward_cruise', .25, 0., 0.),
            ('backward_slow', -.15, 0., 0.), ('backward_fast', -.30, 0., 0.),
            ('left_slow', 0., .10, 0.), ('left_fast', 0., .20, 0.),
            ('right_slow', 0., -.10, 0.), ('right_fast', 0., -.20, 0.),
            ('turn_left_slow', 0., 0., .40), ('turn_left_fast', 0., 0., .60),
            ('turn_right_slow', 0., 0., -.40), ('turn_right_fast', 0., 0., -.60),
            ('forward_left', .15, .08, 0.), ('forward_right', .15, -.08, 0.),
            ('backward_left', -.15, .08, 0.), ('backward_right', -.15, -.08, 0.),
            ('arc_left', .20, 0., .30), ('arc_right', .20, 0., -.30), ('stand', 0., 0., 0.),
        )
    elif args.sweep:
        commands=(("forward",.25,0.,0.),("backward",-.15,0.,0.),
            ("left",0.,.1,0.),("right",0.,-.1,0.),
            ("turn_left",0.,0.,.4),("turn_right",0.,0.,-.4),
            ("forward_left",.15,.08,0.),("forward_right",.15,-.08,0.),
            ("backward_left",-.15,.08,0.),("backward_right",-.15,-.08,0.),
            ("arc_left",.2,0.,.3),("arc_right",.2,0.,-.3),
            ("back_arc_left",-.15,0.,.3),("back_arc_right",-.15,0.,-.3),
            ("stand",0.,0.,0.))
    else:commands=(("fixed",args.vx,args.vy,args.wz),)
    env.set_command_override(torch.tensor(commands[0][1:],device=env.device).repeat(env.num_envs,1))
    obs,_=env.reset();last=None
    print('[native play] public-history student; direction servo requires corrected external localization on hardware.',flush=True)
    segment_steps = round(segment_seconds / env.dt)
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            index=(step//segment_steps)%len(commands)
            if index!=last:
                label,*cmd=commands[index]
                env.set_command_override(torch.tensor(cmd,device=env.device).repeat(env.num_envs,1))
                print('[native play]',label,cmd,flush=True);last=index
            obs,_,_,dones,_=env.step(policy(obs))
            if args.variable_sweep and dones.any():
                print('[native play] reset during', commands[index][0], 'at', round((step+1)*env.dt, 2), 's', flush=True)


def play(args):
    if args.task in ('urdf4_amp_cts_v3','urdf4_amp_cts_v4','urdf4_amp_cts_v5','urdf4_amp_cts_rough','urdf4_amp_cts_transition','urdf4_amp_cts_transition_training'):
        return play_urdf4_v3(args)
    if args.task in ("urdf4_omni_v1", "urdf4_amp_cts_v1", "urdf4_amp_cts_expand_v1", "urdf4_amp_cts_expand_x_v1", "urdf4_amp_cts_direction_v1"):
        return play_urdf4(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=(args.task))
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if args.task in (
        "dog_rs01_trot",
        "dog_rs01_balance",
        "dog_rs01_body_stable",
        "dog_rs01_low_twist",
        "dog_rs01_hip_torque",
        "dog_rs01_straight_balance",
        "dog_rs01_compact_hip",
        "dog_rs01_safe6nm",
        "dog_rs01_safe6nm_v2",
        "dog_rs01_smooth_straight",
        "dog_rs01_smooth_straight_v2",
        "dog_rs01_straight_guarded",
        "dog_rs01_torque_straight_v5",
        "dog_rs01_stage2_actuator_a",
        "dog_rs01_stage2_actuator_b",
        "dog_rs01_straight_stand",
        "dog_rs01_straight_walk",
    ):
        env_cfg.env.num_envs = 1
        env_cfg.commands.ranges.lin_vel_x = [0.15, 0.15]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.stand_probability = 0.0
        env_cfg.commands.pure_sagittal_probability = 1.0
        env_cfg.commands.resampling_time = 1000.0
        env_cfg.domain_rand.randomize_gait_phase_on_reset = False
        if args.task in ("dog_rs01_safe6nm", "dog_rs01_safe6nm_v2"):
            env_cfg.commands.ranges.lin_vel_x = [0.12, 0.12]
        elif args.task == "dog_rs01_straight_stand":
            # Stage 1 only ever learned to hold still.
            env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
            env_cfg.commands.stand_probability = 1.0
        elif args.task == "dog_rs01_straight_walk":
            # The single speed this stage is trained on.
            env_cfg.commands.ranges.lin_vel_x = [0.10, 0.10]
            env_cfg.commands.stand_probability = 0.0
        elif args.task in (
            "dog_rs01_stage2_actuator_a",
            "dog_rs01_stage2_actuator_b",
        ):
            env_cfg.commands.ranges.lin_vel_x = [0.10, 0.10]
        elif args.task in (
            "dog_rs01_smooth_straight",
            "dog_rs01_smooth_straight_v2",
            "dog_rs01_straight_guarded",
            "dog_rs01_torque_straight_v5",
        ):
            env_cfg.commands.ranges.lin_vel_x = [0.16, 0.16]
        env_cfg.env.test = True
        env, _ = task_registry.make_env(name=(args.task), args=args, env_cfg=env_cfg)
        obs = env.get_observations()
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=(args.task), args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=(env.device))
        if EXPORT_POLICY:
            path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "policies",
            )
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print("Exported policy as jit script to: ", path)
        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep", action="store_true", help="URDF4: change direction every 10 seconds")
    parser.add_argument("--variable-sweep", action="store_true", help="URDF4 V3/V4/V5/rough: change speed and direction every 6 seconds")
    parser.add_argument("--march", action="store_true", help="CTS: step in place on a zero velocity command")
    flags, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining
    args = get_args()
    args.sweep = flags.sweep
    args.variable_sweep = flags.variable_sweep
    args.march = flags.march
    play(args)
