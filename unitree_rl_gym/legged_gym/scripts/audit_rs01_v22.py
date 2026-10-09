"""Nominal/perturbed sensor audit with exact actor/reward target assertions."""
import argparse
import json
import sys
import types
import isaacgym
import torch
from legged_gym.envs.base import legged_robot
from legged_gym.utils import get_args, task_registry
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sensor_profile',choices=('clean','normal','robust','mixed'),default='normal')
    p.add_argument('--fast', action='store_true', help='All 12 fast directional commands')
    p.add_argument('--gait_audit', action='store_true', help='Extra contact/foot/cadence telemetry (V24/V25)')
    own,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    args=get_args()
    if args.task not in ('rs01_omni_v22_sensor','rs01_omni_v22_clean','rs01_omni_v23_balance18','rs01_omni_v23_balance20','rs01_omni_v24_support10','rs01_omni_v24_support11','rs01_omni_v25_cadence20','rs01_omni_v25_cadence22','rs01_omni_v26_inward3','rs01_omni_v26_inward5','rs01_omni_v27_foot15','rs01_omni_v27_foot10','rs01_omni_v28_curriculum','rs01_omni_v29_mince','rs01_omni_v30_clearance','rs01_amp_style','rs01_amp_control','rs01_amp_retime'):raise ValueError('V22–V30 only')
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.sensor.enabled=own.sensor_profile!='clean'
    cfg.sensor.clean_fraction=.25 if own.sensor_profile=='mixed' else 0.
    cfg.sensor.robust_fraction={'clean':0.,'normal':0.,'robust':1.,'mixed':.25}[own.sensor_profile]
    cases=((0,0,0),(.2,0,0),(-.2,0,0),(0,.2,0),(0,0,.3),(.2,.08,.25))
    if own.fast:
        cases=((.4,0,0),(-.3,0,0),(0,.3,0),(0,-.3,0),(.3,.15,0),(.3,-.15,0),
               (-.3,.15,0),(-.3,-.15,0),(0,0,.6),(0,0,-.6),(.3,.15,.5),(-.3,-.15,-.5))
    if args.task in ('rs01_omni_v28_curriculum','rs01_omni_v29_mince','rs01_omni_v30_clearance'):
        from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import curriculum_command
        cases=curriculum_command(torch.tensor(cases,dtype=torch.float),cfg.commands.lateral_speed_range_m_s[1],cfg.commands.reverse_combined_axis_caps).tolist()
    if args.task in ('rs01_amp_style','rs01_amp_control','rs01_amp_retime'):cases=cfg.amp.commands
    _set_nominal_eval_cfg(cfg,args.duration_s,len(cases)*args.eval_envs)
    if args.headless:legged_robot.time=types.SimpleNamespace(sleep=lambda s:None)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    command=torch.tensor(cases,device=env.device).repeat_interleave(args.eval_envs,0)
    rows=[];gait_rows=[];resets=torch.zeros(env.num_envs,device=env.device)
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(command,1.);env.compute_observations()
            obs=env.get_observations().clone();target=env.sensor_effective_command.clone()
            # The actor may not see independently sampled/true IMU or joint state.
            torch.testing.assert_close(obs[:,3:6],env.sensor.gyro*env.obs_scales.ang_vel)
            torch.testing.assert_close(obs[:,6:9],env.sensor.gravity)
            torch.testing.assert_close(obs[:,12:24],(env.sensor.q-env.default_dof_pos)*env.obs_scales.dof_pos)
            torch.testing.assert_close(obs[:,24:36],env.sensor.dq*env.obs_scales.dof_vel)
            torch.testing.assert_close(obs[:,9:12],target*env.commands_scale)
            phase=env._gait_phase().clone()
            torch.testing.assert_close(obs[:,48:50],torch.stack((torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase)),1))
            torch.testing.assert_close(env.sensor.skew_s,env.sensor.imu_age_s-env.sensor.motor_age_s)
            sensor=torch.stack((env.sensor.imu_age_s,env.sensor.motor_age_s,env.sensor.skew_s),1).clone()
            new,_,reward,done,_=env.step(policy(obs))
            if hasattr(env,'v25_frequency_hz'):
                alive=~done.bool()
                torch.testing.assert_close(env._gait_phase()[alive],
                    torch.remainder(phase+env.dt*env.v25_frequency_hz,1.)[alive],rtol=0,atol=1e-6)
            torch.testing.assert_close(env.direction_reward_target,target,rtol=0,atol=0)
            error=((target[:,:2]-env.base_lin_vel[:,:2]).square().sum(1)/cfg.rewards.command_planar_tracking_sigma
                +(target[:,2]-env.base_ang_vel[:,2]).square()/cfg.rewards.command_yaw_tracking_sigma)
            alive=~done.bool()
            torch.testing.assert_close(env.v11_tracking_reward[alive],(1/(1+error))[alive])
            if not bool(torch.isfinite(new).all() & torch.isfinite(reward).all()):raise RuntimeError('Nonfinite')
            resets+=done
            if step>=round(2/env.dt):
                if own.gait_audit:
                    contact=env.get_foot_contact_mask()
                    ids=[env.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
                    force=env.contact_forces[:,env.feet_indices,2].clamp(min=0.)
                    gait_rows.append(tuple(x.clone() for x in (
                        getattr(env,'v25_frequency_hz',env._command_gait_frequency()),
                        env.v24_signed_y,env.v24_support_contact,env.v24_touchdown,
                        (env.feet_pos[:,:,2]-cfg.rewards.foot_collision_radius_m)[:,ids],
                        torch.linalg.vector_norm(env.feet_state[:,:,7:9],dim=2)[:,ids],
                        contact[:,ids],(contact!=env._desired_contact_mask()).float().mean(1),
                        force[:,ids])))
                rows.append(tuple(x.clone() for x in (env.base_lin_vel,env.base_ang_vel[:,2],env.rpy[:,0],
                    env.get_foot_contact_mask().sum(1),env.estimated_base_lin_vel,env.estimated_odom_confidence,
                    env.root_states[:,2],env.root_states[:,9],sensor,env.raw_pd_torques.abs().amax(1))))
    values=[torch.stack([r[i] for r in rows]) for i in range(10)]
    result=[]
    for j,cmd in enumerate(cases):
        sl=slice(j*args.eval_envs,(j+1)*args.eval_envs)
        v,w,roll,contact,est,conf,z,vz,sensor,raw=[x[:,sl] for x in values]
        result.append(dict(command=cmd,resets=int(resets[sl].sum()),
            mean_vx_vy_wz=[float(v[:,:,0].mean()),float(v[:,:,1].mean()),float(w.mean())],
            roll_rms_deg=float(roll.square().mean().sqrt()*180/torch.pi),
            flight_ratio=float((contact==0).float().mean()),four_foot_ratio=float((contact==4).float().mean()),
            velocity_estimator_xy_rmse=float((est[:,:,:2]-v[:,:,:2]).square().mean().sqrt()),
            confidence_mean=float(conf.mean()),z_peak_to_peak_m=float((z.max(0).values-z.min(0).values).mean()),
            world_vz_rms=float(vz.square().mean().sqrt()),raw_peak_nm=float(raw.max()),
            imu_motor_skew_mean_ms=(sensor.mean((0,1))*1000).tolist(),
            imu_motor_skew_max_ms=(sensor.amax((0,1))*1000).tolist()))
        if own.gait_audit:
            f,y,contact,land,height,slip,raw_contact,mismatch,force=[torch.stack([r[i] for r in gait_rows])[:,sl] for i in range(9)]
            sample_s=len(gait_rows)*env.dt
            result[-1]['gait']=dict(frequency_mean_hz=float(f.mean()),
                foot_order=['FL','FR','RL','RR'],
                touchdown_rate_hz=(land.sum(0).float().mean(0)/sample_s).tolist(),
                support_inside_10cm_ratio=float(((y<.10)&contact).sum()/contact.sum().clamp(min=1)),
                support_inside_11cm_ratio=float(((y<.11)&contact).sum()/contact.sum().clamp(min=1)),
                contact_mismatch_ratio=float(mismatch.mean()),
                stance_slip_mean_m_s=float((slip*raw_contact).sum()/raw_contact.sum().clamp(min=1)),
                swing_clearance_p95_m=float(torch.quantile(height[~raw_contact],.95)) if (~raw_contact).any() else None,
                mean_vertical_force_n=force.mean((0,1)).tolist())
    print(json.dumps(dict(task=args.task,checkpoint=args.checkpoint,load_run=args.load_run,seed=args.seed,
        profile=own.sensor_profile,duration=args.duration_s,envs_per_case=args.eval_envs,
        aligned=True,finite=True,cases=result),indent=2))


if __name__=='__main__':main()
