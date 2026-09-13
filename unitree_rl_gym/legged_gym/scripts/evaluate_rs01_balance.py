"""Comparable fixed-command audit of signed lean, load, and swing clearance."""
import json
import isaacgym  # noqa: F401
import torch
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args, task_registry


def evaluate(args):
    if args.task not in ('rs01_omni_v16_guarded_support', 'rs01_omni_v17_balance', 'rs01_omni_v18_balance18', 'rs01_omni_v18_balance14', 'rs01_omni_v18_balance_soft', 'rs01_omni_v19_placement', 'rs01_omni_v19_placement_soft'):
        raise ValueError('Use V16, V17, V18 or V19')
    if args.duration_s <= 2 or args.eval_envs < 1:
        raise ValueError('duration_s must exceed 2; eval_envs must be positive')
    commands = ((0.,0.,0.), (.2,0.,0.), (-.2,0.,0.))
    cfg, train = task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg, args.duration_s, 3*args.eval_envs)
    env,_ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if env.num_envs != 3*args.eval_envs:
        raise ValueError('Do not override num_envs; use eval_envs per command')
    train.runner.resume = True
    runner,_ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    cmd = torch.tensor(commands, device=env.device).repeat_interleave(args.eval_envs,0)
    data, finite = [], True
    resets = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(cmd,1.)
            env.compute_observations()
            obs,_,reward,done,_ = env.step(policy(env.get_observations()))
            finite &= bool(torch.isfinite(obs).all() & torch.isfinite(reward).all())
            resets += done
            if step < round(2./env.dt):
                continue
            phase = env._gait_phase()[:,None]
            feet_phase = torch.where(env.diagonal_a_contact_mask[None,:], phase, (phase+.5)%1.)
            duty = cfg.rewards.gait_stance_ratio
            mid = (feet_phase > duty+(1-duty)*.25) & (feet_phase < duty+(1-duty)*.75)
            clearance = env.feet_pos[:,:,2]-cfg.rewards.foot_collision_radius_m
            force = env.contact_forces[:,env.feet_indices,2].clamp(min=0.)
            cost = getattr(env,'v17_balance_cost',torch.zeros_like(resets))
            data.append(tuple(x.clone() for x in (
                env.rpy[:,:2], env.base_lin_vel, env.base_ang_vel[:,2], force, clearance, mid,
                env.get_foot_contact_mask().sum(1), env.dof_pos[:,[0,3,6,9]],
                env.v11_tracking_reward*env.reward_scales['tracking_command_velocity'],
                env.v11_contact_quality_reward*env.reward_scales['phase_two_contact_quality'],
                cost*getattr(cfg.rewards,'balance_prior_weight',0.)*env.reward_scales['phase_two_contact_quality'],
                phase[:,0], env._handoff_mask())))
    stacked = [torch.stack([row[k] for row in data]) for k in range(13)]
    results=[]
    for j,name in enumerate(('march','forward','backward')):
        sl=slice(j*args.eval_envs,(j+1)*args.eval_envs)
        r,v,w,f,h,m,c,q,tracking,quality,balance,phase,handoff = [x[:,sl] for x in stacked]
        results.append(dict(case=name, command=commands[j], resets=int(resets[sl].sum()),
            signed_roll_pitch_deg=(r.mean((0,1))*180/torch.pi).tolist(),
            roll_rms_deg=float(r[:,:,0].square().mean().sqrt()*180/torch.pi),
            mean_vx_vy_wz=[float(v[:,:,0].mean()),float(v[:,:,1].mean()),float(w.mean())],
            vz_rms_m_s=float(v[:,:,2].square().mean().sqrt()),
            mean_foot_force_n=f.mean((0,1)).tolist(), mean_hip_angles_deg=(q.mean((0,1))*180/torch.pi).tolist(),
            mid_swing_clearance_p10_p50_p90_mm=(torch.quantile(h[m],torch.tensor([.1,.5,.9],device=env.device))*1000).tolist(),
            mid_swing_loaded_ratio=float((f[m]>cfg.rewards.foot_contact_force_threshold).float().mean()),
            four_foot_ratio=float((c==4).float().mean()), flight_ratio=float((c==0).float().mean()),
            mean_tracking_per_step=float(tracking.mean()), mean_structure_per_step=float(quality.mean()),
            mean_balance_deduction_per_step=float(balance.mean()),
            handoff_vz_rms_m_s=float(v[:,:,2][handoff].square().mean().sqrt()),
            other_phase_vz_rms_m_s=float(v[:,:,2][~handoff].square().mean().sqrt()),
            phase_bins=[dict(phase_start=k/10., samples=int(((phase>=k/10.) & (phase<(k+1)/10.)).sum()),
                mean_roll_deg=float(r[:,:,0][(phase>=k/10.) & (phase<(k+1)/10.)].mean()*180/torch.pi),
                mean_vz_m_s=float(v[:,:,2][(phase>=k/10.) & (phase<(k+1)/10.)].mean()),
                mean_foot_force_n=f[(phase>=k/10.) & (phase<(k+1)/10.)].mean(0).tolist())
                for k in range(10)]))
    print(json.dumps(dict(task=args.task, run=args.load_run, checkpoint=args.checkpoint, seed=args.seed,
                         duration_s=args.duration_s, skip_s=2., eval_envs=args.eval_envs,
                         foot_slot_by_leg=env.foot_slot_by_leg, finite=finite, cases=results),indent=2))


if __name__ == '__main__':
    evaluate(get_args())
