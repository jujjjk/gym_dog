"""Preview/collect the hardcoded diagonal-VMC teacher (AMP reference source).

Without --record: renders the teacher in the IsaacGym viewer.
With --record PATH: runs headless and saves AMP transition pairs.
"""
import argparse
import sys
import time
import isaacgym
import numpy as np
import torch
import legged_gym.envs  # registers tasks; avoids helpers->envs circular import
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.base import legged_robot
from legged_gym.envs.rs01_omni_v2.rs01_mpc_teacher import Rs01MpcTeacher
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import curriculum_command


def build_command_grid(env, per_band=8, seed=20261005):
    g=torch.Generator().manual_seed(seed)
    bands=[]
    def rnd(lo,hi):
        return lo+(hi-lo)*torch.rand(per_band,generator=g)
    vx=rnd(.10,.60);bands.append(torch.stack((vx,torch.zeros_like(vx),
        torch.zeros_like(vx)),1))
    vb=rnd(.10,.35);bands.append(torch.stack((-vb,torch.zeros_like(vb),
        torch.zeros_like(vb)),1))
    vy=rnd(.08,.15)
    for s in (1,-1):
        bands.append(torch.stack((torch.zeros_like(vy),s*vy,
            torch.zeros_like(vy)),1))
    w=rnd(.20,1.0)
    for s in (1,-1):
        bands.append(torch.stack((torch.zeros_like(w),torch.zeros_like(w),
            s*w),1))
    cf=rnd(.10,.40);cl=rnd(.03,.15)*torch.where(torch.rand(per_band,generator=g)>.5,1,-1)
    cw=rnd(.10,.50)*torch.where(torch.rand(per_band,generator=g)>.5,1,-1)
    bands.append(torch.stack((cf,cl,cw),1))
    cb=rnd(.06,.18);bands.append(torch.stack((-cb,rnd(.03,.08)*torch.where(
        torch.rand(per_band,generator=g)>.5,1,-1),rnd(.10,.25)*torch.where(
        torch.rand(per_band,generator=g)>.5,1,-1)),1))
    return torch.cat(bands)


def amp_obs(env):
    return torch.cat((env.base_ang_vel,env.projected_gravity,
        env.dof_pos-env.default_dof_pos,env.dof_vel),1)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--record',type=str,default=None)
    p.add_argument('--duration_s',type=float,default=60.)
    p.add_argument('--gait_audit',action='store_true')
    p.add_argument('--teacher_cadence',type=float,default=2.5)
    own,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    args=get_args()
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.episode_length_s=int(own.duration_s)
    cfg.noise.noise_level=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    env._command_gait_frequency=lambda:torch.full((env.num_envs,),
        own.teacher_cadence,device=env.device)
    # RS01 spawns at q=0; settle to the default stance so the teacher measures
    # true nominal foot positions before its IK round-trip check.
    zero=torch.zeros(env.num_envs,env.num_actions,device=env.device)
    for _ in range(round(3./env.dt)):env.step(zero)
    drift=(env.dof_pos.reshape(env.num_envs,-1)-env.default_dof_pos.reshape(-1)).abs().max()
    if float(drift)>.35:raise RuntimeError(f"settle drift {float(drift):.2f} rad")
    teacher=Rs01MpcTeacher(env)
    grid=build_command_grid(env)
    env._resample_commands=lambda ids: None
    n=env.num_envs
    ids=torch.arange(n,device=env.device)
    rep=torch.arange(n)%grid.shape[0]
    env.commands[:,:3]=grid[rep.to(grid.device)].to(env.device)
    env.gait_enable[:]=1.
    env._update_turn_mode(ids)
    if args.headless:legged_robot.time=type('t',(),{'sleep':lambda s:None})
    steps=round(own.duration_s/env.dt)
    warm=round(2./env.dt)
    stats={'frames':0,'resets':0}
    if own.record:
        cur=[];nxt=[];cmds=[];freq=[]
    with torch.no_grad():
        dbg=torch.nonzero(env.commands[:,0].abs()>0.3).flatten()
        d=int(dbg[0]) if len(dbg) else 0
        dbg_hist=[]
        for k in range(steps):
            env.commands[:,:3]=grid[rep.to(grid.device)].to(env.device)
            s0=amp_obs(env).clone()
            f=(env.v25_frequency_hz.clone() if hasattr(env,'v25_frequency_hz') else None)
            qt=teacher.step_targets()
            _,_,_,done,_=env.step((qt-teacher.default)/teacher.scale)
            if not args.headless:env.render()
            if own.gait_audit and k>=steps-100:
                qi=teacher.joint_index['FL'];fl=teacher.foot_slot[0]
                dof=env.dof_pos.reshape(env.num_envs,-1)
                dbg_hist.append([float(qt[d,qi["thigh"]]),float(dof[d,qi["thigh"]]),
                    float(qt[d,qi["calf"]]),float(dof[d,qi["calf"]]),
                    float(env.feet_pos[d,fl,2]),float(env._gait_phase()[d]),
                    float((qt[d]-env.default_dof_pos.reshape(-1)).abs().max()),
                    float(env.root_states[d,2])])
            if own.record and k>=warm:
                keep=~done.bool()
                cur.append(s0[keep]);nxt.append(amp_obs(env)[keep])
                cmds.append(env.commands[keep,:3].clone())
                if f is not None:freq.append(f[keep,None])
                stats['frames']+=int(keep.sum())
            stats['resets']+=int(done.sum())
            if own.gait_audit and k==steps-1 and dbg_hist:
                a=torch.tensor(dbg_hist)
                names=['thigh_tgt','thigh_act','calf_tgt','calf_act','footFL_z','phase','|q-def|max','base_z']
                print(("DEBUG env%d: "+" ".join(f"{n}=[%.3f,%.3f]" for n in names))%(d,*[float(x) for pair in zip(a.min(0).values,a.max(0).values) for x in pair]),flush=True)
            if own.gait_audit and k==steps-1:
                band=['fwd','back','lat+','lat-','yaw+','yaw-','fwd-combo','rev-combo']
                for b in range(8):
                    lo,hi=b*8,min((b+1)*8,n)
                    if hi<=lo: continue
                    sel=torch.arange(lo,hi,device=env.device)
                    c=env.commands[sel,:3];v=env.base_lin_vel[sel,:2]
                    w=env.base_ang_vel[sel,2]
                    print(f"{band[b]:>10}: cmd(vx{c[:,0].mean():+.2f},vy{c[:,1].mean():+.2f},"
                          f"wz{c[:,2].mean():+.2f}) -> act(vx{v[:,0].mean():+.3f},"
                          f"vy{v[:,1].mean():+.3f},wz{w.mean():+.3f})",flush=True)
            if k%round(5/env.dt)==0 and own.gait_audit:
                print(f"t={k*env.dt:5.1f} vx={env.base_lin_vel[:,0].mean():+.3f} "
                    f"vy={env.base_lin_vel[:,1].mean():+.3f} wz={env.base_ang_vel[:,2].mean():+.3f} "
                    f"f={f.mean() if f is not None else 0:.2f}Hz resets={stats['resets']}",flush=True)
    if own.record:
        out=np.savez_compressed(own.record,
            obs_cur=torch.cat(cur).cpu().numpy(),obs_next=torch.cat(nxt).cpu().numpy(),
            commands=torch.cat(cmds).cpu().numpy(),
            frequency_hz=torch.cat(freq).cpu().numpy() if freq else np.zeros((0,1)),
            fps=np.array([1./env.dt]),task=np.array([args.task]))
        print(f"Saved {own.record}: {stats['frames']} transitions, "
              f"{stats['frames']*env.dt/1000:.1f}s of motion")


if __name__=='__main__':main()
