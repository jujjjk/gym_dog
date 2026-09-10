"""Repeatable no-reset V15 evaluation, same cold-start protocol as V14."""
import argparse
import contextlib
import json
from pathlib import Path
import torch
import numpy as np
from sim2sim import roll_pitch_yaw
from sim2sim_v15 import V15Sim, run, get_parser


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    p.add_argument('--transitions', action='store_true')
    p.add_argument('--sensor-sync', action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.sensor_sync:
        from sim2sim_sensor_sync import SensorSyncSim
        global V15Sim
        V15Sim = SensorSyncSim
    torch.set_num_threads(1)
    if args.transitions:
        evaluate_transitions(args)
        return
    cases=[('stand',[0,0,0],False),('backward02',[-.2,0,0],False)]
    if args.full:
        cases += [('march',[0,0,0],True), ('forward02',[.2,0,0],False),
                  ('forward04',[.4,0,0],False), ('forward06',[.6,0,0],False),
                  ('left02',[0,.2,0],False), ('right02',[0,-.2,0],False),
                  ('yawleft03',[0,0,.3],False), ('yawright03',[0,0,-.3],False),
                  ('combined',[.3,.1,.25],False), ('reversecombined',[-.3,-.1,-.25],False)]
    results=[]
    for name,command,march in cases:
        for phase in [0,.25,.5,.75]:
            output=args.output/f'{name}_p{int(phase*100):03}'
            a=get_parser().parse_args(['--scene',str(args.scene),'--policy',str(args.policy),
                '--command',*map(str,command),'--phase',str(phase),'--duration','30','--output',str(output)]
                + (['--march'] if march else []))
            with open(str(output)+'.log','w') as f, contextlib.redirect_stdout(f):
                run(a,V15Sim)
            d=json.loads(Path(str(output)+'.json').read_text())
            d['case']=name; results.append(d)
            print(name,phase,d['completed_duration_s'],d.get('mean_velocity'),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


def evaluate_transitions(args):
    sequence=[('stand',[0,0,0],0),('march',[0,0,0],1),('forward',[.2,0,0],1),
              ('stand',[0,0,0],0),('backward',[-.2,0,0],1),('left',[0,.2,0],1),
              ('right',[0,-.2,0],1),('yaw_left',[0,0,.3],1),('yaw_right',[0,0,-.3],1),
              ('combined',[.3,.1,.25],1),('stand',[0,0,0],0)]
    results=[]
    for phase in (0,.25,.5,.75):
        sim=V15Sim(args.scene,args.policy,[0,0,0]); sim.phase_value=phase
        segments=[]; elapsed=0.; reason=None
        for name,cmd,gait in sequence:
            old_phase=sim.phase_value; sim.set_command(cmd,gait)
            assert old_phase==sim.phase_value
            velocities=[]; max_speed=0.; raw_peak=0.
            for step in range(round(5/sim.policy_dt)):
                sim.control_step(); elapsed+=sim.policy_dt
                if not gait: assert sim.phase_value==old_phase
                linear,angular=sim.base_velocity_body()
                velocities.append([*linear[:2],angular[2]])
                roll,pitch,_=roll_pitch_yaw(sim.data.qpos[3:7])
                max_speed=max(max_speed,sim.step_max_speed)
                raw_peak=max(raw_peak,float(np.abs(sim.last_raw_torque).max()))
                if sim.step_overspeed: reason='overspeed'
                elif sim.data.qpos[2]<.18 or max(abs(roll),abs(pitch))>.8: reason='fall'
                elif not np.isfinite(sim.data.qpos).all(): reason='nonfinite'
                if reason: break
            segments.append(dict(name=name,command=cmd,steps=len(velocities),
                mean_velocity=np.mean(velocities,axis=0).tolist(),max_joint_speed_rad_s=max_speed,
                raw_peak_nm=raw_peak))
            if reason: break
        results.append(dict(initial_phase=phase,requested_duration_s=55.,
                            completed_duration_s=round(elapsed,2),stop_reason=reason,segments=segments))
        print('transition',phase,round(elapsed,2),reason,flush=True)
    (args.output/'transitions.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
