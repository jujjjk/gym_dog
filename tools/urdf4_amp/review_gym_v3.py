"""Review the exact 5..55s expert interval of an existing full Gym recording.

All-episode falls and 400Hz torque peaks remain hard gates. Startup contact
statistics are retained separately; no physics is rerun or recorded state edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def starts(mask):
    pad=np.r_[False,mask,False];s=np.flatnonzero(~pad[:-1]&pad[1:]);e=np.flatnonzero(pad[:-1]&~pad[1:])
    return s[(e-s>=2)&(s>0)]


def review(folder,dataset):
    parent=json.loads((folder/'result.json').read_text());m=json.loads((dataset/'manifest.json').read_text())
    z=np.load(folder/'live_rollout.npz');f=z['amp_features'];ct=z['contact'];time=z['time_s'];valid=z['valid'];per=parent['envs_per_clip'];take=(time>=5.)&(time<55.);full=time>=0.
    root=z['root_state_xyzw_world'];feet=z['feet_pos_world_m'];ref=z['reference_xy_world_m'];yawref=z['reference_yaw_rad']
    rows=[]
    for i,(clip,old) in enumerate(zip(m['clips'],parent['cases'])):
        sl=slice(i*per,(i+1)*per);mask=valid[:,sl]&take[:,None];a=f[:,sl];contact=ct[:,sl];cc=contact[mask];count=cc.sum(1);diag=(cc[:,0]&cc[:,3])|(cc[:,1]&cc[:,2]);standing=np.linalg.norm(clip['command'])<1e-8
        row=dict(old);row.update(data_interval_motion_s=[5.,55.],steady_frames=int(mask.sum()),mean_velocity=a[mask][:,[36,37,41]].mean(0).tolist(),
            foot_clearance_p95_mm=(np.quantile((feet[:,sl,:,2]-.02)[mask],.95,axis=0)*1000).tolist(),flight_fraction=float((count==0).mean()),single_fraction=float((count==1).mean()),illegal_two_fraction=float(((count==2)&~diag).mean()))
        allct=contact[full];alln=allct.sum(2);alldiag=(allct[:,:,0]&allct[:,:,3])|(allct[:,:,1]&allct[:,:,2]);row['full_motion_illegal_support_fraction']=float(((alln==1)|((alln==2)&~alldiag)).mean());row['full_motion_flight_fraction']=float((alln==0).mean())
        gaps=[]
        for support in [True,False]:
            values=[]
            for copy in range(per):
                c=contact[take,copy];c=c if support else ~c;t=time[take]
                for left,right in [(0,3),(1,2)]:
                    ea=starts(c[:,left]);eb=starts(c[:,right])
                    if len(ea) and len(eb):values.extend(np.min(abs(t[eb]-t[k])) for k in ea)
            gaps.append(float(np.quantile(values,.95)*1000) if values else None)
        row.update(diagonal_touchdown_p95_ms=gaps[0],diagonal_liftoff_p95_ms=gaps[1])
        delta=root[:,sl,:2]-ref[:,sl];cmd=np.asarray(clip['command']);angle=np.arctan2(cmd[1],cmd[0]);normal=np.stack([-np.sin(yawref[:,sl]+angle),np.cos(yawref[:,sl]+angle)],-1)
        cross=abs((delta*normal).sum(2)) if np.linalg.norm(cmd[:2])>1e-8 else np.linalg.norm(delta,axis=2)
        q=root[:,sl,3:7];yaw=np.arctan2(2*(q[:,:,3]*q[:,:,2]+q[:,:,0]*q[:,:,1]),1-2*(q[:,:,1]**2+q[:,:,2]**2));error=np.arctan2(np.sin(yaw-yawref[:,sl]),np.cos(yaw-yawref[:,sl]))
        row.update(cross_max_m=float(cross[mask].max()),heading_mean_abs_deg=float(abs(error[mask]).mean()*180/np.pi),full_motion_cross_max_m=float(cross[full].max()),full_motion_heading_mean_abs_deg=float(abs(error[full]).mean()*180/np.pi))
        row['lift_events_per_foot']=[min(len(starts((feet[take,i*per+copy,j,2]-.02>.012)&~contact[take,copy,j])) for copy in range(per)) for j in range(4)]
        reject=[];velocity=np.asarray(row['mean_velocity']);clear=row['foot_clearance_p95_mm']
        if row['fallen_count']:reject.append('fall')
        if row['raw_peak_Nm']>17:reject.append('raw_torque')
        if row['flight_fraction']>0 or row['full_motion_flight_fraction']>0:reject.append('flight')
        if row['single_fraction']+row['illegal_two_fraction']>.001:reject.append('support')
        if row['action_clip_fraction']>.001:reject.append('action_envelope')
        if np.linalg.norm(velocity[:2]-cmd[:2])>.05 or abs(velocity[2]-cmd[2])>.12:reject.append('tracking')
        if row['cross_max_m']>.05 or row['heading_mean_abs_deg']>2:reject.append('direction')
        if standing:
            if max(clear)>5:reject.append('standing_feet')
        else:
            if min(clear)<20 or max(clear)>55:reject.append('clearance')
            if any(g is None or g>40.001 for g in gaps):reject.append('diagonal_timing')
            if min(row['lift_events_per_foot'])<50/clip['design']['period_s']*.7:reject.append('alternation')
        row.update(rejection_reasons=reject,passed=not reject);rows.append(row)
    out=dict(parent);out.update(cases=rows,passed=all(r['passed'] for r in rows),statistical_window_motion_s=[5.,55.],
        all_episode_fall_and_torque_gates_preserved=True,parent_result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
        live_rollout_sha256=hashlib.sha256((folder/'live_rollout.npz').read_bytes()).hexdigest(),review='Exact 50s expert interval, with full-motion transients retained separately')
    (folder/'v3_windowed_result.json').write_text(json.dumps(out,indent=2)+'\n');print(folder,sum(r['passed'] for r in rows),'/',len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();review(a.run,a.dataset)
