"""Recreate nominal MuJoCo teacher commands without changing the AMP dataset."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=20);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
src=Path(__file__).with_name('physical_v2.py');source=src.read_text();source=source.replace(' for k in range(round((seconds+2)/dt)):'," schedule=[];initial_root=d.qpos[:7].copy()\n for k in range(round((seconds+2)/dt)):")
source=source.replace('   desired=np.clip(desired,limits[:,0]+.01,limits[:,1]-.01)',"   desired=np.clip(desired,limits[:,0]+.01,limits[:,1]-.01)\n   schedule.append(desired.copy())")
source=source.replace("np.savez_compressed(save/'physical.npz',**a)","np.savez_compressed(save/'physical.npz',**a,command_targets=np.asarray(schedule),actuator_default=q0,initial_root=initial_root,command_dt_s=np.array(.02))")
(a.output/'captured_source.py').write_text(source);ns={'__name__':'capture_teacher_commands','__file__':str(src)};exec(compile(source,str(src),'exec'),ns)
m=json.loads((a.dataset/'manifest.json').read_text());reports=[]
for c in m['clips']:
 dest=a.output/c['clip'];r=ns['run'](ROOT/'artifacts/urdf4_amp_design_20261113/kinematic_v1',np.array(c['command']),a.seconds,c['lead'],c['feed'],c['height'],dest,c['design']);reports.append(r);print(c['clip'],r['stop'],r['mean_velocity'],flush=True)
(a.output/'contract.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),'seconds':a.seconds,'frame_dt_s':.02,'warmup_s':2,'reports':reports},indent=2)+'\n')
