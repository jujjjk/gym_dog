"""Independent FK/schema/provenance and sampled-contact checks for a dataset."""
import argparse,json,tempfile,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
from design import make_model,setup,NAMES,ROOT
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.dataset/'manifest.json').read_text());reports=[]
assert manifest['urdf_sha256']==hashlib.sha256((ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf').read_bytes()).hexdigest()
assert manifest['joint_order']==NAMES and manifest['frame_dt_s']==.02
with tempfile.TemporaryDirectory() as tmp:
 m=make_model(Path(tmp));d=mj.MjData(m);qa,va,feet,limits=setup(m)
 for clip in manifest['clips']:
  f=a.dataset/(clip['clip']+'.npz');z=np.load(f,allow_pickle=False);x=z['amp_features'];assert x.shape==(2500,46) and np.isfinite(x).all();assert len(z['transition_valid'])==2500 and not z['transition_valid'][-1];assert np.allclose(np.diff(z['time_s']),.02);assert np.allclose(np.linalg.norm(z['root_quat_wxyz'],axis=1),1,atol=1e-7);assert np.all(z['joint_pos_rad']>=limits[:,0]-1e-7) and np.all(z['joint_pos_rad']<=limits[:,1]+1e-7)
  errors=[]
  for k in np.linspace(0,len(x)-1,51).astype(int):
   d.qpos[:3]=z['root_pos_world_m'][k];d.qpos[3:7]=z['root_quat_wxyz'][k];d.qpos[qa]=z['joint_pos_rad'][k];mj.mj_forward(m,d);R=np.empty(9);mj.mju_quat2Mat(R,d.qpos[3:7]);actual=(d.xpos[feet]-d.qpos[:3])@R.reshape(3,3);errors.append(float(np.max(abs(actual-x[k,24:36].reshape(4,3)))))
  assert max(errors)<2e-6
  assert hashlib.sha256(f.read_bytes()).hexdigest()==clip['sha256']
  report=dict(clip=clip['clip'],frames=len(x),fk_max_error_m=max(errors),unit_quaternions=True,finite=True,sha256_matches=True);reports.append(report);print(report)
(a.dataset.parent/'format_validation.json').write_text(json.dumps(reports,indent=2)+'\n')
