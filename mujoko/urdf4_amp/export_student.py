"""Export the public-only TorchScript student as exact float32 NumPy MLP weights."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import isaacgym
import numpy as np
import torch
torch.set_num_threads(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path,default=ROOT/'unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000')
    p.add_argument('--checkpoint',type=int,default=17800);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ckpt=a.run/('model_%d.pt'%a.checkpoint);jit=a.run/'exported'/('student_%d.pt'%a.checkpoint)
    meta=json.loads(jit.with_suffix('.pt.json').read_text())
    policy=torch.jit.load(str(jit),map_location='cpu').eval()
    weights={k:v.detach().cpu().numpy() for k,v in policy.state_dict().items()}
    np.savez_compressed(str(a.output/'student.npz'),**weights)
    sys.path.insert(0,str(Path(__file__).resolve().parent));from sim2sim import NumpyStudent
    nn=NumpyStudent(a.output/'student.npz')
    rng=np.random.RandomState(1);x=rng.normal(0,.2,(256,1020)).astype(np.float32)
    x[:,3:6]=[0,0,-1];x[:,50]=1
    with torch.no_grad():expected=policy(torch.from_numpy(x)).numpy()
    actual=nn(x);error=float(abs(actual-expected).max());assert error<2.e-5,error
    urdf=ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
    manifest=ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/manifest.json'
    motions=json.loads(manifest.read_text())
    assert hashlib.sha256(urdf.read_bytes()).hexdigest()==motions['urdf_sha256']
    meta.update(checkpoint=str(ckpt.resolve()),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        torchscript_sha256=hashlib.sha256(jit.read_bytes()).hexdigest(),
        numpy_weights_sha256=hashlib.sha256((a.output/'student.npz').read_bytes()).hexdigest(),
        numpy_export_max_error=error,urdf=str(urdf),urdf_sha256=motions['urdf_sha256'],
        initial_root=[0,0,.30,1,0,0,0],physics_dt_s=.0025,
        motions=[dict(name=c['clip'],command=c['command'],gait_enabled=True) for c in motions['clips']]+[dict(name='stand',command=[0,0,0],gait_enabled=False)])
    (a.output/'contract.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('EXPORTED',a.output,'public-only parity max error',error,flush=True)

if __name__=='__main__':main()
