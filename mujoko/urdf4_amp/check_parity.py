"""Check actual bridge code against Torch actuator fixtures and physical FK data."""
import argparse,json
from pathlib import Path
import numpy as np
from sim2sim import ROOT,StudentSim

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/urdf4_mujoco_20261114');a=p.parse_args()
sim=StudentSim(a.bundle);fixture=np.load(a.bundle/'control_fixture.npz')
cursor=[0]
class FrozenStatePhysics:
    """Isolate the actuator from physics only in this numeric unit check."""
    @staticmethod
    def mj_step(model,data):
        cursor[0]+=1
        data.qpos[sim.qa]=fixture['q'][cursor[0]];data.qvel[sim.va]=fixture['dq'][cursor[0]]
    @staticmethod
    def mj_forward(model,data):pass
sim.mj=FrozenStatePhysics;sim.observe=lambda:None;sim.refresh_contacts=lambda:None
sim.d.qpos[sim.qa]=fixture['q'][0];sim.d.qvel[sim.va]=fixture['dq'][0]
actual=[]
for action in fixture['actions']:actual.append(sim.step(action))
actual=np.concatenate(actual,0);error=float(abs(actual-fixture['torque']).max())
assert error<2.e-5,error

sim=StudentSim(a.bundle);reference=ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2'
manifest=json.loads((reference/'manifest.json').read_text());fk=0.;gyro=0.;gravity=0.;linear=0.;frames=0
for clip in manifest['clips']:
    with np.load(reference/(clip['clip']+'.npz')) as z:
        for i in range(0,len(z['amp_features']),50):
            sim.d.qpos[:3]=z['root_pos_world_m'][i];sim.d.qpos[3:7]=z['root_quat_wxyz'][i]
            sim.d.qpos[sim.qa]=z['joint_pos_rad'][i]
            sim.d.qvel[sim.va]=z['joint_vel_rad_s'][i]
            sim.mj.mj_forward(sim.m,sim.d);R=sim.d.xmat[sim.trunk].reshape(3,3)
            expected=z['amp_features'][i]
            sim.d.qvel[:3]=R@expected[36:39];sim.d.qvel[3:6]=expected[39:42]
            sim.mj.mj_forward(sim.m,sim.d);got=sim.state()
            fk=max(fk,float(abs(got[24:36]-expected[24:36]).max()))
            linear=max(linear,float(abs(got[36:39]-expected[36:39]).max()))
            gyro=max(gyro,float(abs(got[39:42]-expected[39:42]).max()))
            gravity=max(gravity,float(abs(got[42:45]-expected[42:45]).max()));frames+=1
assert fk<1.e-4 and gyro<1.e-6 and gravity<1.e-6 and linear<1.e-6,(fk,gyro,gravity,linear)
r=dict(actuator_torch_numpy_max_torque_error_Nm=error,actuator_physics_ticks=1000,
       fk_frames=frames,foot_position_max_error_m=fk,gyro_body_frame_max_error_rad_s=gyro,
       gravity_body_frame_max_error=gravity,linear_body_frame_max_error_m_s=linear,
       model_mass_kg=float(sim.m.body_mass.sum()),passed=True,
       note='Pose writes are restricted to isolated FK/actuator tests; validation.py never imposes rollout states.')
(a.bundle/'parity_result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
