"""Offline same-state deployment/MuJoCo parity; never opens robot I/O."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from sim2sim_v20 import V20Sim,MOVEMENTS
from sim2sim import quaternion_rotation_matrix,roll_pitch_yaw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'zhenji/mydog_ros2_ws/src/mydog_policy'))
from mydog_policy.rs01_model18000_core import Model18000Contract,Guarded18000PolicyCore
from mydog_policy.rs01_model930_core import Rs01ContinuousTorqueGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--seconds',type=float,default=125.)
    args=p.parse_args();sim=V20Sim(args.scene,args.policy,[0.,0.,0.])
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.policy),sess_options=opt,providers=['CPUExecutionProvider'])
    c=Model18000Contract.from_onnx_session(session,args.policy);core=Guarded18000PolicyCore(session,c)
    guard=Rs01ContinuousTorqueGuard(6.,14.);limits=guard.active_limits()
    core.reset(0.,0.);sequence=[('march',(0.,0.,0.))]
    for move in MOVEMENTS:sequence.extend([move,('march',(0.,0.,0.))])
    maxima=dict(observation=0.,action=0.,limited_target=0.,guard_target=0.,guard_limit=0.)
    for i in range(min(round(args.seconds/.02),6250)):
        command=sequence[i//250][1];sim.set_command(command,1.)
        q=sim.data.qpos[sim.qpos_indices].copy();dq=sim.data.qvel[sim.qvel_indices].copy()
        _,gyro=sim.base_velocity_body();gravity=quaternion_rotation_matrix(sim.data.qpos[3:7]).T@np.array([0.,0.,-1.])
        yaw=roll_pitch_yaw(sim.data.qpos[3:7])[2]
        obs=core.build_observation(i*.02,np.zeros(3),gyro,gravity,command,q,dq,yaw)
        result=core.step(obs,q,dq,limits)
        limits=guard.update(np.maximum(abs(result['torque_info']['safe_pd_torque_nm']),abs(sim.last_motor_torque)),.02)
        sim.control_step()
        for key,a,b in [('observation',obs,sim.last_observation),('action',result['action'],sim.action),
                        ('limited_target',core.actor.target,sim.limited_target),
                        ('guard_target',result['safe_target_policy'],sim.guard_target),
                        ('guard_limit',limits,sim.guard_limit)]:
            maxima[key]=max(maxima[key],float(np.max(abs(np.asarray(a)-b))))
        if sim.step_overspeed:raise RuntimeError('Simulation overspeed')
    print(json.dumps(dict(steps=i+1,seconds=(i+1)*.02,max_abs_errors=maxima,hardware_io=False),indent=2))
    for key,value in maxima.items():
        if value>2e-4:raise AssertionError('%s parity error %g'%(key,value))


if __name__=='__main__':main()
