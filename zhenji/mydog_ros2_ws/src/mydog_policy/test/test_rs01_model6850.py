import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'zhenji/mydog_ros2_ws/src/mydog_policy'))
sys.path.insert(0,str(ROOT/'mujoko/rs01_go2'))
from mydog_policy.rs01_model6850_core import Model6850Contract,Rs01Model6850Core,validate_sample_times
from sim2sim_sensor_sync import SensorSyncSim
from sim2sim import quaternion_rotation_matrix,roll_pitch_yaw


def test_a6850_observation_action_and_target_match_mujoco():
    path=ROOT/'artifacts/rs01_v15_support/stand_only_6850.onnx'
    sim=SensorSyncSim(ROOT/'artifacts/rs01_v14_actuator_parity/scene.xml',path,[0,0,0])
    core=Rs01Model6850Core(sim.session,Model6850Contract.from_onnx_session(sim.session,path))
    core.reset(0.)
    cases=[([0,0,0],0),([0,0,0],1),([.2,0,0],1),([-.2,0,0],1),
           ([0,.2,0],1),([0,-.2,0],1),([0,0,.3],1),([0,0,-.3],1),
           ([.3,.1,.25],1),([-.3,-.1,-.25],1),([0,0,0],0)]
    errors=np.zeros(3)
    for command,gait in cases:
        for _ in range(50):
            sim.set_command(command,gait)
            q=sim.data.qpos[sim.qpos_indices];dq=sim.data.qvel[sim.qvel_indices]
            yaw=roll_pitch_yaw(sim.data.qpos[3:7])[2]
            gravity=quaternion_rotation_matrix(sim.data.qpos[3:7]).T@np.array([0,0,-1])
            result=core.tick(q,dq,sim.base_velocity_body()[1],gravity,yaw,command,gait)
            sim.control_step()
            for i,(a,b) in enumerate([(result['observation'],sim.last_observation),
                                       (result['action'],sim.action),(result['target_policy'],sim.limited_target)]):
                errors[i]=max(errors[i],np.max(np.abs(a-b)))
    print('550-frame parity maxima: observation/action/target',errors)
    assert errors[0]<1e-5 and errors[1]<1e-4 and errors[2]<1e-5,errors


def test_feedback_timing_rejects_stale_future_skew_and_duplicates():
    validate_sample_times(1.,np.full(12,.99),.995)
    for now,motor,imu,last in [
        (1.,np.full(12,.9),.995,None),
        (1.,np.full(12,1.01),.995,None),
        (1.,np.r_[np.full(11,.99),.97],.995,None),
        (1.,np.full(12,.99),.995,np.full(12,.99)),
        (1.,np.full(12,.99),float('nan'),None),
    ]:
        try:validate_sample_times(now,motor,imu,last)
        except ValueError:continue
        raise AssertionError('Invalid timing accepted')


def test_wrong_artifact_cannot_bypass_hash():
    path=ROOT/'artifacts/rs01_v15_support/stand_only_6850.onnx'
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    for override in ('','wrong'):
        try:Model6850Contract.from_onnx_session(session,path,override)
        except RuntimeError:continue
        raise AssertionError('Hash override accepted')
