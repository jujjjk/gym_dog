"""Same sensor-snapshot parity; no real robot I/O is imported or opened."""
import argparse
import json
import sys
from pathlib import Path
from sim2sim_v22 import V22Sim, FAST_MOVEMENTS
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'zhenji/mydog_ros2_ws/src/mydog_policy'))
from mydog_policy.rs01_model23500_core import Model23500Contract, Guarded23500PolicyCore
from mydog_policy.rs01_model930_core import Rs01ContinuousTorqueGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, default=ROOT/'artifacts/rs01_v22_sim2sim/B23500.onnx')
    p.add_argument('--scene', type=Path, default=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml')
    p.add_argument('--seconds', type=float, default=125.)
    p.add_argument('--sensor_profile', choices=('clean', 'normal', 'robust'), default='normal')
    args = p.parse_args()
    V22Sim.sensor_profile = args.sensor_profile
    sim = V22Sim(args.scene, args.policy, [0., 0., 0.])
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=opt, providers=['CPUExecutionProvider'])
    contract = Model23500Contract.from_onnx_session(session, args.policy)
    core = Guarded23500PolicyCore(session, contract)
    guard = Rs01ContinuousTorqueGuard(6., 14.)
    limits = guard.active_limits()
    core.reset(0., float(sim.sensor.yaw[0]))
    sequence = [('march', (0., 0., 0.))]
    for move in FAST_MOVEMENTS:
        sequence.extend([move, ('march', (0., 0., 0.))])
    maxima = dict(observation=0., action=0., limited_target=0., guard_target=0., guard_limit=0.)
    for i in range(min(round(args.seconds/.02), 6250)):
        command = sequence[i//250][1]
        sim.set_command(command, 1.)
        s = sim.sensor
        q, dq, gyro, gravity, torque = [x[0].numpy().copy() for x in (s.q, s.dq, s.gyro, s.gravity, s.torque)]
        obs = core.build_observation(i*.02, np.zeros(3), gyro, gravity, command, q, dq, float(s.yaw[0]))
        result = core.step(obs, q, dq, limits)
        limits = guard.update(np.maximum(abs(result['torque_info']['safe_pd_torque_nm']), abs(torque)), .02)
        sim.control_step()
        for key, a, b in [('observation', obs, sim.last_observation), ('action', result['action'], sim.action),
                          ('limited_target', core.actor.target, sim.limited_target),
                          ('guard_target', result['safe_target_policy'], sim.guard_target),
                          ('guard_limit', limits, sim.guard_limit)]:
            error = float(np.max(abs(np.asarray(a)-b)))
            if not np.isfinite(error):
                raise AssertionError('Non-finite ' + key)
            maxima[key] = max(maxima[key], error)
            if error > 2e-4:
                print(key, 'deployment', a, 'simulation', b, 'maxima', maxima)
                raise AssertionError('%s at step %d: %g' % (key, i, error))
        if sim.step_overspeed:
            raise AssertionError('Simulation overspeed')
    print(json.dumps(dict(steps=i+1, seconds=(i+1)*.02, sensor_profile=args.sensor_profile,
                          max_abs_errors=maxima, hardware_io=False), indent=2))


if __name__ == '__main__':
    main()
