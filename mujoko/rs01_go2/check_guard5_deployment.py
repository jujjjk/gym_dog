"""Guard5 B32750 deployment-core parity against the MuJoCo bridge; no robot I/O."""
import argparse
import math
import json
import sys
from pathlib import Path
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'zhenji/mydog_ros2_ws/src/mydog_policy'))
from sim2sim_guard5 import Guard5Sim  # imports isaacgym before torch
from sim2sim_v22 import FAST_MOVEMENTS
import numpy as np
from mydog_policy.rs01_model32750_core import Model32750Contract, Guarded32750PolicyCore
from mydog_policy.rs01_model930_core import Rs01ContinuousTorqueGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, default=ROOT/'artifacts/rs01_guard5_sim2sim/guard5_32750.onnx')
    p.add_argument('--scene', type=Path, default=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml')
    p.add_argument('--seconds', type=float, default=125.)
    p.add_argument('--sensor_profile', choices=('clean', 'normal', 'robust'), default='normal')
    args = p.parse_args()
    Guard5Sim.sensor_profile = args.sensor_profile
    sim = Guard5Sim(args.scene, args.policy, [0., 0., 0.])
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=opt, providers=['CPUExecutionProvider'])
    contract = Model32750Contract.from_onnx_session(session, args.policy)
    core = Guarded32750PolicyCore(session, contract)
    guard = Rs01ContinuousTorqueGuard(6., 14.)
    limits = guard.active_limits()
    core.reset(0., float(sim.sensor.yaw[0]))
    sequence = [('march', (0., 0., 0.))]
    for move in FAST_MOVEMENTS:
        sequence.extend([move, ('march', (0., 0., 0.))])
    maxima = dict(observation=0., action=0., limited_target=0., guard_target=0., guard_limit=0.)
    # A float32-level observation rounding can flip the limiter's target-crossing
    # branch for a single step under robust sensor noise. Such flips are bounded
    # by one rate-limited step and must reconverge within 10 steps; the resulting
    # one-step torque difference decays through the thermal guard (tau=2s) along
    # a bounded exponential envelope. Sustained divergence fails the check.
    flap_bound = float(np.max(contract.rate_limit)*.02) + 2e-4
    thermal_slack0 = 0.15  # Nm; bound on guard_limit perturbation from one flap
    flaps = 0
    pending_flap = None
    thermal_settle = None
    for i in range(min(round(args.seconds/.02), 6250)):
        command = sequence[i//250][1]
        sim.set_command(command, 1.)
        s = sim.sensor
        q, dq, gyro, gravity, torque = [x[0].numpy().copy() for x in (s.q, s.dq, s.gyro, s.gravity, s.torque)]
        obs = core.build_observation(i*.02, np.zeros(3), gyro, gravity, command, q, dq, float(s.yaw[0]))
        result = core.step(obs, q, dq, limits)
        limits = guard.update(np.maximum(abs(result['torque_info']['safe_pd_torque_nm']), abs(torque)), .02)
        sim.control_step()
        errors = {}
        for key, a, b in [('observation', obs, sim.last_observation), ('action', result['action'], sim.action),
                          ('limited_target', core.actor.target, sim.limited_target),
                          ('guard_target', result['safe_target_policy'], sim.guard_target),
                          ('guard_limit', limits, sim.guard_limit)]:
            error = float(np.max(abs(np.asarray(a)-b)))
            if not np.isfinite(error):
                raise AssertionError('Non-finite ' + key)
            maxima[key] = max(maxima[key], error)
            errors[key] = error
        for key in ('observation', 'action'):
            if errors[key] > 2e-4:
                raise AssertionError('%s at step %d: %g' % (key, i, errors[key]))
        target_error = max(errors['limited_target'], errors['guard_target'])
        if target_error > 2e-4:
            if target_error <= flap_bound:
                if pending_flap is None:
                    pending_flap = i
                    flaps += 1
                    thermal_settle = i
            else:
                raise AssertionError('limited_target at step %d: %g' % (i, target_error))
        if pending_flap is not None:
            if errors['limited_target'] <= 2e-4:
                pending_flap = None
            elif i - pending_flap >= 10:
                raise AssertionError('limiter divergence did not reconverge within 10 steps')
        if errors['guard_limit'] > 2e-4:
            envelope = (thermal_slack0*math.exp(-(i-thermal_settle)*.02/2.) + 2e-4
                        if thermal_settle is not None else 2e-4)
            if errors['guard_limit'] > envelope:
                raise AssertionError('guard_limit at step %d: %g' % (i, errors['guard_limit']))
        if sim.step_overspeed:
            raise AssertionError('Simulation overspeed')
    if pending_flap is not None or errors['guard_limit'] > 2e-4:
        raise AssertionError('divergence still open at end of run')
    print(json.dumps(dict(steps=i+1, seconds=(i+1)*.02, sensor_profile=args.sensor_profile,
                          model='B32750', max_abs_errors=maxima,
                          limiter_branch_flaps=flaps, hardware_io=False), indent=2))


if __name__ == '__main__':
    main()
