"""Offline artifact validation: no ROS, HTTP, serial or actuator access."""
import argparse, json, time
import numpy as np
import onnxruntime as ort
from .rs01_model36100_core import Model36100Contract, Guarded36100PolicyCore, EXPECTED_ONNX_SHA256


def main():
    p = argparse.ArgumentParser(); p.add_argument('onnx_path'); a = p.parse_args()
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    s = ort.InferenceSession(a.onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
    c = Model36100Contract.from_onnx_session(s, a.onnx_path); core = Guarded36100PolicyCore(s, c)
    durations = []
    for i in range(1000):
        start = time.perf_counter()
        obs = core.build_observation(i*.02, np.zeros(3), np.zeros(3), [0, 0, -1], np.zeros(3), c.default, np.zeros(12), 0.)
        result = core.step(obs, c.default, np.zeros(12), np.full(12, 14.))
        if not np.isfinite(result['safe_target_policy']).all():
            raise RuntimeError('Nonfinite target')
        durations.append((time.perf_counter()-start)*1000)
    print(json.dumps(dict(model='B36100', sha256=EXPECTED_ONNX_SHA256, policy_hz=1/c.policy_dt,
        observations=len(obs), actions=len(result['action']), finite=True,
        inward_guard='omnidirectional_inward_scale_v1_5deg',
        core_compute_p95_ms=float(np.percentile(durations[10:], 95)),
        note='Synthetic-input CPU timing, NOT actual motor send frequency', hardware_validated=False), indent=2))


if __name__ == '__main__':
    main()
