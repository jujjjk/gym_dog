"""Short deterministic isolation/parity test, without opening a viewer."""
from play_v22_multi import IndependentSim, display_world, update_display, ROOT
import numpy as np
import torch


def main():
    scene = ROOT/'artifacts/rs01_v20_sim2sim/scene.xml'
    policy = ROOT/'artifacts/rs01_v22_sim2sim/B23500.onnx'
    a = IndependentSim(scene, policy, 16, 'normal')
    b = IndependentSim(scene, policy, 17, 'normal')
    reference = IndependentSim(scene, policy, 17, 'normal')
    assert a.model is not b.model and a.data is not b.data
    assert a.session is b.session  # Stateless weights, not state or history.
    for key in ('action', 'target_rate', 'guard_rms_sq', 'limited_target'):
        assert not np.shares_memory(getattr(a, key), getattr(b, key)), key
    assert a.sensor.q.data_ptr() != b.sensor.q.data_ptr()
    saved = b.data.qpos.copy()
    a.data.qpos[0] += .03
    assert np.array_equal(saved, b.data.qpos)
    global_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        for _ in range(25):
            a.set_command([.2, 0., 0.], 1.)
            a.control_step()
            b.control_step()
            reference.control_step()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert np.array_equal(b.data.qpos, reference.data.qpos)
    assert np.array_equal(b.action, reference.action)
    assert np.array_equal(b.guard_rms_sq, reference.guard_rms_sq)
    model, data, offsets = display_world(scene, [a, b], 2.5)
    update_display(model, data, offsets, [a, b])
    assert np.allclose(data.qpos[:3], a.data.qpos[:3]+offsets[0])
    assert np.allclose(data.qpos[a.model.nq:a.model.nq+3], b.data.qpos[:3]+offsets[1])
    assert not np.allclose(a.data.qpos, b.data.qpos)
    print('PASS: independent physics, actuator history, sensor RNG; interleaved/solo exact parity; grid mapping')


if __name__ == '__main__':
    main()
