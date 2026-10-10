"""No real device access. Contract and mapping tests use the locked artifact."""
import math
from pathlib import Path
import numpy as np
import pytest
import onnxruntime as ort
from mydog_policy.rs01_model36100_core import (
    Model36100Contract, Rs01Model36100Core, Guarded36100PolicyCore, EXPECTED_ONNX_SHA256)
from mydog_policy.rs01_model18000_core import Rs01Model18000Core
from mydog_policy.rs01_model23500_core import Rs01Model23500Core
from mydog_policy.rs01_model32750_core import Rs01Model32750Core

HIP_IDS = [0, 3, 6, 9]
SIDES = np.array([1., -1., 1., -1.])


def make_actor():
    path = Path(__file__).resolve().parents[1]/'resource/B36100.onnx'
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
    return Rs01Model36100Core(s, Model36100Contract.from_onnx_session(s, path))


def reference_guard5_map(target, action_scale, action_clip, limit_rad):
    """Independent transcription of training map_inward_targets (5 deg, all commands)."""
    ranges = action_scale[HIP_IDS]*action_clip
    hip = target[HIP_IDS]
    out = target.copy()
    out[HIP_IDS] = np.where(hip*SIDES < 0, hip*limit_rad/ranges, hip)
    return out


def test_locked_artifact_and_robust_contract():
    path = Path(__file__).resolve().parents[1]/'resource/B36100.onnx'
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
    c = Model36100Contract.from_onnx_session(s, path)
    assert c.raw['task'] == 'rs01_v22_guard5_robust'
    assert c.raw['v20']['mapping'] == 'omnidirectional_inward_scale_v1'
    assert c.raw['v20']['inward_target_rad'] == pytest.approx(math.radians(5.))
    with pytest.raises(RuntimeError, match='override'):
        Model36100Contract.from_onnx_session(s, path, '0'*64)


def test_unconditional_inward_map_matches_training_formula():
    actor = make_actor()
    c = actor.contract
    limit = math.radians(5.)
    rng = np.random.default_rng(20261010)
    for _ in range(500):
        target = c.default + rng.uniform(-.22, .22, 12)
        command = rng.uniform([-.4, -.3, -1.], [.6, .3, 1.])
        actual = actor.project_policy_target(target, command)
        expected = reference_guard5_map(target, c.action_scale, c.action_clip, limit)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        inward = (target[HIP_IDS]-c.default[HIP_IDS])*SIDES < 0
        assert np.all(np.abs(actual[HIP_IDS][inward]-c.default[HIP_IDS][inward]) <= limit+1e-12)
        np.testing.assert_allclose(actual[HIP_IDS][~inward], target[HIP_IDS][~inward], atol=0, rtol=0)


def test_mapping_is_command_independent_and_replaces_conditional_map():
    actor = make_actor()
    c = actor.contract
    target = c.default.copy()
    target[HIP_IDS] = np.array([-.15, .15, -.15, .15])  # all four hips inward
    straight = actor.project_policy_target(target, np.array([.2, 0., 0.]))
    lateral = actor.project_policy_target(target, np.array([0., .3, 0.]))
    turn = actor.project_policy_target(target, np.array([0., 0., 1.]))
    np.testing.assert_allclose(straight, lateral, atol=0, rtol=0)
    np.testing.assert_allclose(straight, turn, atol=0, rtol=0)
    conditional_lateral = Rs01Model18000Core.project_policy_target(actor, target, np.array([0., .3, 0.]))
    assert not np.allclose(lateral, conditional_lateral)
    np.testing.assert_allclose(conditional_lateral, target, atol=1e-12, rtol=0)
    assert Rs01Model36100Core._mix_direction_command is Rs01Model23500Core._mix_direction_command
    assert Rs01Model36100Core.project_policy_target is Rs01Model32750Core.project_policy_target
    assert Rs01Model36100Core.initialize_odometry_on_first_tick
    assert Rs01Model36100Core.target_reached_atol == 1e-12


def test_tick_and_guarded_step_stay_finite():
    actor = make_actor()
    c = actor.contract
    cmd = np.array([.2, 0., 0.])
    r = actor.tick(c.default, np.zeros(12), np.zeros(3), [0, 0, -1], 0., cmd)
    assert r['observation'].shape == (61,)
    assert r['phase'] == 0
    assert np.isfinite(r['target_policy']).all()
    core = Guarded36100PolicyCore(actor.session, c)
    obs = core.build_observation(0., np.zeros(3), np.zeros(3), [0, 0, -1], cmd, c.default, np.zeros(12), 0.)
    result = core.step(obs, c.default, np.zeros(12), np.full(12, 14.))
    assert np.isfinite(result['safe_target_policy']).all()
