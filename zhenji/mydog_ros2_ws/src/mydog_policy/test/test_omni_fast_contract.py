import copy
import ast
from pathlib import Path

import numpy as np

from mydog_policy.omni_fast_contract import (
    DEPLOYMENT_LIMITS,
    PRESETS,
    validate_metadata,
)


def contract():
    return {
        "schema_version": 1,
        "task": "FanfanOmniYawDriftCleanCfg",
        "dimensions": {"observations": 52, "actions": 12},
        "control": {
            "sim_dt": 0.005,
            "decimation": 4,
            "stiffness": [60.0, 70.0, 70.0] * 4,
            "damping": [1.2, 1.6, 1.6] * 4,
            "action_scale": [0.092, 0.215, 0.215, 0.092, 0.215, 0.215,
                             0.092, 0.235, 0.235, 0.092, 0.235, 0.235],
            "output_transform": "tanh",
        },
        "commands": {"ranges": {
            "lin_vel_x": [-0.12, 0.46],
            "lin_vel_y": [-0.12, 0.12],
            "ang_vel_yaw": [-0.85, 0.85],
        }},
        "gait": {"period": 0.54, "calf_amplitude": -0.3},
    }


def test_contract_is_accepted():
    assert validate_metadata(contract()) is True


def test_wrong_damping_is_rejected():
    invalid = copy.deepcopy(contract())
    invalid["control"]["damping"][0] = 1.2
    try:
        validate_metadata(invalid)
    except ValueError as exc:
        assert "Kd" in str(exc)
    else:
        raise AssertionError("invalid Kd was accepted")


def test_presets_stay_inside_deployment_envelope():
    for name, (vx, vy, yaw) in PRESETS.items():
        assert DEPLOYMENT_LIMITS["vx"][0] <= vx <= DEPLOYMENT_LIMITS["vx"][1], name
        assert DEPLOYMENT_LIMITS["vy"][0] <= vy <= DEPLOYMENT_LIMITS["vy"][1], name
        assert DEPLOYMENT_LIMITS["yaw"][0] <= yaw <= DEPLOYMENT_LIMITS["yaw"][1], name


def _safe_target_limiter_class():
    source = Path(__file__).parents[1] / "mydog_policy" / "mydog_policy_node.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SafeTargetLimiter"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["SafeTargetLimiter"]


def test_model_without_absolute_error_limits_uses_torque_budget():
    limiter = _safe_target_limiter_class()()
    kp = np.asarray([60.0, 70.0, 70.0] * 4, dtype=np.float32)
    common = {
        "q_raw": np.full(12, 0.5, dtype=np.float32),
        "q_current": np.zeros(12, dtype=np.float32),
        "dt": 0.02,
        "kp": kp,
        "torque_budget_nm": 8.0,
        "err_limit_safety_factor": 1.0,
        "max_target_rate_rad_s": 1000.0,
        "max_target_accel_rad_s2": 100000.0,
        "err_limit_mul": np.ones(12, dtype=np.float32),
        "target_rate_mul": np.ones(12, dtype=np.float32),
        "target_accel_mul": np.ones(12, dtype=np.float32),
    }
    _, info_none = limiter.limit(absolute_error_limit_rad=None, **common)
    limiter.reset(np.zeros(12, dtype=np.float32))
    _, info_legacy = limiter.limit(
        absolute_error_limit_rad=np.full(12, np.inf, dtype=np.float32), **common
    )
    expected = 8.0 / kp
    np.testing.assert_allclose(info_none["err_limit"], expected, atol=1e-6)
    np.testing.assert_allclose(info_legacy["err_limit"], expected, atol=1e-6)


def test_omni_fast_launch_binds_all_torque_budget_parameters():
    launch_file = Path(__file__).parents[1] / "launch" / "sim2real_omni_fast.launch.py"
    source = launch_file.read_text(encoding="utf-8")
    for parameter in (
        '"motor_torque_limit_nm"',
        '"torque_limit_nm"',
        '"torque_safety_budget_nm"',
        '"expected_active_torque_budget_nm"',
    ):
        assert parameter in source


def test_yaw_clean_launch_uses_requested_initial_safety_limits():
    launch_file = Path(__file__).parents[1] / "launch" / "sim2real_omni_fast.launch.py"
    source = launch_file.read_text(encoding="utf-8")
    assert '"cmd_min_x": "-0.12"' in source
    assert '"cmd_max_x": "0.12"' in source
    assert '"cmd_min_y": "-0.025"' in source
    assert '"cmd_max_y": "0.025"' in source
    assert '"cmd_min_yaw": "-0.25"' in source
    assert '"cmd_max_yaw": "0.25"' in source
    assert 'DeclareLaunchArgument("motor_torque_limit_nm", default_value="8.0")' in source
    assert 'DeclareLaunchArgument("model_kp_scale", default_value="1.0")' in source
    assert 'DeclareLaunchArgument("model_kd_scale", default_value="1.0")' in source
    assert '"enable_zero_cmd_stand_protection": "true"' in source
    assert '"zero_cmd_stand_x_threshold": "0.01"' in source
    assert '"zero_cmd_stand_y_threshold": "0.01"' in source
    assert '"zero_cmd_stand_yaw_threshold": "0.03"' in source
    assert '"enable_policy_action_cmd_gate": "true"' in source
    assert '"deployment_gait_phase_period_scale": "1.35"' in source
    assert '"policy_action_cmd_gate_start_ratio": "0.05"' in source
    assert '"policy_action_cmd_gate_full_ratio": "0.5"' in source
    assert '"policy_action_cmd_gate_max_scale": "0.9"' in source
    assert '"reset_gait_phase_on_command_start": "true"' in source
    assert 'fanfan_yaw_clean_5100.onnx' in source
    assert 'fanfan_yaw_clean_sim2real.csv' in source
