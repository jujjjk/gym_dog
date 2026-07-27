from pathlib import Path
import sys

import isaacgym  # noqa: F401 - must precede torch in this workspace
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mujoko.rs01_go2.export_policy import (  # noqa: E402
    build_contract,
    movable_joint_names,
    sha256_file,
)
from mujoko.rs01_go2.sim2sim import (  # noqa: E402
    Rs01Go2Sim,
    compute_rs01_torques,
    load_scene_contract,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_sim2sim_config import (  # noqa: E402
    Rs01Go2Heading52Cfg,
    Rs01Go2Sim2SimRobustCfg,
)


def test_export_contract_binds_exact_new_rs01_urdf_and_physx_timing():
    contract = build_contract(
        "rs01_go2_sim2sim_robust",
        Rs01Go2Sim2SimRobustCfg,
        ROOT / "dummy_checkpoint.pt",
        ROOT / "dummy_policy.onnx",
    )
    simulator = contract["simulator"]
    assert contract["schema_version"] == 2
    urdf = Path(simulator["urdf"])
    assert urdf == ROOT / "dog_urdf/urdf/dog_rs01.urdf"
    assert simulator["urdf_sha256"] == sha256_file(urdf)
    assert contract["joint_names"] == list(
        Rs01Go2Sim2SimRobustCfg.rs01_actuator.runtime_dof_order
    )
    assert contract["joint_names"] != movable_joint_names(urdf)
    assert simulator["physx"]["physics_step_s"] == 0.005
    assert simulator["physx"]["contact_substeps"] == 2
    assert simulator["physx"]["contact_substep_s"] == 0.0025


def test_export_contract_contains_action_to_real_motor_mapping():
    contract = build_contract(
        "rs01_go2_sim2sim_heading52",
        Rs01Go2Heading52Cfg,
        ROOT / "dummy_checkpoint.pt",
        ROOT / "dummy_policy.onnx",
    )
    mapping = contract["motor_mapping"]
    assert [item["action_index"] for item in mapping] == list(range(12))
    assert [item["joint_name"] for item in mapping] == list(
        Rs01Go2Heading52Cfg.rs01_actuator.runtime_dof_order
    )
    assert [item["motor_id"] for item in mapping] == [
        "0x21", "0x22", "0x23",
        "0x11", "0x12", "0x13",
        "0x31", "0x32", "0x33",
        "0x41", "0x42", "0x43",
    ]
    assert [item["real_feedback_index"] for item in mapping] == [
        3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11,
    ]
    assert contract["real_motor_order"] == [
        "0x11", "0x12", "0x13",
        "0x21", "0x22", "0x23",
        "0x31", "0x32", "0x33",
        "0x41", "0x42", "0x43",
    ]


def test_mujoco_contact_contract_matches_calibrated_new_machine():
    contract = build_contract(
        "rs01_go2_sim2sim_robust",
        Rs01Go2Sim2SimRobustCfg,
        ROOT / "dummy_checkpoint.pt",
        ROOT / "dummy_policy.onnx",
    )
    mujoco_cfg = contract["simulator"]["mujoco"]
    assert mujoco_cfg["integration_timestep_s"] == 0.0025
    assert mujoco_cfg["integration_substeps_per_motor_step"] == 2
    assert mujoco_cfg["contact_solref"] == [0.0065, 1.0]
    assert mujoco_cfg["contact_solimp"] == [
        0.99,
        0.999,
        0.0001,
        0.5,
        2.0,
    ]
    assert (
        mujoco_cfg["integration_timestep_s"]
        * mujoco_cfg["integration_substeps_per_motor_step"]
        == contract["control"]["physics_dt_s"]
    )


def test_heading52_export_contract_uses_sin_cos_layout():
    contract = build_contract(
        "rs01_go2_sim2sim_heading52",
        Rs01Go2Heading52Cfg,
        ROOT / "dummy_checkpoint.pt",
        ROOT / "dummy_policy.onnx",
    )
    observations = contract["observations"]
    assert contract["dimensions"]["observations"] == 52
    assert observations["heading_representation"] == "sin_cos"
    assert observations["layout"][-1] == ["heading_error_sin_cos", 2]


def test_scene_metadata_loader_exposes_rs01_machine_binding(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
<mujoco model="test">
  <custom>
    <text name="rs01_source_urdf_sha256" data="abc123"/>
    <numeric name="rs01_motor_step_s" data="0.005"/>
  </custom>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    metadata = load_scene_contract(scene)
    assert metadata["rs01_source_urdf_sha256"] == "abc123"
    assert metadata["rs01_motor_step_s"] == "0.005"


def test_mujoco_motor_limit_is_electromagnetic_not_second_net_clip():
    raw, motor, applied = compute_rs01_torques(
        response_target=np.array([1.0, -1.0]),
        position=np.zeros(2),
        velocity=np.array([-1.0, 1.0]),
        kp=np.full(2, 40.0),
        kd=np.zeros(2),
        peak_limit=17.0,
        coulomb_friction=np.full(2, 0.2),
        friction_smoothing=0.05,
    )
    assert np.max(np.abs(raw)) == 40.0
    assert np.max(np.abs(motor)) <= 17.0
    # Friction opposes velocity, so net applied torque may differ from the
    # 17 N.m electromagnetic limit. It must not be clipped a second time.
    assert applied[0] > motor[0]
    assert applied[1] < motor[1]


def test_mujoco_world_velocity_is_rotated_into_policy_body_frame():
    class FakeMujoco:
        class mjtObj:
            mjOBJ_BODY = 1

        @staticmethod
        def mj_objectVelocity(model, data, kind, body, output, local):
            assert local == 0
            output[:] = [0.01, -0.02, 0.4, 0.2, -0.1, 0.03]

    original = sys.modules["mujoco"].mj_objectVelocity
    try:
        import mujoco

        mujoco.mj_objectVelocity = FakeMujoco.mj_objectVelocity
        sim = object.__new__(Rs01Go2Sim)
        sim.model = object()
        sim.data = type(
            "Data",
            (),
            {"qpos": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
        )()
        sim.trunk_body = 0
        linear, angular = sim.base_velocity_body()
    finally:
        mujoco.mj_objectVelocity = original

    assert np.allclose(linear, [0.2, -0.1, 0.03])
    assert np.allclose(angular, [0.01, -0.02, 0.4])
