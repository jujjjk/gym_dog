"""Convert dog_rs01.urdf into a floating-base MuJoCo Sim2Sim scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import onnxruntime as ort


METADATA_KEY = "rs01_go2_deployment_config"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values_text(values):
    return " ".join(str(value) for value in values)


def indent_xml(element, level=0):
    """Python 3.8-compatible equivalent of ElementTree.indent."""
    indentation = "\n" + level * "  "
    child_indentation = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indentation
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indentation
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def load_contract(policy):
    metadata = ort.InferenceSession(
        str(policy),
        providers=["CPUExecutionProvider"],
    ).get_modelmeta().custom_metadata_map
    if METADATA_KEY not in metadata:
        raise RuntimeError(f"ONNX is missing {METADATA_KEY}")
    contract = json.loads(metadata[METADATA_KEY])
    if contract.get("schema_version") != 1:
        raise RuntimeError(
            f"Unsupported schema version {contract.get('schema_version')}"
        )
    return contract


def add_floating_world(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.find("link[@name='world']") is not None:
        raise RuntimeError("Source URDF already contains a world link")
    world = ET.Element("link", {"name": "world"})
    floating = ET.Element(
        "joint",
        {"name": "floating_base", "type": "floating"},
    )
    ET.SubElement(floating, "parent", {"link": "world"})
    ET.SubElement(floating, "child", {"link": "Trunk"})
    root.insert(0, world)
    root.insert(1, floating)
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    policy = args.policy.resolve()
    contract = load_contract(policy)
    urdf = Path(contract["simulator"]["urdf"]).resolve()
    expected_urdf_sha256 = contract["simulator"].get("urdf_sha256")
    actual_urdf_sha256 = sha256_file(urdf)
    if (
        expected_urdf_sha256 is not None
        and actual_urdf_sha256 != expected_urdf_sha256
    ):
        raise RuntimeError(
            "The RS01 URDF changed after policy export: "
            f"expected {expected_urdf_sha256}, got {actual_urdf_sha256}"
        )
    mujoco_cfg = contract["simulator"].get("mujoco", {})
    integration_timestep_s = float(
        mujoco_cfg.get(
            "integration_timestep_s",
            contract["control"]["physics_dt_s"],
        )
    )
    integration_substeps = int(
        mujoco_cfg.get("integration_substeps_per_motor_step", 1)
    )
    motor_step_s = float(contract["control"]["physics_dt_s"])
    if integration_substeps < 1 or abs(
        integration_timestep_s * integration_substeps - motor_step_s
    ) > 1.0e-12:
        raise RuntimeError(
            "MuJoCo integration timing must divide the 5 ms RS01 motor step"
        )
    contact_solref = mujoco_cfg.get("contact_solref", [0.02, 1.0])
    contact_solimp = mujoco_cfg.get(
        "contact_solimp", [0.9, 0.95, 0.001, 0.5, 2.0]
    )
    friction = mujoco_cfg.get("friction", [1.0, 0.005, 0.0001])
    contact_attributes = {
        "friction": values_text(friction),
        "condim": str(int(mujoco_cfg.get("contact_dimension", 3))),
        "solref": values_text(contact_solref),
        "solimp": values_text(contact_solimp),
        "margin": str(float(mujoco_cfg.get("contact_margin_m", 0.0))),
        "gap": str(float(mujoco_cfg.get("contact_gap_m", 0.0))),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source_tree = add_floating_world(urdf)
    temporary_urdf = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".urdf",
            prefix=".rs01_sim2sim_",
            dir=urdf.parent,
            delete=False,
        ) as handle:
            temporary_urdf = Path(handle.name)
            source_tree.write(handle, encoding="utf-8", xml_declaration=True)
        converted = mujoco.MjModel.from_xml_path(str(temporary_urdf))
    finally:
        if temporary_urdf is not None:
            temporary_urdf.unlink(missing_ok=True)

    intermediate = output.with_suffix(".converted.xml")
    mujoco.mj_saveLastXML(str(intermediate), converted)
    tree = ET.parse(intermediate)
    root = tree.getroot()
    root.set("model", "rs01_go2_sim2sim")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep=str(integration_timestep_s),
        gravity="0 0 -9.81",
        integrator=str(mujoco_cfg.get("integrator", "implicitfast")),
        solver=str(mujoco_cfg.get("solver", "Newton")),
    )

    worldbody = root.find("worldbody")
    trunk = worldbody.find("body[@name='Trunk']")
    if trunk is None:
        raise RuntimeError("Converted model has no Trunk body")
    trunk.set(
        "pos",
        " ".join(
            str(value)
            for value in contract["initial_state"]["base_position_m"]
        ),
    )

    for geom in trunk.iter("geom"):
        geom.set("contype", "1")
        geom.set("conaffinity", "0")
        geom.attrib.update(contact_attributes)
    for joint in trunk.iter("joint"):
        if joint.get("name") != "floating_base":
            joint.set("armature", "0")
            joint.set("damping", "0")

    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": "ground",
                "type": "plane",
                "size": "20 20 0.1",
                "rgba": "0.78 0.80 0.82 1",
                **contact_attributes,
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {
                "name": "sun",
                "pos": "0 0 3",
                "dir": "0 0 -1",
                "directional": "true",
            },
        ),
    )

    actuator = ET.SubElement(root, "actuator")
    for joint_name in contract["joint_names"]:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{joint_name}_motor",
                "joint": joint_name,
                "gear": "1",
                # Electromagnetic torque is already limited in sim2sim.py.
                # Do not clip the post-friction net joint torque a second
                # time; PhysX applies that independently computed value.
                "ctrllimited": "false",
            },
        )

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(
        custom,
        "text",
        {
            "name": "rs01_source_urdf_sha256",
            "data": actual_urdf_sha256,
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": "rs01_motor_step_s",
            "data": str(motor_step_s),
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": "rs01_integration_substeps",
            "data": str(integration_substeps),
        },
    )

    indent_xml(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    intermediate.unlink(missing_ok=True)

    check = mujoco.MjModel.from_xml_path(str(output))
    expected_names = set(contract["joint_names"])
    actual_names = {
        mujoco.mj_id2name(
            check,
            mujoco.mjtObj.mjOBJ_JOINT,
            index,
        )
        for index in range(check.njnt)
    }
    if not expected_names.issubset(actual_names):
        raise RuntimeError(
            f"Scene is missing joints: {sorted(expected_names - actual_names)}"
        )
    if (check.nq, check.nv, check.nu) != (19, 18, 12):
        raise RuntimeError(
            f"Unexpected model dimensions nq={check.nq}, "
            f"nv={check.nv}, nu={check.nu}"
        )
    if abs(check.opt.timestep - integration_timestep_s) > 1.0e-12:
        raise RuntimeError(
            f"Generated timestep {check.opt.timestep} does not match "
            f"{integration_timestep_s}"
        )

    print(f"scene={output}")
    print(f"nq={check.nq}")
    print(f"nv={check.nv}")
    print(f"nu={check.nu}")
    print(f"motor_step_s={motor_step_s}")
    print(f"integration_timestep_s={check.opt.timestep}")
    print(f"integration_substeps={integration_substeps}")
    print(f"urdf_sha256={actual_urdf_sha256}")


if __name__ == "__main__":
    main()
