#!/usr/bin/env python3
"""Patch fanfan URDF inertials by scaling structure mass and adding RS01 motors."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


LEG_PREFIXES = ("FR", "FL", "RR", "RL")
INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def parse_vec(text: str | None, default=(0.0, 0.0, 0.0)) -> list[float]:
    if not text:
        return [float(x) for x in default]
    return [float(x) for x in text.split()]


def fmt_float(value: float) -> str:
    return f"{value:.12g}"


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def vec_scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def vec_dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def mat_zero() -> list[list[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(3)]


def mat_eye() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def mat_scale(a: list[list[float]], s: float) -> list[list[float]]:
    return [[a[i][j] * s for j in range(3)] for i in range(3)]


def inertia_from_node(node: ET.Element) -> list[list[float]]:
    ixx = float(node.attrib["ixx"])
    ixy = float(node.attrib.get("ixy", 0.0))
    ixz = float(node.attrib.get("ixz", 0.0))
    iyy = float(node.attrib["iyy"])
    iyz = float(node.attrib.get("iyz", 0.0))
    izz = float(node.attrib["izz"])
    return [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]


def inertia_to_dict(mat: list[list[float]]) -> dict[str, float]:
    return {
        "ixx": mat[0][0],
        "ixy": mat[0][1],
        "ixz": mat[0][2],
        "iyy": mat[1][1],
        "iyz": mat[1][2],
        "izz": mat[2][2],
    }


def write_inertia(node: ET.Element, mat: list[list[float]]) -> None:
    vals = inertia_to_dict(mat)
    for key in INERTIA_KEYS:
        node.set(key, fmt_float(vals[key]))


def parallel_axis(mass: float, r: list[float]) -> list[list[float]]:
    rr = vec_dot(r, r)
    out = mat_zero()
    for i in range(3):
        for j in range(3):
            out[i][j] = mass * ((rr if i == j else 0.0) - r[i] * r[j])
    return out


def sphere_inertia(mass: float, radius: float) -> list[list[float]]:
    value = 0.4 * mass * radius * radius
    return [[value, 0.0, 0.0], [0.0, value, 0.0], [0.0, 0.0, value]]


def combine_bodies(bodies: list[dict]) -> tuple[float, list[float], list[list[float]]]:
    total_mass = sum(body["mass"] for body in bodies)
    if total_mass <= 0.0:
        raise ValueError("Cannot combine bodies with non-positive total mass")
    com = [0.0, 0.0, 0.0]
    for body in bodies:
        com = vec_add(com, vec_scale(body["com"], body["mass"]))
    com = vec_scale(com, 1.0 / total_mass)

    inertia = mat_zero()
    for body in bodies:
        shifted = mat_add(body["inertia"], parallel_axis(body["mass"], vec_sub(body["com"], com)))
        inertia = mat_add(inertia, shifted)
    return total_mass, com, inertia


def get_link_map(root: ET.Element) -> dict[str, ET.Element]:
    return {link.attrib["name"]: link for link in root.findall("link")}


def get_joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {joint.attrib["name"]: joint for joint in root.findall("joint")}


def get_inertial(link: ET.Element) -> ET.Element | None:
    return link.find("inertial")


def inertial_data(link: ET.Element) -> dict | None:
    inertial = get_inertial(link)
    if inertial is None:
        return None
    mass_node = inertial.find("mass")
    inertia_node = inertial.find("inertia")
    origin_node = inertial.find("origin")
    if mass_node is None or inertia_node is None:
        return None
    return {
        "mass": float(mass_node.attrib["value"]),
        "com": parse_vec(origin_node.attrib.get("xyz") if origin_node is not None else None),
        "rpy": parse_vec(origin_node.attrib.get("rpy") if origin_node is not None else None),
        "inertia": inertia_from_node(inertia_node),
    }


def set_inertial(link: ET.Element, mass: float, com: list[float], inertia: list[list[float]]) -> None:
    inertial = get_inertial(link)
    if inertial is None:
        raise ValueError(f"Link {link.attrib['name']} has no inertial")
    mass_node = inertial.find("mass")
    inertia_node = inertial.find("inertia")
    origin_node = inertial.find("origin")
    if mass_node is None or inertia_node is None:
        raise ValueError(f"Link {link.attrib['name']} has incomplete inertial")
    if origin_node is None:
        origin_node = ET.SubElement(inertial, "origin", {"rpy": "0 0 0"})
    mass_node.set("value", fmt_float(mass))
    origin_node.set("xyz", " ".join(fmt_float(x) for x in com))
    origin_node.set("rpy", origin_node.attrib.get("rpy", "0 0 0"))
    write_inertia(inertia_node, inertia)


def motor_plan(root: ET.Element, motor_mass: float, motor_radius: float, knee_on_calf: bool) -> dict[str, list[dict]]:
    joints = get_joint_map(root)
    additions: dict[str, list[dict]] = {}
    motor_inertia = sphere_inertia(motor_mass, motor_radius)
    for leg in LEG_PREFIXES:
        additions.setdefault(f"{leg}_hip", []).append(
            {"name": f"{leg}_hip_motor", "mass": motor_mass, "com": [0.0, 0.0, 0.0], "inertia": motor_inertia}
        )
        additions.setdefault(f"{leg}_thigh", []).append(
            {"name": f"{leg}_thigh_motor", "mass": motor_mass, "com": [0.0, 0.0, 0.0], "inertia": motor_inertia}
        )
        calf_joint = joints.get(f"{leg}_calf_joint")
        knee_com = [0.0, 0.0, -0.156]
        if calf_joint is not None and calf_joint.find("origin") is not None:
            knee_com = parse_vec(calf_joint.find("origin").attrib.get("xyz"))
        target_link = f"{leg}_calf" if knee_on_calf else f"{leg}_thigh"
        additions.setdefault(target_link, []).append(
            {"name": f"{leg}_knee_motor", "mass": motor_mass, "com": knee_com, "inertia": motor_inertia}
        )
    return additions


def total_mass(root: ET.Element) -> float:
    total = 0.0
    for link in root.findall("link"):
        data = inertial_data(link)
        if data is not None:
            total += data["mass"]
    return total


def patch_urdf(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    tree = ET.parse(input_path)
    root = tree.getroot()
    links = get_link_map(root)
    warnings: list[str] = []
    link_reports: dict[str, dict] = {}

    original_total = total_mass(root)
    additions = motor_plan(root, args.motor_mass, args.motor_radius, args.knee_motor_on_calf)
    total_motor_mass = sum(motor["mass"] for motors in additions.values() for motor in motors)

    for link in root.findall("link"):
        name = link.attrib["name"]
        data = inertial_data(link)
        if data is None:
            warnings.append(f"Link {name} has no complete inertial; left unchanged.")
            continue
        old_mass = data["mass"]
        old_com = data["com"]
        old_inertia = data["inertia"]
        scaled_mass = old_mass * args.base_mass_scale
        scaled_inertia = mat_scale(old_inertia, args.base_mass_scale)
        bodies = [{"name": "scaled_structure", "mass": scaled_mass, "com": old_com, "inertia": scaled_inertia}]
        for motor in additions.get(name, []):
            bodies.append(motor)
        final_mass, final_com, final_inertia = combine_bodies(bodies)
        set_inertial(link, final_mass, final_com, final_inertia)
        link_reports[name] = {
            "old_mass": old_mass,
            "scaled_mass_before_motor": scaled_mass,
            "added_motor_mass": sum(motor["mass"] for motor in additions.get(name, [])),
            "final_mass": final_mass,
            "old_com": old_com,
            "final_com": final_com,
            "old_inertia": inertia_to_dict(old_inertia),
            "scaled_inertia_before_motor": inertia_to_dict(scaled_inertia),
            "final_inertia": inertia_to_dict(final_inertia),
            "added_motors": [
                {"name": motor["name"], "mass": motor["mass"], "com": motor["com"]} for motor in additions.get(name, [])
            ],
        }

    scaled_structure_total = original_total * args.base_mass_scale
    final_total = total_mass(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    validation = None
    try:
        import check_urdf_mass_com

        with contextlib.redirect_stdout(io.StringIO()):
            validation = check_urdf_mass_com.check_urdf(output_path)
    except Exception as exc:  # pragma: no cover - best-effort report enrichment
        warnings.append(f"Could not run report validation summary: {exc}")

    report = {
        "input_urdf": str(input_path),
        "output_urdf": str(output_path),
        "base_mass_scale": args.base_mass_scale,
        "motor_mass_each": args.motor_mass,
        "motor_count": 12,
        "total_motor_mass": total_motor_mass,
        "original_total_mass": original_total,
        "scaled_structure_total_mass": scaled_structure_total,
        "final_total_mass": final_total,
        "links": link_reports,
        "leg_masses": validation["groups"] if validation is not None else None,
        "trunk_mass": validation["groups"]["trunk"] if validation is not None else None,
        "global_com": validation["global_com"] if validation is not None else None,
        "symmetry_check": validation["symmetry"] if validation is not None else None,
        "inertia_validity_check": {
            "ok": validation["inertia_ok"],
            "invalid_inertia": validation["invalid_inertia"],
        }
        if validation is not None
        else None,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input URDF path")
    parser.add_argument("--output", required=True, help="Output URDF path")
    parser.add_argument("--base-mass-scale", type=float, default=2.20462, help="Scale for existing link masses/inertias")
    parser.add_argument("--motor-mass", type=float, default=0.38, help="RS01 motor mass in kg")
    parser.add_argument("--motor-radius", type=float, default=0.04, help="Equivalent motor sphere radius in meters")
    parser.add_argument("--knee-motor-on-thigh", action="store_true", help="Keep knee motor mass on thigh link")
    parser.add_argument("--knee-motor-on-calf", action="store_true", help="Put knee motor mass on calf link instead")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--backup", default="", help="Optional backup path for the input URDF")
    args = parser.parse_args()
    if args.knee_motor_on_thigh and args.knee_motor_on_calf:
        raise SystemExit("Use only one of --knee-motor-on-thigh or --knee-motor-on-calf")
    if args.backup:
        shutil.copyfile(args.input, args.backup)
    report = patch_urdf(args)
    print(f"input: {report['input_urdf']}")
    print(f"output: {report['output_urdf']}")
    print(f"original_total_mass: {report['original_total_mass']:.6f} kg")
    print(f"scaled_structure_total_mass: {report['scaled_structure_total_mass']:.6f} kg")
    print(f"total_motor_mass: {report['total_motor_mass']:.6f} kg")
    print(f"final_total_mass: {report['final_total_mass']:.6f} kg")
    print(f"report: {args.report}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
