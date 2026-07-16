#!/usr/bin/env python3
"""Check URDF link masses, q=0 COM, symmetry, and inertia validity."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


LEG_PREFIXES = ("FR", "FL", "RR", "RL")


def parse_vec(text: str | None, default=(0.0, 0.0, 0.0)) -> list[float]:
    if not text:
        return [float(x) for x in default]
    return [float(x) for x in text.split()]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def transform_point(t: list[list[float]], p: list[float]) -> list[float]:
    v = [p[0], p[1], p[2], 1.0]
    return [sum(t[i][j] * v[j] for j in range(4)) for i in range(3)]


def rpy_matrix(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def transform_from_origin(origin: ET.Element | None) -> list[list[float]]:
    xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None)
    rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None)
    rot = rpy_matrix(rpy)
    out = [
        [rot[0][0], rot[0][1], rot[0][2], xyz[0]],
        [rot[1][0], rot[1][1], rot[1][2], xyz[1]],
        [rot[2][0], rot[2][1], rot[2][2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return out


def inertia_dict(node: ET.Element) -> dict[str, float]:
    return {
        "ixx": float(node.attrib["ixx"]),
        "ixy": float(node.attrib.get("ixy", 0.0)),
        "ixz": float(node.attrib.get("ixz", 0.0)),
        "iyy": float(node.attrib["iyy"]),
        "iyz": float(node.attrib.get("iyz", 0.0)),
        "izz": float(node.attrib["izz"]),
    }


def inertial_data(link: ET.Element) -> dict | None:
    inertial = link.find("inertial")
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
        "inertia": inertia_dict(inertia_node),
    }


def inertia_is_valid(inertia: dict[str, float], tol: float = 1.0e-12) -> tuple[bool, list[str]]:
    ixx, iyy, izz = inertia["ixx"], inertia["iyy"], inertia["izz"]
    failures: list[str] = []
    if ixx <= 0.0 or iyy <= 0.0 or izz <= 0.0:
        failures.append("non-positive diagonal")
    if ixx + iyy + tol < izz:
        failures.append("ixx + iyy < izz")
    if ixx + izz + tol < iyy:
        failures.append("ixx + izz < iyy")
    if iyy + izz + tol < ixx:
        failures.append("iyy + izz < ixx")
    return not failures, failures


def build_link_transforms(root: ET.Element) -> dict[str, list[list[float]]]:
    links = {link.attrib["name"] for link in root.findall("link")}
    child_links = set()
    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(joint)
        child_links.add(child)
    roots = sorted(links - child_links)
    if not roots:
        raise ValueError("URDF has no root link")
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    transforms = {roots[0]: identity}
    stack = [roots[0]]
    while stack:
        parent = stack.pop()
        parent_t = transforms[parent]
        for joint in children.get(parent, []):
            child = joint.find("child").attrib["link"]
            child_t = mat_mul(parent_t, transform_from_origin(joint.find("origin")))
            transforms[child] = child_t
            stack.append(child)
    return transforms


def group_name(link_name: str) -> str:
    for prefix in LEG_PREFIXES:
        if link_name.startswith(prefix + "_"):
            return prefix
    if link_name == "Trunk":
        return "trunk"
    return "other"


def check_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    transforms = build_link_transforms(root)
    groups = {"trunk": 0.0, "FR": 0.0, "FL": 0.0, "RR": 0.0, "RL": 0.0, "other": 0.0}
    warnings: list[str] = []
    invalid_inertia: dict[str, list[str]] = {}
    total_mass = 0.0
    weighted_com = [0.0, 0.0, 0.0]

    print(f"URDF: {path}")
    print(f"robot: {root.attrib.get('name', '<unnamed>')}")
    print("\nLinks:")
    for link in root.findall("link"):
        name = link.attrib["name"]
        data = inertial_data(link)
        if data is None:
            warning = f"Link {name} has no complete inertial"
            warnings.append(warning)
            print(f"  {name}: WARNING {warning}")
            continue
        ok, failures = inertia_is_valid(data["inertia"])
        if not ok:
            invalid_inertia[name] = failures
        world_com = transform_point(transforms.get(name, transforms[next(iter(transforms))]), data["com"])
        total_mass += data["mass"]
        weighted_com = [weighted_com[i] + data["mass"] * world_com[i] for i in range(3)]
        groups[group_name(name)] += data["mass"]
        inertia = data["inertia"]
        print(
            f"  {name}: mass={data['mass']:.9f} com={data['com']} "
            f"inertia={{ixx:{inertia['ixx']:.9g}, ixy:{inertia['ixy']:.9g}, ixz:{inertia['ixz']:.9g}, "
            f"iyy:{inertia['iyy']:.9g}, iyz:{inertia['iyz']:.9g}, izz:{inertia['izz']:.9g}}}"
        )

    global_com = [x / total_mass for x in weighted_com] if total_mass > 0.0 else [math.nan, math.nan, math.nan]
    symmetry = {
        "fr_fl_diff": abs(groups["FR"] - groups["FL"]),
        "rr_rl_diff": abs(groups["RR"] - groups["RL"]),
        "fr_fl_ok": abs(groups["FR"] - groups["FL"]) < 0.02,
        "rr_rl_ok": abs(groups["RR"] - groups["RL"]) < 0.02,
        "com_y_ok_preferred": abs(global_com[1]) < 0.005,
        "com_y_ok_relaxed": abs(global_com[1]) < 0.01,
    }
    inertia_ok = not invalid_inertia

    print("\nMass summary:")
    print(f"  total: {total_mass:.9f} kg")
    for name in ("trunk", "FR", "FL", "RR", "RL", "other"):
        print(f"  {name}: {groups[name]:.9f} kg")
    print(f"  global_com_q0: [{global_com[0]:.9f}, {global_com[1]:.9f}, {global_com[2]:.9f}]")
    print("\nChecks:")
    print(f"  FR/FL symmetry diff: {symmetry['fr_fl_diff']:.9f} kg ok={symmetry['fr_fl_ok']}")
    print(f"  RR/RL symmetry diff: {symmetry['rr_rl_diff']:.9f} kg ok={symmetry['rr_rl_ok']}")
    print(
        f"  COM y: {global_com[1]:.9f} m "
        f"preferred_ok={symmetry['com_y_ok_preferred']} relaxed_ok={symmetry['com_y_ok_relaxed']}"
    )
    print(f"  inertia validity ok={inertia_ok}")
    if invalid_inertia:
        for name, failures in invalid_inertia.items():
            print(f"    {name}: {', '.join(failures)}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return {
        "total_mass": total_mass,
        "groups": groups,
        "global_com": global_com,
        "symmetry": symmetry,
        "inertia_ok": inertia_ok,
        "invalid_inertia": invalid_inertia,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", help="URDF path to inspect")
    args = parser.parse_args()
    check_urdf(Path(args.urdf))


if __name__ == "__main__":
    main()
