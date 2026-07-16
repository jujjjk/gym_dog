#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


URDF_PATH = Path(__file__).resolve().parents[6] / "fanfan" / "urdf" / "fanfan_mass_scaled_only_trunk_plus_800g.urdf"

LEG_ORDER = ("FR", "FL", "RR", "RL")
FRONT_LEGS = ("FR", "FL")
RIGHT_LEGS = ("FR", "RR")

FOOT_X = {"front": 0.1765, "rear": -0.2219}
FOOT_Y_HALF_WIDTH = 0.138

BODY_SHIFT = {
    "FR": (0.014, 0.040),
    "FL": (0.014, -0.040),
    "RR": (0.080, 0.040),
    "RL": (0.080, -0.040),
}


def read_total_mass_and_com(urdf_path: Path) -> tuple[float, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    link_inertials = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").attrib["value"])
        origin = inertial.find("origin")
        xyz = np.zeros(3)
        if origin is not None and "xyz" in origin.attrib:
            xyz = np.array([float(v) for v in origin.attrib["xyz"].split()], dtype=float)
        link_inertials[link.attrib["name"]] = (mass, xyz)

    children = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = np.zeros(3)
        if origin is not None and "xyz" in origin.attrib:
            xyz = np.array([float(v) for v in origin.attrib["xyz"].split()], dtype=float)
        children.setdefault(parent, []).append((child, xyz))

    poses = {"Trunk": np.zeros(3)}
    stack = ["Trunk"]
    while stack:
        parent = stack.pop()
        for child, xyz in children.get(parent, []):
            poses[child] = poses[parent] + xyz
            stack.append(child)

    total_mass = 0.0
    weighted_com = np.zeros(3)
    for link_name, (mass, local_com) in link_inertials.items():
        total_mass += mass
        weighted_com += mass * (poses.get(link_name, np.zeros(3)) + local_com)

    return total_mass, weighted_com / max(total_mass, 1e-9)


def support_foot_xy(leg: str) -> np.ndarray:
    x = FOOT_X["front"] if leg in FRONT_LEGS else FOOT_X["rear"]
    y = -FOOT_Y_HALF_WIDTH if leg in RIGHT_LEGS else FOOT_Y_HALF_WIDTH
    return np.array([x, y], dtype=float)


def point_margin(point: np.ndarray, triangle: list[np.ndarray]) -> tuple[bool, float]:
    signs = []
    distances = []
    for a, b in zip(triangle, triangle[1:] + triangle[:1]):
        edge = b - a
        rel = point - a
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        signs.append(cross)
        distances.append(abs(cross) / max(float(np.linalg.norm(edge)), 1e-9))
    inside = all(value >= -1e-9 for value in signs) or all(value <= 1e-9 for value in signs)
    return inside, min(distances)


def barycentric_loads(point: np.ndarray, support_legs: list[str]) -> np.ndarray:
    feet = [support_foot_xy(leg) for leg in support_legs]
    matrix = np.array(
        [
            [feet[0][0], feet[1][0], feet[2][0]],
            [feet[0][1], feet[1][1], feet[2][1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    rhs = np.array([point[0], point[1], 1.0], dtype=float)
    return np.linalg.solve(matrix, rhs)


def main() -> None:
    total_mass, com = read_total_mass_and_com(URDF_PATH)
    weight_n = total_mass * 9.80665
    print(f"URDF: {URDF_PATH}")
    print(f"total_mass_kg: {total_mass:.3f}")
    print(f"global_com_m: x={com[0]:+.4f}, y={com[1]:+.4f}, z={com[2]:+.4f}")
    print(
        "gait: crawl order RR -> FR -> RL -> FL, "
        "step_hz=0.55, duty_factor=0.78, stride_length=0.014m, swing_height=0.070m"
    )

    for swing_leg in ("FR", "FL", "RR", "RL"):
        support_legs = [leg for leg in LEG_ORDER if leg != swing_leg]
        shifted_com = np.array(BODY_SHIFT[swing_leg], dtype=float)
        triangle = [support_foot_xy(leg) for leg in support_legs]
        inside, margin = point_margin(shifted_com, triangle)
        load_frac = barycentric_loads(shifted_com, support_legs)

        print(f"\nswing_leg: {swing_leg}")
        print(f"  support_legs: {', '.join(support_legs)}")
        print(f"  shifted_com_xy_m: x={shifted_com[0]:+.3f}, y={shifted_com[1]:+.3f}")
        print(f"  support_margin_mm: {margin * 1000.0:.1f} ({'inside' if inside else 'outside'})")
        for leg, frac in zip(support_legs, load_frac):
            print(
                f"  {leg}: load_frac={frac:.3f}, "
                f"normal_force_n={frac * weight_n:.1f}, kg_equiv={frac * total_mass:.2f}"
            )


if __name__ == "__main__":
    main()
