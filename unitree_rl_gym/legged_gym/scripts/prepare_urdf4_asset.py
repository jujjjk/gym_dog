"""Build the Isaac Gym training asset from the SolidWorks URDF4.0 export.

The exported package is never modified.  This script copies it into
``resources/robots/urdf4`` and applies only presentation-level changes:

* ``package://URDF4.0.SLDASM/meshes/x.STL`` -> ``../meshes/x.STL`` so Isaac Gym
  resolves the meshes relative to the URDF file;
* a visual sphere on each foot link (the export has collision spheres only, so
  the feet would be invisible in the viewer);
* robot name ``urdf4_rs01``.

Masses, inertias, joint origins/axes/limits and collision geometry are copied
verbatim.  Re-run after every new SolidWorks export:

    python legged_gym/scripts/prepare_urdf4_asset.py
"""

import argparse
import hashlib
import json
import os
import re
import shutil

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(LEGGED_GYM_ROOT), "URDF4.0.SLDASM"
)
DEFAULT_TARGET = os.path.join(LEGGED_GYM_ROOT, "resources", "robots", "urdf4")
PACKAGE_PREFIX = "package://URDF4.0.SLDASM/meshes/"
FOOT_LINK_PATTERN = re.compile(
    r'(<link\s+name="(?P<name>[A-Z]{2}_foot)">.*?</inertial>)'
    r'(?P<body>.*?<sphere\s+radius="(?P<radius>[^"]+)"\s*/>.*?</link>)',
    re.DOTALL,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_foot_visual(match):
    visual = (
        "\n    <visual>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0" />\n'
        "      <geometry>\n"
        f'        <sphere radius="{match.group("radius")}" />\n'
        "      </geometry>\n"
        '      <material name="foot_rubber">\n'
        '        <color rgba="0.1 0.1 0.1 1" />\n'
        "      </material>\n"
        "    </visual>"
    )
    return match.group(1) + visual + match.group("body")


def convert(text):
    if PACKAGE_PREFIX not in text:
        raise ValueError(f"Expected mesh prefix {PACKAGE_PREFIX!r} in the URDF")
    text = text.replace(PACKAGE_PREFIX, "../meshes/")
    text, robot_count = re.subn(
        r'<robot\s+name="[^"]*">', '<robot name="urdf4_rs01">', text, count=1
    )
    text, foot_count = FOOT_LINK_PATTERN.subn(add_foot_visual, text)
    if robot_count != 1 or foot_count != 4:
        raise ValueError(
            f"Unexpected URDF structure: robot tags={robot_count}, "
            f"foot links={foot_count} (expected 1 and 4)"
        )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    source_urdf = os.path.join(args.source, "urdf", "URDF4.0.SLDASM.urdf")
    source_meshes = os.path.join(args.source, "meshes")
    target_urdf = os.path.join(args.target, "urdf", "urdf4_rs01.urdf")
    target_meshes = os.path.join(args.target, "meshes")

    with open(source_urdf, encoding="utf-8") as stream:
        converted = convert(stream.read())
    referenced = sorted(set(re.findall(r'\.\./meshes/([^"]+)', converted)))
    missing = [
        name for name in referenced
        if not os.path.isfile(os.path.join(source_meshes, name))
    ]
    if missing:
        raise FileNotFoundError(f"Meshes referenced but not exported: {missing}")

    os.makedirs(os.path.dirname(target_urdf), exist_ok=True)
    os.makedirs(target_meshes, exist_ok=True)
    with open(target_urdf, "w", encoding="utf-8") as stream:
        stream.write(converted)
    for name in referenced:
        shutil.copy2(
            os.path.join(source_meshes, name), os.path.join(target_meshes, name)
        )
    manifest = {
        "source_urdf": os.path.realpath(source_urdf),
        "source_urdf_sha256": sha256(source_urdf),
        "training_urdf_sha256": sha256(target_urdf),
        "meshes": referenced,
        "changes": [
            "mesh paths made relative",
            "foot visual spheres added",
            "robot renamed to urdf4_rs01",
        ],
    }
    with open(
        os.path.join(args.target, "source.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"[urdf4] wrote {target_urdf}")
    print(f"[urdf4] copied {len(referenced)} meshes to {target_meshes}")
    print(f"[urdf4] source sha256 {manifest['source_urdf_sha256']}")


if __name__ == "__main__":
    main()
