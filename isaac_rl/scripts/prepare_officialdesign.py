"""Build the simulation URDF from the untouched 2026-09-30 CAD handoff.

Run with the Isaac conda Python (numpy required); no simulator is needed.
The CSV tensor is already about CoM in link axes, in kg m². Only CoM needs
mm -> m conversion. Never apply a second sign change or parallel-axis shift.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "sim_handoff "
ASSET = ROOT / "assets/officialdesign"


def transform(xyz, rpy):
    """URDF origin transform: translation followed by Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    out = np.eye(4)
    out[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                   [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    out[:3, 3] = xyz
    return out


def fk(robot, angles):
    """Link transforms in Baselink for raw URDF joint angles (radians)."""
    poses = {"Baselink": np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        previous = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            motion = np.eye(4)
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                assert np.isclose(np.linalg.norm(axis), 1), joint.get("name")
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = angles.get(joint.get("name"), 0.0)
                motion[:3, :3] += math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)
            poses[joint.find("child").get("link")] = poses[parent] @ transform(xyz, rpy) @ motion
            pending.remove(joint)
        if len(pending) == previous:
            raise ValueError("URDF is not a connected tree rooted at Baselink")
    return poses


def stl_vertices(path):
    """Read binary SW-exported STL vertices without mesh processing/recentering."""
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little")
    if len(data) != 84 + count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).astype(float)


def build():
    """Validate the handoff, patch all inertials, and derive spawn/sole geometry."""
    cfg = json.loads((ASSET / "meta/calibration.json").read_text())
    tree = ET.parse(HANDOFF / "OFFICIALdesign/urdf/OFFICIALdesign.urdf")
    robot = tree.getroot()
    with (HANDOFF / "inertial_report.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["link"]: row for row in csv.DictReader(stream)}
    links = robot.findall("link")
    assert len(links) == len(rows) == 13
    assert set(rows) == {link.get("name") for link in links}
    masses = []
    for link in links:
        row = rows[link.get("name")]
        mass = float(row["mass_kg"])
        v = {key: float(row[key]) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        tensor = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                           [v['ixz'], v['iyz'], v['izz']]])
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert mass > 0 and np.all(np.isfinite(tensor)) and np.all(eigenvalues > 0), link.get("name")
        assert eigenvalues[-1] <= eigenvalues[:2].sum() + 1e-10, link.get("name")
        inertial = link.find("inertial")
        inertial.find("mass").set("value", str(mass))
        inertial.find("origin").set("xyz", " ".join(str(float(row[f"com_{axis}_mm"])*0.001) for axis in "xyz"))
        inertial.find("origin").set("rpy", "0 0 0")
        inertial.find("inertia").attrib.update({key: str(value) for key, value in v.items()})
        masses.append(mass)
    assert math.isclose(sum(masses), 4.643, abs_tol=1e-9)

    joint_cfg = {joint["name"]: joint for joint in cfg["joints"]}
    movable = [j for j in robot.findall("joint") if j.get("type") == "revolute"]
    assert {j.get("name") for j in movable} == set(joint_cfg) and len(movable) == 10
    for joint in movable:
        item = joint_cfg[joint.get("name")]
        actuator = cfg["actuators"][item["group"]]
        assert item["sign"] in (-1, 1)
        assert item["policy_min_deg"] <= item["policy_default_deg"] <= item["policy_max_deg"]
        lower, upper = sorted(math.radians(item["sign"]*item[key]) for key in ("policy_min_deg", "policy_max_deg"))
        joint.find("limit").attrib.update(lower=str(lower), upper=str(upper),
                                          effort=str(actuator["effort"]), velocity=str(actuator["velocity"]))

    package = ASSET / "OFFICIALdesign"
    (package / "urdf").mkdir(parents=True, exist_ok=True)
    shutil.copytree(HANDOFF / "OFFICIALdesign/meshes", package / "meshes", dirs_exist_ok=True)
    shutil.copy2(HANDOFF / "inertial_report.csv", ASSET / "meta/inertial_report.csv")
    for mesh in robot.findall(".//mesh"):
        prefix = "package://OFFICIALdesign/"
        assert mesh.get("filename").startswith(prefix)
        assert (package / mesh.get("filename")[len(prefix):]).is_file()
    ET.indent(tree)
    output = package / "urdf/OFFICIALdesign.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)

    angles = {j["name"]: math.radians(j["sign"]*j["policy_default_deg"]) for j in cfg["joints"]}
    poses, zero = fk(robot, angles), fk(robot, {})
    min_z, feet = float("inf"), {}
    for link in links:
        name = link.get("name")
        vertices = stl_vertices(package / f"meshes/{name}.STL")
        # The source has identity visual/collision origins and mesh scale 1.
        for kind in ("visual", "collision"):
            origin = link.find(f"{kind}/origin")
            assert np.allclose(np.fromstring(origin.get("xyz"), sep=" "), 0)
            assert np.allclose(np.fromstring(origin.get("rpy"), sep=" "), 0)
            assert np.allclose(np.fromstring(link.find(f"{kind}/geometry/mesh").get("scale", "1 1 1"), sep=" "), 1)
        transformed = vertices @ poses[name][:3, :3].T + poses[name][:3, 3]
        min_z = min(min_z, transformed[:, 2].min())
        if name.startswith("Foot"):
            feet[name] = {"sole_z_local_m": float(vertices[:, 2].min()),
                          "default_origin_in_base_m": poses[name][:3, 3].tolist()}
            assert np.allclose(poses[name][:3, :3], np.eye(3), atol=1e-5)
    com = sum(float(rows[name]["mass_kg"]) * (zero[name] @ np.array(
        [float(rows[name][f"com_{a}_mm"])*0.001 for a in "xyz"] + [1]))[:3] for name in rows) / sum(masses)
    assert np.allclose(com, [0.0046, 0.0044, -0.0586], atol=0.0001), com
    sources = [HANDOFF / "OFFICIALdesign/urdf/OFFICIALdesign.urdf", HANDOFF / "inertial_report.csv",
               ASSET / "meta/calibration.json", *sorted((package / "meshes").glob("*.STL"))]
    report = {"mass_kg": sum(masses), "zero_com_in_base_m": com.tolist(),
              "spawn_height_m": float(-min_z + cfg["spawn_clearance_m"]), "feet": feet,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (ASSET / "meta/build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))
    print(f"Prepared: {output}")


if __name__ == "__main__":
    build()
