"""Offline regression checks for CAD units, signed mapping and simulation geometry.

Run: ./run.sh -m pytest scripts/test_officialdesign_asset.py -q
"""
import csv
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from prepare_officialdesign import ASSET, HANDOFF, fk, stl_vertices


def test_inertials_match_report_in_si_units():
    robot = ET.parse(ASSET / "OFFICIALdesign/urdf/OFFICIALdesign.urdf").getroot()
    with (HANDOFF / "inertial_report.csv").open(encoding="utf-8-sig") as stream:
        rows = {r["link"]: r for r in csv.DictReader(stream)}
    total = 0
    for link in robot.findall("link"):
        row = rows[link.get("name")]
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        assert mass == float(row["mass_kg"]) and mass > 0
        total += mass
        com = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
        np.testing.assert_allclose(com, [float(row[f"com_{a}_mm"])*0.001 for a in "xyz"])
        for key, value in inertial.find("inertia").attrib.items():
            assert float(value) == float(row[key])
    assert abs(total-4.643) < 1e-10


def test_source_kinematics_preserved():
    old = ET.parse(HANDOFF / "OFFICIALdesign/urdf/OFFICIALdesign.urdf").getroot()
    new = ET.parse(ASSET / "OFFICIALdesign/urdf/OFFICIALdesign.urdf").getroot()
    for a, b in zip(old.findall("joint"), new.findall("joint"), strict=True):
        assert a.attrib == b.attrib
        for tag in ("parent", "child", "origin", "axis"):
            assert a.find(tag).attrib == b.find(tag).attrib


def test_signed_limits_and_actuator_groups():
    cfg = json.loads((ASSET / "meta/calibration.json").read_text())
    robot = ET.parse(ASSET / "OFFICIALdesign/urdf/OFFICIALdesign.urdf").getroot()
    for j in cfg["joints"]:
        limit = robot.find(f"joint[@name='{j['name']}']/limit")
        bounds = sorted(math.radians(j["sign"]*j[k]) for k in ("policy_min_deg", "policy_max_deg"))
        np.testing.assert_allclose([float(limit.get(k)) for k in ("lower", "upper")], bounds)
        assert float(limit.get("effort")) == (10.3 if j["group"] == "heavy" else 2.94)
        assert float(limit.get("velocity")) == 4.7


def test_pitch_directions_and_ground_clearance():
    cfg = json.loads((ASSET / "meta/calibration.json").read_text())
    report = json.loads((ASSET / "meta/build_report.json").read_text())
    robot = ET.parse(ASSET / "OFFICIALdesign/urdf/OFFICIALdesign.urdf").getroot()
    poses = fk(robot, {j["name"]: math.radians(j["sign"]*j["policy_default_deg"]) for j in cfg["joints"]})
    zero = fk(robot, {})
    assert zero['Footleft'][1, 3] < 0 < zero['Footright'][1, 3]
    for left, right, child_l, child_r in [('Hipleft_joint', 'Hipright_joint', 'Kneeleft', 'Kneeright'),
                                         ('Kneeleft_joint', 'Kneeright', 'Footleft', 'Footright')]:
        # Positive policy pitch moves both lower-link origins forward in x.
        moved = fk(robot, {left: math.radians(5), right: -math.radians(5)})
        assert moved[child_l][0, 3] > zero[child_l][0, 3]
        assert moved[child_r][0, 3] > zero[child_r][0, 3]
    for name in ('Footleft', 'Footright'):
        vertices = stl_vertices(ASSET / f"OFFICIALdesign/meshes/{name}.STL")
        world = vertices @ poses[name][:3, :3].T + poses[name][:3, 3]
        clearance = world[:, 2].min() + report['spawn_height_m']
        assert 0.0049 < clearance < 0.0053
        np.testing.assert_allclose(poses[name][:3, :3], np.eye(3), atol=1e-5)
