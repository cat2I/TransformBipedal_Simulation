"""Convert the prepared OFFICIALdesign URDF using the installed Isaac Lab 3 API."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Reimport even if only meshes changed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main():
    root = Path(__file__).resolve().parents[2] / "assets/officialdesign"
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(root / "OFFICIALdesign/urdf/OFFICIALdesign.urdf"),
        usd_dir=str(root / "usd"),
        force_usd_conversion=args.force,
        fix_base=False,
        merge_fixed_joints=False,
        collision_type="Convex Decomposition",
        self_collision=True,
        ros_package_paths=[{"name": "OFFICIALdesign", "path": str(root / "OFFICIALdesign")}],
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none", gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
    ))
    stage = Usd.Stage.Open(converter.usd_path)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    masses = {p.GetName(): UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies}
    assert len(bodies) == 13, masses
    assert abs(sum(masses.values()) - 4.643) < 1e-5, masses
    assert all(m > 0 for m in masses.values()), masses
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 10
    with (root / "meta/inertial_report.csv").open(encoding="utf-8-sig") as stream:
        inertials = {r["link"]: r for r in csv.DictReader(stream)}
    corrected_axes = {}
    for prim in bodies:
        row = inertials[prim.GetName()]
        mass_api = UsdPhysics.MassAPI(prim)
        np.testing.assert_allclose(mass_api.GetCenterOfMassAttr().Get(),
                                   [float(row[f"com_{a}_mm"])*0.001 for a in "xyz"], atol=1e-8)
        v = {k: float(row[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        expected = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                             [v['ixz'], v['iyz'], v['izz']]])
        eigenvalues, eigenvectors = np.linalg.eigh(expected)
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        # urdf-usd-converter 0.1.3 passes eigenvector COLUMNS directly to Gf's
        # ROW-vector matrix API, conjugating the principal-axis orientation.
        # Re-author from the authoritative tensor with the required transpose.
        axes = Gf.Quatf(Gf.Matrix3d(*map(float, eigenvectors.T.flatten())).ExtractRotation().GetQuat())
        corrected_axes[prim.GetName()] = (Gf.Vec3f(*map(float, eigenvalues)), axes, expected)
    # Importer 3 emits nested rigid links. Isaac Lab's property/contact traversal
    # stops at the first rigid body, and clone preprocessing can misresolve nested
    # link/mesh names. Author sibling links with the SAME world-space transforms;
    # NamespaceEditor also remaps joint body relationships. Keep importer output.
    interface = root / "usd/OFFICIALdesign.usd"
    output = Usd.Stage.Open(stage.Flatten())
    cache = UsdGeom.XformCache()
    transforms = {str(p.GetPath()): cache.GetLocalToWorldTransform(p) for p in bodies}
    root_path = str(stage.GetDefaultPrim().GetPath())
    for path in sorted(transforms, key=lambda p: p.count("/"), reverse=True):
        destination = root_path + "/" + path.rsplit("/", 1)[-1]
        editor = Usd.NamespaceEditor(output)
        editor.MovePrimAtPath(path, destination)
        assert editor.CanApplyEdits(), path
        editor.ApplyEdits()
        link = UsdGeom.Xformable(output.GetPrimAtPath(destination))
        link.ClearXformOpOrder()
        link.AddTransformOp().Set(transforms[path])
        mass_api = UsdPhysics.MassAPI(link.GetPrim())
        diagonal, axes, expected = corrected_axes[link.GetPrim().GetName()]
        mass_api.GetDiagonalInertiaAttr().Set(diagonal)
        mass_api.GetPrincipalAxesAttr().Set(axes)
        # Quaternion-transformed basis vectors avoid matrix convention ambiguity.
        basis = np.column_stack([axes.Transform(Gf.Vec3f(*map(float, e))) for e in np.eye(3)])
        np.testing.assert_allclose(basis @ np.diag(diagonal) @ basis.T, expected, atol=1e-8)
    # Expand instance geometry so physics materials can bind to collision meshes.
    while instances := [p for p in output.Traverse() if p.IsInstance()]:
        for prim in instances:
            prim.SetInstanceable(False)
    # Bake former instance references before editing geometry names; otherwise
    # shared referenced specs can reintroduce their original descendant names.
    output = Usd.Stage.Open(output.Flatten())
    # PhysX glob '/env_*/Robot/Footleft' can match descendants of the same
    # name too. Give geometry distinct names to avoid false sensor matches.
    geometry = [str(p.GetPath()) for p in output.Traverse()
                if p.GetName() in masses and p.IsA(UsdGeom.Xformable)
                and not p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for path in sorted(geometry, key=lambda p: p.count("/"), reverse=True):
        parent, name = path.rsplit("/", 1)
        editor = Usd.NamespaceEditor(output)
        editor.MovePrimAtPath(path, parent + "/mesh_" + name)
        assert editor.CanApplyEdits(), path
        editor.ApplyEdits()
    assert not [p for p in output.Traverse() if p.GetName() in masses
                and p.IsA(UsdGeom.Xformable) and not p.HasAPI(UsdPhysics.RigidBodyAPI)]
    output.GetRootLayer().Export(str(interface))
    output = Usd.Stage.Open(str(interface))
    actual_joints = {p.GetName() for p in output.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    calibration = json.loads((root / "meta/calibration.json").read_text())
    assert actual_joints == {j['name'] for j in calibration['joints']}, actual_joints
    for prim in output.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                assert all(output.GetPrimAtPath(p).HasAPI(UsdPhysics.RigidBodyAPI) for p in rel.GetTargets())
    output.GetRootLayer().Save()
    inputs = [root / "OFFICIALdesign/urdf/OFFICIALdesign.urdf", root / "meta/calibration.json",
              root / "meta/build_report.json", interface]
    (root / "meta/usd_report.json").write_text(json.dumps({
        "mass_kg": sum(masses.values()), "links": 13, "dof": 10,
        "collision": "Convex Decomposition", "floating_base": True,
        "principal_axes": "reconstructed from CSV tensor using Gf row-vector convention",
        "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }, indent=2)+'\n')
    print(f"USD validated: {len(bodies)} links, 10 DOF, {sum(masses.values()):.6f} kg; {interface}")


if __name__ == "__main__":
    main()
    app.close()
