#!/usr/bin/env python3
"""Xem nhanh cac model URDF cua repo bang MuJoCo viewer.

    python mujoco_view.py --list             # liet ke cac model co san
    python mujoco_view.py FullForm           # mo viewer (chan de tu do + san)
    python mujoco_view.py NewSimple --fixed  # treo base link co dinh vao world
    python mujoco_view.py 3DOFTrans --export /tmp/3dof.xml   # xuat MJCF

Trong viewer: tab "Control" ben phai co slider cho tung khop (position actuator).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import mujoco

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_models():
    """Tim moi file .urdf trong repo, tra ve dict {ten: duong_dan}."""
    models = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".urdf"):
                models[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    return dict(sorted(models.items()))


def mesh_names(urdf_path):
    """Danh sach ten file mesh (basename) ma URDF tham chieu."""
    root = ET.parse(urdf_path).getroot()
    names = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            names.append(os.path.basename(fn.replace("package://", "")))
    return sorted(set(names))


def pick_meshdir(urdf_path, needed):
    """Chon thu muc mesh khop nhat: uu tien duong dan tuong doi trong URDF,
    neu thieu thi do tim moi thu muc mesh khac trong repo."""
    candidates = []

    root = ET.parse(urdf_path).getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            d = os.path.dirname(os.path.normpath(os.path.join(os.path.dirname(urdf_path), fn)))
            candidates.append(d)
            break

    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for d in dirnames:
            if d.lower() in ("meshes", "mesh"):
                candidates.append(os.path.join(dirpath, d))

    best, best_hit = None, -1
    for d in candidates:
        if not os.path.isdir(d):
            continue
        have = {f.lower() for f in os.listdir(d)}
        hit = sum(1 for n in needed if n.lower() in have)
        if hit > best_hit:
            best, best_hit = d, hit
    return best, best_hit


def drop_missing_meshes(spec, meshdir):
    """Xoa mesh (va geom dung no) khi khong tim thay file, de model van compile."""
    have = {f.lower() for f in os.listdir(meshdir)} if os.path.isdir(meshdir) else set()
    missing = set()
    for mesh in list(spec.meshes):
        f = os.path.basename(mesh.file) if mesh.file else mesh.name
        if f.lower() not in have:
            missing.add(mesh.name)

    if not missing:
        return []

    for body in spec.bodies:
        for geom in list(body.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname in missing:
                spec.delete(geom)
    for mesh in list(spec.meshes):
        if mesh.name in missing:
            spec.delete(mesh)
    return sorted(missing)


def build(urdf_path, fixed_base=False, add_actuators=True):
    needed = mesh_names(urdf_path)
    meshdir, hit = pick_meshdir(urdf_path, needed)
    if meshdir is None:
        sys.exit(f"Khong tim thay thu muc mesh nao cho {urdf_path}")
    print(f"  mesh dir : {os.path.relpath(meshdir, ROOT)}  ({hit}/{len(needed)} file)")

    spec = mujoco.MjSpec.from_file(urdf_path)
    spec.meshdir = meshdir
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = os.path.basename(mesh.file.replace("package://", ""))
    spec.compiler.balanceinertia = True   # sua inertia xau tu SolidWorks export
    spec.compiler.discardvisual = False   # giu mesh visual de nhin cho ro
    spec.compiler.fusestatic = False

    dropped = drop_missing_meshes(spec, meshdir)
    if dropped:
        print(f"  !! thieu mesh, bo qua: {', '.join(dropped)}")

    # anh sang + san
    spec.worldbody.add_light(
        pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    if not fixed_base:
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5, 5, 0.1],
            rgba=[0.35, 0.38, 0.42, 1],
        )

    base = spec.worldbody.bodies[0]
    if not fixed_base:
        base.add_freejoint()

    if add_actuators:
        for joint in spec.joints:
            if joint.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                if joint.range[0] >= joint.range[1]:
                    print(f"  !! khop '{joint.name}' co range rong {list(joint.range)}, bo qua actuator (coi nhu khop khoa)")
                    continue
                act = spec.add_actuator()
                act.name = joint.name
                act.target = joint.name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.gainprm[0] = 40.0            # kp
                act.biasprm[1] = -40.0           # -kp
                act.biasprm[2] = -4.0            # -kv
                act.ctrlrange = joint.range
                act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    model = spec.compile()

    if not fixed_base:
        # nhac robot len sao cho diem thap nhat cach san 2cm
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        lowest = min(
            data.geom_xpos[g][2] - model.geom_rbound[g]
            for g in range(model.ngeom)
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
        )
        model.qpos0[2] += 0.02 - lowest

    return spec, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="ten model (khong can duoi .urdf)")
    ap.add_argument("--list", action="store_true", help="liet ke model co san")
    ap.add_argument("--fixed", action="store_true", help="base link gan cung vao world")
    ap.add_argument("--no-actuators", action="store_true", help="khong them slider dieu khien khop")
    ap.add_argument("--export", metavar="FILE.xml", help="ghi ra file MJCF thay vi mo viewer")
    args = ap.parse_args()

    models = find_models()

    if args.list or not args.model:
        print(f"Co {len(models)} model URDF trong repo:\n")
        for name, path in models.items():
            needed = mesh_names(path)
            _, hit = pick_meshdir(path, needed)
            ndof = sum(1 for j in ET.parse(path).getroot().iter("joint")
                       if j.get("type") in ("revolute", "continuous", "prismatic"))
            flag = "" if hit == len(needed) else f"  [thieu {len(needed) - hit} mesh]"
            print(f"  {name:<12} {ndof:>2} DOF   {os.path.relpath(path, ROOT)}{flag}")
        print("\nChay:  python mujoco_view.py <ten>")
        return

    if args.model not in models:
        sys.exit(f"Khong co model '{args.model}'. Co: {', '.join(models)}")

    path = models[args.model]
    print(f"Load {args.model}: {os.path.relpath(path, ROOT)}")
    spec, model = build(path, fixed_base=args.fixed, add_actuators=not args.no_actuators)
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")

    if args.export:
        with open(args.export, "w") as f:
            f.write(spec.to_xml())
        print(f"  da ghi MJCF -> {args.export}")
        return

    import mujoco.viewer
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
