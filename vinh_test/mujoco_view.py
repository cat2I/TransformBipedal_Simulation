#!/usr/bin/env python3
"""Xem nhanh cac model URDF cua repo bang MuJoCo viewer.

    python mujoco_view.py --list             # liet ke cac model co san
    python mujoco_view.py Fulltrans_meshfixed              # model 10 DOF, nen dung
    python mujoco_view.py Fulltrans_meshfixed --mirror     # chan phai bam theo chan trai
    python mujoco_view.py Fulltrans_meshfixed --fixed      # treo than co dinh
    python mujoco_view.py 3DOFTrans --export /tmp/3dof.xml # xuat MJCF
    chạy chế độ mirror: python mujoco_view.py Fulltrans_meshfixed --mirror

Trong viewer, hai bang ben phai:
  * "Control" = LENH ban ra cho dong co ("toi muon khop toi goc nay")
  * "Joint"   = GOC THAT hien tai cua khop ("khop dang o dau")
Hai so bang nhau la binh thuong. Lech xa nhau nghia la co gi do dang chan.

Che do --mirror: chi keo 5 slider '...left...', 5 khop '...right...' tu chay
theo y het. Tien khi dung dang doi xung (dung, ngoi, xoac deu hai ben). Khong
doi dau vi URDF da lo phan doi xung trong truc quay: Bubleft axis = [-1,0,0]
con Bubright axis = [+1,0,0]. Kiem chung: dat ca hai = +0.5 rad thi hai ban
chan ra y = -0.2567 va +0.2562, doi xung dep qua mat phang doc.


================================================================================
HAI THU MODEL THO BI HONG, VA CACH FILE NAY VA LAI
================================================================================

Model doc thang tu URDF chay ra rat xau: robot tha xuong thi CO GIAT khong bao
gio dung, va keo Control khop Bub thi no ket cung o 41 do. Hai nguyen nhan
khac nhau, file nay vá ca hai truoc khi dua model cho viewer.


--- 1. Khop khong co quan tinh dong co -> rung 174 Hz ---------------------------

URDF khong co truong 'armature' nen MuJoCo de bang 0. Khop hoa ra nhe nhu
khong khi: moi buoc tinh, bo giai so day khop qua da roi day nguoc lai, lap di
lap lai. Do duoc 174 Hz — day la rung cua THUAT TOAN, khong phai cua vat ly.

Vá: dat armature = 0.08 cho moi khop (bang gia tri config Isaac dang dung).
Do lai sau khi vá: rung giam tu qvelRMS 13.2 xuong 0.0025.


--- 2. Bo do va cham dung VO LOI, khong dung hinh that -------------------------

MuJoCo (va PhysX cung vay) khong va cham bang mesh that. No boc moi mesh trong
mot VO LOI (convex hull) — tuong tuong boc mang boc thuc pham quanh vat roi hut
chan khong: moi hoc lom bi lap phang thanh khoi dac.

Than robot rong va nhieu hoc, nen vo loi cua no phinh to kinh khung:

    Baselink.STL      the tich that  1876 cm3  ->  vo loi 10539 cm3  = phinh 5.6 lan

Khoi phinh vo hinh do lan ra dung cho chan muon xoac. Hau qua do duoc:

    keo Control Bubleft toi 179.8 do  ->  khop ket cung o 40.9 do
    ly do: MuJoCo bao "Baselink cham Twistleft"
    nhung hai MESH THAT luc do con cach nhau 37.8 mm

Boc mo-men o khop luc ket:  dong co 10.3 Nm / va cham gia -9.1 Nm / trong luc
that chi 1.2 Nm. Tuc 88% suc dong co dang danh nhau voi mot bong ma.

Vá: bo qua va cham giua cac than NAM GAN NHAU TREN CUNG MOT CHAN (ham
nearby_chain_pairs), cong them cac cap long vao nhau san o tu the nghi (ham
overlapping_pairs, vd Baselink long vao Hipleft 72.6 mm).

Quy tac de nho:
    cung mot chan, gan nhau      -> BO   (khop da rang buoc roi, va cham vo nghia)
    chan trai dung chan phai     -> GIU  (va cham that, can de dung dang di/dung len)

Sau khi vá, keo het co tung khop:
    Bubleft  179.8 -> 179.7 do   (truoc khi vá: ket o 40.9)
    Twist    179.9 -> 179.9 do
    Foot      40.3 ->  40.1 do
    Hip       37.3 ->  35.6 do   het tam khop, URDF ghi vay
    Knee     126.1 -> 116.7 do   CHAM THAT: ban chan da vao than, mesh cach 3.2 mm
Tha roi tu do: robot dung yen o z = 0.3968 m, bien do rung 0.07 mm.
(Isaac do doc lap ra 0.397 m — hai engine khop nhau, con so nay dang tin.)


--- Muon xem model tho nguyen ban (chua vá) ------------------------------------

    python mujoco_view.py Fulltrans_meshfixed --armature 0 --keep-overlaps


--- Luu y ve chon model --------------------------------------------------------

Dung 'Fulltrans_meshfixed', dung 'Fulltrans'. Ban cu co loi xuat mesh: file
Hipleft.STL bi nuong dinh ca link Twist vao trong (thua 3434 tam giac, dai gap
doi), nen hong va cham o vung dau goi.
"""

import argparse
import os
import json
import sys
import time
import xml.etree.ElementTree as ET

import mujoco


def _find_root(max_levels=5):
    """Thu muc goc de quet model URDF.

    Script co the nam o thu muc con (vd vinh_test/) chu khong phai goc repo.
    Neu ngay tai cho no dung khong co file .urdf nao thi di nguoc len thu muc
    cha cho toi khi gap, toi da 5 cap. Nho vay dat script o dau cung chay duoc.
    """

    def co_urdf(d):
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if not x.startswith(".")]
            if any(f.endswith(".urdf") for f in filenames):
                return True
        return False

    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(max_levels):
        if co_urdf(d):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # cham goc he thong file
            break
        d = parent
    return here


ROOT = _find_root()


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
            d = os.path.dirname(
                os.path.normpath(os.path.join(os.path.dirname(urdf_path), fn))
            )
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


def overlapping_pairs(model, depth=1e-3):
    """Cac cap than long vao nhau ngay o tu the nghi.

    Chi tiet co khi thuong lap ke nhau — ngam hong om vong quanh than, kieu ban
    le cua — nen mesh chong nhau la chuyen binh thuong. Nhung bo do va cham
    khong biet khop dang rang buoc chung: no chi thay hai khoi dac trung nhau
    roi day ra bang moi gia, khop keo lai, thanh vong lap rung vinh vien.

    Do duoc tren Fulltrans: Baselink long vao Hipleft/Hipright 72.6 mm.

    Cach xu ly chuan cua nganh la loai dung nhung cap do khoi kiem tra va cham.
    Chi xet tu the nghi la du, vi chi tiet lap ke nhau thi chong o moi tu the.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = set()
    for c in range(data.ncon):
        con = data.contact[c]
        if con.dist > -depth:
            continue
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        if b1 == 0 or b2 == 0:  # cham san, khong phai tu dam
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if n1 and n2 and n1 != n2:
            pairs.add(tuple(sorted((n1, n2))))
    return sorted(pairs)


def nearby_chain_pairs(model, max_depth=3):
    """Cac cap than nam gan nhau tren CUNG mot nhanh dong hoc.

    Bo do va cham khong dung mesh that ma dung VO LOI (convex hull) boc quanh
    no. Voi chi tiet rong/lom thi vo loi phinh to hon hinh that rat nhieu — do
    tren Fulltrans: Baselink co the tich that 1876 cm3, vo loi 10539 cm3, tuc
    phinh 5.6 lan. Khoi phinh vo hinh do lan ra dung cho chan muon xoac.

    Hau qua do duoc: keo Control Bubleft toi 179.8°, khop ket o 40.9° vi
    'cham' Twistleft — trong khi hai mesh THAT con cach nhau 37.8 mm. Dong co
    gong 10.3 Nm chi de day mot bong ma (trong luc that chi can 1.2 Nm).

    Cac than gan nhau tren cung mot nhanh thi da bi khop rang buoc, va cham
    giua chung khong mang y nghia vat ly — loai het. Nguoc lai, va chan TRAI
    voi chan PHAI la va cham that (hai chan dam nhau) nen GIU nguyen.

    Args:
        max_depth: cach nhau bao nhieu khop thi con coi la "gan". Baselink toi
            Twist la 3 khop (Baselink -> Bub -> Hip -> Twist).
    """
    name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    pairs = set()
    for body in range(1, model.nbody):
        parent, depth = model.body_parentid[body], 1
        while parent > 0 and depth <= max_depth:
            n1, n2 = name(body), name(parent)
            if n1 and n2 and n1 != n2:
                pairs.add(tuple(sorted((n1, n2))))
            parent, depth = model.body_parentid[parent], depth + 1
    return sorted(pairs)


def build(
    urdf_path, fixed_base=False, add_actuators=True, armature=0.08, drop_overlaps=True
):
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
    spec.compiler.balanceinertia = True  # sua inertia xau tu SolidWorks export
    spec.compiler.discardvisual = False  # giu mesh visual de nhin cho ro
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

    # armature = quan tinh rotor + hop so cua dong co. URDF khong mang truong
    # nay nen mac dinh bang 0, khop hoa ra "nhe nhu khong khi": bo giai so day
    # qua da moi buoc roi day nguoc lai — do duoc 174 Hz, la rung cua thuat
    # toan chu khong phai cua vat ly. 0.08 la gia tri config Isaac dang dung.
    if armature:
        for joint in spec.joints:
            if joint.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                joint.armature = armature

    if add_actuators:
        for joint in spec.joints:
            if joint.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                if joint.range[0] >= joint.range[1]:
                    print(
                        f"  !! khop '{joint.name}' co range rong {list(joint.range)}, bo qua actuator (coi nhu khop khoa)"
                    )
                    continue
                act = spec.add_actuator()
                act.name = joint.name
                act.target = joint.name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.gainprm[0] = 40.0  # kp
                act.biasprm[1] = -40.0  # -kp
                act.biasprm[2] = -4.0  # -kv
                act.ctrlrange = joint.range
                act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    model = spec.compile()

    if drop_overlaps:
        # hai nguon: (1) than long vao nhau san o tu the nghi, (2) than gan
        # nhau tren cung mot nhanh — va cham cua chung la ao do vo loi phinh.
        pairs = sorted(set(overlapping_pairs(model)) | set(nearby_chain_pairs(model)))
        added = []
        for b1, b2 in pairs:
            try:
                exc = spec.add_exclude()
                exc.name = f"{b1}__{b2}"
                exc.bodyname1 = b1
                exc.bodyname2 = b2
                added.append((b1, b2))
            except ValueError:
                # MuJoCo tu them san exclude cho cac cap cha-con luc doc URDF,
                # them lai se bao trung ten — bo qua, no da duoc loai roi.
                pass
        if added:
            print(
                f"  loai {len(added)} cap va cham noi bo (cac than gan nhau tren"
                f" cung mot chan); va cham chan trai <-> chan phai van giu nguyen"
            )
            if len(added) <= 8:
                print("    " + ", ".join(f"{a}<->{b}" for a, b in added))
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


def pose_now(model, data):
    """Tu the hien tai = gia tri o bang Control, ghi theo TEN khop.

    Ghi theo ten chu khong theo chi so, vi MuJoCo va Isaac danh so khop khac
    nhau (MuJoCo xep het chan trai roi toi chan phai; Isaac xep theo cap
    trai-phai). Chep theo chi so giua hai ben la sai.
    """
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): round(
            float(data.ctrl[i]), 6
        )
        for i in range(model.nu)
    }


def load_poses(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"model": None, "poses": []}


def write_poses(path, model_name, poses):
    with open(path, "w") as f:
        json.dump({"model": model_name, "poses": poses}, f, indent=2)


def play_poses(model, data, model_name, path, seg_seconds, pairs):
    """Chay lien tuc qua cac pose da luu, noi suy tuyen tinh giua chung."""
    doc = load_poses(path)
    poses = doc.get("poses", [])
    if len(poses) < 2:
        sys.exit(f"'{path}' chi co {len(poses)} pose, can it nhat 2 de chay.")
    if doc.get("model") and doc["model"] != model_name:
        print(
            f"  !! canh bao: file luu cho model '{doc['model']}', dang mo '{model_name}'"
        )

    idx = {}
    for i in range(model.nu):
        idx[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = i
    thieu = set(poses[0]["ctrl"]) - set(idx)
    if thieu:
        sys.exit(f"File co khop khong ton tai trong model: {sorted(thieu)}")

    # dat robot dung ngay tu the dau, khoi phai roi tu tren xuong
    for name, value in poses[0]["ctrl"].items():
        joint = model.actuator_trnid[idx[name], 0]
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    for name, value in poses[0]["ctrl"].items():
        data.ctrl[idx[name]] = value

    print(
        f"  CHAY {len(poses)} pose tu '{path}', {seg_seconds}s moi doan"
        f" (tong {seg_seconds * (len(poses) - 1):.1f}s), lap lai lien tuc"
    )

    steps_per_frame = max(1, round((1.0 / 60.0) / model.opt.timestep))
    seg_steps = max(1, round(seg_seconds / model.opt.timestep))
    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            for _ in range(steps_per_frame):
                seg = (step // seg_steps) % (len(poses) - 1)
                f = (step % seg_steps) / seg_steps
                a, b = poses[seg]["ctrl"], poses[seg + 1]["ctrl"]
                for name in a:
                    data.ctrl[idx[name]] = a[name] + f * (b[name] - a[name])
                for left, right in pairs:
                    data.ctrl[right] = data.ctrl[left]
                mujoco.mj_step(model, data)
                step += 1
            viewer.sync()
            rest = 1.0 / 60.0 - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="ten model (khong can duoi .urdf)")
    ap.add_argument("--list", action="store_true", help="liet ke model co san")
    ap.add_argument("--fixed", action="store_true", help="base link gan cung vao world")
    ap.add_argument(
        "--no-actuators", action="store_true", help="khong them slider dieu khien khop"
    )
    ap.add_argument(
        "--armature",
        type=float,
        default=0.08,
        help="quan tinh rotor moi khop (mac dinh 0.08 nhu Isaac; 0 = tat)",
    )
    ap.add_argument(
        "--keep-overlaps",
        action="store_true",
        help="giu va cham giua cac than long vao nhau (mac dinh la loai)",
    )
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="che do guong: keo slider chan trai thi chan phai lam theo y het",
    )
    ap.add_argument(
        "--poses",
        metavar="FILE.json",
        help="bat che do luu pose: nhan ENTER trong viewer de ghi tu the hien tai",
    )
    ap.add_argument(
        "--play",
        metavar="FILE.json",
        help="chay lien tuc qua cac pose da luu trong file",
    )
    ap.add_argument(
        "--seg",
        type=float,
        default=1.5,
        help="so giay cho moi doan khi --play (mac dinh 1.5)",
    )
    ap.add_argument(
        "--export", metavar="FILE.xml", help="ghi ra file MJCF thay vi mo viewer"
    )
    args = ap.parse_args()

    models = find_models()

    if args.list or not args.model:
        print(f"Co {len(models)} model URDF trong repo:\n")
        for name, path in models.items():
            needed = mesh_names(path)
            _, hit = pick_meshdir(path, needed)
            ndof = sum(
                1
                for j in ET.parse(path).getroot().iter("joint")
                if j.get("type") in ("revolute", "continuous", "prismatic")
            )
            flag = "" if hit == len(needed) else f"  [thieu {len(needed) - hit} mesh]"
            print(f"  {name:<12} {ndof:>2} DOF   {os.path.relpath(path, ROOT)}{flag}")
        print("\nChay:  python mujoco_view.py <ten>")
        return

    if args.model not in models:
        sys.exit(f"Khong co model '{args.model}'. Co: {', '.join(models)}")

    path = models[args.model]
    print(f"Load {args.model}: {os.path.relpath(path, ROOT)}")
    spec, model = build(
        path,
        fixed_base=args.fixed,
        add_actuators=not args.no_actuators,
        armature=args.armature,
        drop_overlaps=not args.keep_overlaps,
    )
    print(
        f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}"
    )

    if args.export:
        with open(args.export, "w") as f:
            f.write(spec.to_xml())
        print(f"  da ghi MJCF -> {args.export}")
        return

    import mujoco.viewer

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # --- che do guong: keo slider chan trai, chan phai lam theo --------------
    # URDF da lo phan doi xung trong truc quay roi (vd Bubleft axis=[-1,0,0],
    # Bubright axis=[+1,0,0]) nen chi can COPY Y NGUYEN gia tri, khong doi dau.
    # Kiem chung: dat ca hai = +0.5 rad thi hai ban chan doi xung qua mat phang
    # doc (y = -0.2567 va +0.2562).
    pairs = []
    if args.mirror:
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name or "left" not in name:
                continue
            twin = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("left", "right")
            )
            if twin >= 0:
                pairs.append((i, twin))
        if not pairs:
            sys.exit("Model nay khong co cap khop left/right nao de ghep guong.")
        print(f"  che do GUONG: {len(pairs)} khop phai bam theo khop trai")
        print("    -> chi keo slider '...left...', slider '...right...' tu di theo")

    if args.play:
        play_poses(model, data, args.model, args.play, args.seg, pairs)
        return

    # Viewer mac dinh (launch) day du tinh nang hon nhung khong cho ta chen
    # phim tat hay ghi de ctrl moi buoc. Chi doi sang launch_passive khi that
    # su can: che do guong, hoac luu pose bang phim Enter.
    if not pairs and not args.poses:
        mujoco.viewer.launch(model, data)
        return

    path = args.poses
    poses = load_poses(path)["poses"] if path else []
    if path:
        print(f"  LUU POSE -> {path}  (dang co {len(poses)} pose)")
        print("    nhan ENTER trong cua so viewer de luu tu the hien tai")

    def on_key(keycode):
        # 257 = Enter, 335 = Enter ban phim so
        if keycode not in (257, 335) or not path:
            return
        poses.append({"name": f"K{len(poses) + 1}", "ctrl": pose_now(model, data)})
        write_poses(path, args.model, poses)
        print(f"  [luu] pose #{len(poses)} -> {path}")

    steps_per_frame = max(1, round((1.0 / 60.0) / model.opt.timestep))
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            for _ in range(steps_per_frame):
                for left, right in pairs:
                    data.ctrl[right] = data.ctrl[left]
                mujoco.mj_step(model, data)
            viewer.sync()
            rest = 1.0 / 60.0 - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
