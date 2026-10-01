#!/usr/bin/env python3
"""Xem nhanh cac model URDF cua repo bang Isaac Sim — KHONG qua gym/task/reward/policy.

Chi nap thang URDF len stage va cho no roi tu do duoi trong luc (giong mujoco_view.py),
dung isaaclab.sim.spawners.from_files.UrdfFileCfg de chuyen doi + spawn truc tiep.

    python isaacsim_view.py --list             # liet ke cac model co san (khong can boot Isaac Sim)
    python isaacsim_view.py Fulltrans          # mo Isaac Sim, tha robot ro tu do
    python isaacsim_view.py --fixed FullForm   # gan cung base link vao world
    python isaacsim_view.py Fulltrans_meshfixed --headless
    python isaacsim_view.py OFFICIALdesign     # robot moi, dung USD da chuan bi
    python isaacsim_view.py --fixed OFFICIALdesign  # giu base de xem

Luu y: khoi dong Isaac Sim (Omniverse Kit) van cham (~1-2 phut lan dau), khong co
cach nao rut ngan tu phia script — day la thoi gian boot engine, khong phai do
model nang hay nhe.
"""

import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")


def find_models():
    """Tim URDF trong assets, uu tien ban dang dung hon archive; bo qua handoff tho."""
    models = {}
    for dirpath, dirnames, filenames in os.walk(ASSETS):
        dirnames[:] = sorted((d for d in dirnames if not d.startswith(".")), key=lambda d: (d == "_archive", d))
        for fn in sorted(filenames):
            if fn.endswith(".urdf"):
                models.setdefault(os.path.splitext(fn)[0], os.path.join(dirpath, fn))
    return dict(sorted(models.items()))


def prepared_usd(model):
    """OFFICIALdesign dung USD da sua tensor va collision, khong import lai CAD tho."""
    if model != "OFFICIALdesign":
        return None
    path = os.path.join(ASSETS, "officialdesign", "usd", "OFFICIALdesign.usd")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chua co {path}. Chay scripts/convert_officialdesign.py trong isaac_rl truoc.")
    return path


def finish_fixed_root(stage):
    """Remove the inherited Newton root after PhysX moves the root to /World/Robot.

    The prepared USD carries both Physics and Newton root schemas. Isaac Lab's
    fix_root_link helper removes only the former, leaving two detected roots.
    This override belongs to the temporary viewing stage, not the source USD.
    """
    from pxr import UsdPhysics

    base = stage.GetPrimAtPath("/World/Robot/Baselink")
    base.RemoveAppliedSchema("NewtonArticulationRootAPI")
    base.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def find_ros_packages(urdf_path):
    """Doc URDF, tim moi ten package trong 'package://<ten>/...' va tro ve
    danh sach dict {ten: duong_dan} de UrdfConverter phan giai — gia dinh ten
    package trung voi ten mot thu muc nao do trong repo (vd axisnewbipedal/).

    Truoc 2026-09-09 cac package nam ngay o goc repo; gio chung da gom vao
    assets/ nen phai do ca cay thu muc thay vi chi nhin goc."""
    with open(urdf_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    names = sorted(set(re.findall(r'package://([^/"\s]+)/', content)))
    mappings = []
    for name in names:
        candidate = os.path.dirname(os.path.dirname(urdf_path))
        if not os.path.isdir(os.path.join(candidate, "meshes")):
            candidate = os.path.join(ASSETS, name)
        if not os.path.isdir(candidate):
            candidate = None
            for dirpath, dirnames, _ in os.walk(ASSETS):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                if name in dirnames:
                    candidate = os.path.join(dirpath, name)
                    break
        if candidate:
            mappings.append({"name": name, "path": candidate})
        else:
            print(f"  !! khong tim thay thu muc cho package '{name}', bo qua")
    return mappings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="ten model (khong can duoi .urdf)")
    ap.add_argument("--list", action="store_true", help="liet ke model co san")
    ap.add_argument("--fixed", action="store_true", help="gan cung base link vao world")
    ap.add_argument("--steps", type=int, default=0, help="thoat sau N buoc physics; 0 = xem lien tuc")
    args_partial, _ = ap.parse_known_args()

    models = find_models()
    if args_partial.list or not args_partial.model:
        print(f"Co {len(models)} model URDF trong repo:\n")
        for name, path in models.items():
            print(f"  {name:<20} {os.path.relpath(path, ROOT)}")
        print("\nChay:  python isaacsim_view.py <ten>")
        return
    if args_partial.model not in models:
        raise SystemExit(f"Khong co model '{args_partial.model}'. Co: {', '.join(models)}")
    urdf_path = models[args_partial.model]
    usd_path = prepared_usd(args_partial.model)

    # ── chi tu day tro di moi duoc boot Isaac Sim ──────────────────────────
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    
    import sys
    # Tự động ép mở giao diện Isaac Sim (bản 6.x / 4.x) nếu không có cờ ẩn
    if not any(arg.split("=", 1)[0] in ("--headless", "--viz", "--visualizer") for arg in sys.argv):
        sys.argv.extend(["--viz", "kit"])
        
    args_cli = ap.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext

    print(f"Load {args_cli.model}: {os.path.relpath(usd_path or urdf_path, ROOT)}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    # san + den, giong mujoco_view.py
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    if usd_path:
        spawn = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=args_cli.fixed),
        )
    else:
        spawn = sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=args_cli.fixed,
            ros_package_paths=find_ros_packages(urdf_path),
            force_usd_conversion=True,
            collision_type="Bounding Cube",  # TEST: collision re nhat co the, de xac dinh co phai do cook mesh
        )
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        # khong dieu khien gi ca — de khop tu do roi theo trong luc, giong mujoco_view.py
        actuators={"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    robot = Articulation(cfg=robot_cfg)
    if usd_path and args_cli.fixed:
        finish_fixed_root(sim_utils.get_current_stage())

    sim.reset()
    print(f"  njoint={robot.num_joints}  nbody={robot.num_bodies}  fixed={robot.is_fixed_base}", flush=True)
    print("[INFO]: San sang — dang mo phong, dong cua so de thoat.", flush=True)

    count = 0
    while simulation_app.is_running():
        sim.step()
        robot.update(sim_cfg.dt)
        count += 1
        if args_cli.steps > 0 and count >= args_cli.steps:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
