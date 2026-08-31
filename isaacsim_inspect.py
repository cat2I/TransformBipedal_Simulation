#!/usr/bin/env python3
"""Mở robot URDF trong Isaac Sim GUI gốc để xem và chỉnh các khớp tương tác.

Script này dùng TRỰC TIẾP URDF Importer gốc của Isaac Sim (KHÔNG qua
pipeline RL của IsaacLab). Nhờ vậy, tất cả khớp revolute sẽ có đầy đủ
mục 'Angular Drive' trong bảng Property với:
  - Target Position  (kéo thanh trượt để xoay khớp)
  - Stiffness        (độ cứng lò xo)
  - Damping          (độ giảm chấn)

Cách dùng:
    python isaacsim_inspect.py Fulltrans_meshfixed   # mở GUI, base gắn cứng
    python isaacsim_inspect.py --list                 # liệt kê model

Hướng dẫn sau khi GUI mở:
    1. Bảng Stage (phải): click chọn một khớp (joint)
    2. Bảng Property (dưới phải): cuộn tìm mục 'Angular Drive'
    3. Kéo thanh 'Target Position' để xoay khớp
    4. Nhấn PLAY ▶ trên thanh Timeline để thấy robot di chuyển
    5. Đóng cửa sổ để thoát
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_models():
    """Tìm mọi file .urdf trong repo, trả về dict {tên: đường_dẫn}."""
    models = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".urdf"):
                models[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    return dict(sorted(models.items()))


def main():
    ap = argparse.ArgumentParser(
        description="Mở robot URDF trong Isaac Sim GUI để xem/chỉnh khớp."
    )
    ap.add_argument("model", nargs="?", help="tên model (không cần đuôi .urdf)")
    ap.add_argument("--list", action="store_true", help="liệt kê model có sẵn")
    args_partial, _ = ap.parse_known_args()

    models = find_models()
    if args_partial.list or not args_partial.model:
        print(f"Có {len(models)} model URDF trong repo:\n")
        for name, path in models.items():
            print(f"  {name:<20} {os.path.relpath(path, ROOT)}")
        print("\nChạy:  python isaacsim_inspect.py <tên>")
        return

    if args_partial.model not in models:
        raise SystemExit(
            f"Không có model '{args_partial.model}'. Có: {', '.join(models)}"
        )
    urdf_path = models[args_partial.model]

    # ── Boot Isaac Sim với GUI ───────────────────────────────────────────
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    # Ép mở giao diện nếu người dùng không chỉ định
    if not any(arg in sys.argv for arg in ["--headless", "--viz", "--visualizer"]):
        sys.argv.extend(["--viz", "kit"])
    args_cli = ap.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ── Import URDF bằng API gốc của Isaac Sim 6.x ──────────────────────
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    import_config = URDFImporterConfig(
        urdf_path=urdf_path,
        fix_base=True,                        # gắn cứng base link
        merge_fixed_joints=False,              # giữ nguyên tất cả link
        joint_drive_type="force",              # force drive
        joint_target_type="position",          # position target
        override_joint_stiffness=10000.0,      # stiffness cho tất cả khớp
        override_joint_damping=500.0,          # damping cho tất cả khớp
    )

    importer = URDFImporter(import_config)

    print(f"\n  Đang import: {os.path.relpath(urdf_path, ROOT)}")
    usd_path = importer.import_urdf()
    print(f"  → File USD tạm: {usd_path}")

    # ── Nạp USD lên stage hiện tại ──────────────────────────────────────
    import omni.usd
    import isaaclab.sim as sim_utils

    stage_context = omni.usd.get_context()

    # Thêm robot lên stage đang mở bằng cách reference file USD vừa tạo
    stage = stage_context.get_stage()

    from pxr import UsdGeom

    # Tạo SimulationContext: đây là cơ chế chuẩn của IsaacLab để gắn stage
    # hiện tại của Kit vào "current stage" nội bộ của IsaacLab (thread-local).
    # Không làm bước này thì các hàm như GroundPlaneCfg.func() bên dưới sẽ gọi
    # get_current_stage() và nhận về None -> lỗi
    # "AttributeError: 'NoneType' object has no attribute 'GetPrimAtPath'".
    # Tạo context cũng tự tạo PhysicsScene mặc định (gravity -9.81 m/s^2 theo Z),
    # nên không cần tự tạo PhysicsScene bằng tay nữa.
    #
    # device="cpu": mặc định IsaacLab chạy physics trên GPU (để train nhanh),
    # nhưng chế độ GPU bật "Direct GPU API" cho khớp -> PhysX CHẶN việc set
    # Target Position qua Property panel bằng tay, báo lỗi
    # "PxArticulationJointReducedCoordinate::setDriveTarget(): it is illegal
    # to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!".
    # Script này dùng để kéo-thả khớp bằng tay nên cần CPU để đường ghi giá
    # trị "thường" (không qua GPU) hoạt động.
    sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))

    # Thêm robot reference
    robot_prim = stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetReferences().AddReference(usd_path)

    # Nâng robot lên khỏi mặt đất
    xform = UsdGeom.Xformable(robot_prim)
    xform.AddTranslateOp().Set((0.0, 0.0, 0.5))

    # Sàn + đèn
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # ── In hướng dẫn ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HƯỚNG DẪN CHỈNH KHỚP TRONG GIAO DIỆN:")
    print("=" * 60)
    print("  1. Bảng Stage (phải): click chọn một khớp")
    print("     (ví dụ: Hipleft_joint, Kneeleft_joint, ...)")
    print("  2. Bảng Property (dưới phải): cuộn tìm mục 'Angular Drive'")
    print("  3. Kéo thanh 'Target Position' để xoay khớp")
    print("  4. Nhấn PLAY ▶ trên thanh Timeline để thấy robot di chuyển")
    print("  5. Đóng cửa sổ để thoát")
    print("=" * 60 + "\n")

    # ── Vòng lặp GUI ────────────────────────────────────────────────────
    # KHÔNG tự bước physics — để người dùng tự nhấn Play/Stop trong GUI
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
