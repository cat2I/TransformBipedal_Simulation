"""
set_pose.py — Sân chơi vật lý để ĐẶT POSE cho Fulltrans10DOF (KHÔNG train RL)
============================================================================

Mục đích
--------
Dựng một thế giới vật lý tối giản trong Isaac Sim rồi thả robot Fulltrans10DOF
vào đó để bạn tự BẺ CÁC KHỚP tới góc mong muốn và quan sát. Không có phần thưởng,
không có mạng nơ-ron, không training — chỉ vật lý thuần: TRỌNG LỰC + SÀN + MA SÁT.

Thế giới gồm 4 thành phần:
  1. SimulationContext  — bộ máy vật lý, đặt bước thời gian (dt) và TRỌNG LỰC.
  2. GroundPlaneCfg     — SÀN, gắn vật liệu có MA SÁT (static/dynamic friction).
  3. DomeLightCfg       — đèn, để nhìn thấy robot trong cửa sổ viewer.
  4. Articulation       — chính con robot, nạp từ TRANSFORMER_10DOF_CFG có sẵn.

Hai chế độ
----------
  * (mặc định)  VẬT LÝ THẬT: đặt pose xong thả ra, trọng lực tác động. Robot có
                thể đứng vững hoặc ngã — dùng để kiểm tra pose có ỔN ĐỊNH không.
  * --freeze    ĐÓNG BĂNG: mỗi khung hình ép robot về đúng pose. Đứng im như
                tượng, không ngã — dùng để NGẮM hình dáng pose.

Cách chạy (từ thư mục transformer_nam/): thêm viz kit ở cuối vì là bản isaaclab 3.0
  ./run.sh source/bipedal_vinh/scripts/set_pose.py --viz kit                # vật lý thật, robot tự cân bằng/ngã
  ./run.sh source/bipedal_vinh/scripts/set_pose.py --freeze        # đóng băng pose để ngắm
  ./run.sh source/bipedal_vinh/scripts/set_pose.py --headless      # không mở cửa sổ (chạy ngầm)

Muốn ĐỔI POSE: sửa dict POSE_DEG bên dưới (đơn vị ĐỘ) rồi chạy lại. Đó là toàn
bộ những gì bạn cần đụng tới cho công việc set-pose hằng ngày.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# ─────────────────────────────────────────────────────────────────────────────
# 1) THAM SỐ DÒNG LỆNH + KHỞI ĐỘNG ISAAC SIM
#    Bắt buộc: phải bật app TRƯỚC khi import bất cứ thứ gì thuộc isaaclab.assets
#    / isaaclab.sim, vì các module đó cần lõi Omniverse đã chạy.
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Đặt pose cho Fulltrans10DOF trong thế giới vật lý."
)
parser.add_argument(
    "--freeze",
    action="store_true",
    help="Đóng băng: ép robot giữ nguyên pose mỗi khung hình (không ngã).",
)
parser.add_argument(
    "--base_height",
    type=float,
    default=None,
    help="Ghi đè độ cao gốc (base) của robot theo mét. Bỏ trống = dùng giá trị "
    "trong cfg (0.37 m). Tăng lên (vd 0.6) để robot rơi tự do rồi tiếp đất.",
)
AppLauncher.add_app_launcher_args(parser)  # thêm --headless, --device, v.v.
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
# 2) IMPORT PHẦN CÒN LẠI (chỉ hợp lệ SAU khi app đã bật)
# ─────────────────────────────────────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Tái dùng đúng cfg robot đã hiệu chỉnh sẵn (khớp, động cơ, quán tính...).
# Import được là nhờ run.sh đã thêm source/ vào PYTHONPATH.
from bipedal_vinh.tasks.vinh10dof_config import (
    TRANSFORMER_10DOF_CFG,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) POSE MONG MUỐN — đơn vị ĐỘ (độ dễ hình dung hơn radian).
#    0° = tư thế đứng thẳng (theo init_state trong cfg). Sửa số ở đây để đổi pose.
#    Tên khớp phải khớp joint_names trong USD:
#      Bub*   = hông FRONTAL (xoạc chân, trục roll)
#      Hip*   = hông SAGITTAL (bước tới/lui, trục pitch) — khớp chính khi đi
#      Twist* = xoay hông (rẽ hướng)
#      Knee*  = đầu gối
#      Foot*  = cổ chân
# ─────────────────────────────────────────────────────────────────────────────
POSE_DEG: dict[str, float] = {
    "Bubleft_joint": 0.0,
    "Bubright_joint": 0.0,
    "Hipleft_joint": 0.0,
    "Hipright_joint": 0.0,
    "Twistleft_joint": 0.0,
    "Twistright_joint": 0.0,
    "Kneeleft_joint": 0.0,
    "Kneeright_joint": 0.0,
    "Footleft_joint": 0.0,
    "Footright_joint": 0.0,
}


def main() -> None:
    # ── 3a. Bộ máy vật lý + TRỌNG LỰC ────────────────────────────────────────
    # dt = 1/120 s: nửa bước so với 60 Hz mặc định, khớp co giãn ổn định hơn.
    # gravity: vector gia tốc trọng trường, -9.81 theo trục Z (xuống đất).
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.6, 1.0], target=[0.0, 0.0, 0.3])

    # ── 3b. SÀN + MA SÁT ─────────────────────────────────────────────────────
    # RigidBodyMaterialCfg quyết định độ bám giữa chân robot và sàn:
    #   static_friction  — ma sát nghỉ (chống bắt đầu trượt)
    #   dynamic_friction — ma sát trượt (khi đã trượt rồi)
    #   restitution      — độ nảy; 0.0 = không nảy (đất cứng bình thường)
    # Ma sát THỰC TẾ khi va chạm = kết hợp vật liệu sàn NÀY với vật liệu chân
    # robot (định nghĩa trong USD). Nên đây là "một nửa" của độ bám tổng.
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/ground", ground_cfg)

    # ── 3c. ĐÈN (chỉ để nhìn, không ảnh hưởng vật lý) ─────────────────────────
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    # ── 3d. ROBOT ────────────────────────────────────────────────────────────
    # cfg gốc dùng prim_path "{ENV_REGEX_NS}/Robot" (dành cho nhiều env song song
    # khi train). Ở đây chỉ 1 robot nên đổi thành đường dẫn tuyệt đối cố định.
    robot_cfg = TRANSFORMER_10DOF_CFG.replace(prim_path="/World/Robot")
    if args_cli.base_height is not None:
        # copy~ init_state để ghi đè độ cao gốc mà không đụng cfg gốc.
        init = robot_cfg.init_state.replace(
            pos=(
                robot_cfg.init_state.pos[0],
                robot_cfg.init_state.pos[1],
                args_cli.base_height,
            )
        )
        robot_cfg = robot_cfg.replace(init_state=init)
    robot = Articulation(robot_cfg)

    # Dựng xong scene → khởi động vật lý. Sau reset() mới đọc/ghi được state.
    sim.reset()
    print("[set_pose] Thứ tự khớp trong sim:", robot.joint_names)

    # ── 3e. ÁP POSE MONG MUỐN ────────────────────────────────────────────────
    # default_joint_pos lấy từ init_state trong cfg; ta chép ra rồi ghi đè theo
    # POSE_DEG (đổi độ → radian). Dùng find_joints để lấy đúng chỉ số của từng
    # khớp theo TÊN, không phụ thuộc thứ tự khớp trong USD.
    target_q = robot.data.default_joint_pos.clone()  # shape (num_robot, num_dof)
    for joint_name, deg in POSE_DEG.items():
        ids, _ = robot.find_joints(joint_name)  # ('names' hỗ trợ cả regex)
        target_q[:, ids] = math.radians(deg)

    # write_joint_state: "dịch chuyển tức thời" khớp về pose (bỏ qua động lực học).
    robot.write_joint_state_to_sim(target_q, torch.zeros_like(target_q))
    # set_joint_position_target: giao PD của động cơ GIỮ pose này lại.
    robot.set_joint_position_target(target_q)

    # Đặt gốc (base) về pose ban đầu trong cfg (vị trí + quaternion + vận tốc 0).
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_data_to_sim()

    mode = "ĐÓNG BĂNG (giữ pose)" if args_cli.freeze else "VẬT LÝ THẬT (có thể ngã)"
    print(f"[set_pose] Chế độ: {mode}. Đóng cửa sổ hoặc Ctrl+C để thoát.")

    # ── 3f. VÒNG LẶP MÔ PHỎNG ────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if args_cli.freeze:
            # Ép robot về đúng pose mỗi khung hình → đứng im, không ngã.
            robot.write_joint_state_to_sim(target_q, torch.zeros_like(target_q))
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
        else:
            # Vật lý thật: chỉ ra lệnh PD giữ góc, còn lại để trọng lực quyết định.
            robot.set_joint_position_target(target_q)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
