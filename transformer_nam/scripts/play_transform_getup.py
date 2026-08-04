import argparse
import math
import numpy as np

from omni.isaac.lab.app import AppLauncher

# ====================== KHỞI TẠO APP LAUNCHER ======================
parser = argparse.ArgumentParser(description="Test Transformer robot get-up from lying pose")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ====================== IMPORT SAU KHI APP LAUNCHER ======================
from omni.isaac.lab.sim import SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

# Import config của bạn (sửa tên nếu cần)
from config_transformer import TRANSFORMER_CFG   # ← đổi nếu tên file config khác

def main():
    # Tạo môi trường
    sim = SimulationContext()
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.4])

    # Spawn robot
    robot_cfg = TRANSFORMER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot = Articulation(robot_cfg)
    robot.spawn(translations=[(0.0, 0.0, 0.0)])

    sim.reset()
    robot.initialize()

    print("\n=== Robot Get-Up Test Started ===")
    print("Phase 0: Đặt pose nằm giống hình bạn gửi...")

    # ====================== PHASE 0: Pose nằm ban đầu (gần giống hình) ======================
    initial_joint_pos = np.array([
        0.0,                    # Bubleft
        0.0,                    # Bubright
        math.radians(10.0),     # Hipleft
        math.radians(-15.0),    # Hipright
        math.radians(-20.0),    # Kneeleft
        math.radians(-5.0),     # Kneeright
        math.radians(8.0),      # Footleft
        math.radians(2.0),      # Footright
    ])

    joint_names = ["Bubleft_joint", "Bubright_joint", "Hipleft_joint", "Hipright_joint",
                   "Kneeleft_joint", "Kneeright_joint", "Footleft_joint", "Footright_joint"]

    joint_ids = robot.find_joints(joint_names)[0]

    robot.write_joint_state_to_sim(
        joint_pos=initial_joint_pos,
        joint_vel=np.zeros(8),
        joint_ids=joint_ids
    )
    sim.step(render=True)

    input("\nNhấn Enter để bắt đầu Phase 1: Nằm xuống sát sàn...")

    # ====================== PHASE 1: Nằm xuống ======================
    print("Phase 1: Nằm xuống sát sàn...")
    lie_joint_pos = np.array([
        0.0, math.radians(-35.0), math.radians(-30.0), math.radians(-25.0),
        math.radians(-18.0), math.radians(-18.0), math.radians(12.0), math.radians(12.0)
    ])  # thứ tự: BubL, BubR, HipL, HipR, KneeL, KneeR, FootL, FootR

    for _ in range(100):   # di chuyển chậm
        robot.set_joint_position_target(lie_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=True)

    input("\nNhấn Enter để Phase 2: Co chân sâu lấy đà...")

    # ====================== PHASE 2: Crouch thấp ======================
    print("Phase 2: Co chân sâu (frog pose)...")
    crouch_joint_pos = np.array([
        0.0, 0.0,
        math.radians(-28.0), math.radians(-28.0),
        math.radians(-118.0), math.radians(-118.0),
        math.radians(38.0), math.radians(38.0)
    ])

    for _ in range(80):
        robot.set_joint_position_target(crouch_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=True)

    input("\nNhấn Enter để BẬT MẠNH đứng dậy (Phase 3)...")

    # ====================== PHASE 3: Bật mạnh đứng dậy ======================
    print("Phase 3: Explosive Stand-up !!!")
    stand_joint_pos = np.array([
        0.0, 0.0,
        math.radians(20.0), math.radians(20.0),
        math.radians(-60.0), math.radians(-60.0),
        math.radians(15.0), math.radians(15.0)
    ])

    # Boost torque tạm thời cho heavy joints
    heavy_ids = robot.find_joints(["Bubleft_joint", "Bubright_joint", "Hipleft_joint", "Hipright_joint",
                                   "Kneeleft_joint", "Kneeright_joint"])[0]

    for i in range(50):      # ~0.5 giây
        if i < 20:           # boost mạnh ở đầu
            efforts = np.array([18.0, 18.0, 22.0, 22.0, 22.0, 22.0])
            robot.set_joint_effort_target(efforts, joint_ids=heavy_ids)

        robot.set_joint_position_target(stand_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=True)

    print("\n=== Hoàn thành một lần đứng dậy ===")
    print("Nhấn Enter để đóng cửa sổ...")

    input()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        simulation_app.close()