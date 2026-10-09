"""Task đi bộ cho OFFICIALdesign.

File này CHỈ chứa thứ riêng của task đi bộ: phần thưởng, điều kiện kết thúc,
lệnh vận tốc mục tiêu và các buffer phục vụ tính thưởng.

Giao diện điều khiển — thứ tự khớp, dấu, mã hoá obs/action, chuẩn hoá, tần số,
trễ actuator — nằm ở ``interface.py`` và dùng chung cho MỌI task của robot này.
Đó là hợp đồng với firmware trên vi điều khiển; xem phần đầu file đó.

Thêm task mới cho robot này (ví dụ đứng dậy): tạo ``task_standup.py``, kế thừa
``OfficialInterfaceEnv`` và chỉ cài ``_get_rewards`` / ``_get_dones``. Đừng
chép lại phần giao diện.
"""

import torch

from isaaclab.utils.configclass import configclass

from .._shared.lab3 import as_torch
from .interface import OfficialInterfaceCfg, OfficialInterfaceEnv, rpy_xyzw
from .robot import BUILD, CALIBRATION
#thêm gait clock 
from .gait_clock import build_gait_clock, lookup_gait_clock

@configclass
class OfficialWalkEnvCfg(OfficialInterfaceCfg): #config 
    episode_length_s = 10.0
    target_velocity = 0.15  # fixed +x command; no unobserved random direction
    min_base_height = 0.20
    max_tilt = 0.9
    target_height = BUILD["spawn_height_m"] - CALIBRATION["spawn_clearance_m"]

    # cấu hình reward gait clock 
    swing_ratio = 0.35
    strict_relaxer = 0.1 #hệ số chuyển pha 
    stance_mode = "grounded"
    have_incentive = False

    # Một chân chống đơn gánh khoảng trọng lượng robot:
    # 4.643 kg * 9.81 m/s² ≈ 45 N.
    max_foot_force = 45.0
    max_foot_speed = 0.5  # tốc độ bàn chân m/s 
    # trọng số của lực và vận tốc (chỉnh trong rew shaping)
    w_clock_frc = 0.5
    w_clock_vel = 0.5


class OfficialWalkEnv(OfficialInterfaceEnv):
    cfg: OfficialWalkEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Buffer chỉ dùng cho reward -> thuộc task, không thuộc hợp đồng.
        self.feet_ids, _ = self.robot.find_bodies(["Footleft", "Footright"], preserve_order=True)
        self.contact_ids = [self.contact.body_names.index(name) for name in ("Footleft", "Footright")]
        self.sole_offsets = torch.tensor([BUILD["feet"][name]["sole_z_local_m"]
                                          for name in ("Footleft", "Footright")], device=self.device)

        # build gait clock  
        self.clock_table = build_gait_clock(
            swing_ratio=self.cfg.swing_ratio,
            strict_relaxer=self.cfg.strict_relaxer,
            stance_mode=self.cfg.stance_mode,
            have_incentive=self.cfg.have_incentive,
            device=self.device,
        )

    # REWARD SHAPING 
    def _get_rewards(self):
        """Track +x walking, upright height and swing clearance, penalize slip/effort."""
        orientation = rpy_xyzw(as_torch(self.robot.data.root_quat_w))

        position = as_torch(self.robot.data.root_pos_w) - self.scene.env_origins
        velocity = as_torch(self.robot.data.root_com_lin_vel_b)

        gyro = as_torch(self.robot.data.root_com_ang_vel_b)
        # lực và vận tốc của bàn chân 
        feet_vel = as_torch(self.robot.data.body_link_lin_vel_w)[:, self.feet_ids]
        forces = as_torch(self.contact.data.net_forces_w)
        # Contact order follows the sensor; map by names independently of articulation order.
        foot_force = torch.linalg.vector_norm(forces[:, self.contact_ids], dim=-1)
        foot_speed = torch.linalg.vector_norm(feet_vel, dim=-1)
        touching = foot_force > 1.0

        #chặn cho giá trị max trước khi chia để kq trong 0 1 
        force_norm = (foot_force.clamp(max=self.cfg.max_foot_force)/ self.cfg.max_foot_force)
        speed_norm = (foot_speed.clamp(max=self.cfg.max_foot_speed)/ self.cfg.max_foot_speed)

        # lấy pha và lấy bảng tra đồng hồ 
        phase = self._get_gait_phase()
        clock = lookup_gait_clock(self.clock_table, phase)

        # Bảng: [r_frc, r_vel, l_frc, l_vel].
        # Dữ liệu đo: [trái, phải] → lấy hệ số theo cùng thứ tự.
        force_clock = clock[:, [2, 0]]
        speed_clock = clock[:, [3, 1]]

        #công thức lấy chuẩn theo source apex, đưa mỗi chân về điểm trong khoảng -1 1 
        frc_score = torch.tan((torch.pi / 4.0) * force_clock * force_norm).sum(dim=-1)
        vel_score = torch.tan((torch.pi / 4.0) * speed_clock * speed_norm).sum(dim=-1)   
        
        feet_z = as_torch(self.robot.data.body_link_pos_w)[:, self.feet_ids, 2] - self.scene.env_origins[:, None, 2]
        # độ cao của chân: khi chân bay thưởng cả phần nhấc chân
        sole_height = feet_z + self.sole_offsets  # flat-foot approximation of STL sole height

        # hệ số lực âm ở pha cần chân vung -> chuyển thành mức kích hoạt thưởng nhấc chân cao 
        # force clock: chân cần vung -> hệ số lực = -1, chống hệ số phạt  = 0 => với thưởng nhấc chân cao thì đảo ngược lại giống trị tuyệt đối 
        swing_gate = (-force_clock).clamp(0.0, 1.0) # thứ tự: trái. phải 

        # điểm nhấc cao: sai lệch so với mục tiêu chân cao 3.5cm, bình phương dể lệch cao hay thấp đều bị giảm , cấu trúc 1/e mũ 
        height_score = torch.exp(-((sole_height - 0.035) / 0.025)**2)
        swing = (swing_gate * (~touching) * height_score).mean(dim=-1) # touching: tensor boolean 
        
        effort = as_torch(self.robot.data.applied_torque)[:, self.joint_ids]
        
        reward = (1.5*torch.exp(-((velocity[:, 0]-self.cfg.target_velocity)/0.20)**2)
                  + 0.5*torch.exp(-orientation[:, :2].square().sum(-1)/0.12)
                  + 0.5*torch.exp(-((position[:, 2]-self.cfg.target_height)/0.06)**2)
                  # thưởng gait clock 
                  + self.cfg.w_clock_frc * frc_score 
                  + self.cfg.w_clock_vel * vel_score
                  + 1.0 * swing #thưởng khi vung chân cao ở pha nhấc chân
                  # tăng trọng số swing lên 1, cái cũ là 0.15, kết quả: có tiến bộ, đã nhấc đều 

                  - 0.5*velocity[:, 1].square() - 0.1*gyro[:, 2].square()
                  - 0.2*orientation[:, 2].square() - 0.2*position[:, 1].square()
                  - 0.02*(touching*feet_vel[:, :, :2].square().sum(-1)).sum(-1)
                  - 0.001*effort.square().sum(-1)
                  - 0.02*(self.actions-self.previous_actions).square().sum(-1)
                  - 0.05*((self.cmd_actions-self.base_pose)/45).square().mean(-1))
        # Log trung bình toàn batch; không thay đổi reward log ra tensorboard 
        self.extras["log"] = {
            "Gait/frc_score": frc_score.mean(),
            "Gait/vel_score": vel_score.mean(),
            "Gait/swing": swing.mean(),
        }

        return reward * self.step_dt
        
    

    def _get_dones(self):
        height = as_torch(self.robot.data.root_pos_w)[:, 2] - self.scene.env_origins[:, 2]
        tilt = rpy_xyzw(as_torch(self.robot.data.root_quat_w))[:, :2].abs().amax(-1)
        return (height < self.cfg.min_base_height) | (tilt > self.cfg.max_tilt), self.episode_length_buf >= self.max_episode_length-1
