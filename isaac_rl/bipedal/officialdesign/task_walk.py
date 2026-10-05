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


@configclass
class OfficialWalkEnvCfg(OfficialInterfaceCfg):
    episode_length_s = 10.0
    target_velocity = 0.15  # fixed +x command; no unobserved random direction
    min_base_height = 0.20
    max_tilt = 0.9
    target_height = BUILD["spawn_height_m"] - CALIBRATION["spawn_clearance_m"]


class OfficialWalkEnv(OfficialInterfaceEnv):
    cfg: OfficialWalkEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Buffer chỉ dùng cho reward -> thuộc task, không thuộc hợp đồng.
        self.feet_ids, _ = self.robot.find_bodies(["Footleft", "Footright"], preserve_order=True)
        self.contact_ids = [self.contact.body_names.index(name) for name in ("Footleft", "Footright")]
        self.sole_offsets = torch.tensor([BUILD["feet"][name]["sole_z_local_m"]
                                          for name in ("Footleft", "Footright")], device=self.device)

    def _get_rewards(self):
        """Track +x walking, upright height and swing clearance, penalize slip/effort."""
        orientation = rpy_xyzw(as_torch(self.robot.data.root_quat_w))
        position = as_torch(self.robot.data.root_pos_w) - self.scene.env_origins
        velocity = as_torch(self.robot.data.root_com_lin_vel_b)
        gyro = as_torch(self.robot.data.root_com_ang_vel_b)
        feet_vel = as_torch(self.robot.data.body_link_lin_vel_w)[:, self.feet_ids]
        forces = as_torch(self.contact.data.net_forces_w)
        # Contact order follows the sensor; map by names independently of articulation order.
        touching = torch.linalg.vector_norm(forces[:, self.contact_ids], dim=-1) > 1.0
        feet_z = as_torch(self.robot.data.body_link_pos_w)[:, self.feet_ids, 2] - self.scene.env_origins[:, None, 2]
        sole_height = feet_z + self.sole_offsets  # flat-foot approximation of STL sole height
        # TODO: swing dùng .mean(-1) nên HAI chân bay được 1.0 còn MỘT chân bay chỉ 0.5
        #       -> số hạng này thưởng nhảy gấp đôi bước đi. Xem docs/ALGO.md §2.2.1
        #       và bản vá march_alt ở §2.2.2.
        swing = ((~touching) * torch.exp(-((sole_height-0.035)/0.025)**2)).mean(-1)
        effort = as_torch(self.robot.data.applied_torque)[:, self.joint_ids]
        reward = (1.5*torch.exp(-((velocity[:, 0]-self.cfg.target_velocity)/0.20)**2)
                  + 0.5*torch.exp(-orientation[:, :2].square().sum(-1)/0.12)
                  + 0.5*torch.exp(-((position[:, 2]-self.cfg.target_height)/0.06)**2)
                  + 0.15*swing
                  - 0.5*velocity[:, 1].square() - 0.1*gyro[:, 2].square()
                  - 0.2*orientation[:, 2].square() - 0.2*position[:, 1].square()
                  - 0.02*(touching*feet_vel[:, :, :2].square().sum(-1)).sum(-1)
                  - 0.001*effort.square().sum(-1)
                  - 0.02*(self.actions-self.previous_actions).square().sum(-1)
                  - 0.05*((self.cmd_actions-self.base_pose)/45).square().mean(-1))
        return reward*self.step_dt

    def _get_dones(self):
        height = as_torch(self.robot.data.root_pos_w)[:, 2] - self.scene.env_origins[:, 2]
        tilt = rpy_xyzw(as_torch(self.robot.data.root_quat_w))[:, :2].abs().amax(-1)
        return (height < self.cfg.min_base_height) | (tilt > self.cfg.max_tilt), self.episode_length_buf >= self.max_episode_length-1
