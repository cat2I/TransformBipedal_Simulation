"""HỢP ĐỒNG GIAO DIỆN sim ↔ nhúng cho OFFICIALdesign.

Vì sao file này tồn tại
=======================
Vi điều khiển trên robot thật chạy MỘT firmware. Firmware đó giả định một thứ
tự khớp, một cách mã hoá observation, một cách biến action thành góc servo, một
tần số điều khiển. Mọi task của cùng con robot — đi bộ, đứng dậy, xoay người —
PHẢI dùng chung đúng bộ giả định đó, nếu không bạn cần một firmware cho mỗi
task (hoặc một cờ chuyển chế độ, mà quên set cờ một lần là robot nhảy loạn).

`AGENTS.md` §2 ghi yêu cầu này: *"Luôn lưu ý sự tương thích giữa Simulation và
Embedded (đặc biệt là State Space, Action Space, Reward struct, và
Communication Protocol)."*

Chuyện này đã hỏng một lần. Trên cùng một asset `Fulltrans10DOF.usd` từng có:

    transformer_walk10dof_env.py    60 obs / 10 act
    transformer_walk10dof6_env.py   44 obs /  6 act     <-- cùng robot!
    transformer_standup_env.py      60 obs / 10 act

Hai giao diện khác nhau cho một con robot. (Cả ba nay ở `_archive/fulltrans/`.)

Tách hợp đồng ra một file riêng biến yêu cầu đó từ *lời nhắc trong tài liệu*
thành *ràng buộc của cấu trúc code*: task mới kế thừa `OfficialInterfaceEnv`
là tự động dùng đúng giao diện, muốn đổi thì phải sửa file này và thấy ngay
mình đang đổi thứ mà firmware phụ thuộc vào.

Nội dung hợp đồng
=================
* 10 khớp theo thứ tự CAD, kèm dấu đảo cho chân phải (`q_urdf = sign × q_policy`)
* Giới hạn góc policy và pose mặc định, đọc từ `meta/calibration.json`
* Action: 10 số, clip [-1, 1], mỗi bước cộng tối đa 2° vào lệnh góc rồi clamp
* Observation 60D: 4 khung `[roll, pitch, gx, gy, gz]` rồi 4 khung lệnh góc 10D
  đã chuẩn hoá. Thứ tự cũ → mới
* 200 Hz vật lý, decimation 10 → policy 20 Hz
* Trễ actuator 2 substep (10 ms) — giả định, CHƯA đo trên servo thật

Thứ KHÔNG thuộc hợp đồng (mỗi task tự lo): reward, điều kiện kết thúc, lệnh
vận tốc mục tiêu, và mọi buffer chỉ phục vụ việc tính thưởng.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from .._shared.lab3 import as_torch
from .robot import CALIBRATION, JOINTS, JOINT_NAMES, MATERIAL, OFFICIAL_CFG, USD_REPORT


def rpy_xyzw(quat):
    """Euler FLU orientation, using Isaac Lab 3 xyzw without legacy IMU rotation."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
                        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), dim=-1)


@configclass
class OfficialInterfaceCfg(DirectRLEnvCfg): #giao diện điều khiển chung 
    """Phần cấu hình thuộc hợp đồng. Task kế thừa rồi thêm tham số của riêng nó."""

    #step_dt = dt * decimation -> dt = 0.005, decimation = 10 -> step_dt = 0.05
    # tính gait_ste p = step_dt * gait_steps_per_policy_step -> gait_step = 0.05 * 4 = 0.2
    decimation = 10 #mỗi lần policy đưa action, mô phỏng chạy 10 bước vật lí trc khi sang lượt poli tiếp 
    action_space = 10 #policy xuất 10 số, mỗi số điều khiển 1 khớp 
    observation_space = 62 # policy nhận 60 số mỗi lượt. thêm 2 observation của 2 pha sin cos 
    state_space = 0


    sim = sim_utils.SimulationCfg(
        dt=0.005, render_interval=decimation,
        physics=PhysxCfg(gpu_max_soft_body_contacts=2**10, gpu_max_particle_contacts=2**10,
                         gpu_max_rigid_contact_count=2**20, gpu_collision_stack_size=2**24,
                         gpu_temp_buffer_capacity=2**23, gpu_found_lost_aggregate_pairs_capacity=2**22),
    )
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)
    robot = OFFICIAL_CFG
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/IMUleft",
                 update_period=0.005, offset=ImuCfg.OffsetCfg(rot=(0.0, 0.0, 0.0, 1.0)))
    contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Foot.*",
        update_period=0.005, track_air_time=True, track_pose=True, force_threshold=1.0,
    )
    # Persist the complete control interface in each training run's env.yaml.
    joint_names = JOINT_NAMES
    joint_signs = tuple(j["sign"] for j in JOINTS)
    servo_min = tuple(j["policy_min_deg"] for j in JOINTS)
    servo_max = tuple(j["policy_max_deg"] for j in JOINTS)
    start_pos = tuple(j["policy_default_deg"] for j in JOINTS)

    calibration_status = CALIBRATION["calibration_status"]

    asset_sha256 = USD_REPORT["sha256"]
    action_step_deg = 2.0 

    #tạo bộ đếm và hàm tính pha 
    gait_period_s = 3.2

    actuator_delay_steps = 2  # 10 ms simulation assumption, not measured
    orientation_noise_std = 0.015
    gyro_noise_std = 0.01


class OfficialInterfaceEnv(DirectRLEnv):
    """Phần runtime của hợp đồng: giải tên khớp, mã hoá obs, tích luỹ action.

    Task kế thừa lớp này và CHỈ cài ``_get_rewards`` / ``_get_dones``. Cần buffer
    riêng cho việc tính thưởng thì khởi tạo trong ``__init__`` của task, sau khi
    gọi ``super().__init__()``.
    """

    cfg: OfficialInterfaceCfg #dựa vào config ở trên  

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, names = self.robot.find_joints(self.cfg.joint_names, preserve_order=True)
        if names != list(self.cfg.joint_names) or len(names) != self.robot.num_joints:
            raise ValueError(f"Unexpected joint mapping: {names}")
        self.imu_id = self.robot.find_bodies("IMUleft")[0][0]

        #giới hạn góc, dấu khớp và tư thế mựac định robot 
        self.servo_min = torch.tensor(self.cfg.servo_min, device=self.device)
        self.servo_max = torch.tensor(self.cfg.servo_max, device=self.device)
        self.signs = torch.tensor(self.cfg.joint_signs, device=self.device)
        self.base_pose = torch.tensor(self.cfg.start_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        
        if not torch.all((self.base_pose >= self.servo_min) & (self.base_pose <= self.servo_max)):
            raise ValueError("Default pose must lie inside policy angle limits")
        if not 0 <= self.cfg.actuator_delay_steps < self.cfg.decimation:
            raise ValueError("actuator_delay_steps must be in [0, decimation)")
        self.cmd_actions = self.base_pose.clone()
        self.applied_target = self._to_urdf(self.cmd_actions)
        self.previous_target = self.applied_target.clone()
        self.previous_actions = torch.zeros_like(self.cmd_actions)
        self.actions = torch.zeros_like(self.cmd_actions)

        self.imu_history = torch.zeros(self.num_envs, 4, 5, device=self.device)
        
        self.action_history = self._normalize(self.cmd_actions)[:, None, :].repeat(1, 4, 1)
        self._substep = 0
        self._obs_step = -1

        
        if self.cfg.gait_period_s <= 0:
            raise ValueError("gait_period_s must be positive")
        #tạo bộ đếm bước cho mỗi env, ban đầu tensor toàn 0
        self.gait_step = torch.zeros(
        self.num_envs, #cần bao nhiêu bộ đếm: ví dụ chạy x env cần x bộ đếm 
        dtype=torch.int64,
        device=self.device,
        )
        
        self._obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        self._fresh_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Explicitly keep runtime limits in sync with the control interface.
        bounds = torch.stack((self.servo_min*self.signs, self.servo_max*self.signs), dim=-1).sort(dim=-1).values
        self.robot.write_joint_position_limit_to_sim(torch.deg2rad(bounds).repeat(self.num_envs, 1, 1), joint_ids=self.joint_ids)
        print(
            f"OFFICIALdesign: 13 links, 10 DOF, "
            f"{self.cfg.observation_space} observations; "
            f"{self.cfg.calibration_status}"
    )
        print(f"Policy joint order: {names}; Isaac indices: {self.joint_ids}")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.imu = Imu(self.cfg.imu)
        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["imu"] = self.imu
        self.scene.sensors["contact"] = self.contact
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg(physics_material=MATERIAL))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _to_urdf(self, policy_deg):
        """Policy coordinates -> raw CAD joint radians; hardware offsets remain uncalibrated."""
        return torch.deg2rad(policy_deg*self.signs)

    def _normalize(self, policy_deg):
        return (2*(policy_deg-self.servo_min)/(self.servo_max-self.servo_min)-1).clamp(-1, 1)

    # hàm đọc bộ đếm (gait_clock.py) và tính pha 
    # interface.py giữ thời gian: tự đếm bước và tính pha bằng _get_gait_phase().
    def _get_gait_phase(self) -> torch.Tensor: #trả về 1 mảng tensor 
        """Trả pha chuẩn hóa (num_envs,) trong [0, 1).

        Đọc bộ đếm riêng của từng env; không tăng bộ đếm.
        """
        elapsed_s = self.gait_step.to(dtype=torch.float32) * self.step_dt # chuyển số bước sang thời gian 
        return torch.remainder(elapsed_s / self.cfg.gait_period_s, 1.0) # chia chu kì -> tính xem đi được bao nhiêu phần chu kì

    def _get_observations(self):
        # get_observations() can be called twice by a runner; do not shift twice.
        if self._obs_step != self.common_step_counter or self._fresh_reset.any():
            # IMU 
            angles = rpy_xyzw(as_torch(self.robot.data.body_link_quat_w)[:, self.imu_id])[:, :2] # chuyển quat thành rpy -> bỏ yaw
            gyro = as_torch(self.imu.data.ang_vel_b) #lấy vận tốc góc 
            angles = angles + torch.randn_like(angles)*self.cfg.orientation_noise_std #nhiễu cảm biến 
            gyro = gyro + torch.randn_like(gyro)*self.cfg.gyro_noise_std 
            sample = torch.cat((angles.clamp(-1, 1), (gyro/2).clamp(-1, 1)), dim=-1) # ghép thành 5 giá trị 

            normalized = self._normalize(self.cmd_actions) #ánh xạ action của từng khớp sang -1 1 
            self.imu_history[:, :-1] = self.imu_history[:, 1:].clone()
            self.action_history[:, :-1] = self.action_history[:, 1:].clone()
            self.imu_history[:, -1] = sample
            self.action_history[:, -1] = normalized
            self.imu_history[self._fresh_reset] = sample[self._fresh_reset, None, :]
            self.action_history[self._fresh_reset] = normalized[self._fresh_reset, None, :]
            self._fresh_reset[:] = False

            # lấy observation pha 
            phase = self._get_gait_phase()
            phase_angle = 2.0 * torch.pi * phase # chuyển pha từ 0 1 sang 2pi 
            # đồng hồ sin cos cho pha 
            clock_obs = torch.stack(
                (torch.sin(phase_angle), torch.cos(phase_angle)),dim=-1,
            )

            #tổng observation 
            self._obs = torch.cat(
                (
                    self.imu_history.flatten(1),
                    self.action_history.flatten(1),
                    clock_obs,
                ),
                dim=-1,
            )

            
            self._obs_step = self.common_step_counter
        return {"policy": self._obs}

    def _pre_physics_step(self, actions):
        """Integrate bounded commands without accumulating windup beyond joint stops."""
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1, 1)
        self.previous_target.copy_(self.applied_target)
        self.cmd_actions = (self.cmd_actions + self.actions*self.cfg.action_step_deg).clamp(self.servo_min, self.servo_max)
        self.applied_target = self._to_urdf(self.cmd_actions)
        self._substep = 0

        # 1 lần tăng cho mỗi bước của policy 
        self.gait_step += 1 
        # Mỗi bước policy = 10 bước vật lý = 0.05 s.
        # gait_step đếm riêng từng env: tăng 1 trong _pre_physics_step, về 0 khi reset.


    def _apply_action(self):
        target = self.previous_target if self._substep < self.cfg.actuator_delay_steps else self.applied_target
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        self._substep += 1

    def _reset_idx(self, env_ids):
        """Reset physics, actuator targets, delay state and all policy history per environment."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # thêm đồng hồ reset pha 
        self.gait_step[env_ids] = 0 # thiếu -> env reset nhưng vẫn tiếp pha ep trước 
        root = as_torch(self.robot.data.default_root_state)[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = as_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.cmd_actions[env_ids] = self.base_pose[env_ids]
        self.applied_target[env_ids] = self._to_urdf(self.base_pose[env_ids])
        self.previous_target[env_ids] = self.applied_target[env_ids]
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.imu_history[env_ids] = 0
        self.action_history[env_ids] = self._normalize(self.base_pose[env_ids])[:, None, :]
        self._fresh_reset[env_ids] = True
