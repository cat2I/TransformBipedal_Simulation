"""
Transformer 10DOF — Twist + March Environment  (v2 — fixed)
============================================================

FIXES so với v1:
  - spawn height 0.45m + joint_pos = standing pose để không ngã ngay
  - Reward: orientation weight tăng 1.5→3.0, thêm alive_bonus (w=2.0)
  - Reward twist_progress chỉ tính khi robot đang đứng (height > 0.28m)
  - Terminate height threshold 0.12→0.28m (phát hiện ngã sớm hơn)
  - Reset: cmd_deg, gear_deg, last_dir khởi tạo đúng từ standing pose

Task:
  Robot bắt đầu đứng (Twist=90°) và phải:
    1. Xoay Twist_L và Twist_R từ 90° → 0° (hướng về phía trước)
    2. Dậm chân liên tục (alternating steps) để không đổ

Joint order (URDF revolute, 10 khớp, không tính fixed IMU joints):
  [0] Bubleft   [1] Hipleft   [2] Twistleft   [3] Kneeleft   [4] Footleft
  [5] Bubright  [6] Hipright  [7] Twistright   [8] Kneeright  [9] Footright

Obs (62D):
  [ 0:20) imu_hist   — 4 frames × (roll, pitch, gx, gy, gz)  = 20D
  [20:60) jpos_hist  — 4 frames × 10 joints normalized         = 40D
  [60:62) twist_prog — [twist_L_progress, twist_R_progress]    2D
                       (1.0 = start 90°,  0.0 = done 0°)

Action (10D):  delta deg/step ∈ [-2, +2] per joint

Rewards (normalized weighted sum, 7 components):
  w=3.0  alive_bonus      — +1 mỗi step robot còn đứng (height > 0.28m, |rp|<0.95)
  w=3.0  orientation      — giữ thẳng (roll/pitch nhỏ)
  w=1.5  height           — duy trì ~0.39m
  w=2.0  march_alt        — đúng 1 chân trong không khí
  w=1.0  anti_static      — phạt nếu cả 2 chân yên > 0.8s
  w=2.0  twist_progress   — main goal, CHỈ tính khi đang đứng
  w=0.5  joint_limits     — phạt gần giới hạn an toàn
"""

import math
import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, ImuCfg, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise
from isaaclab.sim.spawners import RigidBodyMaterialCfg

from .transformer_config import TRANSFORMER_10DOF_CFG

from ._lab3_compat import as_torch, imu_quat_w


# ============================================================
# CONSTANTS
# ============================================================

# Standing pose (degrees), từ PyBullet Phase-1 IK (L1=0.18051, L2=0.19)
# Order: [Bub_L, Hip_L, Twist_L, Knee_L, Foot_L, Bub_R, Hip_R, Twist_R, Knee_R, Foot_R]
STAND_DEG = [
    120.00,  13.695,  90.00,  -26.693,  12.998,   # Left leg
    120.00,  13.695,  90.00,  -26.693,  12.998,   # Right leg
]

# Soft safety range (degrees) — slightly inside URDF hard limits
SAFE_MIN_DEG = [ -10.0, -35.0, -180.0, -120.0, -60.0,
                 -10.0, -35.0, -180.0, -120.0, -60.0 ]
SAFE_MAX_DEG = [ 175.0,  35.0,  180.0,   20.0,  35.0,
                 175.0,  35.0,  180.0,   20.0,  35.0 ]

# Joint indices
IDX_TWIST_L = 2   # Twistleft_joint
IDX_TWIST_R = 7   # Twistright_joint

TWIST_START_DEG = 90.0
TWIST_GOAL_DEG  =  0.0

# Termination / reward thresholds
HEIGHT_TERMINATE = 0.28   # m — phát hiện ngã sớm
HEIGHT_IDEAL     = 0.39   # m
HEIGHT_MAX_DEV   = 0.20   # m


# ============================================================
# CONFIG
# ============================================================

@configclass
class TransformerTwistMarchEnvCfg(DirectRLEnvCfg):
    """10 DOF env: xoay Twist 90°→0° trong khi dậm chân giữ thăng bằng."""

    episode_length_s  = 15.0
    decimation        = 10         # control 20 Hz (physics 200 Hz)
    num_actions       = 10
    num_observations  = 62
    num_states        = 0

    observation_space = gym.spaces.Box(
        low=-float('inf'), high=float('inf'), shape=(62,), dtype=float
    )
    state_space = gym.spaces.Box(
        low=-float('inf'), high=float('inf'), shape=(0,), dtype=float
    )
    action_space = gym.spaces.Box(
        low=-2.0, high=2.0, shape=(10,), dtype=float
    )

    # Reward weights: [alive, orientation, height, march, anti_static, twist, joint_lim]
    reward_weights = [3.0, 3.0, 1.5, 2.0, 1.0, 2.0, 0.5]

    actuator_delay_min: int   = 2
    actuator_delay_max: int   = 6
    backlash_deg:       float = 2.5

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = TRANSFORMER_10DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Foot.*",
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
        force_threshold=0.001,
        history_length=0,
        debug_vis=False,
    )

    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Baselink",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        update_period=0.012,
    )

    domain_rand: bool = True

    imu_bias_range: dict = {
        "roll":  [-0.10,  0.15],
        "pitch": [-0.42, -0.12],
        "yaw":   [-0.03,  0.03],
    }
    imu_noise_std: dict = {
        "orientation":      0.04,
        "angular_velocity": 0.15,
    }
    imu_drift_rate: float = 0.0001

    # Marching / static
    air_time_threshold:       float = 0.05   # s — chân coi là "đang bay"
    static_penalty_threshold: float = 0.8    # s — phạt nếu cả 2 chân đứng im lâu hơn này


# ============================================================
# ENV
# ============================================================

class TransformerTwistMarchEnv(DirectRLEnv):
    """10-DOF: giữ thăng bằng + xoay Twist 90°→0° + dậm chân liên tục."""

    cfg: TransformerTwistMarchEnvCfg

    def __init__(self, cfg: TransformerTwistMarchEnvCfg,
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, d = self.num_envs, self.device

        # Standing pose tensors
        self.stand_deg    = torch.tensor(STAND_DEG,    device=d).unsqueeze(0).expand(n,-1).clone()
        self.safe_min_deg = torch.tensor(SAFE_MIN_DEG, device=d)
        self.safe_max_deg = torch.tensor(SAFE_MAX_DEG, device=d)

        # Commanded / gear positions (degrees)
        self.cmd_deg  = self.stand_deg.clone()
        self.gear_deg = self.stand_deg.clone()
        self.last_dir = torch.zeros(n, 10, device=d)

        # Actuator timing
        self.act_timer = torch.zeros(n, device=d, dtype=torch.int)
        self.act_delay = torch.zeros(n, device=d, dtype=torch.int)
        self.noisy_cmd = self.stand_deg.clone()

        # Obs history buffers
        self.orient_h = torch.zeros(n, 4, 2, device=d)    # roll, pitch
        self.gyro_h   = torch.zeros(n, 4, 3, device=d)    # gx, gy, gz
        self.jpos_h   = torch.zeros(n, 4, 10, device=d)   # 10 joints normalized

        # IMU domain rand
        self.imu_bias   = torch.zeros(n, 3, device=d)
        self.gyro_drift = torch.zeros(n, 3, device=d)
        self._rand_imu  = getattr(cfg, "domain_rand", False)

        # Domain rand tables
        self.frictions = torch.tensor([0.3 + x/1000 for x in range(201)], device=d)
        self.dampings  = torch.tensor([0.6 + x/1000 for x in range(101)], device=d)

        # Actuator noise
        self.actuator_noise = GaussianNoiseCfg(mean=0.0, std=0.5, operation="add")

        # Static-feet timer
        self.static_timer = torch.zeros(n, device=d)

        print(f"\n{'='*70}")
        print(f"🤖 TRANSFORMER 10DOF — TWIST+MARCH v2 (fixed)")
        print(f"  DOF: 10  |  Obs: {cfg.num_observations}D  |  Act: {cfg.num_actions}D")
        print(f"  Task: Twist L/R 90°→0°, dậm chân giữ thăng bằng")
        print(f"  Episode: {cfg.episode_length_s}s  |  Ctrl: 20Hz  |  Envs: {n}")
        print(f"  Rewards: alive×3.0 orient×3.0 height×1.5 march×2.0 "
              f"static×1.0 twist×2.0 jlim×0.5")
        print(f"{'='*70}\n")

    # ── Scene ────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu_sensor

        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["contact"] = self.contact

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        ground_mat = RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.5,
            restitution=0.05,
            friction_combine_mode="average",
        )
        spawn_ground_plane("/World/ground", cfg=GroundPlaneCfg(physics_material=ground_mat))

        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        print("✅ Scene ready (10 DOF Twist+March v2)")

    # ── Observations ─────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        imu_d       = self.scene.sensors["imu"].data
        orient_raw  = _quat_to_euler(imu_quat_w(self.robot, self.cfg.imu))      # (N,3)
        ang_vel_raw = imu_d.ang_vel_b                     # (N,3)

        # IMU noise + bias
        if self._rand_imu:
            orient = orient_raw + self.imu_bias
        else:
            fixed_bias = torch.tensor([0.0, -0.193, 0.0], device=self.device)
            orient = orient_raw + fixed_bias

        orient = orient + gaussian_noise(
            orient, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["orientation"])
        )
        ang_vel = ang_vel_raw + gaussian_noise(
            ang_vel_raw, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["angular_velocity"])
        )
        self.gyro_drift += self.cfg.imu_drift_rate * torch.randn_like(self.gyro_drift) * self.cfg.sim.dt
        ang_vel = ang_vel + self.gyro_drift

        orient_s  = _scale(orient[:, :2], -1.0, 1.0)    # (N,2) roll,pitch
        ang_vel_s = _scale(ang_vel,       -3.0, 3.0)    # (N,3)

        # Update IMU history
        self.orient_h = torch.roll(self.orient_h, -1, dims=1)
        self.gyro_h   = torch.roll(self.gyro_h,   -1, dims=1)
        self.orient_h[:, -1] = orient_s
        self.gyro_h[:, -1]   = ang_vel_s

        # 20D IMU history: 4 × [roll, pitch, gx, gy, gz]
        imu_hist = torch.cat([self.orient_h, self.gyro_h], dim=2).reshape(self.num_envs, 20)

        # Joint pos history
        cmd_norm = _norm_joints(self.cmd_deg, self.safe_min_deg, self.safe_max_deg)
        self.jpos_h = torch.roll(self.jpos_h, -1, dims=1)
        self.jpos_h[:, -1] = cmd_norm
        jpos_hist = self.jpos_h.reshape(self.num_envs, 40)   # 40D

        # Twist progress
        tw_l = self.cmd_deg[:, IDX_TWIST_L]
        tw_r = self.cmd_deg[:, IDX_TWIST_R]
        prog_l = torch.clamp((tw_l - TWIST_GOAL_DEG) / (TWIST_START_DEG - TWIST_GOAL_DEG), 0.0, 1.0)
        prog_r = torch.clamp((tw_r - TWIST_GOAL_DEG) / (TWIST_START_DEG - TWIST_GOAL_DEG), 0.0, 1.0)
        twist_prog = torch.stack([prog_l, prog_r], dim=1)    # (N,2)

        obs = torch.cat([imu_hist, jpos_hist, twist_prog], dim=1)   # (N,62)
        return {"policy": obs}

    # ── Pre-physics (backlash + delay) ───────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        delta = torch.clamp(actions, -2.0, 2.0)
        self.cmd_deg = torch.clamp(
            self.cmd_deg + delta,
            self.safe_min_deg.unsqueeze(0),
            self.safe_max_deg.unsqueeze(0),
        )

        diff      = self.cmd_deg - self.gear_deg
        direction = torch.sign(diff)
        changed   = (direction != self.last_dir) & (self.last_dir != 0)
        movement  = torch.where(
            changed,
            torch.clamp(torch.abs(diff) - self.cfg.backlash_deg, min=0.0) * direction,
            diff,
        )
        self.gear_deg  = self.gear_deg + movement
        self.last_dir  = torch.where(diff != 0, direction, self.last_dir)
        self.noisy_cmd = torch.clamp(
            gaussian_noise(self.gear_deg, self.actuator_noise),
            self.safe_min_deg.unsqueeze(0),
            self.safe_max_deg.unsqueeze(0),
        )

        self.act_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.act_delay = torch.randint(
            self.cfg.actuator_delay_min,
            self.cfg.actuator_delay_max + 1,
            (self.num_envs,), device=self.device,
        )

    # ── Apply action ──────────────────────────────────────────────────────
    def _apply_action(self):
        mask = self.act_timer >= self.act_delay
        if mask.any():
            targets = torch.deg2rad(self.noisy_cmd)
            cur     = self.robot.data.joint_pos_target.clone()
            cur[mask] = targets[mask]
            self.robot.set_joint_position_target(cur)
        self.act_timer = self.act_timer + 1

    # ── Rewards ───────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        imu_d    = self.scene.sensors["imu"].data
        euler    = _quat_to_euler(imu_quat_w(self.robot, self.cfg.imu))           # (N,3)
        root_pos = self.robot.data.root_pos_w              # (N,3)
        height   = root_pos[:, 2]                          # (N,)
        air_time = self.scene.sensors["contact"].data.current_air_time   # (N,2)

        roll_abs  = torch.abs(euler[:, 0])
        pitch_abs = torch.abs(euler[:, 1])
        standing  = (height > HEIGHT_TERMINATE) & (roll_abs < 0.95) & (pitch_abs < 0.95)

        # ── 1. Alive bonus ───────────────────────────────────────────
        # +1 khi đang đứng, -1 khi ngã (rõ ràng phân biệt sống/chết)
        alive_rew = torch.where(standing, torch.ones_like(height), torch.full_like(height, -1.0))

        # ── 2. Orientation ───────────────────────────────────────────
        angle_sum = roll_abs + pitch_abs
        orient_rew = torch.where(
            angle_sum <= 0.95,
            1.0 - torch.sqrt(torch.clamp(angle_sum / 0.95, 0.0, 1.0)),
            torch.full_like(angle_sum, -1.0),
        )

        # ── 3. Height ────────────────────────────────────────────────
        hdiff      = torch.abs(height - HEIGHT_IDEAL).clamp(0.0, HEIGHT_MAX_DEV)
        height_rew = 1.0 - hdiff / HEIGHT_MAX_DEV

        # ── 4. March alternating ─────────────────────────────────────
        in_air    = (air_time > self.cfg.air_time_threshold)   # (N,2) bool
        n_air     = in_air.sum(dim=1).float()
        march_rew = torch.where(
            n_air == 1, torch.ones_like(n_air),
            torch.where(n_air == 0, torch.full_like(n_air, -0.3),
                        torch.full_like(n_air, -1.0))
        )

        # ── 5. Anti-static ───────────────────────────────────────────
        dt_ctrl = self.cfg.sim.dt * self.cfg.decimation
        self.static_timer = torch.where(
            n_air == 0,
            self.static_timer + dt_ctrl,
            torch.zeros_like(self.static_timer),
        )
        static_rew = torch.where(
            self.static_timer > self.cfg.static_penalty_threshold,
            torch.full_like(self.static_timer, -1.0),
            torch.full_like(self.static_timer,  0.5),
        )

        # ── 6. Twist progress ────────────────────────────────────────
        # CHỈ tính khi robot đang đứng — tránh học cách ngã để xoay twist
        tw_l     = self.cmd_deg[:, IDX_TWIST_L]
        tw_r     = self.cmd_deg[:, IDX_TWIST_R]
        tw_err   = (torch.abs(tw_l) + torch.abs(tw_r)) / 2.0
        raw_prog = torch.exp(-tw_err / 30.0)
        # Bonus khi cả 2 gần đích
        done_bonus = torch.where(
            (torch.abs(tw_l) < 5.0) & (torch.abs(tw_r) < 5.0),
            torch.full_like(raw_prog, 0.5), torch.zeros_like(raw_prog)
        )
        twist_rew = torch.where(standing, raw_prog + done_bonus, torch.zeros_like(raw_prog))

        # ── 7. Joint limits ──────────────────────────────────────────
        margin = 5.0
        near   = ((self.cmd_deg < self.safe_min_deg + margin) |
                  (self.cmd_deg > self.safe_max_deg - margin)).float().sum(dim=1)
        jlim_rew = 1.0 - near / 10.0

        # ── Weighted sum ─────────────────────────────────────────────
        w = torch.tensor(self.cfg.reward_weights, device=self.device, dtype=torch.float32)
       
        rews  = torch.stack([alive_rew, orient_rew, height_rew,
                             march_rew, static_rew, twist_rew, jlim_rew], dim=1)
        total = (rews * w).sum(dim=1)

        # Log
        if self.episode_length_buf[0] % 100 == 0 and self.episode_length_buf[0] > 0:
            i = 0
            print(f"\n{'='*70}")
            print(f"[Step {self.common_step_counter}] ep={self.episode_length_buf[i].item()}")
            print(f"  h={height[i]:.3f}m  standing={'Y' if standing[i] else 'N'}")
            print(f"  TwL={tw_l[i]:+6.1f}°  TwR={tw_r[i]:+6.1f}°  (goal=0°)")
            print(f"  alive={alive_rew[i]:.2f}  orient={orient_rew[i]:.2f}  "
                  f"height={height_rew[i]:.2f}  march={march_rew[i]:.2f}")
            print(f"  static={static_rew[i]:.2f}(t={self.static_timer[i]:.2f}s)  "
                  f"twist={twist_rew[i]:.2f}  jlim={jlim_rew[i]:.2f}")
            print(f"  TOTAL={total[i]:.3f}")
            print(f"{'='*70}")

        return total

    # ── Dones ────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        height = self.robot.data.root_pos_w[:, 2]
        euler  = _quat_to_euler(self.robot.data.root_quat_w)
        fallen = (
            (height < HEIGHT_TERMINATE) |
            (torch.abs(euler[:, 0]) > 0.95) |
            (torch.abs(euler[:, 1]) > 0.95)
        )

        # Early success: cả 2 twist < 3° và vẫn đứng
        tw_l    = self.cmd_deg[:, IDX_TWIST_L]
        tw_r    = self.cmd_deg[:, IDX_TWIST_R]
        success = (torch.abs(tw_l) < 3.0) & (torch.abs(tw_r) < 3.0) & (height > HEIGHT_TERMINATE)

        return fallen | success, truncated

    # ── Reset ────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Reset joint commands → standing pose
        self.cmd_deg[env_ids]  = self.stand_deg[env_ids]
        self.gear_deg[env_ids] = self.stand_deg[env_ids]
        self.last_dir[env_ids] = 0.0
        self.static_timer[env_ids] = 0.0

        # IMU bias randomization
        if self._rand_imu:
            def su(lo, hi): return sample_uniform(lo, hi, (n,), device=self.device)
            self.imu_bias[env_ids, 0] = su(*self.cfg.imu_bias_range["roll"])
            self.imu_bias[env_ids, 1] = su(*self.cfg.imu_bias_range["pitch"])
            self.imu_bias[env_ids, 2] = su(*self.cfg.imu_bias_range["yaw"])
        self.gyro_drift[env_ids] = 0.0

        # Domain randomization
        nj = 10
        fi = torch.randint(0, self.frictions.size(0), (n, nj), device=self.device)
        di = torch.randint(0, self.dampings.size(0),  (n, nj), device=self.device)
        self.robot.write_joint_friction_coefficient_to_sim(self.frictions[fi], None, env_ids)
        self.robot.write_joint_damping_to_sim(self.dampings[di], None, env_ids)

        # Robot state — spawn với standing pose
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.45

        # IMPORTANT:
        # spawn đúng standing pose thay vì URDF default pose
        joint_pos = torch.deg2rad(self.stand_deg[env_ids]).clone()

        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            None,
            env_ids
        )

        # Clear obs history; seed jpos_hist với standing pose normalized
        self.orient_h[env_ids] = 0.0
        self.gyro_h[env_ids]   = 0.0
        stand_norm = _norm_joints(self.stand_deg[env_ids], self.safe_min_deg, self.safe_max_deg)
        self.jpos_h[env_ids] = stand_norm.unsqueeze(1).expand(-1, 4, -1)


# ============================================================
# JIT HELPERS
# ============================================================

@torch.jit.script
def _quat_to_euler(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.norm(q, dim=-1, keepdim=True)
    # IsaacLab 3.0 tra quaternion theo (x, y, z, w), khong con (w, x, y, z)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (torch.ones_like(sinp) * math.pi / 2),
        torch.asin(sinp),
    )
    yaw   = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=1)


@torch.jit.script
def _scale(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp((x - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def _norm_joints(deg: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.clamp((deg - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0)