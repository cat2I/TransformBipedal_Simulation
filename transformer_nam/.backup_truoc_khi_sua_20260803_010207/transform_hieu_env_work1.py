"""
Transformer — Bub Closes, Hip/Knee/Foot Balance  (8 DOF)
=========================================================
Joint order thực tế (từ debug):
  [0] Bubleft   [1] Bubright  [2] Hipleft   [3] Hipright
  [4] Kneeleft  [5] Kneeright [6] Footleft  [7] Footright

Logic:
  - Bub khép dần theo lịch cố định: ±90° → 0° (mỗi N steps giảm bub_step_deg)
  - RL agent điều khiển Hip/Knee/Foot (6 joints) để giữ cân bằng IMU
  - Bub KHÔNG do RL điều khiển — agent chỉ output 6 values cho Hip/Knee/Foot

Reward:
  - Thưởng khi ang_vel nhỏ (đứng yên, không lắc)
  - Thưởng khi Bub đã khép được nhiều (robot cao lên)
  - Không có penalty âm — luôn dương để học ổn định
"""

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

from .transformer_config import TRANSFORMER_CFG

from ._lab3_compat import as_torch, imu_quat_w


@configclass
class TransformerStandEnvCfg(DirectRLEnvCfg):

    episode_length_s: float = 30.0
    decimation:       int   = 10
    num_actions:      int   = 6    # chỉ Hip×2 + Knee×2 + Foot×2
    # obs: roll, pitch, roll_rate, pitch_rate (4)
    #    + hip×2, knee×2, foot×2 norm (6)
    #    + bub_progress (1) + balance_counter_norm (1) = 12
    num_observations: int   = 12
    num_states:       int   = 0

    observation_space = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(12,), dtype=float
    )
    state_space  = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

    # Bub khép lịch cố định
    bub_start_deg:      float = 88.0
    bub_target_deg:     float = 0.0
    bub_step_deg:       float = 3.0    # khép 3°/lần
    balance_hold_steps: int   = 15     # đợi 15 steps ổn định mới khép tiếp
    balance_ang_vel_threshold: float = 0.20  # rad/s

    # Action scale cho Hip/Knee/Foot (độ/step)
    action_scale: float = 3.0  # ±3°/step

    # Joint limits Hip/Knee/Foot
    hip_limit:  float = 30.0   # ±30°
    knee_limit: float = 80.0   # ±80°
    foot_limit: float = 35.0   # ±35°

    # Reward
    w_balance:  float = 3.0
    w_progress: float = 5.0

    sim:   SimulationCfg       = SimulationCfg(dt=0.005, render_interval=decimation, gravity=(0., 0., -9.81))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0, replicate_physics=True)
    robot: ArticulationCfg     = TRANSFORMER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Robot/Foot.*",
        update_period=0.005, track_air_time=False, track_pose=False,
        force_threshold=0.001, history_length=0, debug_vis=False,
    )
    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Robot/Baselink",
        offset=ImuCfg.OffsetCfg(pos=(0., 0., 0.), rot=(0., 0., 0., 1.)),
        debug_vis=False, update_period=0.012,
    )
    imu_noise_std: dict = {"orientation": 0.01, "angular_velocity": 0.02}


class TransformerStandEnv(DirectRLEnv):
    cfg: TransformerStandEnvCfg

    def __init__(self, cfg: TransformerStandEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint order: [BubL, BubR, HipL, HipR, KneeL, KneeR, FootL, FootR]
        # Bub limits
        self.bub_min = -88.0
        self.bub_max =  88.0

        # Hip/Knee/Foot limits (degree)
        hl = cfg.hip_limit
        kl = cfg.knee_limit
        fl = cfg.foot_limit
        # limits cho 6 joints do agent điều khiển: [HipL, HipR, KneeL, KneeR, FootL, FootR]
        self.act_min = torch.tensor([-hl, -hl, -kl, -kl, -fl, -fl], device=self.device)
        self.act_max = torch.tensor([ hl,  hl,  kl,  kl,  fl,  fl], device=self.device)

        # Pose Hip/Knee/Foot hiện tại do agent điều khiển
        self.balance_pose = torch.zeros(self.num_envs, 6, device=self.device)

        # Bub target hiện tại (bắt đầu từ max, khép dần)
        self.bub_target = torch.full(
            (self.num_envs,), cfg.bub_start_deg, device=self.device
        )

        # Đếm steps đã ổn định
        self.balance_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        # Warmup
        self.warmup_steps = 20
        self.imu_data = None

        # Domain rand
        self.frictions = torch.tensor([0.3  + x/1000 for x in range(201)],  device=self.device)
        self.torques   = torch.tensor([9.27 + x/1000 for x in range(1030)], device=self.device)
        self.dampings  = torch.tensor([0.6  + x/1000 for x in range(101)],  device=self.device)

        self.max_ep_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

        print(f"\n{'='*60}")
        print("🤖 TRANSFORMER — BUB CLOSES, HIP/KNEE/FOOT BALANCE")
        print(f"  Bub: {cfg.bub_start_deg}° → {cfg.bub_target_deg}°, step={cfg.bub_step_deg}°")
        print(f"  Agent: Hip±{hl}° Knee±{kl}° Foot±{fl}°")
        print(f"  Balance threshold: {cfg.balance_ang_vel_threshold} rad/s")
        print(f"  Balance hold: {cfg.balance_hold_steps} steps trước khi khép tiếp")
        print(f"{'='*60}\n")

    # ── Scene ─────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot   = Articulation(self.cfg.robot)
        self.imu     = Imu(self.cfg.imu)
        self.contact = ContactSensor(self.cfg.contact)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["imu"]         = self.imu
        self.scene.sensors["contact"]     = self.contact
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.5,
                restitution=0.02, friction_combine_mode="average"
            )
        ))
        light = sim_utils.DomeLightCfg(intensity=2000., color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        print("✅ Scene created")

    # ── Build full 8-joint command ─────────────────────────────

    def _build_full_cmd(self) -> torch.Tensor:
        """
        Ghép Bub (cố định theo lịch) + Hip/Knee/Foot (agent điều khiển).
        Output: (N, 8) degrees
        [BubL, BubR, HipL, HipR, KneeL, KneeR, FootL, FootR]
        """
        bub_l = self.bub_target    # dương → dạng trái, về 0 → khép
        bub_r = self.bub_target    # dương → dạng phải, về 0 → khép

        cmd = torch.stack([
            bub_l,                      # [0] BubL
            bub_r,                      # [1] BubR
            self.balance_pose[:, 0],    # [2] HipL
            self.balance_pose[:, 1],    # [3] HipR
            self.balance_pose[:, 2],    # [4] KneeL
            self.balance_pose[:, 3],    # [5] KneeR
            self.balance_pose[:, 4],    # [6] FootL
            self.balance_pose[:, 5],    # [7] FootR
        ], dim=1)
        return cmd

    # ── Observations ──────────────────────────────────────────

    def _get_observations(self) -> dict:
        self.imu_data = self.scene.sensors["imu"].data
        orient_raw    = quaternion_to_euler(imu_quat_w(self.robot, self.cfg.imu))
        ang_vel_raw   = self.imu_data.ang_vel_b

        orient  = orient_raw  + gaussian_noise(orient_raw,  GaussianNoiseCfg(mean=0., std=self.cfg.imu_noise_std["orientation"],      operation="add"))
        ang_vel = ang_vel_raw + gaussian_noise(ang_vel_raw, GaussianNoiseCfg(mean=0., std=self.cfg.imu_noise_std["angular_velocity"],  operation="add"))

        roll       = orient[:, 0:1].clamp(-2., 2.)
        pitch      = orient[:, 1:2].clamp(-2., 2.)
        roll_rate  = ang_vel[:, 0:1].clamp(-5., 5.)
        pitch_rate = ang_vel[:, 1:2].clamp(-5., 5.)

        # Balance pose normalised [-1, 1]
        bp_norm = self.balance_pose / self.act_max.unsqueeze(0)  # (N,6)

        # Bub progress [0→1]
        bub_prog = (1. - self.bub_target / self.cfg.bub_start_deg).clamp(0., 1.).unsqueeze(1)

        # Balance counter [0→1]
        bal_norm = (self.balance_counter.float() / self.cfg.balance_hold_steps).clamp(0., 1.).unsqueeze(1)

        obs = torch.cat([roll, pitch, roll_rate, pitch_rate,
                         bp_norm, bub_prog, bal_norm], dim=1)  # (N, 12)
        return {"policy": obs}

    # ── Pre-physics ───────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        # ── 1. Agent update balance_pose ──────────────────────
        # actions: (N, 6) trong [-1, 1] → delta degrees
        delta = torch.clamp(actions, -1., 1.) * self.cfg.action_scale
        self.balance_pose = torch.clamp(
            self.balance_pose + delta,
            self.act_min, self.act_max
        )

        # ── 2. State machine: khép Bub khi ổn định ────────────
        imu   = self.scene.sensors["imu"].data
        ang_vel_mag = torch.norm(imu.ang_vel_b[:, :2], dim=1)
        is_balanced = ang_vel_mag < self.cfg.balance_ang_vel_threshold

        # Cập nhật counter
        self.balance_counter = torch.where(
            is_balanced,
            (self.balance_counter + 1).clamp(max=self.cfg.balance_hold_steps),
            torch.zeros_like(self.balance_counter)
        )

        # Khi đủ ổn định → khép Bub thêm
        can_close = (self.balance_counter >= self.cfg.balance_hold_steps) & \
                    (self.bub_target > self.cfg.bub_target_deg)
        self.bub_target = torch.where(
            can_close,
            (self.bub_target - self.cfg.bub_step_deg).clamp(min=self.cfg.bub_target_deg),
            self.bub_target
        )
        # Reset counter sau khi khép
        self.balance_counter = torch.where(
            can_close,
            torch.zeros_like(self.balance_counter),
            self.balance_counter
        )

    def _apply_action(self):
        cmd = self._build_full_cmd()   # (N, 8) degrees
        self.robot.set_joint_position_target(torch.deg2rad(cmd))

    # ── Rewards ───────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        if self.episode_length_buf[0] < self.warmup_steps:
            return torch.zeros(self.num_envs, device=self.device)
        if self.imu_data is None:
            self.imu_data = self.scene.sensors["imu"].data

        ang_vel     = self.imu_data.ang_vel_b
        ang_vel_mag = torch.norm(ang_vel[:, :2], dim=1)

        # Balance: ang_vel nhỏ → robot ổn định
        r_balance = torch.exp(-3. * ang_vel_mag)   # [0,1]

        # Progress: Bub đã khép được bao nhiêu
        r_progress = 1. - self.bub_target / self.cfg.bub_start_deg   # [0,1]

        total = self.cfg.w_balance * r_balance + self.cfg.w_progress * r_progress

        # Logging
        if self.episode_length_buf[0] % 40 == 0 and self.episode_length_buf[0] > 0:
            i = 0
            bub = self.bub_target[i].item()
            av  = ang_vel_mag[i].item()
            bal = self.balance_counter[i].item()
            hp  = self.balance_pose[i, 0].item()
            kn  = self.balance_pose[i, 2].item()
            ft  = self.balance_pose[i, 4].item()
            h   = self.robot.data.root_pos_w[i, 2].item()
            print(
                f"[{self.common_step_counter:6d}] "
                f"Bub={bub:5.1f}° bal={bal:2.0f}/{self.cfg.balance_hold_steps} "
                f"av={av:.3f} h={h:.3f}m | "
                f"Hip={hp:+5.1f} Knee={kn:+5.1f} Foot={ft:+5.1f} | "
                f"R={total[i]:.3f}"
            )
        return total

    # ── Termination ───────────────────────────────────────────

    def _get_dones(self):
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        head_h    = self.robot.data.root_pos_w[:, 2]
        # Chỉ terminate khi bay lên bất thường
        terminated = head_h > 0.8
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2]   = self.scene.env_origins[env_ids, 2] + 0.10

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        n_r = len(env_ids)
        self.robot.write_joint_friction_coefficient_to_sim(
            self.frictions[torch.randint(0, self.frictions.size(0), (n_r, 8), device=self.device)],
            joint_ids=None, env_ids=env_ids)
        self.robot.write_joint_effort_limit_to_sim(
            self.torques[torch.randint(0, self.torques.size(0), (n_r, 8), device=self.device)],
            joint_ids=None, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(
            self.dampings[torch.randint(0, self.dampings.size(0), (n_r, 8), device=self.device)],
            joint_ids=None, env_ids=env_ids)

        self.robot.write_root_link_pose_to_sim   (root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim      (joint_pos, joint_vel, None, env_ids)

        # Reset state
        self.balance_pose   [env_ids] = 0.
        self.bub_target     [env_ids] = self.cfg.bub_start_deg
        self.balance_counter[env_ids] = 0

        self.robot.set_external_force_and_torque(
            torch.zeros(n_r, 1, 3, device=self.device),
            torch.zeros(n_r, 1, 3, device=self.device),
            body_ids=None, env_ids=env_ids,
        )

    # ── Property alias ────────────────────────────────────────
    @property
    def cmd_actions(self):
        return self._build_full_cmd()


# ── Helpers ───────────────────────────────────────────────────

def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.norm(q, dim=-1, keepdim=True)
    # IsaacLab 3.0 tra quaternion theo (x, y, z, w), khong con (w, x, y, z)
    x, y, z, w = q[...,0], q[...,1], q[...,2], q[...,3]
    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = torch.asin(torch.clamp(2*(w*y - z*x), -1., 1.))
    yaw   = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=1)