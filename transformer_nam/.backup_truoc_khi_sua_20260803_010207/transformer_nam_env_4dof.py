"""Transformer bipedal walking - Bimo-style environment"""

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
from isaaclab.sim.utils import bind_physics_material
from random import uniform

from .transformer_config import TRANSFORMER_CFG

from ._lab3_compat import as_torch, imu_quat_w


@configclass
class TransformerWalkEnvCfg(DirectRLEnvCfg):
    """Config for Transformer walking (6 DOF - no Bub)"""
    
    episode_length_s = 10.0
    decimation = 10
    num_actions = 8  
    num_observations = 52 
    num_states = 0
    
    observation_space = gym.spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(52,),  # ✅ CHANGED: 44 → 52
        dtype=float
    )
    
    state_space = gym.spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(0,),
        dtype=float
    )
    
    action_space = gym.spaces.Box(
        low=-3.0,
        high=3.0,
        shape=(8,),  # ✅ CHANGED: 6 → 8
        dtype=float
    )
    
    obj = "walk"
    
    weights = {
        "walk": [1, 1, 1, 0, 2, 1, 1],
    }
    
    actuator_delay_max = 4
    actuator_delay_min = 1
    backlash = 1.6
    
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
    
    robot: ArticulationCfg = TRANSFORMER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Robot/Foot.*",
        update_period=0.005,
        track_air_time=True,
        track_pose=True,
        force_threshold=0.001,
        history_length=0,
        debug_vis=False,
    )
    
    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Robot/Baselink",
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        debug_vis=False,
        update_period=0.012,
    )


class TransformerWalkEnv(DirectRLEnv):
    """Direct RL environment for Transformer (6 DOF)"""
    
    cfg: TransformerWalkEnvCfg
    
    def __init__(self, cfg: TransformerWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.weights = torch.tensor(
            self.cfg.weights[self.cfg.obj], 
            device=self.device
        ).repeat(self.num_envs, 1)
        self.obj = self.cfg.obj
        
        # ✅ CHANGED: servo min/max for 6 joints only!
        # [Hip_L, Hip_R, Knee_L, Knee_R, Ankle_L, Ankle_R]
        self.servo_max = torch.tensor(
            [10, 10, 35, 35, 85, 85, 45, 45],  # ✅ No Bub!
            device=self.device, dtype=torch.int
        )
        self.servo_min = torch.tensor(
            [-10, -10, -35, -35, -85, -85, -45, -45],
            device=self.device, dtype=torch.int
        )
        
        # ✅ CHANGED: base_pose for 6 joints
        # [Hip_L, Hip_R, Knee_L, Knee_R, Ankle_L, Ankle_R]
        start_pos = [0, 0, 25, 25, -50, -50, 25, 25]  # ✅ No Bub!
        self.base_pose = torch.tensor(
            [start_pos for _ in range(self.num_envs)], 
            device=self.device, dtype=torch.float32
        )
        
        self.cmd_actions = self.base_pose.clone()
        self.last_direction = torch.zeros(self.num_envs, 8, device=self.device)  # ✅ 8 joints
        self.gear_position = self.base_pose.clone()
        
        half = self.num_envs // 2
        self.act_direction = torch.cat((
            torch.ones(half, device=self.device),
            -torch.ones(self.num_envs - half, device=self.device)
        ), dim=0)
        
        # ✅ Randomization ranges (unchanged)
        self.frictions = torch.tensor([0.1 + x/1000 for x in range(0, 201)], device=self.device)
        self.torques = torch.tensor([9.27 + x/1000 for x in range(0, 1030)], device=self.device)
        self.dampings = torch.tensor([0.6 + x/1000 for x in range(0, 101)], device=self.device)
        
        self.orient_noise = GaussianNoiseCfg(mean=0.0, std=0.015, operation="add")
        self.gyro_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
        self.actuator_noise = GaussianNoiseCfg(mean=0.0, std=0.5, operation="add")
        
        self.act_timer = 0
        self.act_delay = 0
        
        # ✅ CHANGED: History buffers for 6 joints
        self.orient_h = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.gyro_h = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.act_hist = torch.zeros(self.num_envs, 4, 8, device=self.device)  # ✅ 8 joints
        
        self.act_hist[:, :] = torch.clamp(
            (self.base_pose[0] - self.servo_min) / (self.servo_max - self.servo_min) * 2 - 1, 
            -1, 1
        )
        
        print(f"\n{'='*70}")
        print(f"🤖 TRANSFORMER ENV (8 DOF - WITH BUB)")
        print(f"  Observation dim: {self.cfg.num_observations} (20 IMU + 32 action history)")
        print(f"  Action dim: {self.cfg.num_actions}")
        print(f"  Joints: Bub_L, Hip_L, Knee_L, Ankle_L, Bub_R, Hip_R, Knee_R, Ankle_R")
        print(f"  Objective: {self.obj}")
        print(f"{'='*70}\n")
    
    def _setup_scene(self):
        """Setup scene"""
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu

        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["contact"] = self.contact

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        print(f"\n{'='*70}")
        print("✅ Scene created (6 DOF - no material randomization)")
        print(f"{'='*70}\n")

        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        ground_cfg = RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.4,
            restitution=0.05,
            friction_combine_mode="average",
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(physics_material=ground_cfg))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _get_observations(self) -> dict:
        """✅ CHANGED: 44D observations (20 IMU + 24 action history for 6 joints)"""
        self.imu_data = self.scene.sensors["imu"].data
        orient = quaternion_to_euler(imu_quat_w(self.robot, self.cfg.imu))
        angular_vel = self.imu_data.ang_vel_b
        
        orient = gaussian_noise(orient, self.orient_noise)
        angular_vel = gaussian_noise(angular_vel, self.gyro_noise)
        
        orient = scale_value(orient, -1.0, 1.0)
        angular_vel = scale_value(angular_vel, -2.0, 2.0)
        
        self.update_imu_history(orient, angular_vel)
        
        imu_data = torch.cat((self.orient_h[:, :, :2], self.gyro_h), dim=2)
        imu_data = imu_data.reshape(self.num_envs, 20)
        
        cmd_act = torch.clamp(
            (self.cmd_actions - self.servo_min) / (self.servo_max - self.servo_min) * 2 - 1,
            -1, 1
        )
        
        self.act_hist[:, :-1] = self.act_hist[:, 1:].clone()
        self.act_hist[:, -1] = cmd_act
        
        proc_act = self.act_hist.reshape(self.num_envs, 32)
        
        # 20 + 32 = 52D
        obs_buffer = torch.cat((imu_data, proc_act), dim=1)
        obs_buffer = torch.round(obs_buffer, decimals=4)
        
        return {"policy": obs_buffer}
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Action processing with backlash and delay"""
        actions_cpy = torch.clamp(actions.clone(), -3.0, 3.0)
        self.cmd_actions += actions_cpy * 2 / 3
        
        delta = self.cmd_actions - self.gear_position
        direction = torch.sign(delta)
        direction_changed = (direction != self.last_direction) & (self.last_direction != 0)
        
        movement = torch.where(
            direction_changed,
            torch.clamp(torch.abs(delta) - self.cfg.backlash, min=0) * direction,
            delta
        )
        
        self.gear_position += movement
        self.last_direction = torch.where(delta != 0, direction, self.last_direction)
        
        self.noisy_act = torch.clamp(
            gaussian_noise(self.gear_position, self.actuator_noise),
            self.servo_min,
            self.servo_max
        )
        
        self.act_timer = 0
        self.act_delay = torch.randint(
            low=self.cfg.actuator_delay_min,
            high=self.cfg.actuator_delay_max + 1,
            size=(1,)
        ).item()
    
    def _apply_action(self):
        """Apply action with delay"""
        if self.act_timer >= self.act_delay:
            self.robot.set_joint_position_target(torch.deg2rad(self.noisy_act))
        else:
            self.act_timer += 1
    
    def _get_rewards(self) -> torch.Tensor:
        """Reward calculation"""
        euler_imu_orient = quaternion_to_euler(imu_quat_w(self.robot, self.cfg.imu))
        robot_root_pos = self.robot.data.root_pos_w
        lin_vel = self.robot.data.root_com_vel_w
        contact_pos = self.scene.sensors["contact"].data.pos_w
        air_time = self.scene.sensors["contact"].data.current_air_time
        
        orientation_rew = orientation_reward(euler_imu_orient, self.obj, self.device)
        height_rew = height_reward(robot_root_pos)  # ✅ Fixed ideal_height inside function
        position_rew = joint_position_reward(self.cmd_actions, self.base_pose, self.device)
        sig_extra = sigmoid_extra(self.cmd_actions, self.base_pose)
        vel_rew = velocity_reward(lin_vel, self.act_direction, self.obj)
        feet_h_rew = feet_height_reward(air_time, contact_pos, 0.03, 150)
        dev_rew = deviation_reward(self.scene.env_origins, robot_root_pos, self.obj)
        
        w = self.weights / torch.sum(self.weights, dim=1, keepdim=True)
        
        total_reward = (
            orientation_rew * w[:, 0] + 
            height_rew * w[:, 1] +
            position_rew * w[:, 2] + 
            sig_extra * w[:, 3] +
            feet_h_rew * w[:, 4] + 
            vel_rew * w[:, 5] +
            dev_rew * w[:, 6]
        )
        
        if self.episode_length_buf[0] % 100 == 0 and self.episode_length_buf[0] > 0:
            idx = 0
            print(f"\n{'='*70}")
            print(f"[Step {self.common_step_counter}] Episode {self.episode_length_buf[idx].item()}")
            print(f"  Orientation reward: {orientation_rew[idx].item():.3f}")
            print(f"  Height reward: {height_rew[idx].item():.3f}")
            print(f"  Velocity reward: {vel_rew[idx].item():.3f}")
            print(f"  Feet height reward: {feet_h_rew[idx].item():.3f}")
            print(f"  TOTAL: {total_reward[idx].item():.3f}")
            print(f"{'='*70}")
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Termination conditions"""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        head_heights = self.robot.data.root_pos_w[:, 2]
        height_termination = head_heights < 0.1
        
        root_orientations = self.robot.data.root_quat_w
        euler_angles = quaternion_to_euler(root_orientations)
        x_rotation = torch.abs(euler_angles[:, 0])
        y_rotation = torch.abs(euler_angles[:, 1])
        orientation_termination = (x_rotation > 0.95) | (y_rotation > 0.95)
        
        terminated = height_termination | orientation_termination
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset with randomization"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        reset_ids = env_ids.flatten().long()
        n_reset = reset_ids.shape[0]
        n_joints = 8  
        
        fric_idx = torch.randint(0, self.frictions.size(0), (n_reset, n_joints), device=self.device)
        torque_idx = torch.randint(0, self.torques.size(0), (n_reset, n_joints), device=self.device)
        damp_idx = torch.randint(0, self.dampings.size(0), (n_reset, n_joints), device=self.device)
        
        fric_samples = self.frictions[fric_idx]
        torque_samples = self.torques[torque_idx]
        damp_samples = self.dampings[damp_idx]
        
        self.robot.write_joint_friction_coefficient_to_sim(fric_samples, joint_ids=None, env_ids=env_ids)
        self.robot.write_joint_effort_limit_to_sim(torque_samples, joint_ids=None, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(damp_samples, joint_ids=None, env_ids=env_ids)
        
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.orient_h[env_ids] = 0.0
        self.gyro_h[env_ids] = 0.0
        self.act_hist[env_ids, :] = self.base_pose[0]
        self.cmd_actions[env_ids] = self.base_pose[0]
    
    def update_imu_history(self, new_orient, new_gyro):
        """Update IMU history buffers"""
        self.orient_h[:, :-1] = self.orient_h[:, 1:].clone()
        self.gyro_h[:, :-1] = self.gyro_h[:, 1:].clone()
        
        self.orient_h[:, -1] = new_orient
        self.gyro_h[:, -1] = new_gyro


# ============================================================
# ✅ REWARD FUNCTIONS
# ============================================================

@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor):
    """Convert quaternion to Euler angles"""
    if not isinstance(quat, torch.Tensor):
        quat = torch.tensor(quat)
    
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    # IsaacLab 3.0 tra quaternion theo (x, y, z, w), khong con (w, x, y, z)
    
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp)
    )
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=1)


@torch.jit.script
def scale_value(value: torch.Tensor, min_val: float, max_val: float):
    """Scale value to [-1, 1]"""
    return torch.clamp((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)


@torch.jit.script
def orientation_reward(euler_imu_orient, action: str, device: str):
    """Orientation reward"""
    angle_sums = torch.zeros(euler_imu_orient.shape[0], device=device)
    
    if action == "walk":
        angle_sums = torch.sum(torch.abs(euler_imu_orient), dim=1)
    else:
        angle_sums = torch.sum(torch.abs(euler_imu_orient[:, :2]), dim=1)
    
    orientation_rew = torch.where(
        angle_sums <= 0.95,
        1 - torch.sqrt(angle_sums / 0.95),
        torch.ones_like(angle_sums) * -1
    )
    
    return orientation_rew


@torch.jit.script
def deviation_reward(og_pose, curr_pose, action: str = "walk"):
    """Deviation from origin reward"""
    x_dev = torch.abs(og_pose[:, 0] - curr_pose[:, 0])
    y_dev = torch.abs(og_pose[:, 1] - curr_pose[:, 1])
    
    reward = torch.zeros_like(x_dev)
    
    if action == "walk":
        reward = torch.where(
            y_dev <= 0.3,
            1 - torch.sqrt(y_dev / 0.3),
            torch.ones_like(y_dev) * -1
        )
    else:
        dist = x_dev + y_dev
        reward = torch.where(
            dist <= 0.3,
            1 - torch.sqrt(dist / 0.3),
            torch.ones_like(dist) * -1
        )
    
    return reward


@torch.jit.script
def height_reward(robot_root_pos):
    """Height reward - ✅ FIXED ideal_height for 6 DOF!"""
    heights = robot_root_pos[:, 2]
    ideal_height = 0.392   
    max_deviation = 0.3
    
    height_diff = torch.abs(heights - ideal_height)
    clipped_diff = torch.clamp(height_diff, 0, max_deviation)
    height_rew = scale_value(clipped_diff, 0.3, 0.0)
    height_rew = (height_rew + 1) / 2
    
    return height_rew


@torch.jit.script
def joint_position_reward(pos_buff, start_pos, device: str):
    """Joint position reward - ✅ CHANGED: 6 joints!"""
    # ✅ CHANGED: max_diff for 6 joints [Hip, Hip, Knee, Knee, Ankle, Ankle]
    max_diff = torch.tensor([10, 10, 35, 35, 85, 85, 45, 45], device=device)
    diff = torch.abs(pos_buff - start_pos)
    diff_scaled = 1 - torch.sqrt(torch.clamp(diff / max_diff.unsqueeze(0), 0, 1))
    pos_rew = torch.mean(diff_scaled, dim=1)
    pos_rew = pos_rew * 2 - 1
    
    return pos_rew


@torch.jit.script
def velocity_reward(vel_data, direction, action: str = "walk"):
    """Velocity reward"""
    reward = torch.zeros_like(direction)
    
    if action == "walk":
        vx = vel_data[:, 0]
        vy = torch.abs(vel_data[:, 1])
        
        rew_lin = torch.where(
            vx > 0,
            torch.clamp(vx / (vx + vy + 1e-8), 0, 1.0),
            torch.zeros_like(vx),
        )
        
        rew_ang = torch.clamp(-torch.abs(vel_data[:, 4]) / 2, -1, 0)
        reward = 0.5 * rew_lin + 0.5 * rew_ang
    
    return reward


@torch.jit.script
def sigmoid_extra(pos_buff, start_pos):
    """Sigmoid bonus for joints near ideal position"""
    diff = torch.abs(pos_buff - start_pos)
    greatest_diff, _ = torch.max(diff, dim=1)
    sigmoid_values = 1 / (1 + torch.exp(0.8 * greatest_diff - 6))
    
    return sigmoid_values


@torch.jit.script
def feet_height_reward(air_time, feet_pos, target_h: float, scale: float = 25.0):
    """Feet clearance reward"""
    in_air = (air_time > 0)
    num_in_air = in_air.sum(dim=1)
    
    both_in_air = (num_in_air == 2)
    both_on_ground = (num_in_air == 0)
    
    z_pos = feet_pos[..., 2]
    z_err = torch.abs(z_pos - target_h)
    
    reward_per_leg = torch.where(
        z_pos >= target_h,
        torch.ones_like(z_pos),
        torch.exp(-scale * z_err)
    ) * in_air.float()
    
    reward = reward_per_leg.sum(dim=1)
    
    reward = torch.where(
        both_in_air | both_on_ground,
        torch.zeros_like(reward),
        reward
    )
    
    return reward