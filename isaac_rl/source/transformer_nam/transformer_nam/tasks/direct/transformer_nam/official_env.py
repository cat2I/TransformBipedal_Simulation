"""Walking task for OFFICIALdesign, with an explicit sim/embedded interface.

Policy: 10 incremental actions, clipped to [-1, 1], at 20 Hz. Observations:
4 frames of [roll, pitch, gyro xyz] then 4 frames of normalized policy-angle
targets, 60 floats total. CAD names are retained, including reversed L/R.
All joint writes resolve names; importer traversal order is never assumed.
"""
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from ._lab3_compat import as_torch
from .official_config import BUILD, CALIBRATION, JOINTS, JOINT_NAMES, MATERIAL, OFFICIAL_CFG, USD_REPORT


def rpy_xyzw(quat):
    """Euler FLU orientation, using Isaac Lab 3 xyzw without legacy IMU rotation."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
                        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), dim=-1)


@configclass
class OfficialWalkEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 10
    action_space = 10
    observation_space = 60
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
    target_velocity = 0.15  # fixed +x command; no unobserved random direction
    action_step_deg = 2.0
    actuator_delay_steps = 2  # 10 ms simulation assumption, not measured
    orientation_noise_std = 0.015
    gyro_noise_std = 0.01
    min_base_height = 0.20
    max_tilt = 0.9
    target_height = BUILD["spawn_height_m"] - CALIBRATION["spawn_clearance_m"]


class OfficialWalkEnv(DirectRLEnv):
    cfg: OfficialWalkEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, names = self.robot.find_joints(self.cfg.joint_names, preserve_order=True)
        if names != list(self.cfg.joint_names) or len(names) != self.robot.num_joints:
            raise ValueError(f"Unexpected joint mapping: {names}")
        self.feet_ids, _ = self.robot.find_bodies(["Footleft", "Footright"], preserve_order=True)
        self.contact_ids = [self.contact.body_names.index(name) for name in ("Footleft", "Footright")]
        self.sole_offsets = torch.tensor([BUILD["feet"][name]["sole_z_local_m"]
                                          for name in ("Footleft", "Footright")], device=self.device)
        self.imu_id = self.robot.find_bodies("IMUleft")[0][0]
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
        self._obs = torch.zeros(self.num_envs, 60, device=self.device)
        self._fresh_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Explicitly keep runtime limits in sync with the control interface.
        bounds = torch.stack((self.servo_min*self.signs, self.servo_max*self.signs), dim=-1).sort(dim=-1).values
        self.robot.write_joint_position_limit_to_sim(torch.deg2rad(bounds).repeat(self.num_envs, 1, 1), joint_ids=self.joint_ids)
        print(f"OFFICIALdesign: 13 links, 10 DOF, 60 observations; {self.cfg.calibration_status}")
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

    def _get_observations(self):
        # get_observations() can be called twice by a runner; do not shift twice.
        if self._obs_step != self.common_step_counter or self._fresh_reset.any():
            angles = rpy_xyzw(as_torch(self.robot.data.body_link_quat_w)[:, self.imu_id])[:, :2]
            gyro = as_torch(self.imu.data.ang_vel_b)
            angles = angles + torch.randn_like(angles)*self.cfg.orientation_noise_std
            gyro = gyro + torch.randn_like(gyro)*self.cfg.gyro_noise_std
            sample = torch.cat((angles.clamp(-1, 1), (gyro/2).clamp(-1, 1)), dim=-1)
            normalized = self._normalize(self.cmd_actions)
            self.imu_history[:, :-1] = self.imu_history[:, 1:].clone()
            self.action_history[:, :-1] = self.action_history[:, 1:].clone()
            self.imu_history[:, -1] = sample
            self.action_history[:, -1] = normalized
            self.imu_history[self._fresh_reset] = sample[self._fresh_reset, None, :]
            self.action_history[self._fresh_reset] = normalized[self._fresh_reset, None, :]
            self._fresh_reset[:] = False
            self._obs = torch.cat((self.imu_history.flatten(1), self.action_history.flatten(1)), dim=-1)
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

    def _apply_action(self):
        target = self.previous_target if self._substep < self.cfg.actuator_delay_steps else self.applied_target
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        self._substep += 1

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

    def _reset_idx(self, env_ids):
        """Reset physics, actuator targets, delay state and all policy history per environment."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
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
