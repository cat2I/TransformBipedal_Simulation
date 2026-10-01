"""GPU acceptance checks for OFFICIALdesign; optional viewport capture with --viz kit."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--output", default="/tmp/official-validation.json")
parser.add_argument("--capture", default=None, help="PNG viewport capture (requires --viz kit).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from transformer_nam.tasks.direct.transformer_nam.official_config import BUILD, CALIBRATION, JOINTS
from transformer_nam.tasks.direct.transformer_nam.official_env import OfficialWalkEnv, OfficialWalkEnvCfg, as_torch


def main():
    cfg = OfficialWalkEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 30.0
    cfg.orientation_noise_std = cfg.gyro_noise_std = 0.0
    env = OfficialWalkEnv(cfg)
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (args.num_envs, 60)
        assert env.robot.num_joints == 10 and len(env.robot.body_names) == 13
        assert set(env.contact.body_names) == {'Footleft', 'Footright'}
        masses = as_torch(env.robot.data.body_mass)
        torch.testing.assert_close(masses.sum(-1), torch.full((args.num_envs,), 4.643, device=env.device))
        expected_effort = torch.tensor([CALIBRATION['actuators'][j['group']]['effort'] for j in JOINTS], device=env.device)
        torch.testing.assert_close(as_torch(env.robot.data.joint_effort_limits)[:, env.joint_ids], expected_effort.repeat(args.num_envs, 1))
        torch.testing.assert_close(as_torch(env.robot.data.joint_vel_limits)[:, env.joint_ids], torch.full_like(env.base_pose, 4.7))

        # Hold the initial pose for 5 s. Resets would mask an unstable model, so
        # explicitly fail on ANY termination/time-out instead of averaging them.
        min_height, max_force = float('inf'), 0.0
        for step in range(100):
            obs, reward, died, timeout, _ = env.step(torch.zeros_like(env.cmd_actions))
            assert not died.any() and not timeout.any(), (
                f"Default stance failed at step {step}; previous position={last_position if step else None}; "
                f"previous orientation={last_quat if step else None}; joint angles={last_joint if step else None}")
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            height = as_torch(env.robot.data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
            min_height = min(min_height, height.min().item())
            max_force = max(max_force, as_torch(env.contact.data.net_forces_w).abs().max().item())
            last_position = as_torch(env.robot.data.root_pos_w)[0].tolist()
            last_quat = as_torch(env.robot.data.root_quat_w)[0].tolist()
            last_joint = torch.rad2deg(as_torch(env.robot.data.joint_pos)[0, env.joint_ids]).tolist()
        forces = as_torch(env.contact.data.net_forces_w)
        vertical_force = forces[:, :, 2].sum(-1)
        assert torch.all(vertical_force > 0.6*4.643*9.81), vertical_force
        assert torch.all(vertical_force < 1.4*4.643*9.81), vertical_force
        assert (as_torch(env.robot.data.root_com_lin_vel_b).norm(dim=-1) < 0.1).all()
        stance_height = height.tolist()

        if args.capture:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            env.sim.set_camera_view(eye=(1.0, -1.0, 0.75), target=(0.0, 0.0, 0.22))
            for _ in range(15):
                env.sim.render()
            capture_viewport_to_file(get_active_viewport(), args.capture)
            for _ in range(20):
                app.update()

        # Exercise large actions, saturation and actuator speed/torque clipping.
        falls = 0
        for _ in range(100):
            obs, reward, died, _, _ = env.step(torch.randn_like(env.cmd_actions)*3)
            falls += died.sum().item()
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert (env.cmd_actions >= env.servo_min).all() and (env.cmd_actions <= env.servo_max).all()
            torque = as_torch(env.robot.data.applied_torque)[:, env.joint_ids]
            assert (torque.abs() <= expected_effort + 1e-4).all(), torque

        # Partial reset must clear selected histories/targets without overwriting
        # other environments; delay must not reapply the previous episode target.
        ids = torch.tensor([0], device=env.device)
        others = env.cmd_actions[1:].clone()
        env._reset_idx(ids)
        torch.testing.assert_close(env.cmd_actions[0], env.base_pose[0])
        torch.testing.assert_close(env.cmd_actions[1:], others)
        torch.testing.assert_close(env.previous_target[0], env._to_urdf(env.base_pose[0]))
        torch.testing.assert_close(env.applied_target[0], env.previous_target[0])
        assert torch.count_nonzero(env.actions[0]) == 0
        torch.testing.assert_close(env.action_history[0], env._normalize(env.base_pose[0]).repeat(4, 1))

        # Floating base drop from +10 cm: actually falls before landing, remains finite.
        env.reset()
        root = as_torch(env.robot.data.default_root_state).clone()
        root[:, :3] += env.scene.env_origins
        root[:, 2] += 0.10
        env.robot.write_root_link_pose_to_sim(root[:, :7])
        env.robot.write_root_com_velocity_to_sim(root[:, 7:])
        z_start = root[:, 2].clone()
        for _ in range(5):
            obs, reward, _, _, _ = env.step(torch.zeros_like(env.cmd_actions))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        assert (as_torch(env.robot.data.root_pos_w)[:, 2] < z_start-0.03).all(), 'Base appears fixed'
        for _ in range(35):
            obs, reward, _, _, _ = env.step(torch.zeros_like(env.cmd_actions))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()

        result = {'status': 'passed', 'num_envs': args.num_envs, 'mass_kg': masses.sum(-1).tolist(),
                  'observation_dim': 60, 'action_dim': 10, 'joint_order': env.cfg.joint_names,
                  'isaac_joint_indices': env.joint_ids, 'stance_seconds': 5,
                  'stance_min_base_height_m': min_height, 'stance_final_height_m': stance_height,
                  'stance_final_vertical_contact_N': vertical_force.tolist(),
                  'peak_contact_component_N': max_force, 'random_action_resets': falls,
                  'checks': ['mass', 'joint_mapping', 'effort_velocity_limits', 'stance_no_reset',
                             'foot_contact_weight', 'finite_rollout', 'bounded_actions_torques',
                             'partial_reset', 'floating_base_drop']}
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
