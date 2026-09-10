"""Test initial environment setup - check if robot stands stable and inspect ground"""

import torch
import argparse
import time
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ✅ FIX: Import after AppLauncher
import transformer_nam.tasks.direct.transformer_nam
from transformer_nam.tasks.direct.transformer_nam.transformer_nam_env import (
    TransformerWalkEnv,
    TransformerWalkEnvCfg,
)

def test_initial_environment():
    """Test robot initial pose stability and inspect scene"""
    
    print("\n" + "="*80)
    print("🤖 TESTING INITIAL ENVIRONMENT SETUP")
    print("="*80)
    
    cfg = TransformerWalkEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = TransformerWalkEnv(cfg=cfg)
    env.use_actuator_noise = False  # ✅ Disable noise for stability test
    obs, _ = env.reset()
    
    robot = env.robot
    contact = env.contact
    
    print("\n📊 INITIAL STATE INSPECTION:")
    print("-" * 80)
    
    # 1. Robot base pose
    base_pos = robot.data.root_pos_w[0].cpu().numpy()
    base_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"\n1️⃣  Robot Base Position:")
    print(f"   X: {base_pos[0]:.4f} m")
    print(f"   Y: {base_pos[1]:.4f} m")
    print(f"   Z: {base_pos[2]:.4f} m  (target: {env.cfg.base_height:.4f} m)")
    print(f"\n   Quaternion (x,y,z,w): [{base_quat[0]:.3f}, {base_quat[1]:.3f}, {base_quat[2]:.3f}, {base_quat[3]:.3f}]")
    
    # 2. Joint positions
    joint_pos_deg = torch.rad2deg(robot.data.joint_pos[0]).cpu().numpy()
    joint_names = ["Bubleft", "Hipleft", "Kneeleft", "Footleft", 
                   "Bubright", "Hipright", "Kneeright", "Footright"]
    
    print(f"\n2️⃣  Joint Positions (degrees):")
    for i, name in enumerate(joint_names):
        target = env.base_pose[i].cpu().item()
        actual = joint_pos_deg[i]
        error = abs(actual - target)
        status = "✅" if error < 5.0 else "⚠️"
        print(f"   {status} {name:12s}: {actual:7.2f}° (target: {target:7.2f}°, error: {error:.2f}°)")
    
    # 3. Contact sensor info
    print(f"\n3️⃣  Contact Sensor Configuration:")
    print(f"   Prim path: {env.cfg.contact.prim_path}")
    print(f"   Force threshold: {env.cfg.contact.force_threshold}")
    print(f"   Update period: {env.cfg.contact.update_period}")
    
    # 4. Ground plane info
    print(f"\n4️⃣  Ground Plane Properties:")
    print(f"   Path: /World/GroundPlane (or from USD)")  # ✅ CHANGED
    print(f"   Source: Embedded in USD file")  # ✅ NEW
    print(f"   Static friction: Check USD properties")
    print(f"   Dynamic friction: Check USD properties")
    print(f"   Restitution: Check USD properties")
    
    # 5. Physics settings
    print(f"\n5️⃣  Physics Configuration:")
    print(f"   Timestep: {env.cfg.sim.dt} s")
    print(f"   Decimation: {env.cfg.decimation}")
    print(f"   Effective control rate: {env.cfg.sim.dt * env.cfg.decimation} s")
    print(f"   Gravity: (0, 0, -9.81) m/s²")
    
    print("\n" + "="*80)
    print("🏃 RUNNING STABILITY TEST (10 seconds with HOLD POSE actions)")
    print("="*80)
    
    # ✅ CHANGE: Use actions that HOLD current pose instead of zero
    # Zero actions cause drift due to backlash!
    
    # Get initial gear position (this is what robot tries to maintain)
    hold_actions = torch.zeros(args_cli.num_envs, 8, device=env.device)
    
    test_duration = 10.0
    dt = env.physics_dt * env.cfg.decimation
    num_steps = int(test_duration / dt)
    
    # Statistics tracking
    survived_steps = torch.zeros(args_cli.num_envs, device=env.device)
    min_height = torch.full((args_cli.num_envs,), float('inf'), device=env.device)
    max_height = torch.full((args_cli.num_envs,), float('-inf'), device=env.device)
    max_tilt = torch.zeros(args_cli.num_envs, device=env.device)
    max_joint_drift = torch.zeros(args_cli.num_envs, device=env.device)
    
    print(f"\n⏱️  Test parameters:")
    print(f"   Duration: {test_duration} s")
    print(f"   Control timestep: {dt} s")
    print(f"   Total steps: {num_steps}")
    print(f"   Action: ZERO (should hold base_pose)")
    
    # Record initial state
    initial_joints = robot.data.joint_pos.clone()
    
    try:
        for step in range(num_steps):
            # ✅ Apply ZERO actions (should maintain current cmd_actions)
            obs, reward, terminated, truncated, info = env.step(hold_actions)
            
            # Track joint drift
            current_joints = robot.data.joint_pos
            joint_drift = torch.rad2deg(torch.abs(current_joints - initial_joints)).max(dim=1)[0]
            max_joint_drift = torch.max(max_joint_drift, joint_drift)
            
            # Track statistics
            height = robot.data.root_pos_w[:, 2]
            min_height = torch.min(min_height, height)
            max_height = torch.max(max_height, height)
            
            # Calculate tilt
            # IsaacLab 3.0 tra quaternion theo (x, y, z, w), khong con (w, x, y, z)
            quat = robot.data.root_quat_w
            x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x**2 + y**2)
            roll = torch.atan2(sinr_cosp, cosr_cosp)
            sinp = 2 * (w * y - z * x)
            pitch = torch.asin(torch.clamp(sinp, -1, 1))
            tilt = torch.abs(roll) + torch.abs(pitch)
            max_tilt = torch.max(max_tilt, tilt)
            
            survived_steps += (~(terminated | truncated)).float()
            
            # Print progress every 2 seconds
            if (step + 1) % int(2.0 / dt) == 0:
                alive = (~(terminated | truncated)).sum().item()
                avg_height = height.mean().item()
                avg_reward = reward.mean().item()
                avg_drift = joint_drift.mean().item()
                print(f"   Step {step+1:3d}/{num_steps}: "
                      f"Alive: {alive}/{args_cli.num_envs}, "
                      f"Height: {avg_height:.3f}m, "
                      f"Drift: {avg_drift:.2f}°, "
                      f"Reward: {avg_reward:.3f}")
            
            if not args_cli.headless:
                time.sleep(dt * 0.5)
                
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    
    # Final report
    print("\n" + "="*80)
    print("📊 STABILITY TEST RESULTS")
    print("="*80)
    
    survival_rate = (survived_steps / num_steps * 100).mean().item()
    num_survived = (survived_steps == num_steps).sum().item()
    
    print(f"\n✅ Survival Statistics:")
    print(f"   Survival rate: {survival_rate:.1f}%")
    print(f"   Envs completed: {num_survived}/{args_cli.num_envs}")
    
    print(f"\n📏 Height Statistics:")
    print(f"   Target height: {env.cfg.base_height:.4f} m")
    print(f"   Min height: {min_height.mean().item():.4f} m")
    print(f"   Max height: {max_height.mean().item():.4f} m")
    print(f"   Height range: {(max_height - min_height).mean().item():.4f} m")
    
    print(f"\n📐 Orientation Statistics:")
    print(f"   Max tilt: {torch.rad2deg(max_tilt.mean()).item():.2f}° (threshold: 28.6°)")
    
    print(f"\n🎯 Joint Drift Statistics:")
    print(f"   Max drift: {max_joint_drift.mean().item():.2f}° (target: <2°)")
    
    # Diagnosis
    print("\n" + "="*80)
    print("🔍 DIAGNOSIS")
    print("="*80)
    
    if survival_rate > 95:
        print("\n✅ EXCELLENT: Robot is very stable at initial pose!")
        print("   → Environment setup is correct")
        print("   → Problem likely in reward function or policy training")
        print("\n💡 Suggested next steps:")
        print("   1. Check reward breakdown during training")
        print("   2. Verify that velocity reward is not too negative")
        print("   3. Monitor episode length progression")
        
    elif survival_rate > 70:
        print("\n⚠️  MODERATE: Robot can stand but has some instability")
        print("   → Consider adjusting:")
        print("      - Joint stiffness/damping")
        print("      - Initial height (currently 0.5m)")
        print("      - Base pose joint angles")
        
    elif survival_rate > 30:
        print("\n❌ POOR: Robot struggles to maintain initial pose")
        height_dev = abs(min_height.mean().item() - env.cfg.base_height)
        if height_dev > 0.1:
            print(f"   → Height deviation too large: {height_dev:.3f}m")
            print("   → Increase initial spawn height in transformer_config.py")
        if max_tilt.mean().item() > 0.3:
            print(f"   → Robot tilting too much: {torch.rad2deg(max_tilt.mean()).item():.1f}°")
            print("   → Adjust base_pose joint angles")
            
    else:
        print("\n🔴 CRITICAL: Robot falls immediately!")
        print("   → Initial pose is NOT stable!")
        print("\n🔧 Required fixes:")
        print("   1. Increase spawn height: pos=(0.0, 0.0, 0.6) in transformer_config.py")
        print("   2. Check collision properties (contact_offset, rest_offset)")
        print("   3. Verify ground plane is created properly")
        print("   4. Test with different base_pose angles")
    
    print("\n" + "="*80)
    print("🎬 Scene Hierarchy (for inspection in Isaac Sim GUI):")
    print("="*80)
    print("\n/World")
    print("├── defaultGroundPlane  ← Check this exists")
    print("├── Light               ← Check intensity/color")
    print("└── envs")
    for i in range(min(4, args_cli.num_envs)):
        print(f"    ├── env_{i}")
        print(f"    │   └── Robot")
        print(f"    │       └── Robot")
        print(f"    │           ├── Baselink")
        print(f"    │           ├── Footleft  ← Should have contact sensor")
        print(f"    │           └── Footright ← Should have contact sensor")
    
    print("\n💡 To inspect in GUI, use Isaac Sim windows:")
    print("   Window → Stage (to see hierarchy)")
    print("   Window → Property (to see attributes)")
    
    print("\n" + "="*80)
    
    env.close()

if __name__ == "__main__":
    test_initial_environment()
    simulation_app.close()