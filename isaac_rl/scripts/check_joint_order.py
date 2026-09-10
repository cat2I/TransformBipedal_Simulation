"""Check actual joint order in IsaacLab"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Import after AppLauncher
from transformer_nam.tasks.direct.transformer_nam.transformer_config import TRANSFORMER_CFG
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from dataclasses import replace

# Initialize simulation context
sim_cfg = SimulationCfg(device="cuda:0", dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

# ✅ Create config with resolved prim path (no {ENV_REGEX_NS})
robot_cfg = replace(TRANSFORMER_CFG, prim_path="/World/Robot")

# Create minimal scene
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

# Add robot with resolved path
robot = Articulation(robot_cfg)
scene.articulations["robot"] = robot

# Setup
scene.clone_environments(copy_from_source=False)

# Reset simulation to initialize robot
sim.reset()

print("\n" + "=" * 80)
print("🔍 ACTUAL JOINT ORDER IN ISAACLAB")
print("=" * 80)

print("\n📋 robot.joint_names:")
for i, name in enumerate(robot.joint_names):
    print(f"   [{i}] {name}")

print("\n📋 Expected order (your assumption):")
expected = ["Bubleft", "Hipleft", "Kneeleft", "Footleft",
            "Bubright", "Hipright", "Kneeright", "Footright"]
for i, name in enumerate(expected):
    print(f"   [{i}] {name}")

print("\n🔄 Comparison:")
mismatch = False
for i in range(max(len(robot.joint_names), len(expected))):
    actual = robot.joint_names[i] if i < len(robot.joint_names) else "MISSING"
    expect = expected[i] if i < len(expected) else "EXTRA"
    match = "✅" if actual == expect else "❌"
    if actual != expect:
        mismatch = True
    print(f"   {match} [{i}] Actual: {actual:12s} | Expected: {expect:12s}")

# Build correct mapping if needed
if mismatch:
    print("\n" + "=" * 80)
    print("⚠️  MISMATCH DETECTED! Building correct base_pose order")
    print("=" * 80)
    
    base_pose_dict = {
        "Bubleft": 0, "Hipleft": 0, "Kneeleft": 60, "Footleft": -30,
        "Bubright": 0, "Hipright": 0, "Kneeright": -60, "Footright": 30,
    }
    
    print("\n✅ Correct base_pose for transformer_nam_env.py (line ~118):")
    print("\nself.base_pose = torch.tensor(")
    print("    [", end="")
    correct_values = []
    for j, name in enumerate(robot.joint_names):
        value = base_pose_dict.get(name, 0.0)
        correct_values.append(value)
        if j > 0:
            print("     ", end="")
        print(f"{value:6.1f},  # [{j}] {name}")
    print("    ], device=self.device, dtype=torch.float32")
    print(")")
    
    print("\n" + "=" * 80)
    print("📋 COPY THIS LINE:")
    print("=" * 80)
    values_str = ", ".join([f"{v:.1f}" for v in correct_values])
    print(f"self.base_pose = torch.tensor([{values_str}], device=self.device, dtype=torch.float32)")
    
    print("\n" + "=" * 80)
    print("✅ ALSO UPDATE servo_max and servo_min IN SAME ORDER!")
    print("=" * 80)
    
    # Show servo limits mapping
    servo_max_dict = {
        "Bubleft": 179.806, "Hipleft": 21.257, "Kneeleft": 84.975, "Footleft": 40.262,
        "Bubright": 7.122, "Hipright": 36.389, "Kneeright": 86.671, "Footright": 65.208,
    }
    servo_min_dict = {
        "Bubleft": -7.122, "Hipleft": -36.389, "Kneeleft": -86.671, "Footleft": -65.208,
        "Bubright": -179.806, "Hipright": -21.257, "Kneeright": -84.975, "Footright": -40.262,
    }
    
    servo_max_ordered = [servo_max_dict.get(name, 0.0) for name in robot.joint_names]
    servo_min_ordered = [servo_min_dict.get(name, 0.0) for name in robot.joint_names]
    
    print("\nself.servo_max = torch.tensor(")
    print(f"    {servo_max_ordered},")
    print("    device=self.device, dtype=torch.float32")
    print(")")
    
    print("\nself.servo_min = torch.tensor(")
    print(f"    {servo_min_ordered},")
    print("    device=self.device, dtype=torch.float32")
    print(")")
else:
    print("\n✅ Joint order matches! No changes needed.")

simulation_app.close()