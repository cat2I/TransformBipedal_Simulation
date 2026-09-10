"""Find robot's actual path and default position in USD"""

from pxr import Usd, UsdGeom

usd_path = "/home/tatung/Desktop/Transform_bipedal/Transformer_IsaacLab/asset/Transformer.usd"
stage = Usd.Stage.Open(usd_path)

print("🔍 USD STRUCTURE:\n")
print("=" * 80)

# List ALL prims to find robot
robot_paths = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "Robot" in path or "robot" in path:
        robot_paths.append(path)
        print(f"Found robot prim: {path}")

print("\n" + "=" * 80)
print("\n📍 Checking positions:\n")

# Check each potential robot root
for robot_path in robot_paths:
    prim = stage.GetPrimAtPath(robot_path)
    if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(prim)
        
        # Get all xform ops
        ops = xform.GetOrderedXformOps()
        
        if ops:
            print(f"\n{robot_path}:")
            for op in ops:
                op_type = op.GetOpType()
                value = op.Get()
                
                if op_type == UsdGeom.XformOp.TypeTranslate:
                    print(f"  Position: {value}")
                    print(f"    X: {value[0]:.4f}m")
                    print(f"    Y: {value[1]:.4f}m")
                    print(f"    Z: {value[2]:.4f}m  ← HEIGHT!")
                    
                    if value[2] > 0.4:
                        print(f"    ⚠️  TOO HIGH! Robot spawns at {value[2]:.2f}m")
                        print(f"    → This is why you see Z=0.51m in logs")
                    elif value[2] < 0.3:
                        print(f"    ⚠️  TOO LOW! Robot will fall through ground")
                    else:
                        print(f"    ✅ OK height for standing pose")

print("\n" + "=" * 80)
print("\n💡 Next steps:")
print("   1. Note the robot path (e.g., /Robot or /World/Robot)")
print("   2. Note the Z position")
print("   3. Update transformer_config.py to override this position")