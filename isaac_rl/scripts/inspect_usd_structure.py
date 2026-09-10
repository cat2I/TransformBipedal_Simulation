"""Inspect USD file structure to verify paths"""

from pxr import Usd

usd_path = "/home/tatung/Desktop/Transform_bipedal/Transformer_IsaacLab/asset/Transformer.usd"
stage = Usd.Stage.Open(usd_path)

print("\n🔍 USD FILE STRUCTURE:")
print("=" * 80)

def print_prim_tree(prim, indent=0):
    prefix = "  " * indent
    prim_type = prim.GetTypeName()
    print(f"{prefix}├── {prim.GetName()} ({prim_type})")
    
    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1)

# Print full hierarchy
root = stage.GetPseudoRoot()
for child in root.GetChildren():
    print_prim_tree(child)

print("\n" + "=" * 80)
print("\n💡 Important paths to note:")
print("   - Robot root: /World/Robot (or similar)")
print("   - Ground plane: /World/GroundPlane (or similar)")
print("   - Foot bodies: /World/Robot/.../Footleft, /World/Robot/.../Footright")
print("\n   Update contact sensor path in transformer_nam_env.py accordingly!")