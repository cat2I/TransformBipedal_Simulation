"""Inspect USD file structure to verify paths"""

import os
from pxr import Usd

# Đường dẫn tương đối từ vị trí script → assets/fulltrans/usd/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_USD = os.path.join(_SCRIPT_DIR, "..", "assets", "fulltrans", "usd")
usd_path = os.path.join(_ASSETS_USD, "Fulltrans10DOF.usd")
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