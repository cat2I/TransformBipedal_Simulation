"""OFFICIALdesign 2026-09-30: CAD names, explicit sign mapping, provisional calibration."""
import json
import hashlib
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from ._asset_paths import _assets_root

ASSET_DIR = _assets_root() / "officialdesign"
CALIBRATION = json.loads((ASSET_DIR / "meta/calibration.json").read_text())
BUILD = json.loads((ASSET_DIR / "meta/build_report.json").read_text())
USD_REPORT = json.loads((ASSET_DIR / "meta/usd_report.json").read_text())
for relative, expected in USD_REPORT["sha256"].items():
    if hashlib.sha256((ASSET_DIR / relative).read_bytes()).hexdigest() != expected:
        raise ValueError(f"OFFICIALdesign asset/config changed: {relative}. Run prepare_officialdesign.py then convert_officialdesign.py --force.")
JOINTS = CALIBRATION["joints"]
JOINT_NAMES = [j["name"] for j in JOINTS]
MATERIAL = sim_utils.RigidBodyMaterialCfg(**CALIBRATION["ground"], friction_combine_mode="average")

OFFICIAL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "usd/OFFICIALdesign.usd"),
        activate_contact_sensors=True,
        physics_material=MATERIAL,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, enable_gyroscopic_forces=True,
            linear_damping=0.0, angular_damping=0.0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BUILD["spawn_height_m"]),
        rot=(0.0, 0.0, 0.0, 1.0),  # Isaac Lab 3 uses xyzw.
        joint_pos={j["name"]: math.radians(j["sign"]*j["policy_default_deg"]) for j in JOINTS},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        group: DCMotorCfg(
            joint_names_expr=[j["name"] for j in JOINTS if j["group"] == group],
            effort_limit=values["effort"], effort_limit_sim=values["effort"],
            saturation_effort=values["effort"],
            velocity_limit=values["velocity"], velocity_limit_sim=values["velocity"],
            stiffness=values["stiffness"], damping=values["damping"], armature=values["armature"],
        ) for group, values in CALIBRATION["actuators"].items()
    },
)
