"""Configuration for Transformer robot asset"""
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
import math

from ._asset_paths import asset_path

TRANSFORMER_USD = asset_path("NewSimple.usd")
TRANSFORMER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRANSFORMER_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,  # ✅ ADD THIS!
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,  # 
            sleep_threshold=0.005,              # 
            stabilization_threshold=0.001,      # 
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),           #0.37
        joint_pos={
            # ✅ ADD "_joint" suffix to match USD!
            "Hipleft_joint": math.radians(25),            
            "Kneeleft_joint": math.radians(-50),              #-50
            "Footleft_joint": math.radians(25),           #25
            "Hipright_joint": math.radians(25),
            "Kneeright_joint": math.radians(-50),
            "Footright_joint": math.radians(25),                 #25
            
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # ============================================================
        # ✅ STS3095 Servos (Bub, Hip, Knee) - 105 kg·cm
        # ============================================================
        "heavy_joints": DCMotorCfg(
            joint_names_expr=["Hipleft_joint", "Hipright_joint", 
                             "Kneeleft_joint", "Kneeright_joint"],
            effort_limit=10.3,     #10.3   # stall torque (N·m)
            effort_limit_sim=10.3,   #10.3
            saturation_effort=10.3, #10.3
            velocity_limit=2.7,      
            velocity_limit_sim=2.7,
            armature=0.08,            
            stiffness=60.0,   
            damping=0.85,      
        ),
        
        # ============================================================
        # ✅ STS3125 Servos (Foot) - 30 kg·cm
        # ============================================================
        "light_joints": DCMotorCfg(
            joint_names_expr=["Footleft_joint", "Footright_joint"],
            stiffness=40.0,
            damping=0.65,
            armature=0.08,
            saturation_effort=2.94,
            effort_limit=2.94,
            effort_limit_sim=2.94,
            velocity_limit=2.7,
            velocity_limit_sim=2.7,
        ),
    },
)
"""
Configuration for Transformer bipedal robot — 8 DOF
=====================================================
Từ URDF FullForm.urdf:
  Bubleft  axis=(-1,0,0): dạng sang trái = góc ÂM
  Bubright axis=(+1,0,0): dạng sang phải = góc DƯƠNG
  Hip/Knee/Foot axis=(0,-1,0)
 
Pose START (nằm xoạc như ảnh):
  Bubleft=-88°, Bubright=+88°, tất cả còn lại=0°
 
Limits đã patch trong FullForm.usd: Bub ±90°
"""
import math
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from ._asset_paths import asset_path
 
TRANSFORMER_USD = asset_path("FullForm111.usd")
TRANSFORMER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRANSFORMER_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),   # sát đất — robot nằm ngang
        joint_pos={
            # ── POSE XOẠC (nằm như con ếch) ──────────────────
            # Bubleft  axis=(-1,0,0): âm = dạng sang trái
            "Bubleft_joint":   math.radians(90.0),    # [0] dạng sang trái
            # Bubright axis=(+1,0,0): dương = dạng sang phải
            "Bubright_joint":  math.radians(90.0),    # [1] dạng sang phải
            # Tất cả còn lại = 0
            "Hipleft_joint":   0.0,
            "Hipright_joint":  0.0,
            "Kneeleft_joint":  0.0,
            "Kneeright_joint": 0.0,
            "Footleft_joint":  0.0,
            "Footright_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Bub + Hip + Knee — STS3095
        "heavy_joints": DCMotorCfg(
            joint_names_expr=[
                "Bubleft_joint", "Bubright_joint",
                "Hipleft_joint", "Hipright_joint",
                "Kneeleft_joint", "Kneeright_joint",
            ],
            effort_limit=80,
            effort_limit_sim=80,
            saturation_effort=80,
            velocity_limit=2.7,
            velocity_limit_sim=2.7,
            armature=0.08,
            stiffness=200.0,
            damping=5.0,
        ),
        # Foot — STS3125
        "light_joints": DCMotorCfg(
            joint_names_expr=["Footleft_joint", "Footright_joint"],
            stiffness=120.0,
            damping=3.0,
            armature=0.08,
            saturation_effort=40,
            effort_limit=40,
            effort_limit_sim=40,
            velocity_limit=2.7,
            velocity_limit_sim=2.7,
        ),
    },
)
# """
# transformer_config_10dof.py
# ===========================
# Isaac Lab ArticulationCfg cho robot Transformer 10 DOF
# (Bubleft, Hipleft, Twistleft, Kneeleft, Footleft,
#  Bubright, Hipright, Twistright, Kneeright, Footright)

# Standing pose (from PyBullet Phase-1 IK, L1=0.18051m, L2=0.19m):
#   Bub   = +120.00 deg  →  +2.0944 rad
#   Hip   =  +13.69 deg  →  +0.2390 rad
#   Twist =  +90.00 deg  →  +1.5708 rad   ← start; goal = 0 deg
#   Knee  =  -26.69 deg  →  -0.4659 rad
#   Foot  =  +13.00 deg  →  +0.2269 rad

# Actuator: ImplicitActuatorCfg (position control, stiffness/damping tuned
#           to match real servo specs; domain rand is done at runtime).
# """

# import math
# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets import ArticulationCfg

# # ──────────────────────────────────────────────────────────────────────
# # Standing pose in radians (URDF joint declaration order, NO fixed joints)
# # Isaac Lab skips fixed joints automatically when loading URDF/USD,
# # so the 10 revolute joints map in order:
# #   [Bub_L, Hip_L, Twist_L, Knee_L, Foot_L,
# #    Bub_R, Hip_R, Twist_R, Knee_R, Foot_R]
# # ──────────────────────────────────────────────────────────────────────
# _DEG = math.pi / 180.0

# _STAND_RAD = {
#     # Left leg
#     "Bubleft_joint":    120.00 * _DEG,
#     "Hipleft_joint":     13.695 * _DEG,
#     "Twistleft_joint":   90.00 * _DEG,   # goal = 0 rad
#     "Kneeleft_joint":   -26.693 * _DEG,
#     "Footleft_joint":    12.998 * _DEG,
#     # Right leg
#     "Bubright_joint":   120.00 * _DEG,
#     "Hipright_joint":    13.695 * _DEG,
#     "Twistright_joint":  90.00 * _DEG,   # goal = 0 rad
#     "Kneeright_joint":  -26.693 * _DEG,
#     "Footright_joint":   12.998 * _DEG,
# }

# # ──────────────────────────────────────────────────────────────────────
# # Actuator specs
# # Real servos: ~9.5 kg·cm ≈ 0.93 N·m, 0.1 s/60° ≈ 104 rpm
# # PD gains tuned conservatively for sim-to-real transfer
# # Domain rand of friction/damping is done in env._reset_idx()
# # ──────────────────────────────────────────────────────────────────────
# _BUB_ACT = ImplicitActuatorCfg(
#     joint_names_expr=["Bubleft_joint", "Bubright_joint"],
#     effort_limit=9.27,        # N·m  (torque_samples mean)
#     velocity_limit=10.0,      # rad/s
#     stiffness=120.0,
#     damping=0.65,
#     friction=0.40,
# )

# _HIP_ACT = ImplicitActuatorCfg(
#     joint_names_expr=["Hipleft_joint", "Hipright_joint"],
#     effort_limit=9.27,
#     velocity_limit=10.0,
#     stiffness=120.0,
#     damping=0.65,
#     friction=0.40,
# )

# _TWIST_ACT = ImplicitActuatorCfg(
#     joint_names_expr=["Twistleft_joint", "Twistright_joint"],
#     effort_limit=9.27,
#     velocity_limit=10.0,
#     stiffness=100.0,          # slightly softer — twist is the controlled DOF
#     damping=0.60,
#     friction=0.40,
# )

# _KNEE_ACT = ImplicitActuatorCfg(
#     joint_names_expr=["Kneeleft_joint", "Kneeright_joint"],
#     effort_limit=9.27,
#     velocity_limit=10.0,
#     stiffness=140.0,          # stiffer for weight-bearing
#     damping=0.65,
#     friction=0.40,
# )

# _FOOT_ACT = ImplicitActuatorCfg(
#     joint_names_expr=["Footleft_joint", "Footright_joint"],
#     effort_limit=9.27,
#     velocity_limit=10.0,
#     stiffness=120.0,
#     damping=0.65,
#     friction=0.40,
# )

# # ──────────────────────────────────────────────────────────────────────
# # Main config — point USD_PATH to your Fulltrans10DOF.usd
# # ──────────────────────────────────────────────────────────────────────
# USD_PATH = asset_path("Fulltrans10DOF.usd")
# # If you keep the USD next to the env file, use a relative path e.g.:
# # USD_PATH = f"{os.path.dirname(__file__)}/../../../../asset/Fulltrans10DOF.usd"

# TRANSFORMER_10DOF_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=USD_PATH,
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=4,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.45),   # spawn height; ground contact settles it
#         joint_pos=_STAND_RAD,
#     ),
#     actuators={
#         "bub":   _BUB_ACT,
#         "hip":   _HIP_ACT,
#         "twist": _TWIST_ACT,
#         "knee":  _KNEE_ACT,
#         "foot":  _FOOT_ACT,
#     },
#     soft_joint_pos_limit_factor=0.95,
# )