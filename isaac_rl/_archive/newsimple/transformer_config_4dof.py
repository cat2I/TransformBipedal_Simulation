"""Configuration for Transformer robot asset"""
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
import math

from ._asset_paths import asset_path

TRANSFORMER_USD = asset_path("FullForm.usd")
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
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            # ✅ ADD "_joint" suffix to match USD!
            "Bubleft_joint": 0.0,
            "Hipleft_joint": math.radians(25),
            "Kneeleft_joint": math.radians(-50),
            "Footleft_joint": math.radians(25),
            "Bubright_joint": 0.0,
            "Hipright_joint": math.radians(25),
            "Kneeright_joint": math.radians(-50),
            "Footright_joint": math.radians(25),
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
            joint_names_expr=["Bubleft_joint", "Hipleft_joint", "Bubright_joint", "Hipright_joint",
                             "Kneeleft_joint", "Kneeright_joint"],
            effort_limit=10.3,        # stall torque (N·m)
            effort_limit_sim=10.3,
            saturation_effort=10.3,
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
