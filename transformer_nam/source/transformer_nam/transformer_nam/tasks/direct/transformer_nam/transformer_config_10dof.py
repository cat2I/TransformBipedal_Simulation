"""
Configuration for Transformer robot asset — FULL 10 DOF
==========================================================
Asset: Fulltrans10DOF.usd (thay cho NewSimple.usd 6-DOF trước đây)

Joint list (5 khớp/chân, khớp đúng joint_names_Fulltrans.yaml) — verify thực
nghiệm PyBullet (foot-position test), xem lịch sử chat:
  Bubleft_joint / Bubright_joint     — hông FRONTAL/roll-plane (xoạc), authority
                                        với ROLL, KHÔNG phải sagittal như ghi trước đây
  Hipleft_joint / Hipright_joint     — hông SAGITTAL/pitch-plane (đi bộ thật),
                                        authority với PITCH — khớp chính tạo dáng đi
  Twistleft_joint / Twistright_joint — xoay hông (rẽ hướng, MỚI so với 6DOF)
  Kneeleft_joint / Kneeright_joint   — gối
  Footleft_joint / Footright_joint   — cổ chân

Init pose = tư thế ĐỨNG (kết quả của phase CLOSE trong fulltrans_full.py):
  Bub=0°, Hip(lat)=0°, Twist=0°, Knee=0°, Foot=0°  (chân duỗi thẳng hoàn toàn)
  → Train walk bắt đầu TỪ tư thế đứng này, không phải từ đầu.

Actuator: giá trị effort/velocity lấy ĐÚNG từ Assemtest_new_fixed.urdf
(E:/Fulltrans1/Fulltrans/Assemtest_new/urdf/), không phải suy đoán:
  heavy_joints (STS3095, 105 kg·cm): Bub, Hip(lat), Knee
    effort=10.3 N·m  velocity=3.67 rad/s
  light_joints (STS3125, 30 kg·cm): Twist, Foot
    effort=2.94 N·m  velocity=4.82 rad/s
"""
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
import math

from ._asset_paths import asset_path

TRANSFORMER_10DOF_USD = asset_path("Fulltrans10DOF.usd")
TRANSFORMER_10DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRANSFORMER_10DOF_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
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
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    # ── Tư thế ĐỨNG (BUB đã khép về 0°, không phải tư thế xoạc) ─────────
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            "Bubleft_joint":    0.0,
            "Bubright_joint":   0.0,
            "Hipleft_joint":    0.0,
            "Hipright_joint":   0.0,
            "Twistleft_joint":  0.0,
            "Twistright_joint": 0.0,
            "Kneeleft_joint":   0.0,
            "Kneeright_joint":  0.0,
            "Footleft_joint":   0.0,
            "Footright_joint":  0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    # Giá trị effort/velocity lấy ĐÚNG từ Assemtest_new_fixed.urdf (nguồn
    # ground-truth cơ khí) — không phải suy đoán:
    #   Bub/Hip/Knee : effort=10.3 N·m  velocity=3.67 rad/s  → STS3095 (heavy)
    #   Twist/Foot   : effort=2.94 N·m  velocity=4.82 rad/s  → STS3125 (light)
    actuators={
        # ============================================================
        # STS3095 Servos (Bub, Hip-lat, Knee) - 105 kg·cm
        # ============================================================
        "heavy_joints": DCMotorCfg(
            joint_names_expr=[
                "Bubleft_joint",  "Bubright_joint",
                "Hipleft_joint",  "Hipright_joint",
                "Kneeleft_joint", "Kneeright_joint",
            ],
            effort_limit=10.3,
            effort_limit_sim=10.3,
            saturation_effort=10.3,
            velocity_limit=3.67,
            velocity_limit_sim=3.67,
            armature=0.08,
            stiffness=60.0,
            damping=0.85,
        ),
        # ============================================================
        # STS3125 Servos (Twist, Foot) - 30 kg·cm
        # ============================================================
        "light_joints": DCMotorCfg(
            joint_names_expr=[
                "Twistleft_joint", "Twistright_joint",
                "Footleft_joint",  "Footright_joint",
            ],
            stiffness=40.0,
            damping=0.65,
            armature=0.08,
            saturation_effort=2.94,
            effort_limit=2.94,
            effort_limit_sim=2.94,
            velocity_limit=4.82,
            velocity_limit_sim=4.82,
        ),
    },
)
