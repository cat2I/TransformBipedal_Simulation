# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ── ACTIVE: Fulltrans10DOF.usd, RL CHỈ điều khiển 6/10 khớp (Hip/Knee/Foot) ──
# Bub+Twist khoá cố định 0° — giống cấu trúc 6DOF đã proven (transformer_nam_env.py)
# hội tụ nhanh (~350 iter), nhưng dùng đúng asset thật thay vì NewSimple.usd.
# Chọn hướng này vì bản full-control 10DOF (transformer_walk10dof_env.py) hội tụ
# quá chậm — iteration 499 mới đạt 32% max ep_length, so với 6DOF đạt 91% ở iter 87.
gym.register(
    id="Transformer-Walk10DOF6-Direct-v0",
    entry_point=f"{__name__}.transformer_walk10dof6_env:TransformerWalk10DOF6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transformer_walk10dof6_env:TransformerWalk10DOF6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)

# ── Full 10DOF control (Bub+Hip+Twist+Knee+Foot đều RL điều khiển) ──────────
# Giữ lại để so sánh — hội tụ chậm hơn nhiều so với bản 6/10 ở trên, xem log
# training 2026-07-23 (iter 499: ep_length chỉ 32% max).
gym.register(
    id="Transformer-Walk10DOF-Direct-v0",
    entry_point=f"{__name__}.transformer_walk10dof_env:TransformerWalk10DOFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transformer_walk10dof_env:TransformerWalk10DOFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)

# ── OLD: 6 DOF trên NewSimple.usd (model_349.pt đã train bằng config này) ──
# Giữ lại để rollback / so sánh, không dùng cho training mới.
# gym.register(
#     id="Transformer-Walk-Direct-v0",
#     entry_point=f"{__name__}.transformer_nam_env:TransformerWalkEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.transformer_nam_env:TransformerWalkEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
#     },
# )



# gym.register(
#     id="Transformer-SquatToStand-Direct-v0",
#     entry_point=f"{__name__}.transformer_hieu_env:TransformerStandEnv",  # ✅ FIX
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.transformer_hieu_env:TransformerStandEnvCfg",  # ✅ OK
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
#     },
# )

# import gymnasium as gym

# from . import agents

# from .transformer_hieu_env import (
#     TransformerTwistMarchEnv,
#     TransformerTwistMarchEnvCfg,
# )

# gym.register(
#     id="TransformerTwistMarch-v0",

#     entry_point=
#         f"{__name__}.transformer_hieu_env:TransformerTwistMarchEnv",

#     disable_env_checker=True,

#     kwargs={
#         "env_cfg_entry_point":
#             f"{__name__}.transformer_hieu_env:TransformerTwistMarchEnvCfg",

#         "rsl_rl_cfg_entry_point":
#             f"{agents.__name__}.rsl_rl_ppo_cfg:"
#             "TransformerWalkPPORunnerCfg",
#     },
# )