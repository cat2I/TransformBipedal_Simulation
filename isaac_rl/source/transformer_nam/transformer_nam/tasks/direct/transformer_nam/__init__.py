# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ── Fulltrans10DOF.usd, RL CHỈ điều khiển 6/10 khớp (Hip/Knee/Foot) ──────────
# Bub+Twist khoá cố định 0°, mượn cấu trúc từ bản 6DOF cũ (transformer_nam_env.py).
#
# ĐÍNH CHÍNH 2026-08-03 — ghi chú cũ ở đây nói chọn hướng này vì "6DOF đạt 91%
# ở iter 87, còn 10DOF iter 499 mới 32%". So sánh đó lệch chuẩn: con số 91% là
# của bản 6DOF chạy trên NewSimple.usd (asset đơn giản hơn nhiều), không phải
# bản 10DOF6 chạy trên Fulltrans10DOF.usd đăng ký ở đây.
#
# Đo lại từ tensorboard, cùng asset thật, episode tối đa 200 bước (10.0s / 0.05s):
#
#   2026-07-23_15-23-03  10DOF full  iter 1500  ep_len 83.1  = 41.5%
#   2026-07-23_15-12-33  10DOF6      iter  350  ep_len 32.9  = 16.5%
#
# Tức trên asset thật, 10DOF full đang NHỈNH HƠN, ngược với ghi chú cũ. Cả hai
# đều chưa biết đi (robot ngã trước khi hết 10s). Giữ cả hai task để so tiếp.
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
# Run tốt nhất hiện có: 2026-07-23_15-23-03, 1500 iteration, ep_len 83.1/200.
# Xem bảng đối chiếu ở khối chú thích phía trên.
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
# ── StandUp task: from split (BUB=90) to standing ──
gym.register(
    id="Transformer-StandUp-Direct-v0",
    entry_point=f"{__name__}.transformer_standup_env:TransformerStandUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transformer_standup_env:TransformerStandUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)
