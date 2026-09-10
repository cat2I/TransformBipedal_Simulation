# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import gymnasium as gym

# # Import để trigger registration
# from .transformer_nam import transformer_nam_env  # noqa: F401

# # Register environment
# gym.register(
#     id="Transformer-Walk-Direct-v0",
#     entry_point="transformer_nam.tasks.direct.transformer_nam.transformer_nam_env:TransformerWalkEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "transformer_nam.tasks.direct.transformer_nam.transformer_nam_env:TransformerWalkEnvCfg",
#         "rsl_rl_cfg_entry_point": "transformer_nam.tasks.direct.transformer_nam.agents.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
#     },
# )
# import gymnasium as gym


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