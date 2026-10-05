# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Siêu tham số PPO cho NewSimple."""

from isaaclab.utils.configclass import configclass

from .._shared.ppo import TransformerWalkPPORunnerCfg


@configclass
class NewSimpleWalkPPORunnerCfg(TransformerWalkPPORunnerCfg):
    # CỐ Ý giữ "transformer_walk" thay vì đổi thành "newsimple_walk":
    # 93 run cũ (gồm cả model_349) đang nằm ở logs/rsl_rl/transformer_walk/.
    # Đổi bây giờ là tách log mới khỏi log cũ giữa chừng. Việc đổi tên sẽ làm
    # cùng lúc gom log cũ vào logs/old/ và bỏ tầng rsl_rl/ thừa.
    experiment_name = "transformer_walk"
