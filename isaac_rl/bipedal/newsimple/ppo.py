# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Siêu tham số PPO cho NewSimple."""

from isaaclab.utils.configclass import configclass

from .._shared.ppo import TransformerWalkPPORunnerCfg


@configclass
class NewSimpleWalkPPORunnerCfg(TransformerWalkPPORunnerCfg):
    # Log mới vào logs/newsimple/. Các run CŨ (gồm model_349) nằm ở logs/old/ —
    # 96 run của nhiều task trộn lẫn, không phân biệt được bằng tên nên không
    # tách ra được. play.sh tìm trong cả hai thư mục.
    experiment_name = "newsimple"
