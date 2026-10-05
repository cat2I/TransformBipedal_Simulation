# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""NewSimple — 6 DOF, robot của ``model_349.pt``.

Đây là policy duy nhất đã đi được trên robot thật (``0319.gif``). Truy ngược từ
``trajectory_exports/trajectory_20260319_132712.json``: đối chiếu đầu ra mạng
với ``raw_actions`` đã ghi -> khớp run ``2026-03-19_13-18-11_work``, sai số
trung vị 0.0064 (á quân 1.71, chênh 270 lần).

Phát lại::

    ./play.sh newsimple 2026-03-19_13-18-11_work 349

``play.sh`` tự thêm bộ cờ env ép môi trường về đúng điều kiện lúc train
(``domain_rand=False``, bias 0, nhiễu 0.015/0.01, drift 0). Thiếu bộ đó thì
policy chạy trong môi trường nhiễu khác và trông như hỏng.
"""

import gymnasium as gym

gym.register(
    id="NewSimple-Walk-v0",
    entry_point=f"{__name__}.task_walk:TransformerWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_walk:TransformerWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo:NewSimpleWalkPPORunnerCfg",
    },
)
