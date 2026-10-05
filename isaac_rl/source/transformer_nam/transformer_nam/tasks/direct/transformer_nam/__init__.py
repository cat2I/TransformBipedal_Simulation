# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Đăng ký task. Mỗi task thuộc về ĐÚNG MỘT robot — đó là đường biên cứng của
dự án này: đổi robot là checkpoint, reward scale, pose và thứ tự khớp đều phải
làm lại từ đầu, dù shape obs/action có trùng nhau đi nữa.

Tên task theo mẫu `<Robot>-<Task>-v0`.

Ba task của fulltrans (Walk10DOF, Walk10DOF6, StandUp) đã đóng băng vào
`_archive/fulltrans/` ngày 2026-10-05: không policy nào biết đi (tốt nhất
ep_len 83.1/200), và chúng dùng hai giao diện khác nhau (60/10 và 44/6) trên
cùng một asset.
"""

import gymnasium as gym

from . import agents

# ── OFFICIALdesign — robot mới, đang train ──────────────────────────────────
# 13 link, 10 DOF, obs 60 / act 10. Asset: assets/officialdesign/.
# Mới chỉ có smoke test 2 iteration; chưa có policy biết đi.
gym.register(
    id="Official-Walk-v0",
    entry_point=f"{__name__}.official_env:OfficialWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.official_env:OfficialWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.official_ppo_cfg:OfficialWalkPPORunnerCfg",
    },
)

# ── NewSimple — 6 DOF, asset của model_349.pt ───────────────────────────────
# Đây là policy đứng sau 0319.gif (robot thật đi bộ). Truy ngược từ
# trajectory_exports/trajectory_20260319_132712.json: đối chiếu đầu ra mạng với
# raw_actions đã ghi -> khớp run 2026-03-19_13-18-11_work, sai số trung vị
# 0.0064 (á quân 1.71, chênh 270 lần).
#
# Phát lại bằng:  ./play.sh newsimple 2026-03-19_13-18-11_work 349
# (play.sh tự thêm bộ cờ env ép về đúng điều kiện lúc train — thiếu nó thì
# policy chạy trong môi trường nhiễu khác và trông như hỏng.)
gym.register(
    id="NewSimple-Walk-v0",
    entry_point=f"{__name__}.transformer_nam_env:TransformerWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transformer_nam_env:TransformerWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)
