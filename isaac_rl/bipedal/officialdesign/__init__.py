# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""OFFICIALdesign — robot mới, handoff CAD 2026-09-30.

13 link, 10 DOF, obs 60 / act 10, policy 20 Hz. Asset: ``assets/officialdesign/``.
Mới chỉ có smoke test 2 iteration; chưa có policy biết đi.

- ``robot``      ArticulationCfg, đọc từ ``meta/calibration.json``, kiểm SHA-256
- ``task_walk``  môi trường đi bộ (obs/action/reward/termination)
- ``ppo``        siêu tham số, ``experiment_name = "officialdesign_walk"``

Cấu hình đọc từ JSON chứ không viết cứng trong Python, nên sửa giới hạn góc hay
thông số servo là sửa ``meta/calibration.json``. Nhưng ``robot.py`` kiểm SHA-256
lúc load, nên sửa xong PHẢI chạy lại::

    ./run.sh scripts/prepare_officialdesign.py
    ./run.sh scripts/convert_officialdesign.py --headless --force
"""

import gymnasium as gym

gym.register(
    id="Official-Walk-v0",
    entry_point=f"{__name__}.task_walk:OfficialWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_walk:OfficialWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo:OfficialWalkPPORunnerCfg",
    },
)
