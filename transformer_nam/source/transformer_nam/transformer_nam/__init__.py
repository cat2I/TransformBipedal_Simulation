# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# UI extension (ui_extension_example.py) CỐ Ý không import ở đây.
#
# File đó là boilerplate của template IsaacLab và project không dùng UI extension
# nào. Nó `import omni.ext`, mà omni.ext chỉ nạp được SAU khi SimulationApp khởi
# động — nên để dòng import ở đây thì `import transformer_nam` gãy với mọi công
# cụ đứng ngoài simulator (script chuyển checkpoint, linter, kiểm tra nhanh...).
# Train/play không bị ảnh hưởng vì chúng import package sau AppLauncher.
#
# Nếu sau này thực sự cần UI extension, import nó bên trong hàm chạy sau khi app
# đã khởi động, đừng đưa lên mức module.
