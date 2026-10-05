# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Môi trường RL IsaacLab cho robot hai chân.

Cấu trúc: MỖI ROBOT MỘT THƯ MỤC
--------------------------------
Robot là đường biên cứng của dự án này. Đổi robot thì checkpoint, reward scale,
pose, giới hạn góc và thứ tự khớp đều phải làm lại từ đầu — kể cả khi shape
obs/action trùng nhau. Đổi task trên cùng một robot thì giữ gần hết.

Thứ gì thay đổi cùng nhau thì ở cùng nhau, nên robot nằm ngoài, task nằm trong::

    bipedal/
        _shared/           dùng chung MỌI robot — cố tình để mỏng
        officialdesign/    robot mới, đang train
        newsimple/         robot của model_349 (0319.gif)

``_shared/`` phải mỏng. Mỗi lần định thêm gì vào đó, hỏi: "thứ này có đúng với
MỌI robot không?" Lưỡng lự thì chép vào từng robot. Trùng lặp 30 dòng rẻ hơn
nhiều so với một ``_shared/`` phình to rồi sửa cho robot A làm gãy robot B.

IsaacLab gốc chia ngược lại (task-first, robot nằm ở ``isaaclab_assets``) vì nó
là thư viện benchmark: task là hằng số, robot là biến. Dự án này ngược — robot
là hằng số của cả một giai đoạn (CAD chốt rồi, hàn rồi), task là biến.

Đăng ký task
------------
Import tường minh, KHÔNG dùng ``isaaclab_tasks.utils.import_packages``. Bộ quét
đó lọc blacklist theo CHUỖI CON (``any(b in name for b in ["utils", ".mdp"])``),
nên một thư mục tên chứa "utils" bị bỏ qua **im lặng**. Thiếu một ``__init__.py``
ở tầng trung gian cũng cho kết quả y hệt: import thành công, nhưng ``gym.make``
báo ``NameNotFound`` — thông báo chỉ sai hướng, làm người ta đi soi tên task.

Thêm robot mới = tạo thư mục + thêm một dòng dưới đây.
"""

from . import newsimple, officialdesign  # noqa: F401

__all__ = ["newsimple", "officialdesign"]
