# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Thứ dùng chung cho MỌI robot. Cố tình để mỏng.

- ``lab3``  lớp tương thích IsaacLab 2.x -> 3.0 (``as_torch``, ``imu_quat_w``)
- ``paths`` dò tìm thư mục ``assets/`` của project
- ``ppo``   cấu hình PPO nền, mỗi robot kế thừa rồi đổi ``experiment_name``

Trước khi thêm gì vào đây, hỏi: "thứ này có đúng với MỌI robot không?"
Lưỡng lự thì chép vào từng robot — trùng lặp vài chục dòng rẻ hơn nhiều so với
một ``_shared`` phình to rồi sửa cho robot này làm gãy robot kia.
"""
