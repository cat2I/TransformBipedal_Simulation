"""
Lớp tương thích giữa IsaacLab 2.x và 3.0
=========================================
IsaacLab 3.0 đổi BA thứ ảnh hưởng trực tiếp tới code của project:

1. ``ImuData`` không còn ``quat_w``. Sensor ``Imu`` giờ là IMU "thật", chỉ xuất
   gyro (``ang_vel_b``) và accelerometer (``lin_acc_b``); phần pose chuyển sang
   sensor ``Pva`` mới. Đây là lỗi ném exception ngay nên dễ phát hiện.

2. Quy ước quaternion đổi từ ``(w, x, y, z)`` sang ``(x, y, z, w)``. Đây là lỗi
   ÂM THẦM — không có exception, chỉ ra góc sai. Đã kiểm chứng thực nghiệm:
   robot spawn ở tư thế thẳng đứng cho ``root_quat_w ≈ [0, 0, 0, 1]``.

3. ``robot.data.*`` không còn trả ``torch.Tensor`` mà trả ``ProxyArray`` bọc
   quanh mảng warp. Nó bắc cầu được phép toán và hàm torch thông thường, nhưng
   KHÔNG lọt qua hàm ``@torch.jit.script`` và không có các method như
   ``clone()``. Dùng :func:`as_torch` ở mọi chỗ đọc ``robot.data``.

Module này dựng lại ``quat_w`` bằng đúng công thức bản 2.x đã dùng bên trong
sensor Imu::

    quat_imu_w = quat_mul(quat_thân_gắn_IMU, offset.rot)

nhờ đó observation giữ nguyên ý nghĩa mà các checkpoint cũ đã được train.

Về offset — chỗ dễ sai nhất
---------------------------
Các env khai báo ``ImuCfg.OffsetCfg(rot=(0.0, 0.0, 0.0, 1.0))``:

* Bản 2.x đọc theo ``(w, x, y, z)`` → ``w=0, z=1`` → xoay 180° quanh trục Z,
  tức ĐỔI DẤU roll và pitch trong observation.
* Bản 3.0 đọc cùng bộ số đó theo ``(x, y, z, w)`` → identity, không xoay gì.

Nhiều khả năng tác giả chỉ định viết identity và bộ số này là nhầm lẫn thứ tự,
nhưng policy đã train với dấu bị đảo — muốn checkpoint cũ chạy đúng thì phải
giữ nguyên hành vi cũ. Vì vậy mặc định ``LEGACY_WXYZ_OFFSET = True``.

Đặt ``LEGACY_WXYZ_OFFSET = False`` nếu train lại từ đầu và muốn dùng ngữ nghĩa
đúng của 3.0. Khi đó các checkpoint train trước đây sẽ KHÔNG dùng lại được.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_mul

LEGACY_WXYZ_OFFSET = True

# (id(robot), tên thân) -> chỉ số thân. Tra tên thân qua regex khá tốn, mà giá
# trị không đổi trong suốt vòng đời env nên cache lại.
_BODY_IDX_CACHE: dict[tuple[int, str], int] = {}


def as_torch(value):
    """Trả về ``torch.Tensor`` thật từ một giá trị đọc ở ``robot.data``.

    IsaacLab 3.0 bọc dữ liệu trong ``isaaclab.utils.warp.ProxyArray``. Lớp này
    có bắc cầu sang torch, nhưng chỉ ở mức "deprecation bridge": nó KHÔNG lọt
    qua ``@torch.jit.script`` (kernel jit đòi đúng kiểu ``Tensor``), và mọi
    method không phải toán tử — ``clone()``, ``view()``, ``unsqueeze()``… — đi
    qua ``__getattr__`` nên chạy được nhưng ném ``DeprecationWarning``.

    Thuộc tính ``.torch`` là view zero-copy (``wp.to_torch``) và được cache sẵn
    bên trong ProxyArray, nên gọi hàm này mỗi bước sim không tốn thêm gì.

    Với ``torch.Tensor`` sẵn có thì trả lại nguyên vẹn, nhờ đó code vẫn chạy
    bình thường trên IsaacLab 2.x.
    """
    if isinstance(value, torch.Tensor):
        return value
    inner = getattr(value, "torch", None)
    if isinstance(inner, torch.Tensor):
        return inner
    return torch.as_tensor(value)


def _offset_quat_xyzw(imu_cfg, device, dtype) -> torch.Tensor:
    """Đổi ``imu_cfg.offset.rot`` sang quaternion ``(x, y, z, w)``."""
    rot = tuple(float(v) for v in imu_cfg.offset.rot)
    if LEGACY_WXYZ_OFFSET:
        w, x, y, z = rot
        rot = (x, y, z, w)
    return torch.tensor(rot, device=device, dtype=dtype)


def imu_quat_w(robot, imu_cfg) -> torch.Tensor:
    """Quaternion ``(x, y, z, w)`` của khung IMU trong hệ world.

    Thay cho ``sensor.data.quat_w`` đã bị bỏ ở IsaacLab 3.0.

    Args:
        robot: Articulation mang IMU (thường là ``self.robot``).
        imu_cfg: Cấu hình IMU (thường là ``self.cfg.imu``). Tên thân gắn IMU
            lấy từ đoạn cuối của ``prim_path``.

    Returns:
        Tensor shape ``(num_envs, 4)`` theo thứ tự ``(x, y, z, w)``.
    """
    body_name = imu_cfg.prim_path.rstrip("/").rsplit("/", 1)[-1]
    key = (id(robot), body_name)
    idx = _BODY_IDX_CACHE.get(key)
    if idx is None:
        matches = robot.find_bodies(body_name)[0]
        if not matches:
            raise ValueError(
                f"Không tìm thấy thân '{body_name}' (suy ra từ imu prim_path "
                f"'{imu_cfg.prim_path}') trong robot. Các thân có sẵn: {robot.body_names}"
            )
        idx = matches[0]
        _BODY_IDX_CACHE[key] = idx

    body_quat = as_torch(robot.data.body_link_quat_w)[:, idx]
    offset = _offset_quat_xyzw(imu_cfg, body_quat.device, body_quat.dtype)
    return quat_mul(body_quat, offset.repeat(body_quat.shape[0], 1))
