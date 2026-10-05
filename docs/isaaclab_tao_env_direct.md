# Hướng dẫn tạo môi trường IsaacLab Độc lập (Direct)

Tài liệu này hướng dẫn cách tạo một package môi trường hoàn toàn mới, kế thừa từ môi trường 10DOF cũ nhưng tách biệt để có thể thoải mái tinh chỉnh vật lý (stiffness, mass, v.v.) mà không làm hỏng mô hình đã train.

## Cấu trúc thư mục mục tiêu
Ví dụ ta tạo một package tên là `bipedal_vinh` nằm trong thư mục `source/`:
```text
source/
└── bipedal_vinh/
    ├── __init__.py               <-- (Cửa ngõ)
    └── tasks/
        ├── __init__.py           <-- (Giấy khai sinh - Đăng ký môi trường)
        ├── vinh10dof_config.py   <-- (Bản vẽ cấu hình vật lý)
        ├── vinh10dof_env.py      <-- (Logic môi trường RL)
        └── _asset_paths.py       <-- (Đường dẫn tới file USD)
```

---

## Các bước thực hiện thủ công (Ví dụ trên VS Code)

### Bước 1: Tạo cây thư mục
1. Trong cột bên trái của VS Code, mở thư mục `source/`.
2. Tạo New Folder tên là `bipedal_vinh`.
3. Bên trong `bipedal_vinh`, tạo tiếp New Folder tên là `tasks`.

### Bước 2: Copy "nguyên liệu" từ môi trường cũ
1. Tìm đến thư mục chứa môi trường gốc (vd: `source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/`).
2. Copy 3 file cốt lõi:
   * `transformer_config_10dof.py`
   * `transformer_walk10dof_env.py`
   * `_asset_paths.py`
3. Paste 3 file đó vào thư mục `source/bipedal_vinh/tasks/` vừa tạo.
4. Đổi tên 2 file đầu thành `vinh10dof_config.py` và `vinh10dof_env.py` cho dễ phân biệt.

### Bước 3: Móc nối các file trong nhà mới
Mở file `vinh10dof_env.py` lên. Tìm dòng import file config ở phần đầu:
```python
from .transformer_config_10dof import TRANSFORMER_10DOF_CFG
```
Sửa nó thành tên file mới của bạn:
```python
from .vinh10dof_config import TRANSFORMER_10DOF_CFG
```
Lưu file lại.

### Bước 4: Viết "Giấy khai sinh" (`tasks/__init__.py`)
Tạo một file mới tên là `__init__.py` nằm trong thư mục `source/bipedal_vinh/tasks/`.
Dán đoạn code đăng ký (register) này vào:
```python
import gymnasium as gym

gym.register(
    id="VinhRobot-10DOF-v0",  # Tên môi trường độc lập của bạn
    entry_point="bipedal_vinh.tasks.vinh10dof_env:TransformerWalk10DOFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bipedal_vinh.tasks.vinh10dof_env:TransformerWalk10DOFEnvCfg",
        # Phần thuật toán RL: Kế thừa từ thư mục cũ
        "rsl_rl_cfg_entry_point": "transformer_nam.tasks.direct.transformer_nam.agents.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)
```

### Bước 5: Tạo "Cửa ngõ" cho package (`bipedal_vinh/__init__.py`)
Tạo một file mới tên là `__init__.py` nằm ở thư mục ngoài cùng `source/bipedal_vinh/` (không nằm trong tasks).
Dán đúng 1 dòng này vào:
```python
from . import tasks
```

### Bước 6: Khai báo với hệ thống trung tâm
Mở file "khai sinh gốc" của project (nơi IsaacLab luôn đọc lúc khởi động), ví dụ:
`source/transformer_nam/transformer_nam/tasks/__init__.py`
Thêm dòng này xuống dưới cùng:
```python
import bipedal_vinh
```

🎉 **Hoàn tất!** Bây giờ bạn có thể sửa `vinh10dof_config.py` thoải mái và gọi môi trường mới bằng lệnh:
`./run.sh scripts/rsl_rl/train.py --task VinhRobot-10DOF-v0`

---
---

# SỰ THẬT VỀ 2 FILE `__init__.py` (Hiệu ứng Domino)

Tại sao lại cần đến 2 file `__init__.py` ở Bước 4 và Bước 5? Bản chất của chúng là gì?

Hãy tưởng tượng hệ thống **IsaacLab** là một **Cục quản lý dân cư**, còn thư mục **`bipedal_vinh`** của bạn là một **ngôi nhà mới xây**.
Đặc tính của Python là: **File `__init__.py` giống như một người Quản gia (Lễ tân) của thư mục đó.** Cứ mỗi khi có ai "gõ cửa" thư mục (bằng lệnh `import`), người Quản gia này sẽ tự động thức dậy và chạy tất cả code được viết bên trong nó.

### 1. File `tasks/__init__.py` (Tờ đơn đăng ký)
**Code bên trong:** Gọi hàm `gym.register(...)`
* **Mục đích:** Cục quản lý (IsaacLab/Gymnasium) không tự động lục lọi ổ cứng tìm code của bạn. Hàm `gym.register` chính là tờ đơn báo cáo: *"Tôi có môi trường tên là `VinhRobot-10DOF-v0`. Ai gọi nó thì lấy file `vinh10dof_env.py` ra chạy."*
* **Tại sao phải nằm trong `__init__.py`?** Nếu viết hàm đăng ký vào một file bình thường, nó sẽ nằm im vĩnh viễn. Việc để nó vào `__init__.py` giúp hàm này **tự động được kích hoạt** ngay khi thư mục `tasks` bị hệ thống chạm tới.

### 2. File `bipedal_vinh/__init__.py` (Người dẫn đường)
**Code bên trong:** `from . import tasks`
* **Mục đích chống lại "sự lười biếng" của Python:** Ở Bước 6, khi hệ thống chạy dòng `import bipedal_vinh`, Python chỉ chạy đến gõ cửa ngôi nhà `bipedal_vinh` và gặp người Quản gia ở cổng (file `__init__.py` ngoài cùng). 
* Nếu file này trống rỗng, Python sẽ cho rằng *"Không có dặn dò gì cả"* và bỏ đi. Nó **KHÔNG BAO GIỜ** tự mò vào các phòng ban bên trong (thư mục `tasks/`). 
* Nếu điều đó xảy ra, file `tasks/__init__.py` sẽ vĩnh viễn không được gọi -> Môi trường không được đăng ký -> IsaacLab báo lỗi *"Không tìm thấy môi trường"*.
* Dòng chữ `from . import tasks` chính là lời dặn của Quản gia: *"Xin mời đi tiếp vào căn phòng `tasks`"*.

### 🚀 Phản ứng dây chuyền (Domino Effect)
Nhờ thiết kế này, một phản ứng hoàn hảo xảy ra mỗi khi khởi động IsaacLab:
1. File hệ thống gốc chạy lệnh `import bipedal_vinh` (Bước 6).
2. Python gõ cửa thư mục ngoài cùng, gặp `__init__.py` (Bước 5).
3. File này ra lệnh `from . import tasks`, ép Python đi sâu vào thư mục con.
4. Python gõ cửa thư mục `tasks`, gặp `__init__.py` thứ hai (Bước 4).
5. File này lập tức kích hoạt `gym.register(...)`, chính thức ghi tên môi trường của bạn vào danh bạ của hệ thống!
