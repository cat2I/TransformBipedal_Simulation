# Tạo môi trường IsaacLab Direct mới

> **Lưu ý 2026-10-05.** Tài liệu gốc của Vinh (từ mục "Bản gốc" trở xuống) mô
> tả bố cục `source/<package>/tasks/` — bố cục đó **đã bỏ**. Phần giải thích
> *Hiệu ứng Domino* về `__init__.py` ở cuối vẫn đúng nguyên lý và rất đáng đọc.
> Mục ngay dưới đây là quy trình áp dụng cho bố cục hiện tại.

---

## Quy trình hiện tại: thêm một ROBOT mới

Trong bố cục này **mỗi robot một thư mục**, task nằm bên trong robot. Thêm
robot mới là thêm một thư mục dưới `isaac_rl/bipedal/`:

```text
isaac_rl/bipedal/
├── __init__.py          <-- thêm 1 dòng: from . import <robot_moi>
├── _shared/             <-- dùng chung, ĐỪNG phình to
└── <robot_moi>/
    ├── __init__.py        gym.register("<Robot>-<Task>-v0")
    ├── robot.py           ArticulationCfg: USD, actuator, pose, giới hạn khớp
    ├── task_walk.py       env: obs / action / reward / termination / reset
    └── ppo.py             siêu tham số; experiment_name = "<robot_moi>"
```

### Các bước

1. **Đặt asset** vào `assets/<robot_moi>/usd/`. Không để trong `isaac_rl/` —
   `mjc_rl/` và `inspect/` cũng đọc chung thư mục `assets/`.

2. **`robot.py`** — chép từ `bipedal/officialdesign/robot.py` làm mẫu. Nó đọc
   thông số từ `meta/calibration.json` thay vì viết cứng trong Python, và kiểm
   SHA-256 để không train nhầm với config lệch USD.

3. **`task_walk.py`** — chép từ robot gần nhất. Sửa đường dẫn import:
   `from .._shared.lab3 import as_torch` và `from .robot import <CFG>`.

4. **`ppo.py`** — kế thừa `.._shared.ppo` và **bắt buộc** đặt
   `experiment_name = "<robot_moi>"`. Quên là log rơi vào
   `logs/_chua_dat_ten/`, dấu hiệu để phát hiện.

5. **`<robot_moi>/__init__.py`** — `gym.register`, `entry_point` dùng
   `f"{__name__}.task_walk:<Class>"` để tự khớp khi đổi chỗ thư mục.

6. **`bipedal/__init__.py`** — thêm `from . import <robot_moi>`.
   **Bước này KHÔNG được quên** — xem phần Hiệu ứng Domino ở cuối tài liệu.

7. **`play.sh`** — thêm một dòng vào mỗi mảng `ORDER`, `TASK`, `LOGDIRS`,
   `ASSET`, `NOTE` (và `EXTRA` nếu cần cờ env riêng). Cập nhật `play.md`.

8. **Nghiệm thu:**
   ```bash
   ./run.sh scripts/list_envs.py                                   # task có trong bảng?
   ./run.sh scripts/rsl_rl/train.py --task <Robot>-<Task>-v0 \
            --num_envs 256 --headless --max_iterations 2           # exit 0?
   ```

### Khác với tài liệu gốc

| Bản gốc (Vinh) | Hiện tại |
|---|---|
| Package riêng trong `source/` | Một thư mục con trong `bipedal/` |
| `tasks/__init__.py` đăng ký | `<robot>/__init__.py` đăng ký |
| `import_packages` tự quét | `bipedal/__init__.py` import **tường minh** |
| Chép `_asset_paths.py`, `_lab3_compat.py` vào package mới | Dùng chung `bipedal/_shared/` |

Vì sao bỏ bộ quét tự động: nó lọc blacklist theo **chuỗi con**
(`any(b in name for b in ["utils", ".mdp"])`), nên thư mục tên chứa `"utils"`
bị bỏ qua **im lặng** — task biến mất mà không có lỗi nào.

---
# Bản gốc — Vinh, trước 2026-10-05

## Hướng dẫn tạo môi trường IsaacLab Độc lập (Direct)

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
