# `_archive/` — code đông lạnh

Thư mục này **không phải code đang bảo trì**. Nó là **văn bản tham khảo**:
những bản cài đặt cũ còn giá trị đọc lại, nhưng không còn ai chạy.

## ⚠️ Code ở đây KHÔNG chạy được

Mọi file đều dùng relative import (`from .transformer_config import ...`).
Chuyển ra khỏi package là dấu `.` mất nghĩa. Muốn chạy lại phải tự nối lại
import — và đó là việc có chủ đích, không phải sơ suất.

Ba lớp bảo đảm thư mục này không bao giờ gây lỗi im lặng:

| Cơ chế | Vì sao `_archive/` nằm ngoài |
|---|---|
| `_archive/` không có `__init__.py` ở gốc, và `bipedal/__init__.py` import tường minh từng robot | không nhánh nào dẫn tới nó |
| Không còn bộ quét tự động nào (đã bỏ `import_packages` ở pha B) | ngoài tầm |
| `pyproject.toml`: `ruff extend-exclude`, `pytest testpaths=["scripts"]` | đã loại trừ |

---

## `newsimple/` — dòng tiến hoá dẫn tới `model_349`

`model_349` là policy duy nhất đã đi được trên robot thật (`0319.gif`). Bản
đang sống của nhánh này là `bipedal/newsimple/task_walk.py` (6 act / 44 obs,
`NewSimple.usd`) — **vẫn nằm trong repo**, task `Transformer-Walk-Direct-v0`.
Dưới đây là các nấc trước nó và hai nhánh rẽ của Hiếu.

| File | Dòng | obs / act | Asset | Là gì |
|---|---:|---:|---|---|
| `transformer_nam_env_4dof.py` | 549 | 52 / 8 | NewSimple.usd | Nấc sớm nhất còn giữ. "4dof" = **4 DOF mỗi chân × 2**, không phải 4 khớp |
| `transformer_nam_env_3dof.py` | 549 | 44 / 6 | NewSimple.usd | Nấc kế. "3dof" = 3 DOF mỗi chân × 2. Cùng giao diện 44/6 với bản sống → so reward hai bên là thấy được thứ gì đã đổi |
| `transformer_config_3dof.py` | 79 | — | NewSimple.usd | Khai báo robot. **Mồ côi**: không env nào import nó (hai env trên đều dùng `transformer_config.py`) |
| `transformer_config_4dof.py` | 82 | — | **FullForm.usd** | Khai báo robot. Mồ côi. ⚠️ Trỏ asset **khác họ** với phần còn lại của thư mục này |
| `transform_hieu_env_work1.py` | 357 | 12 / 6 | NewSimple.usd | Nhánh rẽ của Hiếu, v1: *"Bub Closes, Hip/Knee/Foot Balance"*. Obs chỉ 12D — **không xếp chồng lịch sử**, thiết kế khác hẳn |
| `transformer_hieu_env.py` | 542 | 62 / 10 | *(gãy)* | Nhánh rẽ của Hiếu, v2: *"Twist + March"*. Obs 62D = 60 + 2 ô tiến độ twist |

### `transformer_hieu_env.py` — hỏng sẵn từ trước khi đóng băng

Nó `from .transformer_config import TRANSFORMER_10DOF_CFG`, nhưng trong
`transformer_config.py` dòng 280 là `# TRANSFORMER_10DOF_CFG = ArticulationCfg(`
— **đã bị comment**. Import là `ImportError` ngay. Giữ lại thuần tuý để đọc.

### Giá trị đã được trích ra

Ba ý tưởng trong `transformer_hieu_env.py` đã chép vào
[`docs/ALGO.md`](../../docs/ALGO.md) §2.2.1–2.2.3, không cần mở file này nữa:

- **`march_alt`** — thưởng khi đúng MỘT chân bay (+1.0), phạt hai chân chạm
  đất (−0.3) và hai chân bay (−1.0). Hiếu cho trọng số 2.0.
  Đi kèm là phát hiện: số hạng `swing` của `bipedal/officialdesign/task_walk.py` dùng `.mean(-1)`
  nên **nhảy được điểm gấp đôi bước đi**.
- **Đồng hồ chống đứng ì** — phạt theo *thời gian liên tục* hai chân cùng chạm
  đất, bắt được kiểu nhúc nhích tại chỗ cho có vận tốc tức thời.
- **Khoá reward theo điều kiện** — `where(standing, rew, 0)`, chống việc agent
  học cách ngã để ăn điểm dễ hơn.

---

## `vinh/` — nhánh của Vinh

Fork độc lập của `transformer_walk10dof_env.py`, cùng asset `Fulltrans10DOF.usd`,
đăng ký task riêng `VinhRobot-10DOF-v0`.

| File | Là gì |
|---|---|
| `vinh10dof_env.py` | 60 obs / 10 act. Diff với bản gốc là 226 dòng nhưng **chủ yếu là định dạng lại**; mọi giá trị thực chất (`frictions`, `torques`, `dampings`, `imu_bias_range`, buffer PhysX) giống bản gốc |
| `vinh10dof_config.py` | Khai báo robot — **giống hệt từng byte** `transformer_config_10dof.py` |
| `tasks_init.py` | Nguyên là `tasks/__init__.py`. **Đổi tên để `vinh/` không vô tình thành package Python** |
| `__init__.py` | Vỏ package cũ |
| `set_pose.py` | ⭐ Công cụ, xem dưới |

Hai file `_lab3_compat.py` và `_asset_paths.py` của nhánh này đã **xoá** —
chúng giống hệt từng byte bản đang sống trong `bipedal/_shared/`.

### `set_pose.py` — thứ duy nhất có thể hồi sinh rẻ

Dựng một thế giới vật lý tối giản (trọng lực + sàn + ma sát), thả robot vào,
ép 10 khớp về bộ góc viết cứng trong `POSE_DEG`. Hai chế độ: vật lý thật (xem
pose **có đứng vững không**) và `--freeze` (đóng băng để ngắm hình dáng).

Nó **không** làm được imitation learning hay thiết kế quỹ đạo — chỉ một pose
tĩnh, không trục thời gian, không keyframe, không nội suy, không xuất file.

Chỉnh sang OFFICIALdesign tốn ~10 phút: đổi dòng import config, và thay 10 key
trong `POSE_DEG` (robot mới không có `Twistleft_joint`/`Twistright_joint`, mà
là `Nhat_rotate_left`/`rotate_right`).

**Trước khi làm thế, kiểm tra xem đã đủ chưa:**
[`inspect/isaacsim_inspect.py`](../../inspect/isaacsim_inspect.py) mở GUI Isaac
Sim với thanh trượt Angular Drive → Target Position cho từng khớp, và **đã hỗ
trợ OFFICIALdesign sẵn**:

```bash
cd inspect && python isaacsim_inspect.py OFFICIALdesign
```

---

## Lấy lại lịch sử

Mọi file ở đây được chuyển bằng `git mv` nên giữ nguyên lịch sử:

```bash
git log --follow -- isaac_rl/_archive/<đường/dẫn/file>
```

Các file đã **xoá hẳn** (CartPole template, `.backup_.../`, bản trùng byte)
vẫn nằm trong git:

```bash
git log --all --diff-filter=D --name-only | grep <tên-file>
git show <hash>^:<đường/dẫn/cũ>
```
