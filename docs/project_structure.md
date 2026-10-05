# Cấu trúc dự án

Dự án Học tăng cường (Reinforcement Learning) cho robot hai chân. Mô phỏng
chạy trên **Isaac Sim + IsaacLab 3.0**, thuật toán dùng **rsl_rl** (PPO).
Đích cuối là xuất trọng số sang vi điều khiển trên robot thật.

> Cập nhật 2026-10-05 sau khi dọn lại cấu trúc. Bản trước mô tả bố cục cũ
> (thư mục `source/`, `urdf/`, `FullForm/`... ở gốc) — đã không còn đúng.

---

## 1. Tổng quan

```text
Transform_bipedal_todai/
├── isaac_rl/        Mô phỏng + train RL (Isaac Sim). Phần chính.
├── mjc_rl/          Nhánh riêng: MuJoCo + Stable-Baselines3, bài toán twist recovery.
├── assets/          Mô hình robot dùng chung — mỗi robot một thư mục.
├── inspect/         Công cụ xem/chỉnh robot trong GUI Isaac Sim, không train.
├── sim_handoff /    CAD gốc do bên cơ khí bàn giao. KHÔNG sửa.   (tên có dấu cách cuối)
├── docs/            Tài liệu.
├── AGENTS.md        Quy trình làm việc đa agent (Claude / codex).
├── PLAN.md          Kế hoạch công việc chính.
├── FIX_AFTER_DIFF.md  Tổng hợp những chỗ codex làm sai, theo thứ tự.
└── 0319.gif         Robot thật đi bộ (policy model_349).
```

**`assets/` phục vụ ba client độc lập** — `isaac_rl`, `mjc_rl`, `inspect` — cả
ba đều dò ngược cây thư mục để tìm nó, nên chạy từ đâu cũng được:

```text
assets/<robot>/
├── usd/       cho Isaac Sim
├── mjcf/      cho MuJoCo
├── meshes/    STL
└── meta/      calibration, báo cáo quán tính, hash          (riêng officialdesign)
```

Robot hiện có: `officialdesign` (robot mới, đang train), `newsimple`
(robot của `model_349`), `fulltrans`, `simpletrans`, `trans3dof`, `_archive`.

---

## 2. `isaac_rl/`

```text
isaac_rl/
├── bipedal/              package Python — `import bipedal`
│   ├── _shared/            dùng chung MỌI robot, cố tình để mỏng
│   │   ├── lab3.py           lớp tương thích IsaacLab 2.x → 3.0
│   │   ├── paths.py          dò tìm thư mục assets/
│   │   └── ppo.py            cấu hình PPO nền
│   ├── officialdesign/     robot mới
│   │   ├── robot.py          ArticulationCfg, đọc meta/calibration.json
│   │   ├── task_walk.py      env đi bộ: obs / action / reward / termination
│   │   ├── ppo.py            siêu tham số
│   │   └── __init__.py       gym.register("Official-Walk-v0")
│   └── newsimple/          robot của model_349, cùng bố cục
├── scripts/              script chạy trực tiếp
├── _archive/             code đông lạnh — KHÔNG chạy được, chỉ để đọc
├── logs/                 kết quả train (gitignore)
├── trajectory_exports/   quỹ đạo JSON xuất cho robot thật
├── run.sh                bật conda env + biến môi trường rồi gọi python
├── play.sh / play.md     phát lại checkpoint; play.md là bảng tra
├── setup.py              cài package (pip install -e isaac_rl)
└── pyproject.toml        config pytest / ruff / pyright
```

### Vì sao chia theo robot

Robot là **đường biên cứng** của dự án. Đổi robot thì checkpoint, reward
scale, pose, giới hạn góc và thứ tự khớp đều phải làm lại từ đầu — kể cả khi
shape obs/action trùng nhau. Đổi task trên cùng một robot thì giữ gần hết.

Thứ gì thay đổi cùng nhau thì ở cùng nhau → robot nằm ngoài, task nằm trong.

IsaacLab gốc chia ngược lại (task-first, robot nằm ở `isaaclab_assets`) vì nó
là **thư viện benchmark**: task là hằng số, robot là biến. Dự án này ngược —
robot là hằng số của cả một giai đoạn (CAD chốt rồi, hàn rồi), task là biến.

`_shared/` phải mỏng. Trước khi thêm gì vào đó, hỏi: *"thứ này có đúng với
MỌI robot không?"* Lưỡng lự thì chép vào từng robot — trùng lặp vài chục dòng
rẻ hơn nhiều so với một `_shared` phình to rồi sửa cho robot này làm gãy robot kia.

### Đăng ký task

Import **tường minh** trong `bipedal/__init__.py`, không dùng
`isaaclab_tasks.utils.import_packages`. Bộ quét đó lọc blacklist theo *chuỗi
con*, nên một thư mục tên chứa `"utils"` bị bỏ qua **im lặng**; thiếu một
`__init__.py` ở tầng trung gian cũng cho kết quả y hệt — import thành công
nhưng `gym.make` báo `NameNotFound`, một thông báo chỉ sai hướng.

Thêm robot mới = tạo thư mục + thêm một dòng `from . import <tên>`.

| Task | obs / act | Asset | Tình trạng |
|---|---:|---|---|
| `Official-Walk-v0` | 60 / 10 | `assets/officialdesign` | Robot mới, mới có smoke test |
| `NewSimple-Walk-v0` | 44 / 6 | `assets/newsimple` | `model_349` — đã đi được trên robot thật |

---

## 3. `isaac_rl/scripts/`

### Train / play
| File | Việc |
|---|---|
| `rsl_rl/train.py` | Khởi động huấn luyện PPO |
| `rsl_rl/play.py` | Chạy thử một policy đã train (`.pt`) |
| `rsl_rl/cli_args.py` | Tham số dòng lệnh. **Phải cùng thư mục với `train.py`** — `train.py` `import cli_args` trần |
| `rsl_rl/export_trajectory.py` | Xuất quỹ đạo JSON cho robot thật |

### Pipeline asset OFFICIALdesign
Chạy theo thứ tự khi CAD hoặc `calibration.json` thay đổi:

| File | Việc |
|---|---|
| `prepare_officialdesign.py` | CAD handoff → URDF (mass, CoM mm→m, tensor quán tính) |
| `convert_officialdesign.py` | URDF → USD cho PhysX, dựng lại principal axes |
| `test_officialdesign_asset.py` | 4 test offline: đơn vị CAD, dấu khớp, hình học |
| `validate_officialdesign.py` | 9 check trên GPU: mass, mapping khớp, giữ pose, contact, thả rơi |

> `bipedal/officialdesign/robot.py` kiểm **SHA-256** của URDF/USD/calibration
> lúc load. Sửa `calibration.json` mà không chạy lại `prepare` + `convert` thì
> task ném lỗi ngay — cơ chế này chặn việc train với config lệch USD.

### Tiện ích
`list_envs.py` (liệt kê task đã đăng ký) · `random_agent.py` / `zero_agent.py`
(chạy robot với action ngẫu nhiên hoặc bằng 0, test vật lý) ·
`check_joint_order.py` · `debug_robot_position.py` · `inspect_usd_structure.py` ·
`convert_checkpoint_rslrl5.py` (chuyển checkpoint rsl_rl < 4.0 sang định dạng mới) ·
`test_initial_env.py` · `play_transform_getup.py`

---

## 4. `logs/`

```text
logs/
├── old/              96 run cũ lẫn lộn nhiều task (có model_349)
├── newsimple/        run mới
└── officialdesign/   run mới
```

Thư mục con = `agent_cfg.experiment_name`, đặt ở `bipedal/<robot>/ppo.py`.

`logs/old/` không tách ra được: 96 run đó thuộc nhiều task khác nhau từ thời
cả nhóm cùng dùng chung một `experiment_name`, mà tên run không cho biết task
nào. Phải mở `<run>/params/env.yaml` xem `num_actions` mới chắc.

`play.sh` tìm trong mọi thư mục của một robot nên bạn không cần biết checkpoint
nằm ở đâu:

```bash
./play.sh                                          # bảng robot
./play.sh newsimple 2026-03-19_13-18-11_work 349   # phát lại 0319.gif
```

---

## 5. `isaac_rl/_archive/`

Code đông lạnh: bản cài đặt cũ còn giá trị đọc lại, không còn ai chạy.

| | |
|---|---|
| `fulltrans/` | 3 task của robot Fulltrans. Không policy nào biết đi (tốt nhất ep_len 83.1/200) |
| `newsimple/` | Dòng tiến hoá dẫn tới `model_349` + hai nhánh rẽ của Hiếu |
| `vinh/` | Fork của Vinh + `set_pose.py` |

**Code ở đó KHÔNG chạy được** — relative import gãy khi ra khỏi package. Có
chủ đích: nó là văn bản tham khảo. Xem `_archive/README.md`.

---

## 6. Tài liệu

| File | Nội dung |
|---|---|
| `docs/ALGO.md` | Thuật toán, reward, gait, sim-to-real cho robot mới |
| `docs/isaaclab.md` | Cài đặt, vận hành IsaacLab, bảng tra lỗi |
| `docs/isaaclab_tao_env_direct.md` | Quy trình dựng một env Direct mới |
| `docs/git.md` | Thao tác git của dự án |
| `assets/officialdesign/README.md` | Mapping khớp, thông số servo, lệnh train/play |
| `isaac_rl/play.md` | Bảng tra `play.sh` |

---

## 7. Hai nhánh mô phỏng

| | `isaac_rl/` | `mjc_rl/` |
|---|---|---|
| Engine | Isaac Sim + PhysX (GPU) | MuJoCo (CPU) |
| Thư viện RL | rsl_rl | Stable-Baselines3 |
| Bài toán | Đi bộ | Twist recovery (lật dậy) |
| Asset | `assets/<robot>/usd/` | `assets/<robot>/mjcf/` |

Hai nhánh độc lập, chỉ dùng chung thư mục `assets/`.
