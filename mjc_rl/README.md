# mjc_rl — twist recovery bằng MuJoCo + Stable-Baselines3

Bài toán: robot **sau khi đã đứng dậy** thì thân đang vặn ngang (khớp `Twist` ≈ 1.51 rad ≈ 86°).
Cần xoay về hướng thẳng — bằng cách **dậm chân tại chỗ** để triệt tiêu ma sát tĩnh, thay vì
vặn trực tiếp dưới đất (dễ hỏng motor, trượt ngã).

Đây là bước **sau** standup, và tách khỏi `isaac_rl/` vì dùng simulator + framework khác:

| | `isaac_rl/` | `mjc_rl/` (thư mục này) |
|---|---|---|
| Simulator | Isaac Sim | MuJoCo |
| Framework RL | rsl_rl (GPU, 4096 env song song) | Stable-Baselines3 (CPU, 1 env) |
| Bài toán | walk, standup | twist recovery |

## ⚠️ Môi trường: venv `~/mujoco_env`, KHÔNG phải conda

Đây là chỗ dễ vấp nhất, vì **hai file cần env khác nhau**:

| Chạy gì | Cần gói | Env nào chạy được |
|---|---|---|
| `mujoco_view.py` | chỉ `mujoco` | **cả hai** — venv `~/mujoco_env` hoặc conda `isaacsim` |
| `trainppo.py`, `play.py` | `mujoco` + `stable_baselines3` | **chỉ** venv `~/mujoco_env` |

`stable_baselines3` không có trong env conda nào (`base` và `isaacsim` đều thiếu).

```bash
conda deactivate                 # thoát conda trước, nếu đang trong env nào đó
source ~/mujoco_env/bin/activate
```

> **venv** và **conda env** là hai hệ quản lý môi trường Python khác nhau, cài chồng lên nhau
> được nhưng dễ lẫn. Nếu đang đứng trong conda env mà `source` venv, biến `PATH` có thể trộn
> hai bên → import ra nhầm phiên bản. Nên `conda deactivate` trước.

Đã cài trong `~/mujoco_env` (Python 3.12.3):

| Gói | Phiên bản |
|---|---|
| `stable_baselines3` | 2.7.1 |
| `mujoco` | 3.7.0 |
| `gymnasium` | 1.2.3 |
| `torch` | 2.10.0+cu128 |

⚠️ **Lệch phiên bản `mujoco`**: venv có 3.7.0, conda `isaacsim` có 3.8.0. Cùng một model mà
mở ở hai env có thể ra kết quả vật lý hơi khác. Khi so số đo, nhớ ghi rõ chạy ở env nào.

## Bố cục

```
mjc_rl/
├── sb3/                      code
│   ├── twist_Env.py          môi trường Gymnasium: quan sát, hành động, hàm thưởng
│   ├── trainppo.py           train
│   ├── play.py               chạy thử policy đã train, mở cửa sổ 3D
│   ├── mujoco_view.py        xem model URDF, dựng tư thế bằng slider
│   └── getup_poses.json      6 keyframe chuỗi đứng dậy (mujoco_view đọc/ghi)
├── model/                    kết quả train
│   ├── ppo_twist_model.zip
│   └── ppo_twist_tensorboard/
└── docs/
    ├── note.md                     hướng dẫn dựng tư thế bằng mujoco_view
    └── RL_Twist_Recovery_Plan.md   thiết kế môi trường, hàm thưởng, lý do chọn SB3
```

## Chạy

```bash
cd mjc_rl/sb3

python trainppo.py    # train, ghi vào ../model/
python play.py        # nạp ../model/ppo_twist_model.zip, mở cửa sổ 3D

python mujoco_view.py --list                                   # liệt kê model
python mujoco_view.py Fulltrans_meshfixed --fixed               # treo thân, kéo slider
python mujoco_view.py Fulltrans_meshfixed --play getup_poses.json --seg 1.5
```

Đường dẫn tới trọng số và tới file MJCF đều tính từ vị trí file `.py`, nên **chạy từ thư mục
nào cũng đúng** — không bắt buộc phải `cd sb3` trước.

## Ghi chú

- MJCF của robot nằm ở `assets/fulltrans/mjcf/Fulltrans_RL.xml`, không nằm trong thư mục này.
  `twist_Env.py` tự dò ngược lên tìm `assets/`.
- SB3 cảnh báo khi train: PPO với `MlpPolicy` chạy **CPU nhanh hơn GPU** (mạng quá nhỏ, chi phí
  chuyển dữ liệu qua lại GPU lớn hơn phần tính toán tiết kiệm được). Muốn tắt cảnh báo và chạy
  đúng chỗ, thêm `device="cpu"` vào `PPO(...)` trong `trainppo.py`.
