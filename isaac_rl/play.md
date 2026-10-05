# `play.sh` — phát lại checkpoint

Bảng tra nhanh cho [`play.sh`](play.sh). Thay cho `play6.sh` / `play10.sh` cũ
(hai file đó khoá cứng vào log dir của robot cũ nên thêm robot là phải viết thêm script).

## Cú pháp

```
./play.sh                              # bảng robot
./play.sh <robot>                      # danh sách run + iter có sẵn của robot đó
./play.sh <robot> <run>                # phát checkpoint MỚI NHẤT của run
./play.sh <robot> <run> <iter>         # phát đúng iteration đó
./play.sh <robot> <run> <iter> --...   # cờ thừa đẩy thẳng sang play.py
```

## Bảng robot

| `<robot>` | Gym task | Log dir | Asset | obs/act | Tình trạng |
|---|---|---|---|---:|---|
| `official` | `Official-Walk-v0` | `logs/officialdesign/` | `assets/officialdesign` | 60 / 10 | Robot mới. Mới chỉ smoke test 2 iteration |
| `newsimple` | `NewSimple-Walk-v0` | `logs/newsimple/` + `logs/old/` | `assets/newsimple` | 44 / 6 | **`model_349` — policy DUY NHẤT đã đi được trên robot thật** (`0319.gif`) |

> **Ba task `fulltrans` đã đóng băng** vào `_archive/fulltrans/` ngày 2026-10-05
> (`Walk10DOF`, `Walk10DOF6`, `StandUp`). Không policy nào biết đi, và chúng dùng
> hai giao diện khác nhau (60/10 và 44/6) trên cùng một asset.

## Ví dụ

```bash
# Phát lại policy đứng sau 0319.gif
./play.sh newsimple 2026-03-19_13-18-11_work 349

# Robot mới, 4 env cùng lúc
./play.sh official 2026-10-01_01-27-28_handoff_smoke --num_envs 4

# Quên tên run? Bỏ trống, nó liệt kê cho
./play.sh official
```

## Hai thứ script tự lo

**1. Chọn đúng định dạng checkpoint.** `rsl_rl` < 4.0 lưu *một* module `ActorCritic`
gộp; bản ≥ 4.0 đòi `actor` / `critic` tách rời. `scripts/convert_checkpoint_rslrl5.py`
sinh ra bản `model_N_rslrl5.pt`. Script thử bản `_rslrl5` trước, không có thì lấy
`model_N.pt`.

| Robot | File có sẵn |
|---|---|
| Robot cũ (train bằng rsl_rl cũ) | `model_N.pt` **và** `model_N_rslrl5.pt` → dùng bản `_rslrl5` |
| `official` (train thẳng bằng rsl-rl mới) | chỉ `model_N.pt` |

**2. Cờ env của `newsimple` được nhúng sẵn:**

```
env.domain_rand=False
env.imu_fixed_bias=[0.0,0.0,0.0]
env.imu_noise_std.orientation=0.015
env.imu_noise_std.angular_velocity=0.01
env.imu_drift_rate=0.0
```

Không có bộ này, policy cũ sẽ chạy trong môi trường nhiễu/bias khác lúc nó học,
và trông như "policy hỏng" dù thực ra chỉ sai điều kiện phát lại.

## ⚠️ Log dir dùng chung

`logs/old/` chứa **96 run của nhiều task khác nhau** — gồm cả ba task
`fulltrans` đã đóng băng — vì cả nhóm từng kế thừa
`TransformerWalkPPORunnerCfg` mà không override `experiment_name`.

**Tên run không cho biết nó thuộc task nào.** Phải mở `<run>/params/env.yaml`
xem `num_actions` mới chắc. Vì không phân biệt được nên không tách ra được —
cứ để nguyên một đống.

Run MỚI thì sạch: `logs/<robot>/`. `play.sh` tìm trong cả hai nên bạn không
cần biết checkpoint nằm ở đâu:

```
logs/
├── old/              96 run cũ lẫn lộn (có model_349)
├── newsimple/        run mới của newsimple
└── officialdesign/   run mới của robot mới
```

Tầng `rsl_rl/` ở giữa đã bỏ — đó là quy ước IsaacLab (`logs/<thư viện RL>/`)
để chứa nhiều thư viện song song, mà project chỉ dùng `rsl_rl`.

## Thêm robot mới

Sửa 5 mảng ở đầu [`play.sh`](play.sh) (dòng 26–68), mỗi mảng thêm một dòng:

| Mảng | Nội dung |
|---|---|
| `ORDER` | thêm tên vào danh sách (quyết định thứ tự hiển thị) |
| `TASK` | gym task id |
| `EXPERIMENT` | tên thư mục log = `agent_cfg.experiment_name` |
| `ASSET` | mô tả asset + obs/act, chỉ để hiển thị |
| `NOTE` | ghi chú tình trạng, chỉ để hiển thị |
| `EXTRA` | *(tuỳ chọn)* cờ hydra ép env về đúng điều kiện lúc train |

Rồi cập nhật bảng trong file này.
