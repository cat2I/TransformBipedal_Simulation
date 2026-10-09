# Cách chạy OFFICIALdesign (Official-Walk-v0)

Mọi lệnh chạy **trong thư mục `isaac_rl/`**. `run.sh` tự bật conda env `isaaclab30`.

```bash
cd ~/Desktop/TransformBipedal_Simulation/isaac_rl
```

## 1. Smoke test (kiểm code chạy được, ~10 giây)
```bash
./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 --num_envs 256 --headless --max_iterations 2
```
Phải thấy: `62 observations`, `Linear(in_features=62, ...)`, 3 dòng `Gait/...`.

## 2. Train
```bash
./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 --num_envs 256 --headless --max_iterations 500
```
- 500 iteration × 256 env ≈ 15 phút.
- Log lưu ở `logs/officialdesign_clock/<ngày_giờ>/` (tên thư mục = `experiment_name` trong `bipedal/officialdesign/ppo.py`).
- Đổi số cho riêng 1 run, không sửa code (chỉ được với trường có trong cfg):
  ```bash
  ... --max_iterations 500 env.target_velocity=0.09 env.w_clock_frc=1.0
  ```
- Số gõ cứng trong code (vd trọng số `swing`) **không** override được → sửa file, ghi comment giá trị cũ.

## 3. Xem đồ thị (TensorBoard)
Mở terminal thứ hai:
```bash
tensorboard --logdir logs/officialdesign_clock
```
Mở `http://localhost:6006`. Xem:
- Nhóm **Gait**: `frc_score`, `vel_score` đi lên về 0; `swing` tăng.
- **Train**: `Mean episode length` (tối đa 200 = 10 s), `Mean reward`.

## 4. Play (xem robot đi)
```bash
./run.sh scripts/rsl_rl/play.py --task Official-Walk-v0 --num_envs 1 --load_run <ngày_giờ>
```
- `<ngày_giờ>`0 = tên thư mục run, xem bằng `ls -t logs/officialdesign_clock` (mới nhất ở trên).
- Mặc định load checkpoint mới nhất của run đó.
- Cách đọc log play (in 2 bước/dòng, 1 chu kỳ 3.2 s ≈ 32 dòng):
  - `clock=[sin, cos]`: `sin > 0` → lượt chân phải vung; `sin < 0` → lượt chân trái.
  - `contact L=Y R=n`: `n` = chân đang bay.
  - ⚠️ L/R của play.py lấy theo vị trí sensor, chưa đối chiếu tên; và CAD "left" = chân phải thật (AC4.3).

## 5. Khi sửa `assets/officialdesign/meta/calibration.json`
`robot.py` kiểm SHA-256 → phải chạy lại trước khi train, không là crash lúc load:
```bash
./run.sh scripts/prepare_officialdesign.py
./run.sh scripts/convert_officialdesign.py --headless --force
```
Kiểm số mới đã ăn: bảng "Simulation Joint Information" in lúc train (cột Velocity/Effort Limits).
Thư mục nháp `assets/officialdesign/usd/OFFICIALdesign_1/` do convert sinh ra — không commit.

## Lịch sử run chẩn đoán
| Run | Thay đổi | Kết quả |
|---|---|---|
| `2026-10-09_15-16-11` | `swing` 0.15, `action_step_deg` 6.61 | Đứng/lê chân phải, chân trái gõ; có tiến lên |
| `2026-10-09_18-07-07` | `swing` 1.0 | (chưa ghi) |
