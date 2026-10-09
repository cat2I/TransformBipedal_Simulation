# FIX_AFTER_DIFF.md — Sửa những gì agy làm sai/thiếu so với PLAN.md

Append-only. Mỗi lần review thêm một mục, không xoá mục cũ.

---

## Lần 1 — 2026-09-13 (review sau khi agy chạy `fix_env.py`)

Cách làm của agy: viết `fix_env.py` dùng `re.sub` để sửa file. `re.sub` **không báo lỗi khi pattern không khớp** —
nó im lặng bỏ qua. Vì vậy có mục trong PLAN.md bị bỏ sót mà agy không biết.

### F1.1 — A4 ma sát sàn CHƯA đổi (AC1.3 chưa pass)
Regex trong `fix_env.py` tìm `static_friction=2.0, dynamic_friction=2.5` trên **một dòng**,
nhưng trong ENV hai tham số nằm **hai dòng** (dòng ~234–235) → không khớp → giữ nguyên `2.0 / 2.5`.

Sửa bằng tay trong `_setup_scene`:
```python
ground_cfg = RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.4,
    restitution=0.05,
    friction_combine_mode="average",
)
```
- [x] AC-F1.1: `grep -n "friction=" ENV` → `static_friction=0.8` và `dynamic_friction=0.4`.

### F1.2 — Khối comment "SỬA 2026-09-10" chưa rút gọn (Việc 2, phần cuối)
PLAN yêu cầu rút còn ≤ 3 dòng. Hiện vẫn 13 dòng (từ `# SỬA 2026-09-10` tới `# Bản tháng 3 (commit 1752c9a)...`).
Thay toàn bộ khối đó bằng đúng 1 dòng đã có sẵn trong `fix_env.py`:
```python
# gaussian_noise() đã trả về data + nhiễu; không cộng thêm lần nữa.
```
- [x] AC-F1.2: `grep -c "SỬA 2026-09-10\|commit 1752c9a" ENV` → 0.

### F1.3 — Xoá `fix_env.py`
Là công cụ tạm của agy, không thuộc repo. Xoá file, không commit.
- [x] AC-F1.3: `git status --short` không còn `?? fix_env.py`.

### F1.4 — Chạy lại Việc 4 (AC4.1–4.3) sau khi sửa F1.1, dán output.
Lần trước nếu agy đã chạy thì kết quả đó dựa trên ma sát sai → phải chạy lại.
- [ ] AC-F1.4: `env.yaml` không chứa friction (nó không nằm trong cfg) → kiểm bằng `grep` ở F1.1 là đủ; train 2 iter không lỗi.

### Đã pass (không cần làm lại)
- [x] AC1.1 servo_max/min
- [x] AC1.2 weights + comment
- [x] AC1.4 feet_height 0.04
- [x] AC1.5 max_diff 10
- [x] AC1.6 termination 0.2
- [x] AC2.1 5 giá trị IMU
- [x] AC2.2 không còn `imu_legacy_double`
- [x] AC2.3 không còn cộng nhiễu hai lần
- [x] AC3.1 comment đầu file

---

## Lần 2 — 2026-09-13 (review `git diff --cached` sau khi agy sửa Lần 1)

Kết quả: F1.1, F1.2, F1.3 **pass**. Toàn bộ AC1.x, AC2.x, AC3.1 của PLAN.md **pass**.
Diff env đúng 9 hunk, không có thay đổi ngoài phạm vi. `fix_env.py` đã xoá.

### F2.1 — Việc 4 (AC4.1, AC4.2) chưa có bằng chứng
`logs/rsl_rl/transformer_walk/` không có run nào sau `2026-09-09_14-49-57`, và agy chưa dán output.
Chạy lại và dán **nguyên văn** 3 thứ: (a) dòng `Observation dim` / `Action dim`, (b) dòng cuối của iteration 2,
(c) `grep -n "domain_rand\|servo_max\|servo_min\|orientation\|angular_velocity" <run>/params/env.yaml`.
Xoá run thử sau khi dán.
- [ ] AC-F2.1: 3 output trên có mặt trong báo cáo của agy.

### F2.2 — Xoá `review.diff` ở gốc repo
File tạm agy tạo để đưa diff, không thuộc repo.
- [ ] AC-F2.2: `git status --short` không còn `?? review.diff`.

### Ghi chú (không phải lỗi agy)
- `transformer_config.py` có diff 146 dòng: đó là User comment khối FullForm111 từ trước PLAN, không phải agy sửa. AC4.3 chấp nhận.
- Dòng 182–183 `self.orient_noise` / `self.gyro_noise` giờ không còn được dùng (ENV đã lấy std từ cfg). Để nguyên, ngoài phạm vi.

---

## Lần 3 — 2026-10-08 (review commit `b6b2805`, B5a Việc 1 `gait_clock.py`)

Kết quả: logic LUT **đúng**. Bug dòng 72 của apex đã được tránh (`between_swings` grounded = `[inc, −1, inc, −1]`).
QA chạy thử 6 tổ hợp `stance_mode × have_incentive`: shape `(4, 4096)`, không NaN, min/max ∈ [−1, 1], đối xứng lệch 0.0.
Dịch chu kỳ đúng 1.0 (apex dịch `last_knot + offset` = cùng giá trị, viết gọn hơn). AC1.1–1.6 **pass**.

### F3.1 — Thiếu hàm tra pha → hệ số
PLAN Việc 1 yêu cầu "kèm hàm tra `(num_envs,) pha → (num_envs, 4)`". File mới có hàm dựng bảng, chưa có hàm tra.
Không có hàm này thì Việc 4 (reward) không dùng được bảng.
- [ ] AC-F3.1 (= PLAN AC1.7): hàm tra vector hoá, wrap được `p ≥ 1` và `p < 0`, trả `(num_envs, 4)` đúng thứ tự cột.

### F3.2 — Comment sai / dở dang
- Dòng 79: comment ghi "start − offset", code là `starts + offsets`.
- Dòng 83: câu "làm mượt đoạn chuyển pha thay vì" bị cụt.
- Dòng 126, 148: dòng `#` rỗng.
- Dòng 34–42: ghi "hệ số phạt" cho cả hàng, nhưng hàng có cả 0/+1. Ghi "hệ số thưởng/phạt".
- Dòng 140: "24 mốc mỗi pha" → thực ra 24 mốc cho cả 3 chu kỳ, nội suy cho cả 4 cột.
- [ ] AC-F3.2: 5 chỗ trên được sửa, không đổi code.

### Ghi chú
- Việc 0 (phép tính `T`, `swing_ratio`) chưa có kết quả. Không chặn Việc 1, nhưng phải xong trước Việc 3.
  - (2026-10-09) Nới cổng: Việc 0 phải xong trước **AC5.2 / Việc 6**, không còn chặn Việc 3. Xem PLAN Việc 0.
  - F3.1 (hàm tra) phải xong trước **Việc 4**. Việc 3 không cần nó.
