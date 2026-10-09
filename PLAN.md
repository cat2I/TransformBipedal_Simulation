# PLAN.md — Áp config đã đi bộ được (bản Isaac cũ) vào env chạy IsaacLab 3.0

## Công việc mới — B5a Gait clock kiểu Siekmann cho OFFICIALdesign (2026-10-08)

Người lập: Claude (Tech Lead). Người thực thi: Vinh (tự code, Claude review diff).
Nhánh: `vinh_dev`. Tham chiếu: `docs/ALGO.md` §2.3–2.4, `docs/gait-clock-rl-thao-luan.md`,
`isaac_rl/bipedal/officialdesign/gaitclockref/` (code apex gốc + `RLdiscuss.md`).

### Quyết định đã chốt (user, 2026-10-08)
1. **Reward theo pha = Siekmann**: hệ số đồng hồ (spline PCHIP, ∈[−1,1]) × lực/tốc độ bàn chân.
   Không làm XNOR. Không làm quỹ đạo mẫu (B5b để sau).
2. **Đồng hồ nằm trong `interface.py`** (hợp đồng sim ↔ nhúng). Obs **60 → 62** cho mọi task.
   Reward theo pha nằm trong `task_walk.py`.
3. **Đứng yên: hoãn.** Lệnh vẫn cố định +x 0.15 m/s, không thêm lệnh vào obs.

### Phạm vi
Được sửa/tạo:
- `isaac_rl/bipedal/officialdesign/interface.py` — đồng hồ + 2 ô obs.
- `isaac_rl/bipedal/officialdesign/gait_clock.py` — **file mới**: dựng bảng hệ số đồng hồ (torch).
- `isaac_rl/bipedal/officialdesign/task_walk.py` — reward theo pha, bỏ số hạng `swing`.
- `isaac_rl/bipedal/officialdesign/ppo.py` — chỉ `experiment_name`.
- `isaac_rl/scripts/validate_officialdesign.py` — chỉ chỗ ghi cứng `60`.
- `isaac_rl/scripts/check_gait_clock.py` — **file mới**: kiểm tra offline bảng đồng hồ.
- `docs/ALGO.md` — chỉ tick B5a và ghi quyết định đã chốt ở §2.4.1.

**Cấm:** sửa `gaitclockref/` (đó là tài liệu tham chiếu, giữ nguyên bản gốc apex).
**Cấm:** `import` bất cứ thứ gì từ `gaitclockref/` (file đó import `cassie`, `matplotlib`, sẽ gãy).
Cấm đụng action, framestack, actuator delay, DR, PPO hyperparameter.

---

### Việc 0 — Tính chu kỳ `T` trước khi code (giấy bút, không code)
Lý do: khớp chỉ đổi tối đa `action_step_deg × 20 Hz = 40°/s`. Chọn `T` quá ngắn → robot không theo kịp lịch → bị phạt mãi, học tệ (ALGO §2.3.1 mục 3).
- Ước lượng góc Knee + Hip cần gập rồi duỗi để nhấc bàn chân ~3 cm.
- `t_swing_min = (góc gập + góc duỗi) / 40°/s`. Với lịch 4 pha, một chân vung trong `swing_ratio × T`.
- Ghi kết quả vào comment cạnh `gait_period_s` kèm `TODO: thay bằng tốc độ servo đo ở A1`.
- [ ] AC0.1: `gait_period_s` và `swing_ratio` có comment ghi phép tính, không phải số chọn bừa.
  (Gợi ý điểm xuất phát: `T = 1.0 s`, `swing_ratio = 0.35` → chống kép `0.15` mỗi lần.)
- **2026-10-08 — User đề xuất `T = 3.2 s`, `s = 0.35`** (vung 1.12 s, chống kép 0.48 s). Nhấc chân 3 cm: đủ dư.
  ⚠️ Còn thiếu ràng buộc **ngang**: chân vung phải đuổi kịp thân. Tốc độ bàn chân so với hông khi vung ≈ `v·(1−s)/s`,
  tốc độ góc hông ≈ `v·(1−s)/(s·L)` — **không phụ thuộc T**. Với `v=0.15`, `s=0.35`, `L≈0.33 m` → ≈ 48°/s > 40°/s.
  Bước = `v·T/2` = 24 cm (~0.73·L). User phải tự kiểm lại `L` (trục hip pitch → đế bàn chân trong URDF) và chọn
  một trong: tăng `s`, giảm `v`, hoặc nâng `action_step_deg` nếu servo thật nhanh hơn (TODO A1). AC0.1 chưa pass.
- **2026-10-08 — User chọn nâng `action_step_deg`.** Servo thật (theo User): Bub/Hip/Knee = **STS3120 C001**, còn lại = STS3215.
  ⚠️ `assets/officialdesign/meta/calibration.json` nhóm `heavy` đang ghi **STS3095** (effort 10.3, velocity 4.7) → lệch phần cứng.
  User tra datasheet STS3120 C001 (tốc độ s/60° và stall torque **ở đúng điện áp đang cấp**) trước khi chọn số.
  Điều kiện chọn: `action_step_deg × 20 ≥ tốc độ cần / 0.7` (chừa 30%) **và** `≤ tốc độ servo chậm nhất trong các khớp chân, có tải`.
  `action_step_deg` là trường **hợp đồng** (`interface.py:94`) → firmware Pi đổi cùng số. Ghi phép tính vào comment cạnh nó.
- **2026-10-09 — User tạm hoãn Việc 0, làm Việc 3 trước.** Được, vì Việc 3 chỉ cần `T` là một biến, không cần giá trị đúng.
  Điều kiện: `gait_period_s` để giá trị tạm kèm `TODO: Việc 0 chưa chốt`; test AC3.3 tính theo `cfg.gait_period_s`, không gõ cứng số.
  **Cổng mới: AC0.1 phải pass trước AC5.2 (train smoke) và Việc 6.** Đổi `T` sau khi train = policy cũ vô dụng, phải train lại.

### Việc 1 — `gait_clock.py`: port `phase_function.py` sang dạng dùng được trên GPU
Ý tưởng: **không** viết lại PCHIP bằng torch. Lúc khởi tạo, dùng scipy PCHIP tính sẵn
hệ số tại N điểm pha (bảng tra, LUT) → đưa lên GPU. Lúc chạy chỉ tra bảng theo chỉ số.
Lý do: PCHIP chỉ cần tính 1 lần; tra bảng thì vector hoá cho 4096 env miễn phí.

Yêu cầu:
- Trục x là **pha chuẩn hoá `p ∈ [0,1)`**, không dùng `FREQ`/giây như apex.
  Lịch 4 pha: phải-vung `[0, s]` → chống kép `[s, 0.5]` → trái-vung `[0.5, 0.5+s]` → chống kép `[0.5+s, 1]`, với `s = swing_ratio`.
- Giữ 3 tham số của apex: `strict_relaxer`, `stance_mode` (`grounded`/`aerial`/`zero`), `have_incentive`.
- Nối 3 chu kỳ (trước–hiện tại–sau) trước khi nội suy, dịch **đúng 1 chu kỳ** (= 1.0 với trục chuẩn hoá).
- Trả về tensor `(4, N)` theo thứ tự cố định `[r_frc, r_vel, l_frc, l_vel]`, kèm hàm tra `(num_envs,) pha → (num_envs, 4)`.
- Docstring ghi rõ: dấu −1 phạt / 0 kệ / +1 thưởng, nguồn apex, và vì sao dùng LUT.
- ⚠️ Trước khi port, so sánh nhánh `grounded` + `have_incentive=False` của **chống kép thứ nhất** (`gaitclockref/phase_function.py` dòng 70–72) với **chống kép thứ hai** (dòng 110–112). Lẽ ra hai khối giống nhau, chỉ khác cột. **Tự tìm chỗ khác**, đừng chép nguyên. AC1.6 sẽ bắt lỗi này.

- [x] AC1.1: Không import `gaitclockref`, không import `matplotlib`. scipy chỉ dùng lúc dựng bảng.
- [x] AC1.2: Mọi giá trị bảng ∈ [−1, 1] (PCHIP không vọt lố).
- [x] AC1.3: Tuần hoàn: giá trị tại `p=0` ≈ giá trị tại `p→1` (sai < 1e-3).
- [x] AC1.4: Đối xứng: cột trái tại `p` == cột phải tại `p + 0.5` (cả frc lẫn vel).
- [x] AC1.5: `grounded`, `have_incentive=False`, giữa pha phải-vung: `r_frc=−1, r_vel=0, l_frc=0, l_vel=−1`.
- [x] AC1.6: `grounded`, `have_incentive=False`, giữa chống kép: `r_frc=0, l_frc=0, r_vel=−1, l_vel=−1`.
  (Review 2026-10-08, commit `b6b2805`: AC1.2–1.6 do QA chạy thử cả 6 tổ hợp mode×incentive. Việc 2 vẫn phải tự viết script kiểm.)
- [x] AC1.7: Có hàm tra `(table (4,N), p (num_envs,)) → (num_envs, 4)`, chạy trên GPU, không vòng lặp Python;
  `p = 1.0`, `p = 1.3`, `p = −0.2` cho cùng kết quả với `p = 0`, `0.3`, `0.8`. → xem FIX_AFTER_DIFF Lần 3 / F3.1.
  (Review 2026-10-09, commit `d02e426`: QA chạy `lookup_gait_clock` — wrap sai lệch 0.0, khớp đúng cột bảng sai lệch 0.0, `p=−1e−9` không tràn chỉ số.)

### Việc 2 — `check_gait_clock.py`: kiểm tra offline (chạy không cần Isaac)
- Kiểm AC1.2–AC1.6 bằng `assert`, in `PASS` từng mục.
- Lưu 1 ảnh PNG 4 đường (r_frc, r_vel, l_frc, l_vel theo `p`) vào thư mục scratch/log, **không** commit ảnh.
  Đây là lúc dùng matplotlib (chỉ trong script, không trong `gait_clock.py`).
- [ ] AC2.1: Script chạy exit 0, dán output vào báo cáo.
- [ ] AC2.2: Ảnh khớp Hình 3 của paper: vùng vung của chân nào thì `frc` của chân đó ở −1; vùng chống thì `vel` ở −1.
- **2026-10-08 — User quyết định BỎ Việc 2.** AC1.2–1.6 đã được QA kiểm thay (xem trên).
  Bù lại: bằng chứng AC1.7 (hàm tra, wrap pha) = dán output chạy thử 6 giá trị `p` trong REPL khi nộp diff.

### Việc 3 — `interface.py`: đồng hồ vào hợp đồng
- `OfficialInterfaceCfg`: thêm `gait_period_s` (Việc 0), `observation_space = 62`.
- Bộ đếm riêng `self.gait_step` (int, shape `(num_envs,)`). **Không** dùng `episode_length_buf`:
  `scripts/rsl_rl/train.py:220` gọi `init_at_random_ep_len=True` nên `episode_length_buf` bị random lúc đầu train, còn Pi thì đếm từ 0.
- Một hàm duy nhất trả `p = (gait_step × step_dt / gait_period_s) mod 1`. Cả obs (Việc 3) và reward (Việc 4) **cùng gọi hàm này**.
- Obs = 60 số cũ (giữ nguyên thứ tự) + `[sin(2πp), cos(2πp)]` ở **cuối**. Chỉ khung hiện tại, **không** stack 4 khung (pha là tất định, stack là thừa).
- `_reset_idx`: `gait_step = 0` cho env bị reset.
- Tự quyết chỗ tăng `gait_step` sao cho thoả AC3.3 — nghĩ xem `DirectRLEnv.step()` gọi `_pre_physics_step` → vật lý → `_get_rewards` → reset → `_get_observations` theo thứ tự nào.
- Cập nhật docstring đầu file mục "Nội dung hợp đồng": obs 62D, thứ tự, công thức pha, `p=0` lúc reset. Thêm `TODO: firmware Pi phải nối 2 số này, cùng gait_period_s, cùng bộ đếm reset về 0`.

- [x] AC3.1: `obs['policy'].shape == (num_envs, 62)`; 60 cột đầu tính y hệt trước.
- [x] AC3.2: Ngay sau reset: 2 cột cuối = `(0, 1)`.
- [x] AC3.3: Sau k bước (không reset): `p = k·step_dt/T mod 1`. Reward của bước đó dùng **cùng** `p` với obs trả về ở bước đó.
- [x] AC3.4: Reset một phần env thì chỉ env đó về `p=0`, env khác chạy tiếp.
  (Review 2026-10-09, commit `d02e426`: pass qua đọc code — `gait_step += 1` trong `_pre_physics_step` (chạy trước `_get_rewards` và `_reset_idx`),
  `gait_step[env_ids] = 0` trong `_reset_idx` (chạy trước `_get_observations`), obs và reward cùng gọi `_get_gait_phase()`.
  Bằng chứng chạy thật đi cùng AC5.1. Phần docstring hợp đồng của Việc 3 **chưa làm** → FIX_AFTER_DIFF Lần 4 / F4.1.)

### Việc 4 — `task_walk.py`: reward Siekmann
- Cfg task thêm: `swing_ratio`, `strict_relaxer` (0.1), `stance_mode="grounded"`, `have_incentive=False`,
  `max_foot_force` (≈ trọng lượng robot `4.643 × 9.81 ≈ 45 N`, vì pha chống đơn một chân gánh cả thân — apex dùng 250 N cho Cassie, **không** chép số đó),
  `max_foot_speed` (gợi ý 0.5 m/s, `TODO`), `w_clock_frc`, `w_clock_vel` (gợi ý 0.5 mỗi cái, `TODO tune`).
- `__init__`: dựng bảng đồng hồ 1 lần, đưa lên `self.device`.
- `_get_rewards`:
  - Đo: lực = chuẩn `net_forces_w` của 2 chân (đã có `forces[:, self.contact_ids]`); tốc độ = chuẩn `body_link_lin_vel_w` của 2 chân (đã có `feet_vel`).
  - Chuẩn hoá: `min(đo, max) / max` → ∈ [0, 1]. **Cap trước khi chia** (chống policy dậm thật mạnh để ăn điểm).
  - Điểm mỗi chân: `tan(π/4 · clock · đo_chuẩn_hoá)` như apex. Thứ tự chân trong `contact_ids`/`feet_ids` là `(Footleft, Footright)`, còn bảng là `[r, …, l, …]` — **ghép đúng trái với trái**.
  - Cộng `w_clock_frc·(frc_L+frc_R) + w_clock_vel·(vel_L+vel_R)` vào reward.
  - **Bỏ** số hạng `0.15*swing` và TODO của nó (thưởng nhảy gấp đôi bước đi — ALGO §2.2.1; đồng hồ thay thế nó).
  - Các số hạng khác giữ nguyên.
- (Nên làm) Ghi trung bình `frc_score`, `vel_score` vào `self.extras["log"]` để xem trên TensorBoard robot có theo nhịp không.

- **2026-10-09 — User giữ `swing` nhưng gate theo đồng hồ** (working tree, chưa commit): chỉ chân có `frc` âm (đang tới lượt vung) mới được thưởng độ cao.
  Chấp nhận: lỗi "nhảy ăn gấp đôi" (ALGO §2.2.1) biến mất vì hai chân không bao giờ cùng có gate > 0 (với `grounded`, `s < 0.5`).
  Lý do giữ: phạt lực chỉ bắt chân vung *không chạm đất*, không bắt *nhấc cao* → robot có thể lê chân sát sàn. AC4.1 sửa lại như dưới.
- [x] AC4.1: ~~Không còn `swing` trong reward.~~ `swing` chỉ thưởng chân có `−frc_clock > 0`. Không đổi số hạng nào khác.
- [x] AC4.2: Không có vòng `for` qua env; mọi thứ là tensor `(num_envs, …)`.
- [ ] AC4.3: Ghép chân đúng: in thử 1 env lúc `p` giữa pha phải-vung, chân **phải** chạm đất → `frc_R < 0`.

### Việc 5 — Phụ trợ
- `ppo.py`: `experiment_name = "officialdesign_clock"` (checkpoint 60D cũ không load được vào mạng 62D, tách thư mục để không nhầm).
- `validate_officialdesign.py` dòng 31, 110: thay `60` bằng `env.cfg.observation_space` (hợp đồng đổi thật, không phải sửa test cho qua).
- `docs/ALGO.md`: tick B5a; ở §2.4.1 ghi quyết định 1 và 2 đã chốt (Siekmann, hoãn đứng yên).
- [ ] AC5.1: `validate_officialdesign.py` pass trên GPU, dán output.
- [ ] AC5.2: Train smoke `--num_envs 256 --max_iterations 2 --headless` exit 0, log ghi `Observation dim 62`.

### Việc 5b — Train chẩn đoán (2026-10-09, User muốn xem đồng hồ có hoạt động không, trước khi chốt Việc 0)
Mục đích: **chỉ** trả lời "dây nối đồng hồ → reward có đúng không". Policy này **bỏ đi**, không deploy, không thay Việc 6.
Không vi phạm cổng Việc 0: cổng chặn train *thật*; run này không giữ lại.
- Điều kiện trước: (a) log `frc_score`, `vel_score`, `swing` vào `self.extras["log"]` (PLAN Việc 4, "nên làm" → **bắt buộc** cho 5b);
  (b) `ppo.py` `experiment_name = "officialdesign_clock"` (Việc 5); (c) smoke 2 iter exit 0.
- Chạy với số **khả thi** qua override dòng lệnh, **không sửa code hợp đồng**: `env.target_velocity=0.09 env.action_step_deg=2.25`.
  Lý do: với `v=0.15`, `action_step_deg=2.0` thì hông cần ≈48°/s > 40°/s (Việc 0) → robot không theo kịp lịch,
  run fail cũng không biết là do dây nối sai hay do vật lý không cho phép.
  (Nếu Việc M xong trước thì **bỏ** override `action_step_deg`; tính lại `v` khả thi theo tốc độ mới.)
- [x] AC5b.1: TensorBoard có 3 đường `frc_score`, `vel_score`, `swing`.
  (Smoke 2026-10-09 `officialdesign_clock/2026-10-09_15-11-17`: exit 0, MLP `in_features=62`, bảng khớp heavy `2.31 / 11.77`, light `4.7 / 2.94`,
  `env.yaml` `action_step_deg: 6.61`. Iter 1: `frc −0.34`, `vel −0.64`, `swing 0.10`, **episode length ≈ 15 bước (0.75 s)** — theo dõi: nghi `action_step_deg` 6.61 × nhiễu khám phá std 1.0 làm robot run gấp 3.3 lần lúc đầu.)
- [ ] AC5b.2: Báo cáo: `frc_score`, `vel_score` có đi lên về 0 không; play 1 env, hai chân có luân phiên theo nhịp `T` không.
  **Run 1 (`2026-10-09_15-16-11`, 256 env × 481 iter, `v=0.15`, `action_step_deg=6.61`):** kẹt ở **đứng chôn chân**.
  ep length 15 → 185/200; `frc` −0.34 → −0.285 (iter 100) → −0.30; `vel` −0.64 → −0.45 → −0.49; `swing` 0.10 → 0.017 → 0.026; action std 0.98 (gần như không giảm).
  `frc` khớp dự đoán "hai chân đè đất suốt" (`tan(π/4·−0.5)·~0.66 ≈ −0.28`). Nghi nguyên nhân: thưởng vận tốc σ=0.20 cho đứng yên ~57% điểm tối đa.
  Còn thiếu trước khi đổi nút: thí nghiệm "trọng tài" (frc từng chân theo `p`, robot đứng, 64 bước) = AC4.3. Play chưa xem.
- **2026-10-09 — mở phạm vi `scripts/rsl_rl/play.py`** (chỉ phần in log, không đụng vòng điều khiển):
  dòng 488–489 crash `'int' object has no attribute 'shape'` — `cfg.observation_space`/`action_space` của Direct env là `int`.
  Lỗi có từ commit `bcdae629` (2026-08-04), không do B5a. Dòng 344–345, 562–566 gọi `obs[60:62]` là "twist progress" — với OFFICIALdesign đó là `[sin, cos]` đồng hồ → đổi nhãn.
  - [x] AC5b.3: play chạy được với `Official-Walk-v0`; nhãn cột 60–61 không còn ghi "twist". (Claude sửa theo yêu cầu User — ngoại lệ AGENTS §0.)
- **Play run 1 (bước 100–240):** đồng hồ obs đúng — 5.625°/bước = 64 bước/chu kỳ = 3.2 s; sau timeout bước 200 in `p = 1/64` (reset về 0 một bước trước). Bằng chứng chạy cho AC3.2/3.3.
  Robot đứng `h ≈ 0.37`, chân chỉ "gõ" 0.05–0.15 s (lịch vung 1.12 s). Chân nhấc **ngược lượt** nhiều hơn: 18 lần sai lượt / 9 lần đúng lượt (yếu, p≈0.06).
  Nghi: play.py gán `air_time[0]=L` theo vị trí, không theo tên; hoặc nhiễu. **Lịch đối xứng → hoán đổi L/R trong reward không làm hỏng việc học**
  (chỉ đổi chân đi trước), miễn `frc`, `vel`, `swing_gate` cùng một cách ghép — code đang nhất quán (`[2,0]`, `[3,1]`, gate từ `force_clock`).
  → AC4.3 **không còn chặn run 2**; vẫn phải pass trước Việc 6 (đặt tên đúng cho firmware/debug).
  User quan sát (nhìn theo hướng đi): **chân phải thật chống, chân trái thật luôn là chân gõ; robot có tiến lên** → lê chân ăn được điểm vận tốc.
  ⚠️ `calibration.json` notes: "CAD left is robot right" → body `Footleft` có thể là chân **phải thật**. Dấu `roll` chưa đối chiếu được với khung IMU → gộp vào AC4.3.
  User quan sát bằng mắt: **lê chân + một chân gõ**. Khớp `vel_score` −0.45 (chân chống trượt) và `roll ≈ −0.09 rad` không đổi (nghiêng hẳn một bên → chân bên nhẹ gõ).

### Việc 5c — Run chẩn đoán 2: cà rốt `swing` (2026-10-09, User chọn)
Lý do (sổ sách run 1): lê chân lãi ≈ +0.09/bước so với đứng (vận tốc +0.32, phạt đồng hồ −0.23). `swing` là số hạng **duy nhất**
đòi chân rời đất (`~touching`) **và** đúng lượt (`swing_gate`), nên đứng/lê không ăn được. Hiện trọng số 0.15 × `.mean` → tối đa 0.075, quá nhỏ.
Loại `have_incentive=True`: thưởng chân chống gánh lực + chân vung chạy nhanh → đứng hết bị phạt, lê đúng lượt được thưởng.
- `task_walk.py`: đưa `0.15` thành trường cfg `w_swing` (mặc định **0.15** để run 1 tái lập được), reward dùng `self.cfg.w_swing`. Không đổi gì khác.
- Run 2: `... --max_iterations 500 env.w_swing=1.0` (tối đa thực 0.5/bước; nhấc 1 cm đã ≈ 0.18).
- [ ] AC5c.1: `grep -n "0.15 \* swing" task_walk.py` → 0; `w_swing: 0.15` có trong `env.yaml` khi chạy không override.
  **2026-10-09 — User chọn sửa thẳng số `0.15 → 1.0`** thay cho trường cfg. Chấp nhận cho run chẩn đoán.
  AC5c.1 thay bằng: comment cạnh số ghi giá trị cũ (0.15, run 1) và run 2. Truy vết qua `<run>/git/*.diff` (env.yaml không ghi số gõ cứng).
  Trước Việc 6: nên đưa lên cfg (`w_swing`) để tune bằng dòng lệnh.
- **Run 2 `2026-10-09_18-07-07` (`1.0 * swing`), play bước 328–378:** hai chân **luân phiên** gõ (run 1: chỉ một chân) nhưng theo **nhịp riêng 8 bước = 0.4 s**
  (L nhấc đúng bước 328, 336, 344, 352, 360, 368; R nhấc 2 bước sau), **bỏ qua đồng hồ 3.2 s**: đúng lượt 9 / sai lượt 9. Air ≤ 0.15 s, `h` dao động cùng nhịp 8 bước.
  Nghi: `T = 3.2 s` chậm gấp ~3 lần nhịp con lắc tự nhiên của chân (`2π√(L/g)` ≈ 1.15 s với L≈0.33) → vung 1.12 s đứng một chân quá khó; gõ nhanh hai chân "rải đều" vẫn trúng cửa sổ gate một phần.
  → Việc 0 (chọn `T`) quay lại thành nút chính. Chưa có TensorBoard run 2.
- [ ] AC5c.2: Báo cáo run 2 so với run 1: `swing`, `frc`, `vel`, episode length, play (chân nào nhấc, có đúng lượt, có tiến không).

### Việc M — Giới hạn động cơ theo datasheet, bản "tối đa" (2026-10-09, tạm — sửa lại khi đo A1)
User yêu cầu: đặt giới hạn sim = **datasheet không tải**, cho policy dùng hết công suất servo, tinh chỉnh sau.
Nằm ngoài B5a nhưng **chặn Việc 0**: đổi tốc độ trần thì phép tính `T`/`v` của Việc 0 đổi theo.

Datasheet (User, `gaitclockref/note.txt`): STS3120 C001 (Bub/Hip/Knee) 0.454 s/60°; STS3215 (Foot, rotate) 0.222 s/60°.

Mô hình `DCMotorCfg` của Isaac (`actuator_pd.py:299`): `τ_max(ω) = saturation_effort·(1 − ω/velocity_limit)`, cắt ở `effort_limit`.
→ `velocity_limit` = tốc độ **không tải**, `saturation_effort` = mô-men **kẹt (stall)**. "Có tải chậm hơn" do mô hình **tự sinh ra**.
**Không** điền tốc độ có tải (~½) vào `velocity_limit` — làm thế là chia đôi hai lần.

Phạm vi được sửa:
- `assets/officialdesign/meta/calibration.json` → `actuators.heavy`: `servo`, `velocity`, `effort`; `actuators.light`: kiểm lại `effort` theo datasheet STS3215; mục `notes` ghi nguồn số.
- `interface.py` → `action_step_deg` (trường hợp đồng): `= tốc độ không tải servo chậm nhất / 20 Hz`. Comment ghi phép tính + `TODO: firmware Pi đổi cùng số; giảm lại sau khi đo A1`.
- `scripts/test_officialdesign_asset.py:49-50`, `scripts/validate_officialdesign.py:38`: đang gõ cứng `10.3 / 2.94 / 4.7` → đọc từ `CALIBRATION` (hợp đồng đổi thật, không phải sửa test cho qua).
- Chạy lại `prepare_officialdesign.py` rồi `convert_officialdesign.py --headless --force` (robot.py kiểm SHA-256, không chạy là crash lúc load).
Cấm: đụng `stiffness`, `damping`, `armature` (tuning sim, để sau).

- **2026-10-09 — QA kiểm số User tính:** heavy `2.31 rad/s` ✅, `action_step_deg 6.61` ✅. STS3120 @12V: stall 120 kg·cm = **11.77 N·m**, rated 40 kg·cm = 3.92 N·m.
  Bản tối đa: `heavy.effort = 11.77` (stall). `TODO` sau A1: tách `effort_limit` = rated, `saturation_effort` = stall (cần sửa `robot.py`, ngoài Việc M).
  STS3215 @12V: stall 30 kg·cm = **2.94 N·m** (khớp số đang có, giữ nguyên), rated 10 kg·cm = 0.98 N·m.
  User xác nhận `0.454 s/60° (22 RPM) @12V`. `0.222 s/60°` của STS3215: chưa xác nhận điện áp (`TODO`).
- [ ] ACM.1: `heavy.servo = "STS3120 C001"`, `heavy.velocity` = rad/s tính từ 0.454 s/60° (ghi phép tính trong `notes`).
- [ ] ACM.2: `effort` hai nhóm = mô-men kẹt datasheet **ở đúng điện áp đang cấp**, đổi kg·cm → N·m (`× 0.0981`). Ghi điện áp vào `notes`.
- [ ] ACM.3: `action_step_deg` = `velocity_heavy(°/s) / 20`, có comment phép tính và `TODO` firmware.
- [ ] ACM.4: 2 test không còn số gõ cứng; `test_officialdesign_asset.py` và `validate_officialdesign.py` pass, dán output.
  **2026-10-09 — User hoãn ACM.4 để train chẩn đoán (5b) trước.** Được: train không chạy 2 test này.
  Thay bằng bằng chứng rẻ: `grep` `velocity_limit`/`effort_limit` trong `<run>/params/env.yaml` của run smoke. **ACM.4 phải pass trước Việc 6.**

### Việc 6 — Train thật & đọc kết quả (sau khi AC0–AC5 pass)
- Train `--num_envs 2048`, ~1500 iteration. Xem TensorBoard: `frc_score`, `vel_score` phải **tăng về 0** (ít bị phạt hơn).
- Play và quan sát: hai chân luân phiên, đúng nhịp `T`? Hay nhảy / lê / đứng 1 chân?
- Đối chiếu bảng ALGO §2.4.2 để quyết bước tiếp.
- [ ] AC6.1: Báo cáo gồm đường cong reward, 2 score clock, và mô tả dáng đi khi play.

---

## Công việc mới — OFFICIALdesign handoff 2026-09-30 (2026-10-01)

User yêu cầu tích hợp robot mới để train. Phạm vi đợt này độc lập với kế hoạch
NewSimple bên dưới: thêm `assets/officialdesign/`, script chuẩn bị/kiểm thử,
task `Transformer-Official-10DOF-Direct-v0`, đăng ký task và tài liệu chạy.
Giữ các thay đổi có sẵn của user. Không sửa dữ liệu handoff gốc.

- [x] Chuẩn bị URDF 13 link từ handoff: mass, CoM mm→m, tensor tại CoM;
      kiểm tra tensor vật lý và tổng 4.643 kg, mesh và cây 10 DOF.
- [x] Lưu mapping theo tên joint thật, dấu hai chân, giới hạn train tạm có
      nguồn rõ ràng; không gọi chúng là giới hạn cơ khí đã đo.
- [x] Import USD floating base, giữ IMU, collision convex, kiểm tra mass USD.
- [x] Task mới: actuator heavy/light riêng, reset sạch, lệnh theo tên khớp,
      IMU không nhân đôi, pose/chiều cao theo hình học robot mới, PPO riêng.
- [x] Chạy kiểm tra hướng khớp, reset, số hữu hạn, giới hạn torque/position,
      đứng giữ pose/thả rơi và train PPO ngắn trên GPU.
- [x] Ghi lệnh train/play và các mục CAD/phần cứng còn chưa xác nhận.

Giới hạn nghiệm thu: chạy được train không đồng nghĩa policy đã biết đi hay
thông số đã đủ để triển khai robot thật. Hip/Foot CoM và script SolidWorks
vẫn chờ kiểm tra phía CAD theo handoff.

Kết quả: offline 4/4; GPU validation 8 env pass (giữ pose 5 s, contact,
reset, torque, thả rơi); PPO 256 env × 2 iterations = 12,288 steps, exit 0.
Báo cáo: `assets/officialdesign/meta/validation_report.json` và
`assets/officialdesign/meta/train_smoke_report.json`.

---

Người lập: Claude (Tech Lead). Người thực thi: agy (Gemini). Ngày: 2026-09-10.

## Nguồn sự thật

Folder bản gốc của bạn user, train trên Isaac cũ, **robot đi bộ được**:
`/home/cat21/Documents/projects/Transformer/IsaacSim_TransformerBipedal/` (gọi tắt **REF**).
File quan trọng: `REF/transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_nam_env.py`
(gọi tắt **REF_ENV**). Git log của REF ghi rõ từng lần tune (2026-07-29 → 07-30).

Claude đã diff toàn bộ REF với repo này. Kết quả:
- `NewSimple.usd` + 4 layer `configuration/`: **md5 giống hệt** → asset không phải nguyên nhân.
- `transformer_config.py` (robot, actuator, pose): **giống** giá trị.
- `agents/rsl_rl_ppo_cfg.py`: **giống** giá trị, chỉ khác API rsl_rl 5.
- `transformer_nam_env.py`: **khác 8 chỗ** — toàn bộ việc nằm trong file này.

## Phạm vi

Chỉ sửa **1 file**:
`isaac_rl/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_nam_env.py` (gọi tắt **ENV**).

**Cấm sửa** mọi file khác. Trong ENV, **cấm đụng** những phần thuộc "port sang IsaacLab 3.0" (bảng B) —
đó là lý do REF_ENV không chạy được trực tiếp trên máy này.

---

## Bảng A — 8 giá trị phải đổi (REF_ENV là chuẩn)

| # | Vị trí trong ENV | Hiện tại | Đổi thành | Số dòng REF_ENV | Ghi chú |
|---|---|---|---|---|---|
| A1 | `EnvCfg.servo_max` | `(30, 30, -45, -45, 30, 30)` | `(35, 35, -40, -40, 35, 35)` | 131 | commit "tăng góc servo max ở knee" |
| A2 | `EnvCfg.servo_min` | `(20, 20, -55, -55, 20, 20)` | `(15, 15, -70, -70, 15, 15)` | 135 | commit "thay đổi giới hạn khớp" |
| A3 | `EnvCfg.weights["walk"]` | `[1, 1, 1, 0, 2, 1.2, 1]` | `[1, 1, 0.2, 0, 2, 6, 1]` | 62 | commit "tăng điểm velo" |
| A4 | `_setup_scene` → `RigidBodyMaterialCfg` | `static_friction=2.0, dynamic_friction=2.5` | `static_friction=0.8, dynamic_friction=0.4` | 209–210 | ma sát sàn |
| A5 | `_get_rewards` → `feet_height_reward(air_time, contact_pos, 0.03, 150)` | `0.03` | `0.04` | 313 | commit "tăng feetheight reward" |
| A6 | `joint_position_reward` → `max_diff` | `[5, 5, 5, 5, 5, 5]` | `[10, 10, 10, 10, 10, 10]` | 521 | commit "đổi độ lệch tối đa lên 10" |
| A7 | `_get_dones` → `height_termination` | `head_heights < 0.1` | `head_heights < 0.2` | 354 | commit "sửa điều kiện chết" |
| A8 | Khối IMU trong `EnvCfg` + `_get_observations` | có bias, drift, nhiễu 0.04/0.15, cờ legacy | xem Việc 2 | 223–227 | REF không có bias/drift |

`start_pos` giữ `(25, 25, -50, -50, 25, 25)` — REF dòng 142 giống.

## Bảng B — KHÔNG được đụng (port IsaacLab 3.0, REF_ENV không có vì chạy bản cũ)

| Thứ | Trong ENV | Trong REF_ENV | Lý do giữ bản ENV |
|---|---|---|---|
| `from ._lab3_compat import as_torch, imu_quat_w` | có | không | IsaacLab 3.0 trả `ProxyArray`, không phải Tensor |
| `as_torch(self.robot.data.…)` (mọi chỗ) | có | không | như trên |
| `imu_quat_w(self.robot, self.cfg.imu)` | có | `self.imu_data.quat_w` | như trên |
| `quaternion_to_euler`: `x, y, z, w = quat[...]` | xyzw | wxyz | IsaacLab 3.0 đổi quy ước quaternion |
| `from isaaclab.utils.configclass import configclass` | có | `from isaaclab.utils import configclass` | đường import 3.0 |
| `servo_max/min/start_pos` đọc từ `self.cfg.*` | có | hard-code | giữ, chỉ đổi giá trị mặc định (A1, A2) |

---

## Việc 1 — Áp A1..A7

Sửa đúng 7 con số/tuple theo Bảng A. Không sửa gì khác quanh đó.

- [x] AC1.1: `grep -n "servo_max\|servo_min" ENV` cho ra tuple đúng A1, A2.
- [x] AC1.2: `weights = {"walk": [1, 1, 0.2, 0, 2, 6, 1]}`. Thêm 1 dòng comment ngay trên:
      `# [orientation, height, joint_position, sigmoid_extra, feet_height, velocity, deviation] — theo REF 2026-07-30`
- [x] AC1.3: ma sát sàn `0.8 / 0.4`.
- [x] AC1.4: `feet_height_reward(..., 0.04, 150)`.
- [x] AC1.5: `max_diff = torch.tensor([10, 10, 10, 10, 10, 10], device=device)`.
- [x] AC1.6: `height_termination = head_heights < 0.2`.
- [x] AC1.7: Ngoài 7 dòng này, phần `_get_rewards`, `_get_dones` và các hàm sau `# REWARD FUNCTIONS` **không có diff nào khác**.

## Việc 2 — IMU về đúng REF (A8)

Trong `class TransformerWalkEnvCfg`:

| Trường | Hiện tại | Đổi thành |
|---|---|---|
| `domain_rand` | `True` | `False` |
| `imu_noise_std["orientation"]` | `0.04` | `0.015` |
| `imu_noise_std["angular_velocity"]` | `0.15` | `0.01` |
| `imu_drift_rate` | `0.0001` | `0.0` |
| `imu_fixed_bias` | `(0.0, -0.193, 0.0)` | `(0.0, 0.0, 0.0)` |
| `imu_legacy_double` | `True` | **xoá trường + xoá khối comment "TƯƠNG THÍCH NGƯỢC"** |
| `imu_bias_range` | giữ | giữ (chỉ có tác dụng khi `domain_rand=True`) |

Sửa comment cạnh `imu_noise_std`: bỏ "Tăng từ 0.015 → match real std", thay bằng
`# mặc định = REF (đã đi bộ được). Muốn tăng: env.imu_noise_std.orientation=0.04 trên dòng lệnh`.

Trong `_get_observations`, thay khối:
```python
orient_n = gaussian_noise(orient, GaussianNoiseCfg(...))
gyro_n = gaussian_noise(angular_vel_raw, GaussianNoiseCfg(...))
if self.cfg.imu_legacy_double:
    orient = orient + orient_n
    angular_vel = angular_vel_raw + gyro_n
else:
    orient = orient_n
    angular_vel = gyro_n
```
bằng đúng 2 dòng (đây chính là REF_ENV dòng 226–227, chỉ khác lấy std từ cfg):
```python
orient = gaussian_noise(orient, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["orientation"]))
angular_vel = gaussian_noise(angular_vel_raw, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["angular_velocity"]))
```
Rút khối comment "SỬA 2026-09-10" còn ≤ 3 dòng, ý: *`gaussian_noise()` đã trả về `data + nhiễu`; không cộng thêm lần nữa.*

Giữ nguyên: khối `if self._should_randomize_imu:` (bias random) và khối drift — chúng tắt khi giá trị mặc định = 0/False.

- [x] AC2.1: 5 giá trị đúng bảng.
- [x] AC2.2: `grep -n imu_legacy_double ENV` → 0 kết quả.
- [x] AC2.3: Trong `_get_observations` không còn `orient += ...` hay `orient + orient_n`.

## Việc 3 — Dọn comment sai đầu file

Khối comment trên `from .transformer_config import TRANSFORMER_CFG` (bắt đầu "transformer_config.py định nghĩa
TRANSFORMER_CFG HAI LẦN…") mô tả tình trạng cũ. Thay bằng:
```python
# TRANSFORMER_CFG = NewSimple.usd, 6 khớp, pose Hip 25° / Knee -50° / Foot 25°. Giống REF từng số.
# Khối FullForm111 (8 khớp) trong transformer_config.py đã comment — KHÔNG mở lại (lệch shape [1,6] != [1,8]).
```
- [x] AC3.1: Không còn chữ `transformer_config_3dof` trong ENV.

## Việc 4 — Kiểm chứng (agy chạy, dán output kèm diff)

```bash
cd isaac_rl
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk-Direct-v0 --num_envs 64 --headless --max_iterations 2
```
- [ ] AC4.1: Chạy hết 2 iteration không lỗi; log in `Observation dim: 44`, `Action dim: 6`.
- [ ] AC4.2: Trong `logs/rsl_rl/transformer_walk/<run mới>/params/env.yaml` có:
      `domain_rand: false`, `imu_drift_rate: 0.0`, `orientation: 0.015`, `angular_velocity: 0.01`,
      `servo_max: [35, 35, -40, -40, 35, 35]`, `servo_min: [15, 15, -70, -70, 15, 15]`,
      `weights.walk: [1, 1, 0.2, 0, 2, 6, 1]`.
- [ ] AC4.3: `git diff --stat` chỉ có `transformer_nam_env.py` (ngoài `PLAN.md`, `AGENTS.md`). Xoá run thử 2 iteration sau khi xác nhận; không commit log.

---

## Sau nghiệm thu — lệnh train thật (User chạy)

REF `agent.yaml`: `num_envs 512`, `max_iterations 350`. Run hôm qua (`2026-09-09_14-49-57`) chạy `num_envs 64`
= ít dữ liệu hơn **8 lần** mỗi iteration → không so sánh được. Bắt buộc 512:

```bash
cd isaac_rl
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk-Direct-v0 --num_envs 512 --headless --max_iterations 350
```
RTX 4050 6GB OOM → `--num_envs 256 --max_iterations 700` (giữ tổng số mẫu).

Play: `./run.sh scripts/rsl_rl/play.py --task Transformer-Walk-Direct-v0 --num_envs 1 --checkpoint "$PWD/logs/rsl_rl/transformer_walk/<run>/model_349.pt"`

Nếu sau 350 iter × 512 env vẫn không đi: config đã khớp REF 100% ở tầng Python → phần còn lại là backend
vật lý IsaacLab 2.x → 3.0. Lúc đó mới tune (ma sát sàn, `solver_position_iteration_count`, stiffness) — Claude lập plan mới.
