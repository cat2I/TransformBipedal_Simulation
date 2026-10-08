# RLdiscuss — Ghi chép thảo luận: Periodic Reward Composition

> Nguồn chính: **Siekmann et al.** — *"Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"* (file `periodic rew.pdf`).
> Code "nhà": **osudrl/apex** (cùng lab OSU DRL, chạy trên robot Cassie).
> Mục tiêu ghi chép: nền tảng để tự dựng reward train bipedal trong `TransformBipedal_Simulation`.

---

## 0. Vấn đề bài này giải

Dạy robot đi bộ kiểu cũ: thu một đoạn đi mẫu (**reference trajectory** — quỹ đạo tham chiếu) rồi bắt robot **chép y hệt**.
Dở ở chỗ:
- Robot bị trói vào đúng mẫu đó → gặp địa hình lạ là ngã.
- Mỗi dáng đi (walk, run, hop...) phải thu một mẫu riêng.

**Ý tưởng của bài:** đừng chép mẫu. Mô tả dáng đi như một **cái nhịp lặp lại**; ở mỗi khoảnh khắc trong nhịp chỉ nói *cái gì nên đúng*, không nói *cử động khớp ra sao*. Robot tự mò ra cách cử động.

---

## 1. Đồng hồ nhịp φ (phase clock) — xương sống chung

Công thức:
```
p = (step · dt  mod  T) / T          ∈ [0, 1)
```
- `step·dt` = thời gian đã trôi. `mod T` = cắt vòng theo chu kỳ T. Chia T = chuẩn hóa về [0,1).
- Là **răng cưa (sawtooth)**: chạy 0→1 rồi nhảy về 0, lặp mãi.
- **Dùng xuyên suốt cả 3 nguồn** (periodic rew, humanoid gym, Unitree). Cái khác nhau là *xây gì lên trên* nó.

---

## 2. Mã hóa tuần hoàn sin/cos

Răng cưa có **điểm nhảy giật** 1→0. Mạng neuron ghét điểm gãy đó. Nên bọc nó lên vòng tròn:
```
sin(2πp) , cos(2πp)
```
- Cần **cả hai** mới xác định duy nhất một điểm trên vòng (một mình sin thì 2 góc trùng giá trị → nhập nhằng).
- Hết vòng nối liền mượt, không còn điểm gãy.

---

## 3. Viên gạch cơ bản của reward

Mỗi "luật thưởng/phạt" = tích của **3 mảnh**:
```
R_i(s, φ) = c_i · I_i(φ) · q_i(s)
```
| Mảnh | Tên | ELI5 |
|------|-----|------|
| `q_i(s)` | số đo (measurement) | Đo một đại lượng vật lý lúc này: lực bàn chân, tốc độ bàn chân... |
| `I_i(φ)` | chỉ báo pha (indicator), 0/1 | Công tắc theo nhịp: "có đang ở pha này không?" |
| `c_i` | hệ số pha (coefficient) | Dấu quyết định: `−1` phạt, `+1` thưởng, `0` kệ |

Nhân lại: *chỉ khi đúng pha (I=1) thì mới lấy số đo q nhân hệ số c cộng vào điểm.* Sai pha (I=0) → luật ngủ.

Tổng reward: `R(s,φ) = β + Σ_i R_i(s,φ)` (β = baseline sống sót).

---

## 4. Hai luật duy nhất đẻ ra dáng đi

Chân có 2 pha lặp: **swing (chân bay)** và **stance (chân trụ)**.

| Pha | `c_frc` (lực) | `c_spd` (tốc độ) | Dịch ra tiếng người |
|-----|:---:|:---:|---------|
| **Swing** (bay) | `−1` | `0` | Bay thì **cấm có lực** (không chạm gì). Tốc độ kệ. |
| **Stance** (trụ) | `0` | `−1` | Trụ thì **cấm trượt** (tốc độ ≈ 0). Lực kệ. |

"Bay thì đừng có lực, trụ thì đừng trượt" — áp đúng lúc → **tự nhiên ra dáng đi**. Không cần mẫu. Đây là điểm đẹp nhất của bài.

Ghép lực + tốc độ cho một chân = `R_unipedal` (uni = một chân).

---

## 5. Làm mượt ranh giới bằng Von Mises

Nếu công tắc I bật/tắt **cứng đét** ở mốc pha → reward **nhảy giật** → robot học bị rung.

Cách xử lý: mốc bắt đầu `A` và kết thúc `B` của mỗi pha thành **biến ngẫu nhiên**, rải theo phân phối **Von Mises** (quả chuông bell-curve nhưng **uốn vòng quanh vòng tròn**, vì nhịp là vòng lặp).
```
A_i ~ Φ(2π·a_i, κ) ,  B_i ~ Φ(2π·b_i, κ)
P(A_i < φ < B_i) = P(A_i < φ)·(1 − P(B_i < φ))
```
- Dùng **E[I_i(φ)] = xác suất đang ở trong pha** → trườn mượt 0↔1, không giật.
- `κ` (kappa) = độ sắc/mờ của ranh giới. κ lớn → sắc; κ nhỏ → mờ nhòe.
- Huấn luyện bằng **kỳ vọng E[R]** (mượt). Nhờ *linearity of expectation*, robot học ra y hệt bản ngẫu nhiên nhưng êm hơn.

`r` = **tỉ lệ swing** kéo dài bao lâu trong nhịp (swing dài `r`, stance dài `1−r`). Chính là `r_swing` trong công thức ω (mục 8).

---

## 6. Hai chân + offset θ → mọi dáng đi

Hai chân chạy **cùng một** chu trình swing/stance, chỉ **lệch pha** nhau bằng `θ_left`, `θ_right`:
- **Walking (đi):** lệch nửa nhịp → một chân bay thì chân kia trụ.
- **Hopping (nhảy cóc):** lệch 0 → hai chân bay/trụ cùng lúc.
- **Galloping / skipping:** các độ lệch khác (skipping cần 4 pha thay vì 2).

➡️ Chỉ đổi `θ` → ra dáng khác. **Một khung, mọi dáng đi.**

Gộp cả 2 chân (lực + tốc độ × trái + phải) = **`R_bipedal`** (eq. 2):
```
E[R_bipedal] = E[C_frc(φ+θ_left)]·q_left_frc  + E[C_frc(φ+θ_right)]·q_right_frc
             + E[C_spd(φ+θ_left)]·q_left_spd  + E[C_spd(φ+θ_right)]·q_right_spd
```
với `C_frc(φ) = c_swing_frc·E[I_swing] + c_stance_frc·E[I_stance]` (và tương tự `C_spd`).

---

## 7. Hình 3 — bức tranh của tất cả

**Đồng hồ tròn (trên):** φ vẽ thành vòng (vì lặp). Nửa **xanh** = swing (`c_swing_frc = −1`), nửa **cam** = stance (`c_stance_frc = 0`). Kim gạch đỏ = φ hiện tại. Nhãn `a/b` = mốc vào/ra pha; chỗ nối màu nhòe = vùng mờ Von Mises.

**Đường cong (dưới): `E[C_frc(φ)]`** — hệ số phạt lực theo nhịp:
- Vùng swing (xanh): đường ở **−1** → đang bay, phạt lực mạnh nhất.
- Vùng stance (cam): đường lên **0** → đang trụ, bỏ qua lực.
- Chuyển **hình chữ S mượt** (nhờ Von Mises), không nhảy vuông góc.

Lưu ý: Hình 3 **chỉ vẽ LỰC**. Bản song sinh cho **TỐC ĐỘ** `E[C_spd(φ)]` thì **ngược lại**: `0` khi swing, `−1` khi stance (luật "trụ thì cấm trượt").

---

## 8. ω — công tắc mềm walking ↔ standing

Vấn đề: muốn **một policy** vừa đi được vừa **đứng yên** được. Khi đứng, mọi luật "đi" phải tắt.

```
ω = ( 1 + exp(−50·(r_swing − 0.15)) )^(−1)
```
- Là **sigmoid** (hàm chữ S). `ω ≈ 1` khi đang đi (r_swing lớn), `ω ≈ 0` khi đứng (r_swing ≈ 0).
- `−0.15` = ngưỡng; `50` = độ dốc (chuyển rất gắt quanh ngưỡng).
- Là cái **núm chỉnh sáng (dimmer)**, đấu **2 chỗ cùng lúc**:
  1. **Bên trong** `q_frc, q_spd, q_ẋ, q_ẏ` (ω nhân trong `exp`): đứng → ω≈0 → `exp(0)=1` → cost = 0 → **tắt** mấy phạt locomotion.
  2. **Qua hệ số** `(ω−1)` trên `q_standing_cost`: đứng → `(ω−1)≈−1` → **bật** phạt đứng-không-yên.

*Chỉ bài periodic rew dùng ω* (vì train 1 policy cho nhiều dáng + đứng). Humanoid gym và Unitree **không có ω** (mỗi bên một dáng cố định).

---

## 9. Công thức tổng `E[R_multi(s,φ)]`

```
E[R_multi] = 0.400 · E[R_bipedal(s,φ)]      ← đi đúng nhịp (quan trọng nhất)
           + 0.300 · E[R_cmd(s)]            ← nghe lệnh
           + 0.100 · E[R_smooth(s)]         ← đi êm
           + 0.100 · (ω−1) · q_standing_cost(s)   ← phạt đứng-không-yên (CÓ công tắc ω)
           + 0.100 · (−1)  · q_hop_sym(s)         ← phạt nhảy lệch (LUÔN bật)
           + 1                               ← baseline sống sót β
```
Đọc ELI5: **điểm = (cộng điểm làm đúng) − (trừ điểm làm sai) + (điểm vì còn sống)**.
- 3 dòng đầu: hệ số `0.4 > 0.3 > 0.1` → thứ tự ưu tiên.
- Dòng 4 **tự bật/tắt** theo ω; dòng 5 **luôn bật**.
- `+1`: giữ tổng luôn dương → robot không học kiểu "thà ngã cho xong".

**`q_standing_cost`:**
```
q_standing_cost = 1 − exp( −( err_sym + 20·q_action_diff ) )
```
`err_sym` = đứng cân/lệch, đúng tư thế không. `20·q_action_diff` = tay chân run giật không (×20 → lúc đứng mà giật bị phạt rất nặng).

---

## 10. R_cmd và R_smooth — hai "túi điểm trừ"

Lưu ý: hai cái này **KHÔNG phải điểm cộng**. Mỗi số hạng con có dấu `(−1)` → đều là **phạt**; càng làm đúng càng gần 0.

**R_cmd** — "làm đúng cái tao BẢO":
```
R_cmd = (−1)·q_ẋ + (−1)·q_ẏ + (−1)·q_orientation
```
đi đúng tốc độ tới / đúng tốc độ ngang / quay đúng hướng.

**R_smooth** — "làm cho ÊM, kệ đi đâu":
```
R_smooth = (−1)·q_action_diff + (−1)·q_torque + (−1)·q_pelvis_acc
```
không giật cục / không gồng hết lực / hông không rung lắc.

**Bảng q_i (khuôn chung `1 − exp(−k·lỗi)`):**

| q_i | Công thức | Có ω? |
|-----|-----------|:---:|
| q_left/right frc | `1 − exp(−ω·‖raw_foot_frc‖²/100)` | ✅ |
| q_left/right spd | `1 − exp(−2·ω·‖raw_foot_spd‖²)` | ✅ |
| q_ẋ | `1 − exp(−2·ω·\|ẋ_des − ẋ_act\|)` | ✅ |
| q_ẏ | `1 − exp(−2·ω·\|ẏ_des − ẏ_act\|)` | ✅ |
| q_orientation | `1 − exp(−3·(1 − (quat_actᵀ·quat_des)²))` | ❌ |
| q_action_diff | `1 − exp(−5·‖a_t − a_{t−1}‖)` | ❌ |
| q_torque | `1 − exp(−0.05·‖τ‖)` | ❌ |
| q_pelvis_acc | `1 − exp(−0.10·(‖pelvis_rot‖+‖pelvis_acc‖))` | ❌ |

**Quy tắc:** có ω = "chỉ bắt buộc khi đi; đứng thì thôi". Không ω = "lúc nào cũng phải tuân". → Trong R_cmd, chỉ `q_orientation` còn sống khi đứng; cả R_smooth luôn sống.

Khuôn `1 − exp(−k·lỗi)`: lỗi 0 → cost 0 (hoàn hảo); lỗi lớn → cost → 1 (bão hòa). `k` = độ nhạy.

---

## 11. Lý thuyết (paper) vs thực tế (code apex)

| Paper nói | Code thật làm |
|---|---|
| Làm mượt pha bằng **Von Mises** | Dùng **spline PCHIP** (`cassie/phase_function.py`, `create_phase_reward()`, 40Hz) |
| `R_i = c_i·I_i(φ)·q_i` | **ĐANG DÙNG: `np.tan(π/4 · clock · measurement)`** — cũng là cổng-nhịp × số đo. Bản `np.tanh` **bị comment** (xem mục 15). |
| 2 pha swing/stance | **4 pha**: right-swing → double-stance → left-swing → double-stance |
| ω cho standing | Trong `clock_rewards.py` **không thấy ω**; đứng tách **env riêng** (`CassieStanding-v0`, `standing_rewards.py`) |

Trọng số reward thật (`clock_rewards.py`): `0.2 frc + 0.2 vel + 0.2 orient + 0.15 pelvis + 0.15 com_vel + 0.05 hip_roll + 0.025 torque + 0.025 action`. Cùng họ với `E[R_multi]`, chỉ chia nhỏ & cân lại. Có bản `early_clock_reward` cho giai đoạn đầu (curriculum).

Thư mục `rewards/` có ~16 bản thử nghiệm → research thật thì lộn xộn, nhiều trial-and-error.

**Bài học lớn:** lý thuyết sạch ≠ code thật. Cùng mục tiêu (cổng nhịp mượt, bay-cấm-lực/trụ-cấm-trượt) nhưng cơ chế làm mượt khác nhau.

---

## 12. Kiến trúc policy: LSTM hay MLP?

Câu hỏi: nhịp có **bắt buộc** LSTM không? → **Không.** MLP chạy được, với điều kiện.

- **Nhịp cần được *biết*, không cần được *nhớ*.** Nếu **nhét `sin(2πφ), cos(2πφ)` vào observation** thì MLP (không trí nhớ) vẫn biết đang ở đâu trong nhịp → đủ. Chính Unitree làm vậy (MLP + sin/cos phase trong obs).
- LSTM trong apex/paper dùng cho **lý do khác**: robot thật không đo được vận tốc toàn cục, lực chân, ma sát, khối lượng (những thứ bị **dynamics randomization**). LSTM nhớ lịch sử để **đoán mấy thứ ẩn** này → xử lý **quan sát thiếu thông tin (POMDP)**. Không phải để đếm nhịp.

**MLP + frame stack vs LSTM:** cùng họ "dùng quá khứ", **không đồng nhất**.
| | Frame stack + MLP | LSTM |
|---|---|---|
| Nhớ xa | Cứng, đúng N bước | Không giới hạn |
| Cách nhớ | Chép nguyên xi N khung | Tóm tắt thành vector gọn, có cổng quên/nhớ |
| Input | Phình theo N | Cố định |
| Train | Dễ, ổn định | Khó, chậm hơn |

→ Nhớ **ngắn** (vận tốc = hiệu 2 khung; ma sát = vài khung) thì frame stack **xấp xỉ** LSTM. Nhớ **dài** hoặc muốn gọn nhẹ thì LSTM thắng. Locomotion phần lớn là nhớ ngắn → frame stack thường đủ.

---

## 13. So sánh nhanh 3 nguồn

| | periodic rew (Siekmann) | humanoid gym | Unitree (rl_gym/rl_lab) |
|---|---|---|---|
| Đồng hồ `p = (step·dt mod T)/T` | ✅ | ✅ | ✅ |
| Làm mượt pha | Von Mises (xác suất) | periodic stance mask I_p(t) | ngưỡng cứng `phase < 0.55` |
| Reward tiếp xúc | c·I(φ)·q (force/speed) | theo mask pha | **XNOR**(contact, is_stance) |
| ω standing switch | ✅ (1 policy nhiều dáng + đứng) | ❌ | ❌ |
| Policy | LSTM | — | MLP (+ sin/cos phase trong obs) |

---

## 14. Đọc source: `phase_function.py` (chặng 1 — xưởng đúc "đồng hồ")

File này là **cái máy DỰNG đồng hồ** — đẻ ra đúng mấy đường cong `E[C_frc(φ)]`, `E[C_spd(φ)]` ở Hình 3, nhưng bằng **spline PCHIP** chứ không phải Von Mises.

Hàm lõi: `create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, have_incentive, FREQ=40, for_viz=False)`.

**5 tham số (ELI5):**
| Tham số | ELI5 |
|---|---|
| `swing_duration`, `stance_duration` | Bay bao lâu / trụ bao lâu (giây). Quyết định nhịp dài cỡ nào. |
| `strict_relaxer` | **Núm chỉnh độ nghiêm của ranh giới.** Thụt 2 mép mỗi pha vào trong → tạo đoạn dốc (ramp) thay cho bậc thang. Nhỏ = ranh giới sắc; lớn = mờ/mượt. **Là nghịch đảo của κ** (Von Mises): κ lớn↔sắc, strict_relaxer nhỏ↔sắc. |
| `stance_mode` | Kiểu trụ: `grounded` / `aerial` / `zero` (có `encode/decode_stance_mode`). |
| `have_incentive` | Có **thưởng (+1)** hay chỉ **phạt (−1/0)**. Dạy chó: chỉ phạt lỗi, hay vừa phạt lỗi vừa thưởng ngoan. |

**Cấu trúc trong ruột:**
- `phase_points` shape **(2, 8)**: hàng 0 = x (mốc thời-gian-pha), hàng 1 = y (hệ số `−1 / 0 / +1`). **8 cột = 4 pha × 2 đầu mút**.
- **4 pha** (khác paper 2 pha): `right-swing → double-stance → left-swing → double-stance`.
- Mỗi pha: `offset = (slice[1] − slice[0]) · strict_relaxer`; đặt lại 2 mép thành `mép_đầu + offset`, `mép_cuối − offset` → **thụt vào trong** → sinh đoạn dốc mượt ở chỗ nối (đây chính là mấy dòng nhân `strict_relaxer` lặp đi lặp lại).
- Đúc **4 spline**: `r_frc, r_vel, l_frc, l_vel` (force + vel × phải + trái).
- Dùng **`PchipInterpolator`** (PCHIP = nội suy Hermite bậc 3 từng khúc). Chọn nó vì **KHÔNG vọt lố (no overshoot)** → giữ hệ số nằm gọn trong `[−1, +1]`. Cubic spline thường thì vọt lố ra ngoài.
- **Kéo dài 3 chu kỳ** (prev + current + next rồi `hstack`) trước khi nội suy → spline **tuần hoàn mượt** ở chỗ nối đầu↔cuối, không gãy.
- Trả về: `[r_frc_spline, r_vel_spline], [l_frc_spline, l_vel_spline], phaselength`.

➡️ Xong chặng này: ta có 4 "cây kim đồng hồ", gọi `spline(phase)` ra hệ số `∈ [−1,+1]` tại bất kỳ pha nào.

---

## 15. Đọc source: `clock_rewards.py` (chặng 2 — lắp reward cuối)

File này **dùng đồng hồ** (4 spline trên) để **chấm điểm**. Hàm chính `clock_reward` làm 3 bước:

**Bước 1 — Đọc & chuẩn hóa số đo:**
```
normed_foot_frc = foot_frc / desired_max_foot_frc   # 250 N
normed_foot_vel = foot_vel / desired_max_foot_vel   # 2.0 m/s
```
Chia cho mức tối đa mong muốn → kéo số đo về cỡ `~[0,1]` để bước sau `tan` không nổ.

**Bước 2 — Lấy giá trị đồng hồ rồi chấm điểm:**
```
left_frc_clock = self.left_clock[0](self.phase)        # gọi spline → ∈[−1,+1]
left_frc_score = np.tan(np.pi/4 · left_frc_clock · normed_left_frc)   # ĐANG DÙNG
# left_frc_score = np.tanh(...)   ← BỊ COMMENT
```
- Vì sao `tan(π/4 · x)`: khi `x ∈ [−1,1]` thì `π/4·x ∈ [−π/4, π/4]`, `tan` ra `∈ [−1,+1]`. Map `[−1,1] → [−1,1]` nhưng **cong** (nhạy hơn ở gần hai mép ±1).
- **Insight cốt lõi:** `clock` giữ **DẤU** (−1 phạt / +1 thưởng / 0 kệ), `measurement` giữ **ĐỘ LỚN**. Tích của chúng chính là `R_i = I(φ)·q(s)`. (Chính chỗ bạn nhầm "0.8": 0.8 là measurement, không phải clock.)

**Bước 3 — Tổng có trọng số, 8 số hạng, cộng lại = 1.0:**
```
reward = 0.200·foot_frc_score + 0.200·foot_vel_score
       + 0.200·exp(−(com_orient_err + foot_orient_err))
       + 0.150·exp(−pelvis_motion) + 0.150·exp(−com_vel_err)
       + 0.050·exp(−hip_roll_penalty) + 0.025·exp(−torque_penalty)
       + 0.025·exp(−action_penalty)
```

**Hai LOẠI số hạng — phải phân biệt:**
| Loại | Gồm | Dấu | Vai trò |
|---|---|:---:|---|
| **foot** (clock-gated) | `foot_frc`, `foot_vel` | **CÓ THỂ ÂM** | Cổng theo nhịp. **Phạt thật duy nhất** — cái đẻ ra dáng đi. |
| **regularizer** | 6 cái còn lại | **chỉ ≥ 0** | Khuôn `exp(−error) ∈ (0,1]`. Chỉ thưởng: càng ít lỗi càng gần 1. |

**Đảo gói (phải nhớ):** code dùng `exp(−error)` → **thưởng**, `∈ (0,1]`. Paper dùng `1 − exp(−k·error)` → **phạt**, `∈ [0,1)`. **Cùng một thông tin, gói ngược nhau.**

**Không có ω** trong `clock_rewards` → đứng yên là **env riêng**, không nhét chung vào đây.

(Còn `early_clock_reward`, `no_speed_clock_reward`, `aslip_clock_reward`, `max_vel_clock_reward` = các biến thể curriculum / thử nghiệm.)

---

## 16. Đã "bê" phase_function + clock_rewards sang TransformBipedal_Simulation chưa?

**Chưa** — grep `PchipInterpolator / create_phase_reward / clock_reward` trong project = **rỗng**.

Hai task đi bộ hiện có (`officialdesign/task_walk.py`, `newsimple/task_walk.py`) **đều không có phase clock**: không `self.phase`, không `sin/cos`, không swing/stance theo thời gian. Chúng ép dáng bằng **tiếp xúc thật** (`~touching`, `air_time > 0`) — tức là đọc chân có chạm đất không, chứ không xem đồng hồ.

**"Bê sang" ở đây = THÊM cái đang thiếu, không phải thay cái cũ.** Và vì:
1. Khác framework: apex là **numpy + scipy PCHIP** chạy **1 robot trên CPU**; project là **Isaac Lab + torch** chạy **hàng nghìn robot trên GPU** → **không copy thẳng được**, phải **viết lại bằng tensor torch** (kiểu Unitree `g1_env`).
2. Obs hiện 44D (IMU + lịch sử action) **không có thông tin pha** → nếu thêm reward theo nhịp mà không thêm `sin/cos(pha)` vào obs thì **policy bị chấm điểm theo một biến nó không nhìn thấy** (POMDP) → học không ra nhịp. Nối lại bài học mục 12.

---

## Việc tiếp theo (gợi ý)

- ~~Đọc `phase_function.py` từng dòng~~ ✅ (mục 14) · ~~đọc `clock_rewards.py`~~ ✅ (mục 15).
- Vẽ/kiểm `E[C_spd(φ)]` (bản tốc độ, ngược với Hình 3) — chưa làm.
- **Quyết định đường port cho `TransformBipedal_Simulation`:** port đầy đủ periodic-rew (ω, nhiều dáng) hay nâng cấp reward tiếp-xúc có sẵn thành có-pha? Dù chọn đường nào, nếu thêm reward theo nhịp thì **phải thêm `sin/cos(pha)` vào obs** (mục 16).
- Nếu port: viết lại `phase_function` thành hàm torch (vector hóa theo `num_envs`), bỏ scipy.
