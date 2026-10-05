# ALGO.md — Toàn bộ cách robot OFFICIALdesign học đi

Người lập: Claude (Tech Lead). Cập nhật: 2026-10-05.
Robot: OFFICIALdesign, 10 DOF. Task `Official-Walk-v0`.
File code chính: `bipedal/officialdesign/task_walk.py`, `bipedal/officialdesign/robot.py`, `bipedal/officialdesign/ppo.py`
(cùng thư mục `isaac_rl/bipedal/`).

> File này đã gộp toàn bộ `docs/RL_CONFIG.md` cũ (rà soát 2026-08-16, viết cho env 10DOF cũ). Nội dung dùng được đã cập nhật theo env mới. Mẫu code config rsl_rl nằm ở **Phụ lục A**, lỗi riêng của env cũ ở **Phụ lục B**. Bản gốc RL_CONFIG.md vẫn lấy lại được từ git.

---

## 0. Bản đồ

### 0.1 Ba tầng, giải thích bằng ví dụ dạy trẻ đi xe đạp

| Tầng | Ví dụ | Câu hỏi nó trả lời | Sửa ở đâu |
|---|---|---|---|
| **1. Cách học** | Phương pháp học: mỗi lần sửa một chút, có thầy chấm điểm | Mạng nơ-ron cập nhật thế nào? | `bipedal/officialdesign/ppo.py` (config rsl_rl) |
| **2. Bài tập** | Tập gì, có bánh phụ không, làm đúng được khen gì | Robot thấy gì, ra lệnh gì, được thưởng vì cái gì? | `bipedal/officialdesign/task_walk.py`: obs / action / reward |
| **3. Sân tập giống sân thật** | Sân có dốc, có gió, xe hơi rơ y như đường thật | Policy học trong sim có dùng được ngoài đời không? | `bipedal/officialdesign/task_walk.py` (DR, delay, IMU) + code trên Pi |

Tầng 1 **không biết gì về robot**: cùng một config dùng được cho robot 4 chân. Tầng 2 và tầng 3 là nơi đặc thù của robot này.

**Nguyên tắc: không fork, không sửa thư viện rsl_rl.** Mọi cơ chế tầng 1 đều bật được bằng config. Mọi thứ tầng 2 và 3 đều nằm trong env. Nhờ vậy nâng cấp thư viện không làm vỡ dự án.

### 0.2 Tóm tắt quyết định
- **Tầng 1:** giữ PPO + MLP. Bật asymmetric critic, symmetry, chuẩn hóa obs cho critic. Tăng `num_envs`, giảm `init_std`.
- **Tầng 2:** thêm gait clock trước (B5). CPG + residual RL (B9) chỉ làm khi bị kẹt ở giới hạn tốc độ servo (mục 2.4.2). ZMP chỉ dùng làm reward.
- **Tầng 3:** đo servo và độ trễ trước. Khôi phục domain randomization (env mới đang thiếu gần hết). Delay riêng từng chân. Gộp 2 IMU.

### 0.3 Phần cứng (không thay đổi được)

Robot gồm 2 module chân riêng. Mỗi module từng là xe + tay máy, nay dock (ghép) lại thành biped.

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Khớp | 10, mỗi chân 5: Bub, Hip, rotate, Knee, Foot | `assets/officialdesign/meta/calibration.json` |
| Khối lượng | 4.643 kg | `sim_handoff /SIM_HANDOFF.md` |
| Servo nặng | STS3095, khoảng 10.3 Nm stall: Bub, Hip, Knee | `calibration.json` |
| Servo nhẹ | STS3215, khoảng 2.94 Nm stall: rotate, Foot | `calibration.json` |
| IMU | 2 con, `IMUleft`/`IMUright`, gắn cứng vào Baselink, y = ±16.5 mm, cùng hướng | URDF |
| Máy tính | 2 Raspberry Pi, mỗi Pi lo 1 chân (1 bus servo + 1 IMU) | user |

Thuật ngữ:
- **Servo bus** (Feetech STS): nhiều servo nối chung một dây tín hiệu. Mình ra lệnh "quay tới góc X", bộ PID bên trong servo lo phần còn lại. **Không ra lệnh lực (torque) trực tiếp được.**
- **Stall torque**: lực xoắn lớn nhất khi servo bị giữ đứng yên. Lúc đang quay thì lực thấp hơn.

### 0.4 Cái gì lên robot thật, cái gì chỉ để train

**Quy tắc:** thứ gì là **đầu vào của actor** hoặc **một phần của action** lúc train thì bắt buộc phải có lúc deploy, và phải tính **giống hệt** sim. Thứ chỉ dùng để "mớm" lúc train thì chỉ được nằm ở **reward** hoặc **critic**.

Mẹo phân loại: thứ đó có nằm trên đường đi từ cảm biến tới servo không? Có thì phải deploy. Không thì bỏ được.

| Chỉ dùng lúc train (deploy bỏ đi) | Phải mang lên robot thật |
|---|---|
| Critic (1.2) | Actor (file ONNX) |
| Reward, kể cả ZMP reward (2.5) | **Gait clock**, vì pha là đầu vào của actor (2.3) |
| Contact, obs đặc quyền của critic (2.6) | **CPG + bộ sinh quỹ đạo**, vì góc mẫu là một nửa của lệnh góc (2.4) |
| Symmetry augmentation, domain randomization | Bộ gộp 2 IMU (3.5), giới hạn góc/tốc độ, framestack |

---

# TẦNG 1 — THUẬT TOÁN HỌC (rsl_rl / PPO)

## 1.1 Chọn thuật toán

| Thuật toán | Nói đơn giản | Hợp không? | Lý do |
|---|---|---|---|
| **PPO + MLP** | Cho hàng nghìn robot ảo tập đi. Làm đúng được thưởng, ngã bị phạt. | ✅ **Lõi** | Output là góc khớp, khớp với servo vị trí. Xuất ONNX nhẹ, chạy trên Pi được. |
| **SAC + MLP** | RL học lại từ kinh nghiệm cũ, tiết kiệm mẫu. | ❌ | Isaac chạy song song nên dư dữ liệu. Khi đó PPO nhanh và ổn định hơn. |
| **Whole-body MPC** | Mỗi vài ms robot "nhìn trước tương lai" bằng mô hình vật lý, rồi tính lực cho từng khớp. | ❌ | Cần điều khiển torque, mô hình khối lượng chính xác, vòng lặp ≥200 Hz. Không có cái nào. |
| **ZMP** | Giữ "điểm đè" luôn nằm trong bàn chân thì không ngã. | ⚠️ Phụ | Chỉ dùng trong sim làm reward (mục 2.5). |
| **CPG** | Bộ tạo nhịp sinh sẵn quỹ đạo chân đều nhịp. | ✅ Bổ trợ | Thuộc tầng 2 (mục 2.4). |
| **Policy riêng mỗi chân** (decentralized) | Mỗi chân một bộ não. | ❌ | Biped cần hai chân phối hợp rất chặt để giữ thăng bằng. |

## 1.2 PPO là gì

Robot có **hai bộ não** tách rời:
- **Actor** (diễn viên): nhìn tình hình rồi quyết định nhúc nhích khớp. **Chỉ actor được đem lên robot thật.**
- **Critic** (giám khảo): chấm "tình thế này về sau tốt hay xấu". Critic chỉ dùng để dạy actor, deploy thì bỏ đi.

Vòng lặp:
1. Cho N robot chạy 24 bước để thu dữ liệu.
2. Critic chấm từng khoảnh khắc.
3. Hành động **tốt hơn mức critic dự đoán** thì tăng xác suất lặp lại. Tệ hơn thì giảm.
4. Lặp lại.

**"P" = Proximal (gần):** mỗi lần chỉ sửa não *một chút*. RL rất dễ sập: sửa mạnh tay một lần là policy đang đi tạm được có thể "quên sạch" và không hồi phục, vì dữ liệu tiếp theo lại do chính policy hỏng đó sinh ra. `clip_param = 0.2` là sợi dây xích giữ xác suất mới không lệch quá ±20% so với cũ.

Một iteration chạy như sau:
```
THU THẬP   N env × 24 bước → (obs, action, reward, done)
           hết giờ (không phải ngã) → reward += γ·V(s)      ← time-limit bootstrapping
ADVANTAGE  δ = r + γ·V(s') − V(s);  A = δ + γ·λ·A (duyệt ngược); chuẩn hóa A
TỐI ƯU     5 epoch × 4 minibatch = 20 bước gradient
           L = L_actor(clip 0.2) + 1.0·L_critic(clip) − 0.01·entropy
           đo KL → tự chỉnh learning rate; clip grad norm 1.0
```

## 1.3 Hyperparameter: hiện tại và đề xuất

`official_ppo_cfg.py` kế thừa `rsl_rl_ppo_cfg.py` (chỉ đổi `experiment_name` và `max_iterations = 3000`).

| Tham số | Hiện tại | Đề xuất | Nghĩa / lý do |
|---|---:|---:|---|
| `scene.num_envs` | **256** | **2048–4096** | Hiện chỉ 256 × 24 = 6 144 mẫu mỗi lần update, nên gradient nhiễu. Chuẩn legged_gym khoảng 98k. **Đây là đòn bẩy lớn nhất.** (Có thể đặt bằng `--num_envs` khi train.) |
| `num_steps_per_env` | 24 | 24 | 1.2 s kinh nghiệm mỗi robot mỗi vòng. |
| `gamma` | 0.99 | 0.99 | Chiết khấu. Chân trời ≈ 1/(1−γ) = 100 bước = **5 s** ở 20 Hz. Đổi nếu đổi tần số (mục 1.6). |
| `lam` (GAE) | 0.95 | 0.95 | Trộn "tin số liệu thật" (đúng nhưng nhiễu) với "tin dự đoán critic" (mượt nhưng có thể sai). |
| `clip_param` | 0.2 | 0.2 | Dây xích ±20%. |
| `use_clipped_value_loss` | True | True | Critic cũng bị xích. |
| `learning_rate` | 1e-3 | 1e-3 | Chỉ là giá trị khởi đầu, bị adaptive ghi đè. |
| `schedule` / `desired_kl` | adaptive / 0.01 | giữ | KL đo "não mới khác não cũ bao nhiêu". Đổi nhiều thì giảm LR, ít thì tăng. **Muốn học nhanh/chậm hơn thì chỉnh `desired_kl`, không chỉnh `learning_rate`.** |
| `num_learning_epochs` × `num_mini_batches` | 5 × 4 | giữ | 20 bước gradient mỗi vòng. |
| `entropy_coef` | 0.01 | 0.01 → 0.005 nếu robot rung | Thưởng cho việc "còn phân vân", chống chốt sớm vào một dáng đi tệ. |
| `max_grad_norm` | 1.0 | 1.0 | Chặn update đột biến. |
| `init_std` | **1.0** | **0.5** | Action bị clip [-1, 1]. Với std 1.0, khoảng 1/3 action ngẫu nhiên bị chặn ở biên, nên robot tập "giật cục". |
| `hidden_dims` | [256, 256, 128] ELU | giữ | Khoảng 115k tham số mỗi mạng. Infer trên Pi < 1 ms. |
| `obs_normalization` | False | critic: **True**; actor: tùy | Xem mục 1.4. |

## 1.4 Các cơ chế của rsl_rl: dùng gì

Đã kiểm tra: IsaacLab đang cài có đủ `RslRlSymmetryCfg`, `RslRlDistillation*`, `RslRlRndCfg`, `RslRlRNNModelCfg`.

| Cơ chế | Quyết định | Giải thích |
|---|---|---|
| **Time-limit bootstrapping** | ✅ Tự chạy | Hết giờ không phải là ngã, nên không bị phạt. Env đã trả đúng cờ `time_out`. |
| **Asymmetric actor-critic** (`obs_groups`) | ✅ **Làm sớm** | Critic xem thông tin đặc quyền (privileged: thứ sim biết mà robot thật không đo được). Critic chấm chính xác hơn → actor học nhanh hơn. Actor không đổi. Hiện env chỉ trả `{"policy": ...}`, nên critic mù ngang actor. Nội dung nhóm `critic` xem mục 2.6. |
| **Symmetry augmentation** (`symmetry_cfg`) | ✅ Làm, **phải test** | Lật mỗi mẫu trái↔phải → dữ liệu gấp đôi, dáng đi cân hai bên. Hàm lật phụ thuộc layout obs (mục 2.7). Bật `use_data_augmentation`. `mirror_loss` để nhẹ hoặc tắt, vì CAD không đối xứng hoàn hảo. |
| **Obs normalization** (chuẩn hóa) | ⚠️ Critic: bật. Actor: tùy | Obs actor đã tự đưa về [-1, 1], nên lợi ích nhỏ. Obs critic lẫn nhiều thang đo, nên bật. Nếu bật cho actor, phải xác nhận file ONNX chứa bộ chuẩn hóa (`exporter.py` có hỗ trợ). |
| **Orthogonal init** | ⚪ Thấp | rsl_rl không gọi hàm init này, nhưng vẫn train tốt. Giảm `init_std` cho tác dụng gần giống. |
| **Recurrent** (LSTM/GRU, mạng có trí nhớ) | ❌ Chưa | Phải quản lý hidden state (trạng thái nhớ) khi chạy ONNX trên Pi. Framestack + gait clock là đủ. |
| **Distillation** (teacher → student) | ⏳ Để sau | Teacher thấy hết, student bắt chước chỉ bằng cảm biến thật. Mạnh hơn asymmetric critic. Chỉ làm khi asymmetric critic + DR vẫn không qua được sim-to-real. |
| **RND** (thưởng tò mò) | ❌ | Reward đã dày. RND dành cho bài toán reward thưa. |
| **State-dependent std** | ❌ | Phức tạp thêm mà lợi ích không rõ. |
| **CNN / multi-GPU** | ❌ | Không có camera, chỉ 1 GPU. |

## 1.5 Framestack (số frame lịch sử trong obs)

- **Là gì:** MLP không có trí nhớ, nên nhét N frame gần nhất vào input để nó tự suy ra xu hướng ("đang nghiêng thêm hay đang gượng lại").
- **Hiện tại:** 4 frame × 50 ms = **200 ms**.
- **Lên 8 frame chưa nên làm ngay.**
  - Phần lịch sử action là lệnh policy tự gửi, ít thông tin mới. Nó chỉ giúp biết "lệnh nào còn đang bay" do trễ.
  - Lịch sử IMU (và góc servo nếu thêm) mới đáng giá. Nếu tăng, hãy **stack lệch**: IMU 6–8 frame, action 2–3 frame.
  - Khi đã có gait clock và góc servo, policy cần ít lịch sử hơn.
- **Thứ tự:** giữ 4 → thêm gait clock + góc servo → ablation (thí nghiệm bỏ/thêm một thành phần để đo tác dụng) 4 vs 8.
- Đổi framestack là đổi shape obs, nên checkpoint cũ không dùng được nữa.

## 1.6 Tần số điều khiển: 20 Hz hay 50 Hz

- Biped thường chạy **50 Hz**. Ở 20 Hz, mỗi 50 ms robot mới "nhìn" một lần, nên phản xạ giữ thăng bằng chậm.
- Ngân sách trễ ước tính của hệ 2 Pi là 5–15 ms, nên 50 Hz (chu kỳ 20 ms) khả thi.
- **Quyết sau khi đo A4.** Trễ end-to-end < 12 ms thì cân nhắc 50 Hz.
- Đổi tần số kéo theo: `decimation` 10 → 4, `action_step_deg`, `gamma` (0.99 ở 50 Hz ≈ 2 s, chuẩn legged_gym), số frame history, `episode_length`.

## 1.7 Đọc TensorBoard

| Chỉ số | Ý nghĩa | Dấu hiệu tốt |
|---|---|---|
| `Train/mean_episode_length` | Robot trụ được bao lâu | Tăng đều, tiến tới 200 |
| `Train/mean_reward` | Tổng thưởng | Tăng; đọc kèm episode_length |
| `Loss/value` | Critic đoán sai bao nhiêu | Giảm rồi ổn định |
| `Loss/surrogate` | Loss của actor | Dao động quanh 0, không phân kỳ |
| `Loss/learning_rate` | LR tự chỉnh | Nhảy lung tung là **bình thường** |
| `Policy/mean_noise_std` | Độ ngẫu nhiên | Giảm dần |

`mean_noise_std` tụt nhanh về ~0 mà `episode_length` không tăng nghĩa là policy chốt sớm vào một dáng tệ. Khi đó tăng `entropy_coef`.

---

# TẦNG 2 — THIẾT KẾ CHUYỂN ĐỘNG (obs / action / reward)

## 2.1 Giao diện hiện tại của env

| Đại lượng | Giá trị |
|---|---|
| Vật lý | `dt = 0.005` (200 Hz) |
| Policy | `decimation = 10` → **20 Hz** |
| Episode | 10 s = 200 bước |
| **Action** | 10 số trong [-1, 1]. Mỗi bước góc lệnh thay đổi `action × 2°`, rồi clamp vào biên khớp. Tối đa **40°/s**. |
| **Obs actor (60)** | 4 frame × [roll, pitch, gyro xyz / 2] + 4 frame × 10 lệnh góc đã chuẩn hóa [-1, 1] |
| Lệnh đi | +x, 0.15 m/s, cố định (policy không cần đoán hướng) |
| Kết thúc | thân thấp hơn 0.20 m, hoặc nghiêng quá 0.9 rad |

Biên khớp (độ, hệ tọa độ policy). Đây là "training envelope" (vùng cho phép khi train), **chưa phải giới hạn cơ khí đã đo**:

| Khớp | min | max | default | Nhóm |
|---|---:|---:|---:|---|
| Bub L/R | −12 | 12 | 0 | nặng |
| Hip L/R | −15 | 45 | 20 | nặng |
| rotate L/R | −15 | 15 | 0 | nhẹ |
| Knee L/R | −75 | 0 | −40 | nặng |
| Foot L/R | −25 | 45 | 20 | nhẹ |

## 2.2 Reward hiện tại

Tổng reward nhân với `step_dt`.

| Thành phần | Hệ số | Mục đích |
|---|---:|---|
| Bám vận tốc x = 0.15 | +1.5 (Gauss σ 0.2) | Đi về phía trước |
| Thân thẳng (roll, pitch) | +0.5 | Không nghiêng |
| Chiều cao thân | +0.5 (Gauss σ 0.06) | Không khuỵu |
| Chân nhấc khoảng 3.5 cm khi không chạm đất | +0.15 | Không lê chân |
| vy², wz², yaw², lệch y | −0.5, −0.1, −0.2, −0.2 | Đi thẳng |
| Trượt chân khi đang chạm đất | −0.02 | Chống trượt |
| Torque² | −0.001 | Tiết kiệm servo |
| **`action_rate`** (đổi action giữa hai bước) | −0.02 | **Lệnh mượt (đã có)** |
| Lệch khỏi tư thế mặc định | −0.05 | Không đi tư thế quái |

**Đang thiếu:** reward theo **pha bước** (chân nào nên chạm đất lúc nào) và reward thăng bằng theo CoM. Hai mục dưới bổ sung.

### 2.2.1 `swing` đang thưởng NGƯỢC với dáng đi

Số hạng `+0.15 swing` ở bảng trên không trung lập với dáng đi — nó nghiêng về
phía nhảy. Code hiện tại (`bipedal/officialdesign/task_walk.py`, trong `_get_rewards`):

```python
swing = ((~touching) * torch.exp(-((sole_height - 0.035)/0.025)**2)).mean(-1)
```

`.mean(-1)` lấy trung bình trên **hai** bàn chân. Với bàn chân ở đúng 3.5 cm:

| Tình huống | Phép tính | `swing` |
|---|---|---:|
| Hai chân bay (nhảy) | `(1.0 + 1.0) / 2` | **1.00** |
| Một chân bay (bước đi) | `(1.0 + 0) / 2` | **0.50** |
| Hai chân chạm đất | `(0 + 0) / 2` | 0.00 |

Nhảy được điểm **gấp đôi** bước đi.

Hệ số chỉ 0.15 nên số hạng này không áp đảo. Nhưng không có số hạng nào khác
nói "phải có một chân chạm đất", mà "nhảy về phía trước" lại thoả mãn cả ba số
hạng lớn: bám vận tốc x, thân thẳng, và chiều cao thân (nhảy thì lên cao chứ
không khuỵu). Ngưỡng kết thúc (`min_base_height 0.20`, `max_tilt 0.9`) cũng
không chặn.

### 2.2.2 `march_alt` — bản vá 3 dòng

Nhặt từ env "Twist + March" của Hiếu, nay ở
`isaac_rl/_archive/newsimple/transformer_hieu_env.py` (đóng băng 2026-10-05,
không chạy được vì relative import gãy — chỉ để đọc).
Hiếu cho nó trọng số 2.0 — hạng mục chính, không phải phụ.

```python
in_air    = air_time > 0.05            # (N, 2) bool, cần track_air_time=True
n_air     = in_air.sum(dim=1).float()
march_rew = torch.where(n_air == 1, torch.ones_like(n_air),              # bước đi
            torch.where(n_air == 0, torch.full_like(n_air, -0.3),        # đứng ì
                                    torch.full_like(n_air, -1.0)))       # nhảy
```

Hình dạng ngược hẳn với `swing`: nó thưởng đúng cái `swing` đang phạt.

**Quan hệ với §2.3 (gait clock):** gait clock là lời giải tốt hơn về lâu dài vì
nó cho policy biết *đang ở đâu trong chu kỳ* (đổi observation, 60D → 62D, phải
train lại từ đầu và đổi cả firmware nhúng). `march_alt` chỉ đổi reward, giữ
nguyên giao diện 60/10, dùng được ngay. Hai thứ không loại trừ nhau — có gait
clock rồi thì `march_alt` thành số hạng dư, bỏ đi được.

### 2.2.3 Hai mẹo khác từ cùng nguồn

**Đồng hồ chống đứng ì.** Hiện ta chống đứng ì *gián tiếp* qua bám vận tốc.
Đồng hồ phạt theo **thời gian liên tục** hai chân cùng chạm đất, nên bắt được
kiểu "nhúc nhích tại chỗ cho có vận tốc tức thời":

```python
self.static_timer = torch.where(n_air == 0, self.static_timer + self.step_dt,
                                torch.zeros_like(self.static_timer))
static_rew = torch.where(self.static_timer > NGUONG, -1.0, +0.5)
```

**Khoá reward theo điều kiện — chống reward hacking.** Hiếu khoá số hạng phụ,
chỉ tính khi robot còn đứng:

```python
# CHỈ tính khi robot đang đứng — tránh học cách ngã để xoay twist
twist_rew = torch.where(standing, raw_prog + done_bonus, 0)
```

Không khoá thì agent phát hiện "ngã xuống làm việc này dễ hơn nhiều" và đi
thẳng tới đó. Áp dụng được cho mọi số hạng phụ: `swing` chẳng hạn, nên khoá
theo `height > ngưỡng` để robot không ăn điểm swing trong lúc đang đổ.


## 2.3 Gait clock (đồng hồ nhịp bước) — thay đổi OBS

**“Pha” chỉ có nghĩa là: đang ở đâu trong một vòng động tác lặp lại.** Hãy tưởng tượng đếm nhịp: “nhấc trái → đặt trái xuống → nhấc phải → đặt phải xuống → lặp lại”. Ở đây, **một chu kỳ** là trọn một vòng trái–phải, từ một thời điểm đến khi động tác của cùng chân bắt đầu lặp lại.

Ta dùng một số `p` để chỉ tiến độ của vòng đó:

- `p = 0`: bắt đầu vòng.
- `p = 0.25`: đã qua 1/4 vòng.
- `p = 0.5`: đã qua nửa vòng.
- `p = 0.75`: đã qua 3/4 vòng.
- Khi hết vòng, `p` quay về `0` để bắt đầu vòng tiếp theo.

Số này gọi là **pha chuẩn hóa**, chạy từ `0` đến sát `1`. Nó là tiến độ thời gian, **không phải góc khớp hay góc nghiêng của robot**.

**Ví dụ dễ hình dung:** giả sử một vòng dài 1 giây, ta đặt lịch như sau. Đây chỉ là lịch minh họa, chưa phải thông số đã chọn cho robot:

| Thời gian trong vòng | Pha `p` | Chân trái nên làm gì? | Chân phải nên làm gì? |
|---|---|---|---|
| 0–0.1 giây | 0–0.1 | Chạm đất, chuẩn bị chuyển trọng lượng sang chân phải | Chạm đất, nhận trọng lượng |
| 0.1–0.5 giây | 0.1–0.5 | Nhấc lên, đưa về trước, rồi hạ xuống để chạm đất ở cuối khoảng | Chống đất, đỡ thân |
| 0.5–0.6 giây | 0.5–0.6 | Chạm đất, nhận trọng lượng | Chạm đất, chuẩn bị chuyển trọng lượng sang chân trái |
| 0.6–1 giây | 0.6–1 | Chống đất, đỡ thân | Nhấc lên, đưa về trước, rồi hạ xuống để chạm đất ở cuối khoảng |

Trong một chu kỳ, mỗi chân có **giai đoạn chống** (*stance*: chân chạm đất để đỡ thân) và **giai đoạn vung** (*swing*: chân rời đất để bước sang chỗ mới). Có thể gặp cách gọi “pha chống”, “pha vung”: đó là **các khoảng trong chu kỳ**, còn `p` là con số cho biết hiện đang đến khoảng nào. Khoảng cả hai chân chạm đất gọi là **chống kép**.

**Vậy “nhấc chân đúng pha” là sao?** Theo lịch ví dụ:

- Ở `p = 0.3`, đang đến lượt chân trái vung: chân trái nên rời đất, chân phải nên chống. Nếu chân trái vẫn kéo lê trên sàn thì chưa làm đúng lịch.
- Ở `p = 0.8`, đang đến lượt chân phải vung: chân phải nên rời đất, chân trái nên chống.
- Đúng pha không có nghĩa là giữ chân thật cao suốt khoảng vung. Chân cần nhấc lên, đi qua phía trước, rồi hạ xuống đúng lúc để nhận trọng lượng.

**Đồng hồ tạo pha bằng cách nào?** Chương trình tự tăng `p` theo thời gian:

```text
p_mới = (p_cũ + Δt / T) modulo 1
T: thời gian một vòng trái–phải; Δt: thời gian giữa hai lần cập nhật.
“modulo 1”: hết một vòng thì quay lại đầu vòng.
```

Ví dụ `T = 1 giây`, cập nhật ở 20 Hz (`Δt = 0.05 giây`), thì mỗi lần `p` tăng `0.05`. Giảm `T` nghĩa là đếm nhịp nhanh hơn. Máy tính tạo được đồng hồ này ở cả sim và robot thật, không cần cảm biến tiếp đất.

**Vai trò của pha đối với policy và reward:**

1. **Đưa lịch vào đầu vào của policy.** Policy là mạng chọn lệnh góc khớp. Ngoài thông tin tư thế/chuyển động, nó được biết hiện đang đến đoạn nào của nhịp bước. Nếu không có đồng hồ, policy phải tự học cách tạo và duy trì nhịp từ những thông tin còn lại.
2. **Dùng lịch để chấm điểm khi train.** Sim đo chân thực sự có chạm đất hay không, rồi so với lịch mong muốn. Ví dụ ở `p = 0.3`, thưởng cho chân trái vung và chân phải chống đúng lịch; vẫn cần các reward khác để học giữ thăng bằng, tiến về trước và tránh trượt chân. Đúng nhịp riêng nó chưa đủ để đi được.

Để đưa pha vào mạng, dùng hai số **`sin(2πp)` và `cos(2πp)`**. Có thể hình dung chúng là hai tọa độ của đầu kim đồng hồ chạy quanh một vòng tròn. Gần cuối vòng và đầu vòng là hai vị trí rất gần nhau; cách biểu diễn này giúp mạng thấy sự liên tục đó, thay vì thấy số `p` đột ngột nhảy từ gần `1` về `0`. `2πp` là góc tính bằng radian tương ứng với tiến độ `p`.

**Giới hạn cần hiểu:** đồng hồ cho biết chân nào **nên** chạm đất theo lịch, không cho biết chân nào **đang thực sự** chạm đất. Nếu chân bị vướng, đồng hồ vẫn chạy. Nó không thay thế cảm biến contact và không tự bảo đảm robot giữ thăng bằng. Khi chỉ thêm gait clock, đồng hồ cũng chưa sinh lệnh góc khớp; policy vẫn phải học cách cử động để làm theo nhịp.

### 2.3.1 Nhược điểm của gait clock

1. **Nhịp cố định, không phản ứng.** Bị đẩy thì robot nên bước gấp một bước, nhưng đồng hồ vẫn đếm đều. Cách vá: cho policy thêm 1 action chỉnh tốc độ đồng hồ.
2. **Đứng yên vẫn dậm chân.** Đồng hồ chạy liên tục nên robot luôn bị "bảo" phải nhấc chân. Xử lý ở 2.3.4.
3. **Chọn sai `T` thì robot không theo kịp.** Khớp đổi tối đa 40°/s (2.1). Lịch đòi nhấc chân nhanh hơn mức đó thì robot bị phạt mãi dù cố hết sức, nên học rất tệ. Kiểm tra trước khi chọn `T`:
   ```text
   thời gian vung tối thiểu = (góc gập + góc duỗi) / tốc độ khớp tối đa
   T tối thiểu = thời gian vung tối thiểu / tỉ lệ vung trong chu kỳ
   ```
   TODO: tính lại với biên độ khớp thật và tốc độ servo đo ở A1.
4. **Dáng đi do người áp đặt.** Tỉ lệ chống/vung do người chọn, chưa chắc tối ưu cho robot này. Đổi kiểu đi (đi ↔ chạy) khó.
5. **Thêm việc đồng bộ sim ↔ Pi.** Obs đổi shape nên phải train lại từ đầu. Pi phải tính pha giống hệt sim: cùng `T`, cùng `p` lúc reset, cùng cách xử lý khi vòng lặp bị trễ.

### 2.3.2 So với chỉ dùng reward (không có đồng hồ)

| | Chỉ reward (`air_time`, `march_alt` ở 2.2.2) | Gait clock |
|---|---|---|
| Tốc độ học | Chậm, hay kẹt ở dáng lạ (nhảy, lê chân, khập khiễng) | Nhanh, ổn định |
| Linh hoạt | Cao: nhịp tự nảy sinh, tự bước gấp khi bị đẩy | Thấp: bị khóa theo nhịp |
| Đứng yên | Tự nhiên | Phải xử lý thêm |
| Đổi obs / firmware | Không | Có |
| Tham số phải chọn | Ít | `T`, tỉ lệ chống/vung |

**Lý do mạnh nhất để thêm đồng hồ cho robot này:** actor là MLP không có trí nhớ, chỉ thấy 4 frame = 200 ms (1.5). Một chu kỳ bước dài khoảng 1 s. Không có đồng hồ thì actor phải đoán "đang ở đâu trong vòng" chỉ từ tư thế thân, việc này khó hơn nhiều.

### 2.3.3 Bỏ đồng hồ khi deploy được không?

**Không**, nếu actor đã được train với pha trong obs. Policy không thuộc lòng dáng đi. Thứ nó học là ánh xạ `(tư thế + sin/cos pha) → góc khớp`. Bỏ đồng hồ thì 2 ô pha thành rác, policy coi như thời gian đứng yên, nên không bước hoặc bước loạn. Đồng hồ trên Pi chỉ tốn 2 dòng code.

Muốn dùng đồng hồ/CPG **chỉ để mớm lúc train**, có 3 cách đúng:

| Cách | Làm gì | Cái giá |
|---|---|---|
| Đồng hồ chỉ ở reward + critic | Actor xuất góc đầy đủ, không thấy pha | Actor tự giữ nhịp ~1 s với trí nhớ 200 ms. Cần thêm lịch sử hoặc LSTM (A.4) |
| Giảm dần CPG (curriculum) | `góc = α·góc_mẫu + action`, α giảm từ 1 về 0 khi train | Vướng vấn đề trí nhớ như trên |
| Teacher → student (A.5) | Teacher train có CPG/đồng hồ. Student chỉ dùng cảm biến thật, bắt chước góc cuối của teacher | Thêm một giai đoạn train, student cần trí nhớ |

Cả 3 cách đều phải trả giá bằng trí nhớ của mạng. Hiện chưa có lý do để làm.

### 2.3.4 Điều khiển đi / dừng / rẽ

Gait clock dùng được với lệnh điều khiển. Đây là cách phổ biến. Cần thêm:
1. **Lệnh vào obs actor**: `(vx, vy, ωz)`. Lúc train phải random lệnh, **kể cả lệnh 0**. Hiện env cố định +x 0.15 m/s (mục 4, lỗi 8.5).
2. **Khi lệnh ≈ 0**: reward theo pha đổi thành "hai chân cùng chống", có thể dừng luôn đồng hồ. Chỉ dừng ở **pha chống kép**. Dừng giữa lúc đang vung thì robot đứng một chân và đổ.

TODO (user chốt): danh sách lệnh cần có (chỉ đi tới/dừng, hay thêm rẽ, đi ngang). Danh sách này quyết định obs thêm mấy số.

## 2.4 CPG + residual RL — thay đổi ACTION

**Có: góc mẫu và quỹ đạo mẫu liên quan trực tiếp đến các góc khớp tạo thành dáng đi.** Cách viết “CPG sinh chuyển động mẫu” ở đây là cách gọi gọn cho **bộ tạo nhịp CPG + phần chuyển nhịp thành chuyển động của robot**. CPG (Central Pattern Generator) tạo tín hiệu dao động có nhịp; muốn dùng tín hiệu đó để điều khiển robot, phải quy định nó tương ứng với chuyển động nào. Chỉ biết thời gian và pha thì chưa biết Hip, Knee, Foot phải quay bao nhiêu độ.

**“Góc mẫu” và “quỹ đạo mẫu” khác nhau ở đâu?**

- **Góc mẫu:** góc mục tiêu cơ sở của một khớp **ở một thời điểm**, trước khi RL sửa. Với robot 10 khớp, tại mỗi thời điểm ta có một bộ 10 góc mẫu; một số khớp có thể giữ nguyên góc.
- **Quỹ đạo góc mẫu:** cách các góc mẫu đó **thay đổi suốt cả chu kỳ**. Đây có thể là công thức hoặc đường cong được nội suy, không nhất thiết là một bảng góc lưu sẵn.
- **Dáng đi:** hình thành khi các khớp phối hợp theo các quỹ đạo đó và robot tương tác với mặt đất. Một bộ góc tại một thời điểm chỉ mô tả tư thế mục tiêu; cần cả chuỗi chuyển động mới tạo thành bước đi. Có quỹ đạo mẫu chưa bảo đảm robot thực hiện được hay giữ được thăng bằng.

Ví dụ với **một khớp giả định**, chỉ để hiểu khái niệm, không phải góc dùng cho OFFICIALdesign:

| Pha `p` | Góc mẫu của khớp |
|---|---|
| 0 | 20° |
| 0.25 | 30° |
| 0.5 | 20° |
| 0.75 | 10° |
| Hết vòng, quay về 0 | 20° |

Ở `p = 0.25`, **30° là một góc mẫu**. Cả đường cong đi qua các giá trị trên và nối mượt giữa chúng là **quỹ đạo góc mẫu**. Những khớp khác có đường cong riêng, phối hợp để gập chân, đưa chân về trước, đặt chân xuống và chuyển trọng lượng. Không phải tất cả khớp cùng tăng/giảm góc một lúc.

**Quỹ đạo mẫu lấy từ đâu?** Có hai cách thường gặp:

1. **Tạo trực tiếp trong không gian góc khớp:** quy định mỗi khớp dao động quanh góc nào, biên độ bao nhiêu, sớm/muộn so với khớp khác ra sao. Tham số có thể do người thiết kế, tối ưu hoặc học. Một ví dụ toán học đơn giản là `q_mẫu,i(p) = q_giữa,i + A_i × sin(2πp + φ_i)`, trong đó `i` chỉ khớp, `A_i` là biên độ và `φ_i` là độ lệch nhịp riêng của khớp. Đây chỉ là ví dụ biểu diễn, chưa phải công thức dáng đi phù hợp cho robot này.
2. **Tạo đường đi của bàn chân trước:** quy định bàn chân nhấc cao bao nhiêu, bước tới đâu theo pha, rồi dùng **động học nghịch** (tính góc khớp cần có để bàn chân đến vị trí mong muốn) để đổi thành góc khớp. Trong cách này, “quỹ đạo bàn chân” là đường đi trong không gian, còn “quỹ đạo góc khớp” là các góc thu được sau bước tính đó.

Như vậy, **pha cho biết đang đến đoạn nào; quỹ đạo quy định tại đoạn đó các khớp nên ở góc nào**. Giữ nguyên nhịp mà đổi biên độ, độ lệch giữa các khớp hoặc đường đi của bàn chân sẽ cho chuyển động khác.

**Residual RL sửa gì?** Trong phương án đang nói ở đây, policy nhìn trạng thái robot và pha rồi xuất phần góc cần cộng thêm/bớt đi:

```text
Thời gian → pha → bộ sinh quỹ đạo → 10 góc mẫu
Trạng thái robot + pha → policy → 10 góc sửa

Góc mục tiêu = góc mẫu + góc sửa
```

Ví dụ, tại một thời điểm, góc mẫu của một khớp là `30°`, policy xuất phần sửa `−3°`, thì góc mục tiêu sau khi cộng là `27°`. Đây là **lệnh mục tiêu**, góc thật còn phụ thuộc phản ứng của servo và tải. Khi triển khai còn phải áp dụng giới hạn góc và tốc độ phù hợp. Phần sửa có thể thay đổi theo tình trạng robot dù đang ở cùng một pha; đó là cách RL thích nghi chuyển động cơ sở. Ngoài phần sửa góc, có thể thiết kế policy chỉnh cả tần số và biên độ.

Để phân biệt hai hướng trong tài liệu:

| Hướng | Phần được cung cấp sẵn | RL phải học gì? |
|---|---|---|
| **Gait clock + RL** | Pha và lịch chân nào nên chống/vung | Học cách phát lệnh góc khớp để tạo chuyển động theo nhịp |
| **CPG + bộ sinh quỹ đạo + residual RL** | Nhịp và chuyển động cơ sở được biểu diễn bằng các góc mẫu | Học phần điều chỉnh quanh chuyển động cơ sở; có thể học chỉnh tham số nhịp/quỹ đạo nếu được thiết kế như vậy |

**Đối với OFFICIALdesign, mục này mới mô tả phương án:** chưa chốt bộ quỹ đạo hay bộ góc mẫu cho 10 khớp. Cần thiết kế và kiểm chứng phần đó khi triển khai B9. CPG không tự suy ra một dáng đi phù hợp chỉ từ số lượng khớp và đồng hồ.

Tham khảo cách kết hợp bộ tạo dáng đi với RL sửa góc mục tiêu: [Kasaei và cộng sự, 2021](https://arxiv.org/abs/2103.00928).

Vì sao hợp robot này:
1. **Gỡ giới hạn tốc độ.** Hiện `action_step_deg = 2.0` ở 20 Hz, nên mỗi khớp đổi tối đa **40°/s**. Nhấc chân như vậy khá chậm. Nếu CPG lo phần chuyển động lớn, policy chỉ cần sửa nhỏ.
2. **Lệnh mượt hơn.** CPG có thể chạy trên Pi ở 100 Hz trở lên, trong khi policy vẫn sửa ở 20 Hz. Servo ít giật, ít nóng.
3. **Kết quả train ổn định, lặp lại được.** RL không phải tự mò ra dáng đi.
4. **Nhẹ cho servo yếu.** Foot và rotate chỉ khoảng 3 Nm.

Rủi ro cần chú ý:
- Phần sửa cho phép quá nhỏ: robot bị "khóa" vào dáng CPG, không tự cứu được khi bị đẩy.
- Phần sửa quá lớn: policy bỏ qua CPG luôn.
- Nên cho policy chỉnh được cả **tần số và biên độ** CPG.

### 2.4.1 CPG + residual cần gì khi deploy

CPG vốn **là** đồng hồ + bộ sinh góc mẫu. Policy chỉ xuất phần sửa (vài độ). Bỏ CPG trên Pi thì servo chỉ nhận mấy độ sửa đó, nên robot đứng quanh tư thế mặc định hoặc khuỵu.

Pi phải chạy y hệt sim:
1. **Đồng hồ pha**: cùng `T`, cùng `p` lúc bắt đầu.
2. **Bộ sinh quỹ đạo**: cùng công thức, cùng tham số.
3. **Phép cộng**: `góc mục tiêu = góc mẫu + góc sửa`, rồi clamp vào cùng giới hạn góc và tốc độ như sim.

CPG cần từng phần của gait clock (2.3) ở mức khác nhau:

| Phần của gait clock | CPG + residual có cần không? |
|---|---|
| Bộ đếm pha `p` | **Bắt buộc.** Không có `p` thì không có góc mẫu |
| Pha trong obs actor (`sin/cos`) | **Nên có.** Phần sửa phụ thuộc thời điểm trong vòng bước (sắp chạm đất khác đang vung). Không có thì actor tự đoán từ 200 ms lịch sử |
| Reward theo lịch chống/vung | **Không bắt buộc.** Góc mẫu đã áp nhịp. Giữ trọng số nhỏ để policy không sửa phá nhịp |

### 2.4.2 Gait clock hay CPG + residual: chọn cái nào

Hai phương pháp **khác nhau nhưng lồng nhau**. CPG chứa sẵn một đồng hồ. Khác nhau ở chỗ đồng hồ dùng để làm gì:

| | 2.3 Gait clock | 2.4 CPG + residual |
|---|---|---|
| Đồng hồ đưa vào đâu | Obs + reward | Bộ sinh quỹ đạo → **góc mẫu trong action** (thường đưa cả vào obs) |
| Policy xuất | **Góc đầy đủ** | **Góc sửa** |
| Ai tạo dáng đi | Policy tự học | Người thiết kế quỹ đạo mẫu, policy chỉ sửa |

Có thể coi là hai núm vặn độc lập:
- **Núm A: actor có thấy pha không?** Có (khuyến nghị) / không (chỉ ở reward + critic, xem 2.3.3).
- **Núm B: có dáng đi mẫu không?** Không (chỉ reward lịch tiếp đất) / có, đặt trong **reward** (thưởng bám quỹ đạo góc mẫu) / có, đặt trong **action** (CPG + residual).

**Vì sao tồn tại hai cách.** Hai cách ra đời ở hai thời kỳ:
- **CPG + residual có trước.** CPG gốc từ sinh học và robotics cổ điển. Hồi đó RL còn yếu, tốn dữ liệu, nên giữ bộ tạo dáng đi sẵn có và cho RL sửa thêm.
- **Gait clock + RL có sau (khoảng từ 2020).** Sim song song (Isaac Gym) làm dữ liệu gần như miễn phí, nên mớm cho RL ít nhất có thể và để nó tự tìm dáng đi.

**Khi nào dùng cách nào:**

| Tình huống | Hợp với |
|---|---|
| Ít tài nguyên train, sim chậm | CPG + residual |
| Servo yếu/chậm, cần dáng đi an toàn, đoán trước được | CPG + residual |
| Đã có sẵn dáng đi tốt (ZMP, tay chỉnh), muốn RL làm cứng cáp hơn | CPG + residual |
| Train song song nhiều env | Gait clock |
| Cần nhiều lệnh: đi, dừng, rẽ, đi ngang, đổi tốc độ | Gait clock. CPG phải thiết kế thêm quỹ đạo mẫu cho từng kiểu |
| Cần phản ứng mạnh khi bị đẩy, nền gồ ghề | Gait clock |

**Các nhóm chuyên nghiệp chọn gì:**
- 2018–2020: bộ sinh quỹ đạo + RL. Google Minitaur dùng PMTG (Iscen et al. 2018). ETH ANYmal (Lee et al. 2020) dùng bộ sinh quỹ đạo bàn chân theo pha, RL chỉnh tần số và cộng phần sửa.
- Từ 2021: phần lớn biped/humanoid chuyển sang gait clock (hoặc periodic reward) + RL: Cassie (Siekmann et al. 2021), unitree_rl_gym cho G1/H1 (pha `sin/cos` trong obs + reward tiếp đất theo pha). Robot 4 chân nhiều nhóm bỏ luôn đồng hồ (legged_gym ANYmal chỉ thưởng `feet_air_time`).
- Dạng lai: Humanoid-Gym (2024) dùng gait clock trong obs + **reward** bám quỹ đạo góc mẫu đơn giản, tức mớm dáng đi qua reward chứ không qua action.

**Quyết định cho OFFICIALdesign:** gait clock trong obs actor + reward theo lịch tiếp đất (B5), chưa có dáng mẫu. Lý do: chưa có quỹ đạo mẫu cho 10 khớp, chưa có số đo servo (A1), và cần điều khiển đi/dừng. Sau khi train B5, đọc kết quả:

| Thấy gì | Nghĩa là | Làm gì |
|---|---|---|
| Bước đúng nhịp, đi ổn | B5 đủ | Chuyển sang DR / sim-to-real |
| Đúng nhịp nhưng nhấc chân thấp, lê, chậm | Kẹt ở 40°/s | Tăng `action_step_deg` trước. Vẫn không được thì B9 (CPG) |
| Dáng kỳ quặc, mỗi lần train ra một dáng | RL tự mò khó | Thêm dáng mẫu: reward bám quỹ đạo hoặc B9 |

## 2.5 ZMP — thay đổi REWARD

**ZMP** (Zero Moment Point): điểm trên sàn nơi các lực làm robot lật triệt tiêu nhau. ZMP nằm trong vùng bàn chân chống (support polygon) thì robot không lật.

- **Không dùng làm bộ điều khiển trên robot thật:** cần cảm biến lực bàn chân và CoM chính xác, mà CoM Hip trái/phải còn lệch khoảng 8 mm chưa xác nhận.
- **Dùng làm reward trong sim:** sim biết chính xác CoM và điểm tiếp xúc, nên thưởng khi hình chiếu CoM/ZMP nằm trong vùng chống. Chỉ dùng lúc train, robot thật không cần cảm biến thêm.
- (Tùy chọn) Dùng LIPM (Linear Inverted Pendulum Model, coi robot như con lắc ngược) sinh quỹ đạo mẫu offline cho CPG. Công sức nhiều, lợi thêm ít.

## 2.6 Obs của critic (thông tin đặc quyền)

Nhóm `critic` (env trả thêm, config khai báo `obs_groups`):
- vận tốc dài và vận tốc góc thật của thân;
- roll, pitch thật (không nhiễu);
- chiều cao thân;
- góc và vận tốc khớp thật;
- contact hai chân, lực tiếp xúc;
- các tham số đã random: delay từng chân, Kp/Kd, ma sát, khối lượng;
- pha gait clock.

Cấu hình: `obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}`.

**Quy tắc:** actor **không bao giờ** được nhận thứ robot thật không đo được (contact, vận tốc thân thật).

## 2.7 Hàm lật đối xứng (cho symmetry ở tầng 1)

| Thành phần | Phép lật |
|---|---|
| Hip, Knee, Foot | Đổi trái↔phải, **giữ dấu** (hệ tọa độ policy) |
| Bub (nghiêng ngang), rotate (xoay yaw) | Đổi trái↔phải, **đổi dấu** |
| IMU roll, gyro x, gyro z | **Đổi dấu** |
| IMU pitch, gyro y | Giữ nguyên |
| Gait clock | Lệch nửa chu kỳ (đổi vai hai chân) |

⚠️ Trong `calibration.json`, `rotate_right` có `sign = +1` giống bên trái, trong khi mọi khớp phải khác đều là `-1`. **Phải kiểm chứng** trước khi viết hàm lật.

Test bắt buộc: lật một trạng thái, cho sim chạy, rồi kiểm tra kết quả có đúng là ảnh gương của trạng thái gốc không. Phải map đúng index trong cả 4 frame lịch sử.

---

# TẦNG 3 — SIM-TO-REAL (sim phải giống robot thật)

## 3.1 Servo và PID nội bộ

**Vấn đề:** policy học trong sim rằng "ra lệnh X thì khớp phản ứng kiểu Y". Servo thật phản ứng kiểu Z, và policy chưa từng gặp Z. Hiện `stiffness 60/40`, `damping 1.5/1.0`, `velocity 4.7 rad/s` và delay 10 ms đều là **số đoán** (`calibration.json` ghi "provisional").

Làm theo thứ tự:
1. **System identification** (đo hành vi thật rồi chỉnh sim cho khớp).
   - Gửi lệnh step (nhảy góc đột ngột 20–30°), đọc present position ở ≥200 Hz.
   - Đo **từng nhóm** STS3095 và STS3215, cả **không tải** lẫn **đang chịu trọng lượng robot**.
   - Làm y hệt trong Isaac, chỉnh `stiffness`, `damping`, `effort_limit`, `velocity_limit`, `armature`, friction cho hai đường cong trùng nhau.
2. **Backlash** (rơ bánh răng): env cũ có mô hình rơ 2.5°, env mới **không có**. Đo độ rơ thật rồi thêm lại. Nhớ reset trạng thái rơ mỗi episode (đây là lỗi của env cũ).
3. **Delay:** xem 3.4.
4. **Domain randomization:** xem 3.6.
5. **Để sau: actuator network** (mạng nơ-ron nhỏ học cách servo phản ứng: `ActuatorNetMLP`/`LSTM`). Chỉ làm khi các bước trên chưa đủ.

**Feedback dòng điện (đã kiểm tra datasheet):**
- STS **có** trả dòng điện: thanh ghi 69–70, 1 đơn vị = 6,5 mA.
- Thanh ghi "load" (60–61) là **% PWM** đưa vào motor, không phải dòng điện, cũng không phải mô-men.
- Hãng **không công bố độ chính xác**. Chưa biết giá trị có dấu hay không, cũng chưa biết tần số lấy mẫu. Phải tự đo.
- ST-3120-C001: tỉ số truyền 1/399, `Kt = 25,5 kg·cm/A` (đo ở trục ra, đã gộp hộp số). Quay không tải tốn 250 mA, tương đương khoảng 5% mô-men kẹt cứng, hoàn toàn do ma sát.
- Hộp số tỉ số lớn làm dòng điện **không phản ánh đúng mô-men ngoài tác dụng lên khớp**:
  - trong vùng chết của servo, bánh răng có thể tự giữ tải, nên dòng ≈ 0 dù khớp đang chịu tải;
  - ma sát làm dòng khác nhau theo chiều quay (trễ — hysteresis);
  - quán tính rotor bị nhân với bình phương tỉ số truyền, nên khi đổi tốc độ nhanh, dòng chủ yếu dùng để tăng tốc rotor.
- **Cách dùng:** làm công cụ đo khi sysid, bảo vệ quá tải ở LL. **Không** đưa vào obs actor trước khi có mô hình dòng trong sim (Kt random, ma sát theo chiều quay, vùng chết, lượng tử hóa 6,5 mA, nhiễu).

Mẹo: có thể giảm hệ số P trong EEPROM (bộ nhớ cấu hình) của servo cho khớp "mềm" hơn. **Đổi xong phải đo lại.**

## 3.2 Feedback góc servo

- STS **có** encoder từ (magnetic encoder, cảm biến đo góc bằng nam châm), đọc qua ô nhớ *Present Position*. Độ phân giải khoảng 0.09°.
- Dữ liệu "không đáng tin lắm" chủ yếu do đường truyền:
  - trễ, vì đọc lần lượt từng servo;
  - mất gói, do timeout hoặc sai checksum (mã kiểm tra lỗi);
  - vận tốc tính từ hiệu góc rất nhiễu.
- **Cách dùng:** đưa **góc** vào obs actor. Trong sim cố ý làm "bẩn":
  - nhiễu 0.5–1°;
  - trễ 1–2 bước;
  - thỉnh thoảng giữ giá trị cũ (giả mất gói).
- **Không** đưa vận tốc khớp vào actor. Phương án an toàn hơn: chỉ đưa góc thật cho critic.
  - ⚠️ Chưa chốt. Vận tốc khớp giúp nhận ra cú va chạm khi chân chạm đất (3.3). Quyết định sau khi đo nhiễu `dq` thật (A1). Đừng cấm ngay từ đầu.
- **Hiện obs actor chỉ có lịch sử góc *lệnh*, chưa có góc *đo*.** Đưa `q` đo vào actor là **điều kiện bắt buộc** để actor tự suy ra contact (3.3). Vì vậy B10 không còn là việc tùy chọn.
- Đọc thêm vị trí, tốc độ, dòng trong cùng một lệnh `SYNC_READ` gần như không tốn băng thông: 6 servo, 15 byte mỗi servo, 1 Mbps, mất khoảng 1,3 ms truyền thuần. Phải đo lại trên Pi thật.

## 3.3 Contact (robot không có cảm biến tiếp đất)

- Trong sim, contact chỉ dùng để tính reward và điều kiện kết thúc. Ngoài đời không tính reward, nên không cần cảm biến.
- Actor không nhận contact. Env mới đang đúng điều này.
- Gait clock (mục 2.3) cung cấp lịch "chân nào **nên** chống, chân nào **nên** vung". Nó không đo được chân nào **đang thực sự** chạm đất; contact trong sim mới là thông tin dùng để kiểm tra robot có làm đúng lịch khi train hay không.

**Trạng thái: bỏ ngỏ, chờ thí nghiệm trong sim.**

Actor có thể tự suy ra contact (suy ngầm) từ các dấu hiệu sau, với điều kiện obs có `q` đo (mục 3.2):
- **Chân nào thấp hơn:** từ `q` và roll/pitch, mạng tính được chân nào thấp hơn. Trên nền phẳng, chân thấp hơn gần như chắc chắn là chân chống.
- **Khớp bị đè lệch:** sai số `q_lệnh − q_đo` lớn và có hướng ổn định khi chân gánh trọng lượng. Khi chân ở trên không, sai số này nhỏ.
- **Va chạm:** gyro và sai số khớp tăng vọt ở thời điểm chân chạm đất. Frame stack cho mạng thấy được mẫu "trước và sau" va chạm.

Không có câu lệnh `if` nào trong mạng. Reward phụ thuộc vào contact, nên gradient đẩy actor tự dùng những dấu hiệu tương quan với contact.

Các phương án, xếp theo mức tốn công:

| # | Phương án | Thêm thông tin mới? | Độ phức tạp |
|---|---|---|---|
| 1 | Actor nhận `q` đo + IMU + lịch sử. Critic nhận contact thật (§2.6) | Không | Thấp, vì đằng nào cũng phải làm 3.2 |
| 2 | Contact estimator: MLP nhỏ, input là obs actor, output là xác suất contact của 2 chân, học có giám sát bằng nhãn contact của sim. Output đưa vào actor | **Không**. Chỉ dùng lại thông tin actor đã có, nhưng làm tường minh để kiểm tra được và có thể giúp học nhanh hơn | Trung bình: code train riêng, export thêm mạng |
| 3 | Công tắc hành trình hoặc FSR ở đế chân | **Có** | Sim: thấp (contact sensor + ngưỡng + trễ + random hỏng). Phần cứng: công tắc dễ (GPIO). FSR khó hơn (Pi không có ADC, cảm biến dễ hỏng, phi tuyến, lắp cơ khí khó) |
| 4 | Dòng servo trong obs | Có nhưng nhiễu nặng | Cao (xem 3.1) |

**Thí nghiệm quyết định (làm trong sim trước, chưa cần mua phần cứng):**
1. Baseline: phương án 1 + gait clock.
2. Baseline + contact lý tưởng trong obs actor, có làm bẩn (trễ, nhiễu, random hỏng). Đây là **giới hạn trên** của mọi cảm biến contact.
3. So sánh khi bị đẩy, trên nền gồ ghề, và khi chân chạm đất sớm hoặc muộn hơn lịch của gait clock. **Không** chỉ nhìn tổng reward.
- Nếu (2) không tốt hơn (1) rõ rệt: dừng, không cần FSR hay estimator.
- Nếu (2) tốt hơn rõ rệt: chọn phương án 2 hoặc 3.

FSR vẫn có ích để làm nhãn đúng khi kiểm tra estimator ngoài đời, kể cả khi không đưa vào obs.

**Gait clock và contact thật bổ sung cho nhau, không thay thế nhau.** Clock giống bản nhạc: nó nói "**đến lượt** chân trái". Contact cho biết chân trái **đã thật sự** chạm đất hay chưa.
- Đi đều trên nền phẳng: contact gần trùng với lịch, nên clock cộng với suy ngầm là đủ.
- Bị đẩy, nền gồ ghề, giẫm phải vật: chân chạm đất lệch nhịp. Đây là lúc contact thật có thể giúp.

TODO (user chốt): chỉ số nào, chênh lệch bao nhiêu thì coi là "tốt hơn rõ rệt". Ví dụ: thời gian trước khi ngã khi bị đẩy, sai số vận tốc, tỉ lệ trượt chân.

## 3.4 Kiến trúc 2 Pi

**Hai giai đoạn:**
- **Giai đoạn 1:** laptop chạy policy, nối Ethernet tới hai Pi. Mỗi Pi đọc servo + IMU của chân mình, gửi lên, nhận lệnh, rồi ghi xuống servo.
- **Giai đoạn 2:** Pi A vừa infer vừa điều khiển chân A. Pi B nối Pi A qua Ethernet, có dây **sync** và **GND chung**.

Đây là mẫu chuẩn "**bộ não trung tâm + nút I/O phân tán**" (I/O node: bo mạch chỉ đọc cảm biến và điều khiển motor tại chỗ). ANYmal và robot công nghiệp làm y như vậy qua EtherCAT (chuẩn Ethernet thời gian thực).

**Quy tắc thiết kế:**
1. **Dây sync làm tiếng còi xuất phát.**
   - Pi A phát xung GPIO mỗi chu kỳ, kể cả ở giai đoạn 1 (laptop không có GPIO).
   - Cả hai Pi **đọc cảm biến đúng lúc có xung**, nên dữ liệu hai chân cùng một khoảnh khắc, chính xác cỡ micro-giây.
2. **UDP, gói nhỏ cố định.**
   - UDP gửi thẳng, không chờ xác nhận. Mất một gói không làm nghẽn các gói sau.
   - Gói gồm: `seq` (số thứ tự), timestamp, 5 góc servo, dữ liệu IMU, cờ lỗi.
3. **Giao thức giống hệt nhau hai giai đoạn.** Pi B không cần biết đầu kia là laptop hay Pi A. Chuyển giai đoạn chỉ đổi IP, không train lại.
4. **Watchdog** (bộ canh giờ): quá khoảng 100 ms không có lệnh thì Pi giữ tư thế an toàn.
5. **Lợi thế sẵn có:** hai bus servo chạy song song, mỗi bus 5 servo.

Ngân sách thời gian (ước tính, **phải đo**):
```
xung sync → đọc 5 servo + IMU → UDP lên → infer → UDP xuống → ghi servo
               ~3–8 ms           <1 ms     <1 ms    <1 ms      ~1–2 ms
tổng: khoảng 5–15 ms trong chu kỳ 50 ms
```

**Trong sim:**
- Hiện env dùng `actuator_delay_steps = 2` (10 ms), **chung cho mọi env và mọi chân, không random**.
- Cần: delay **riêng từng env và từng chân**, cố định trong một episode, random quanh giá trị đo.
- Thỉnh thoảng giả mất gói: chân đó giữ lệnh cũ.

## 3.5 Hai IMU

Hai IMU gắn cứng cùng Baselink, cách nhau 33 mm, cùng hướng, nên gộp đơn giản. Chạy **cùng một đoạn code** ở cả sim và Pi:
1. Hiệu chỉnh offset gyro và hướng lắp của từng con khi robot đứng yên.
2. **Lấy trung bình gyro và accel thô.** Nhiễu giảm khoảng √2 lần. Lệch do khoảng cách 16.5 mm không đáng kể.
3. Chạy **một** bộ lọc complementary hoặc Madgwick (trộn gyro với accel để ra góc nghiêng ổn định) → roll, pitch.
4. **Obs giữ nguyên 5 số.** Không đưa hai IMU thô vào policy.

Chẩn đoán:
- Hai IMU lệch nhau lớn nghĩa là một con hỏng, **hoặc mối dock giữa hai module đang rơ/uốn**. Nên log lại.
- Một con chết thì dùng con còn lại.

Trong sim: mô phỏng hai IMU với nhiễu và bias (độ lệch cố định) độc lập, rồi chạy code gộp. Hiện env chỉ dùng `IMUleft`, lấy góc thật cộng nhiễu trắng, chưa mô phỏng bộ lọc.

## 3.6 Domain randomization (DR): env mới đang thụt lùi

DR (ngẫu nhiên hóa): mỗi robot ảo có tham số hơi khác nhau, nên policy không phụ thuộc vào một bộ số chính xác.

| Đại lượng | Env 10DOF cũ | **Env mới** | Cần |
|---|---|---|---|
| Ma sát khớp | 0.30–0.50 | ❌ | ±20% quanh số đo A1 |
| Damping khớp | 0.60–0.70 | ❌ | ±20% quanh số đo A1 |
| Giới hạn torque | 9.27–10.30 | ❌ | ±10–20% |
| Kp servo | ❌ | ❌ | ±20% |
| Delay actuator | 2–6 substep (lỗi: một giá trị chung) | cố định 2 | Riêng từng env/chân (3.4) |
| Backlash | 2.5° | ❌ | Theo số đo |
| Nhiễu servo | σ 0.5° | ❌ | Theo số đo |
| Bias IMU | có | ❌ | Mỗi episode |
| Drift gyro | có | ❌ | Có |
| Nhiễu orientation / gyro | σ 0.04 / 0.15 | σ **0.015 / 0.01** | Theo số đo IMU thật |
| Ma sát sàn | cố định | cố định 0.8 / 0.4 | Random khoảng 0.4–1.2 |
| Khối lượng / CoM | ❌ | ❌ | ±10% / ±1 cm |
| Đẩy ngẫu nhiên (push) | ❌ | ❌ | Có, khi đã đi được |

Policy train trên env mới hiện tại gần như chắc chắn sẽ "giòn" khi lên robot thật.

---

# 4. Lỗi của env 10DOF cũ: tình trạng trên env mới

Chi tiết từng lỗi và code sửa cho env cũ: Phụ lục B.

| Lỗi env cũ | Env mới |
|---|---|
| 8.1 backlash không reset | ⚪ Không áp dụng: env mới không có backlash (xem 3.1) |
| 8.2 chuẩn hóa `act_hist` lệch | ✅ Đã sửa: init và reset cùng dùng `_normalize()` |
| 8.3 delay dùng chung | ⚠️ Vẫn thiếu: delay cố định, không random (xem 3.4) |
| 8.4 thiếu `obs_groups` | ❌ Vẫn còn (xem 1.4, 2.6) |
| 8.5 thiếu command trong obs | ✅ Né được nhờ lệnh +x cố định. Muốn rẽ thì phải thêm command. |

---

# 5. Việc phải làm (theo thứ tự)

### A. Phần cứng / đo đạc — user làm
- [ ] A1. Step response STS3095 và STS3215, không tải và có tải, log ≥200 Hz. Đo luôn độ rơ (backlash).
- [ ] A2. Đo thời gian đọc 5 servo trên một bus, tỉ lệ lỗi đọc (%).
- [ ] A3. Dựng giao thức 2 Pi: xung sync GPIO, UDP + `seq`, watchdog.
- [ ] A4. Đo trễ end-to-end (từ xung sync đến lúc servo nhận lệnh). Dùng số này để quyết 20 hay 50 Hz.
- [ ] A5. Hiệu chỉnh hai IMU. Log độ lệch giữa hai con (để kiểm tra mối dock có rơ không). Đo nhiễu và bias thật.
- [ ] A6. Đo giới hạn cơ khí thật và điểm 0 servo. Xác nhận CoM Hip. Xác nhận dấu của `rotate_right`.
- [ ] A7. Xác nhận model servo theo từng ID: nhóm heavy là STS3095 hay ST3120? (`calibration.json` đang ghi STS3095.)
- [ ] A8. Đo dòng điện trên bàn thử (3.1). Đọc thanh ghi 69–70, khoảng 500 mẫu mỗi trường hợp:
  1. Treo tải tĩnh 0,5 / 1 / 2 kg ở cánh tay 10 cm. Dòng có tuyến tính không? Có lúc nào treo tải mà dòng ≈ 0 không?
  2. Cùng tải đó, nâng lên chậm rồi hạ xuống chậm. Hai chiều chênh nhau bao nhiêu?
  3. Dùng tay đẩy cánh tay servo theo hai chiều. Giá trị có đổi dấu không?
  4. Quay nhanh qua lại, không tải. Dòng đỉnh là bao nhiêu?

### B. Sim + config — agy làm qua `PLAN.md`

Làm được ngay, không cần chờ số đo:
- [ ] B1. Config PPO: `num_envs` 2048–4096, `init_std` 0.5.
- [ ] B2. Asymmetric critic: env trả nhóm `critic` (2.6), khai báo `obs_groups`, bật `obs_normalization` cho critic.
- [ ] B3. Khung domain randomization (3.6) với dải tạm, đánh dấu `TODO: thay bằng số đo A*`.
- [ ] B4. Delay riêng từng env/chân + giả mất gói (3.4).
- [ ] B5. Gait clock trong obs + reward theo pha (2.3).

Sau khi có số đo A:
- [ ] B6. Actuator theo A1 (Kp, Kd, tốc độ, torque, backlash). Thu hẹp dải DR quanh số đo.
- [ ] B7. Mô phỏng hai IMU + code gộp chung sim/Pi (3.5).
- [ ] B8. Symmetry augmentation (2.7), sau khi A6 xác nhận dấu.
- [ ] B9. CPG + residual RL (2.4).
- [ ] B10. **(Bắt buộc, là nền cho 3.3)** Góc servo đo được vào obs actor, kèm nhiễu, trễ và mất gói (3.2).
- [ ] B11. (Tùy chọn) Reward CoM/ZMP (2.5). Ablation framestack 4 vs 8 (1.5).
- [ ] B12. (Để sau) Distillation, actuator network.
- [ ] B13. Thí nghiệm contact trong sim (3.3), sau B2, B5 và B10: so baseline với bản có contact lý tưởng (đã làm bẩn) trong obs actor. Kết quả quyết định có làm estimator hoặc cảm biến đế chân hay không.

---

# 6. Giả định chưa xác nhận

| Giả định | Giá trị hiện tại | Xác nhận bằng |
|---|---|---|
| Trễ actuator | 10 ms, chung | A4 |
| Kp / Kd servo | 60 / 1.5 (nặng), 40 / 1.0 (nhẹ) | A1 |
| Tốc độ servo | 4.7 rad/s | A1 |
| Torque | stall trong handoff | A1 (có tải) |
| Backlash | không mô phỏng | A1 |
| Nhiễu IMU | σ 0.015 rad / 0.01 rad/s | A5 |
| Giới hạn góc | training envelope | A6 |
| CoM Hip trái/phải | lệch khoảng 8 mm theo z | A6 / CAD |
| Dấu `rotate_right` | +1 | A6 |
| Mối dock cứng tuyệt đối | giả định | A5 |
| Model servo nhóm heavy | STS3095 (`calibration.json`), user nói ST3120 | A7 |
| Dòng servo: độ chính xác, có dấu không, tần số lấy mẫu | hãng không công bố, chỉ biết 6,5 mA/đơn vị | A8 |
| Nhiễu `dq` đủ thấp để đưa vào actor | chưa biết (3.2 tạm cấm) | A1 |
| Actor tự suy được contact, không cần cảm biến | chưa biết | B13 |

---

# 7. Tra cứu nhanh

```bash
cd isaac_rl
# Train
./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 --headless --num_envs 4096
# Play
./run.sh scripts/rsl_rl/play.py --task Official-Walk-v0 --num_envs 1 \
    --checkpoint "$PWD/logs/officialdesign/<run>/model_<N>.pt"
# TensorBoard
tensorboard --logdir=logs/officialdesign_walk --port=6006
```

| Muốn | Sửa |
|---|---|
| Học nhanh/chậm hơn | `desired_kl` (không phải `learning_rate`) |
| Khám phá nhiều/ít hơn | `entropy_coef`, `init_std` |
| Nhìn xa hơn | `gamma` |
| Nhiều dữ liệu hơn | `num_envs`, `num_steps_per_env` |
| Thêm thông tin cho critic mà giữ actor | nhóm `critic` + `obs_groups` |

Đọc thêm:
- Schulman et al. 2017 — PPO.
- Schulman et al. 2015 — GAE.
- Huang et al. — *The 37 Implementation Details of PPO*.
- Rudin et al. 2021 — *Learning to Walk in Minutes* (gốc của rsl_rl).
- Mittal et al. 2024 — symmetry trong RL robot (tác giả của `RslRlSymmetryCfg`).
- Siekmann et al. 2021 — *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition* (gait clock, Cassie).
- Iscen et al. 2018 — *Policies Modulating Trajectory Generators* (PMTG).
- Lee et al. 2020 — *Learning Quadrupedal Locomotion over Challenging Terrain* (bộ sinh quỹ đạo + RL, ANYmal).
- Gu et al. 2024 — *Humanoid-Gym* (gait clock + reward bám quỹ đạo mẫu).
- Source: `rsl_rl/algorithms/ppo.py`.

---

# Phụ lục A — Mẫu config rsl_rl (dùng khi làm B1, B2, B8, B12)

Đã đối chiếu với IsaacLab đang cài: `~/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/`.

## A.1 Asymmetric critic (B2)

Env trả **hai nhóm** obs:

```python
# official_env.py → _get_observations()
return {
    "policy": actor_obs,        # 60 chiều — GIỮ NGUYÊN, chỉ chứa thứ đo được ngoài đời
    "critic": privileged_obs,   # thông tin đặc quyền, xem mục 2.6
}
```

Config ánh xạ nhóm của env vào actor/critic:

```python
# agents/official_ppo_cfg.py
obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
```

- Không khai báo `obs_groups` thì rsl_rl gán nhóm `"policy"` cho **cả hai**, và critic mù ngang actor. Đây chính là tình trạng hiện tại.
- Actor không đổi shape, nên checkpoint actor cũ vẫn nạp được. Chỉ critic phải học lại.
- File mẫu: `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py`.

## A.2 Hyperparameter (B1)

```python
actor = RslRlMLPModelCfg(
    hidden_dims=[256, 256, 128], activation="elu", obs_normalization=False,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),   # 1.0 → 0.5
)
# num_envs: truyền --num_envs 4096 khi train, hoặc sửa InteractiveSceneCfg trong official_env.py
```

Nếu bật `obs_normalization=True` cho actor, thì state_dict có thêm buffer `_mean`, `_var`, `_std`, `count`, nên checkpoint cũ không nạp được. Phải xác nhận file ONNX xuất ra có chứa bộ chuẩn hóa (`exporter.py` nhận tham số `normalizer`).

## A.3 Symmetry augmentation (B8)

```python
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

algorithm = RslRlPpoAlgorithmCfg(
    ...,
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,          # CAD không đối xứng hoàn hảo → để tắt hoặc hệ số nhỏ
        data_augmentation_func=official_symmetry.compute_symmetric_states,
    ),
)
```

Chữ ký hàm (file mới `official_symmetry.py`):

```python
@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    """Trả về (obs_aug, actions_aug). Batch gốc nối với batch đã lật trái↔phải.

    num_aug = 2 cho biped (gốc + gương). ANYmal 4 chân dùng num_aug = 4.
    obs là TensorDict: phải lật CẢ nhóm "policy" lẫn "critic".
    Phải map đúng index trong cả 4 frame lịch sử. Quy tắc lật: mục 2.7.
    """
```

- Mẫu để chép: `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py`.
- Bật cả hai cờ = False thì rsl_rl vẫn gọi hàm để **log chỉ số đối xứng** mà không ảnh hưởng train. Dùng cách này kiểm tra hàm lật trước khi bật thật.

## A.4 Recurrent policy (để sau)

Thay framestack bằng LSTM/GRU (mạng có trí nhớ, cửa sổ không giới hạn):

```python
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

actor = RslRlRNNModelCfg(
    hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1,
)
```

Chưa dùng, vì ONNX trên Pi phải tự quản lý hidden state (mục 1.4).

## A.5 Distillation teacher → student (B12, để sau)

Bước 1: train teacher bằng PPO, teacher thấy cả obs đặc quyền. Bước 2: chưng cất sang student, student chỉ dùng cảm biến thật.

```python
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlDistillationAlgorithmCfg

obs_groups = {"student": ["policy"], "teacher": ["policy", "critic"]}
algorithm = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=2, learning_rate=1e-3, gradient_length=15,
)
```

Mẫu: `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_distillation_cfg.py`.

## A.6 Chi tiết thư viện nên biết

| Điểm | Thực tế trong rsl_rl 5.0.1 |
|---|---|
| Adaptive LR | `kl > 2·desired_kl` thì `lr /= 1.5` (sàn 1e-5); `kl < desired_kl/2` thì `lr *= 1.5` (trần 1e-2). Xem `rsl_rl/algorithms/ppo.py`. |
| LR annealing tuyến tính (giảm dần learning rate theo thời gian) | Không có. Adaptive thay thế. |
| Grad clipping | Clip **riêng** actor và critic, không clip chung như CleanRL. |
| Orthogonal init | Hàm có, nhưng không ai gọi. Dùng init mặc định của PyTorch. |
| Value / return normalization | Không có. Reward do env tự lo thang đo. |
| Advantage normalization | Bật, toàn batch (có tùy chọn theo minibatch). |
| NaN check | Bật mặc định (`check_for_nan`). |
| Optimizer | adam mặc định; chọn được adamw/sgd/rmsprop. |
| Std của actor | Học được, **không phụ thuộc obs**. `play.py` bỏ ngẫu nhiên, lấy thẳng tâm phân phối. |
| Kích thước mạng [256, 256, 128] | Actor ~115.6k tham số, critic ~114.4k (với 60 obs). |

---

# Phụ lục B — Lỗi riêng của env 10DOF cũ

(nay ở `isaac_rl/_archive/fulltrans/transformer_walk10dof_env.py`)

Chỉ cần nếu train lại robot cũ. Env mới đã né hoặc không có các lỗi này (mục 4).

| # | Lỗi | Mức | Sửa |
|---|---|---|---|
| B.1 | `gear_position`, `last_direction` (trạng thái mô hình backlash) **không reset** trong `_reset_idx`. Khớp vật lý đã về 0, nhưng mô hình rơ vẫn tưởng bánh răng ở vị trí episode trước. | Nghiêm trọng | Thêm vào cuối `_reset_idx`: `self.gear_position[env_ids] = self.base_pose[0]`; `self.last_direction[env_ids] = 0.0` |
| B.2 | `act_hist` lúc init dùng công thức chuẩn hóa, lúc reset dùng giá trị thô. Knee lệch 0.867 → 0.000, Foot 0.200 → 0.000, nên policy thấy một "cú giật" giả đầu mỗi episode. | Trung bình | Dùng chung một hàm: `clamp((pose − min)/(max − min)·2 − 1, −1, 1)` |
| B.3 | `act_delay = torch.randint(..., size=(1,)).item()`: một độ trễ **chung cho mọi env**, đổi **mỗi bước**. Mất phần lớn giá trị của DR. | Trung bình | Sample theo từng env trong `_reset_idx`, giữ cố định suốt episode; `_apply_action` so sánh theo vector |
| B.4 | Không khai báo `obs_groups` → critic mù | Nghiêm trọng (hiệu quả) | Như A.1 |
| B.5 | Command hướng đi (`act_direction`) dùng để tính reward nhưng policy không thấy | Nghiêm trọng (thiết kế) | Đưa command vào obs (đổi shape) |

Thêm về env cũ:
- Action được clamp ±3 rồi nhân 2/3, tức tối đa 2°/bước.
- Chuỗi xử lý action: clamp → cộng dồn → backlash 2.5° → nhiễu σ 0.5° → clamp biên → delay 2–6 substep.
- Ma sát sàn cố định ở static 2.0 / dynamic 2.5.
