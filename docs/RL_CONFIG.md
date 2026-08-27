# RL Config — Transformer Bipedal

Tài liệu tham chiếu về cấu hình PPO, kiến trúc mạng, và các "gia vị" (implementation
details) của thư viện đang dùng. Viết để đọc hiểu và để quyết định nên chỉnh gì.

**Ngày rà soát:** 2026-08-16
**Thư viện:** `rsl_rl_lib 5.0.1` (env conda `isaacsim`, Python 3.12) + `isaaclab_rl`
**File config chính:** [`agents/rsl_rl_ppo_cfg.py`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/agents/rsl_rl_ppo_cfg.py) → class `TransformerWalkPPORunnerCfg`
**Env chính:** [`transformer_walk10dof_env.py`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_walk10dof_env.py)

> File PPO ở `tasks/manager_based/.../rsl_rl_ppo_cfg.py` là cartpole mẫu (mạng 32×32,
> `experiment_name="cartpole_direct"`). **Không liên quan** robot đi bộ. Đừng sửa nhầm.

---

## Mục lục

1. [PPO là gì — trực giác trước](#1-ppo-là-gì--trực-giác-trước)
2. [Kiến trúc mạng](#2-kiến-trúc-mạng)
3. [Observation & Action space](#3-observation--action-space)
4. [Toàn bộ hyperparameter](#4-toàn-bộ-hyperparameter)
5. [Vòng đời một iteration](#5-vòng-đời-một-iteration)
6. [Kho "gia vị" của rsl_rl 5.0.1](#6-kho-gia-vị-của-rsl_rl-501)
7. [Domain randomization hiện có](#7-domain-randomization-hiện-có)
8. [Lỗi và khoảng trống đã phát hiện](#8-lỗi-và-khoảng-trống-đã-phát-hiện)
9. [Roadmap nâng cấp theo ưu tiên](#9-roadmap-nâng-cấp-theo-ưu-tiên)
10. [Tra cứu nhanh](#10-tra-cứu-nhanh)

---

## 1. PPO là gì — trực giác trước

Robot có **hai bộ não** tách rời:

- **Actor** (diễn viên) — nhìn tình hình, quyết định nhúc nhích khớp thế nào.
  Đây là cái duy nhất bạn đem lên phần cứng thật.
- **Critic** (giám khảo) — nhìn tình hình, chấm điểm "tình thế này về sau tốt hay xấu".
  Critic **không điều khiển gì cả**. Nó tồn tại chỉ để dạy Actor, rồi bị vứt đi khi deploy.

Vòng lặp:

1. Cho 4096 robot chạy thử 24 bước → thu dữ liệu.
2. Critic chấm điểm từng khoảnh khắc.
3. Hành động nào **tốt hơn mức Critic dự đoán** → tăng xác suất lặp lại. Tệ hơn → giảm.
4. Lặp lại.

**Chữ "P" đầu trong PPO = Proximal (gần).** Đây là toàn bộ ý tưởng cốt lõi: mỗi lần
chỉ cho phép sửa não *một chút*. RL rất dễ sập — sửa mạnh tay một lần, policy đang đi
tàm tạm bỗng "quên sạch" và không hồi phục được, vì dữ liệu tiếp theo do chính policy
hỏng đó sinh ra. `clip_param=0.2` là cái dây xích: xác suất mới không lệch quá ±20%
so với xác suất cũ.

Vì sao cần dây xích mà không chỉ hạ learning rate? Vì trong RL, cùng một learning rate
có thể gây thay đổi hành vi nhỏ xíu ở vùng này nhưng khổng lồ ở vùng khác — phụ thuộc
độ dốc của phân phối. Clipping đo trực tiếp cái ta quan tâm (hành vi đổi bao nhiêu),
không đo gián tiếp qua kích thước bước.

---

## 2. Kiến trúc mạng

Cả Actor và Critic đều là **MLP thuần** — không CNN, không LSTM.

```
ACTOR (điều khiển)
  obs 60 ──► Linear(60,256) ─ELU─► Linear(256,256) ─ELU─► Linear(256,128) ─ELU─► Linear(128,10) ──► mean 10 chiều
                                                                                       │
                                                    std: nn.Parameter(10,) khởi tạo 1.0 ─┘  (KHÔNG phụ thuộc input)

CRITIC (chấm điểm)
  obs 60 ──► Linear(60,256) ─ELU─► Linear(256,256) ─ELU─► Linear(256,128) ─ELU─► Linear(128,1) ──► value 1 chiều
```

### Đếm tham số

| Lớp | Actor | Critic |
|---|---:|---:|
| Linear(60, 256) | 15.616 | 15.616 |
| Linear(256, 256) | 65.792 | 65.792 |
| Linear(256, 128) | 32.896 | 32.896 |
| Linear(128, out) | 1.290 (out=10) | 129 (out=1) |
| `std_param` | 10 | — |
| **Tổng** | **115.604** | **114.433** |

Tổng ~230k tham số. Rất nhỏ — chạy được trên vi điều khiển, phù hợp deploy thật.

### Ba điểm cần biết về Actor

**`init_std=1.0` — độ ngẫu nhiên khám phá.**
Actor không xuất một con số cứng. Nó xuất *tâm* của phân phối chuẩn rồi bốc thăm quanh
đó. `std=1.0` nghĩa là lúc mới train nó quẫy khá mạnh. Std là 10 tham số **học được**
và **không phụ thuộc observation** (`std_type="scalar"`) — mạng tự giảm dần khi tự tin.
Lúc `play.py` thì bỏ ngẫu nhiên, lấy thẳng tâm (`stochastic_output=False`).

Muốn std thay đổi theo tình huống (mạnh dạn ở tư thế an toàn, dè dặt khi sắp ngã)?
Có sẵn `HeteroscedasticGaussianDistribution` (`state_dependent_std=True`).

**Không có orthogonal init.**
`MLP.init_weights()` tồn tại trong thư viện nhưng **không ai gọi**. `MLPModel.__init__`
chỉ gọi `distribution.init_mlp_weights()`, mà với `GaussianDistribution` hàm đó là `pass`.
Kết quả: dùng init mặc định PyTorch (Kaiming uniform). Đây là sai lệch so với công thức
PPO chuẩn (thường dùng orthogonal, gain nhỏ ở lớp cuối). Thực tế rsl_rl vẫn train tốt
không cần, nhưng đáng biết.

**Actor và Critic hoàn toàn tách rời** — không chia sẻ lớp nào. Đúng chuẩn cho robotics
(chia sẻ trunk hay gây xung đột gradient giữa hai mục tiêu).

---

## 3. Observation & Action space

### Observation: 60 chiều

Xem [`_get_observations`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_walk10dof_env.py#L262).

| Phần | Kích thước | Chi tiết |
|---|---:|---|
| Lịch sử IMU | 20 | **4 khung** × (roll, pitch = 2 + gyro xyz = 3) |
| Lịch sử hành động | 40 | **4 khung** × 10 khớp |

**Framestack = 4, không phải 8.** Buffer khai báo tại dòng 209-211:

```python
self.orient_h = torch.zeros(self.num_envs, 4, 3, ...)   # yaw bị bỏ khi ghép obs
self.gyro_h   = torch.zeros(self.num_envs, 4, 3, ...)
self.act_hist = torch.zeros(self.num_envs, 4, 10, ...)
```

Cửa sổ thời gian: 4 khung × 0,05 s = **0,2 giây**. Ngắn — một chu kỳ bước chân người
khoảng 1 s, robot chỉ thấy 1/5 chu kỳ.

Framestack thay cho RNN: MLP không có trí nhớ, nên phải nhét lịch sử vào input để nó
suy ra *xu hướng* (đang nghiêng thêm hay đang gượng lại). Rẻ hơn LSTM, nhưng cửa sổ cố định.

**Nâng framestack lên 8** → `num_observations = 120`, `observation_space.shape = (120,)`,
lớp Linear đầu thành `Linear(120, 256)`. **Mọi checkpoint cũ hỏng.**

**Những gì policy KHÔNG thấy** (quan trọng):

- Không có **command vận tốc**. Hướng đi nằm trong `act_direction`, dùng để tính reward,
  nhưng policy không nhìn thấy. Robot phải đoán mình đang được yêu cầu làm gì.
  Mọi công thức locomotion chuẩn đều đưa command vào obs.
- Không có **trạng thái khớp thật** (`joint_pos`, `joint_vel`). Obs chỉ chứa *lệnh đã gửi*
  (`act_hist`). Với backlash 2,5° và delay, lệnh gửi ≠ vị trí thật → robot mù về chân mình.
- Không có vận tốc thân, không có lực tiếp xúc, không có yaw.

### Action: 10 chiều

Xem [`_pre_physics_step`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_walk10dof_env.py#L307).

Action **không phải góc khớp** mà là **lượng thay đổi góc**:

```python
actions_cpy = torch.clamp(actions, -3.0, 3.0)
self.cmd_actions += actions_cpy * 2 / 3     # cộng dồn, tối đa 2°/bước
```

Thứ tự khớp và biên servo (độ):

| Index | Khớp | min | max | Vai trò |
|---:|---|---:|---:|---|
| 0,1 | Bub_L/R | -15 | 15 | frontal/roll — hỗ trợ cân bằng |
| 2,3 | Hip_L/R | -30 | 30 | sagittal — khớp chính tạo dáng đi |
| 4,5 | Twist_L/R | -20 | 20 | yaw — dự phòng rẽ hướng |
| 6,7 | Knee_L/R | -70 | 5 | chỉ gập một chiều âm |
| 8,9 | Foot_L/R | -45 | 30 | |

`base_pose` = toàn 0° (đứng thẳng chân).

Chuỗi xử lý action mỗi bước: **clamp → cộng dồn → mô hình backlash → nhiễu Gaussian
σ=0.5° → clamp vào biên servo → delay 2–6 substep → `set_joint_position_target`**.

### Nhịp thời gian

| Đại lượng | Giá trị |
|---|---|
| `sim.dt` | 0,005 s (200 Hz vật lý) |
| `decimation` | 10 |
| **Bước RL** | **0,05 s → điều khiển 20 Hz** |
| `episode_length_s` | 10 s = **200 bước** |
| Rollout mỗi iteration | 24 bước = 1,2 s kinh nghiệm |

> Rollout (1,2 s) ngắn hơn episode (10 s) rất nhiều. Critic phải gánh phần "nhìn xa" —
> nó bù cho đoạn tương lai chưa quan sát được. Đây là lý do chất lượng Critic quan trọng
> hơn nhiều so với cảm giác trực giác ban đầu.

---

## 4. Toàn bộ hyperparameter

### Thu thập dữ liệu

| Tham số | Giá trị | Nghĩa |
|---|---:|---|
| `scene.num_envs` | 4096 | robot chạy song song trên GPU |
| `num_steps_per_env` | 24 | mỗi robot đi 24 bước rồi dừng lại học |
| `max_iterations` | 5000 | tổng số vòng học |
| `save_interval` | 50 | 50 vòng lưu checkpoint |
| `experiment_name` | `transformer_walk` | thư mục log |

Suy ra:

- **98.304 mẫu** mỗi iteration (24 × 4096)
- **491 triệu bước môi trường** nếu chạy hết 5000 iteration
- 1,2 giây kinh nghiệm mỗi robot mỗi iteration

### Tính "hành động này tốt cỡ nào"

| Tham số | Giá trị | Nghĩa |
|---|---:|---|
| `gamma` | 0.99 | Chiết khấu. Phần thưởng 1 bước sau × 0.99, 2 bước sau × 0.99²… Chân trời hiệu dụng ≈ 1/(1−γ) = **100 bước = 5 giây**. Ngã sau 5 s thì bây giờ đã "thấy đau". |
| `lam` | 0.95 | GAE lambda. Cân bằng giữa "tin số liệu thật đo được" (đúng nhưng nhiễu) và "tin dự đoán Critic" (mượt nhưng có thể sai). 0.95 nghiêng về số liệu thật. |

Chỉnh `gamma`: robot đi bộ thường 0.99–0.995. Tăng lên 0.995 → chân trời 200 bước = 10 s
= trọn episode, nhưng Critic khó học hơn.

### Cập nhật mạng

| Tham số | Giá trị | Nghĩa |
|---|---:|---|
| `num_learning_epochs` | 5 | nhai lại dữ liệu 5 lượt |
| `num_mini_batches` | 4 | chia thành 4 lô, mỗi lô 24.576 mẫu |
| | | → **20 bước gradient** mỗi iteration |
| `clip_param` | 0.2 | dây xích ±20% |
| `value_loss_coef` | 1.0 | Critic học nặng ngang Actor |
| `use_clipped_value_loss` | True | Critic cũng bị xích |
| `entropy_coef` | 0.01 | thưởng cho việc "còn phân vân" — chống chốt sớm một dáng đi tệ |
| `max_grad_norm` | 1.0 | cắt gradient, chống nổ số |
| `optimizer` | adam (mặc định) | chọn được adam/adamw/sgd/rmsprop |

`entropy_coef=0.01` cao gấp đôi mặc định ANYmal (0.005). Với bài toán chưa hội tụ thì hợp lý.

### Learning rate — cơ chế tự lái

```python
learning_rate = 1e-3
schedule = "adaptive"
desired_kl = 0.01
```

`1e-3` chỉ là **giá trị khởi đầu**. `schedule="adaptive"` bật cơ chế tự chỉnh tốc độ học
đo bằng **KL divergence** — con số cho biết "não mới khác não cũ bao nhiêu".

Mục tiêu: mỗi lần cập nhật, não chỉ nên đổi đúng mức `desired_kl = 0.01`.
Logic thật ([`ppo.py:281-284`](file:///home/cat21/miniconda3/envs/isaacsim/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py)):

```python
if kl_mean > desired_kl * 2.0:                      # đổi quá nhiều
    learning_rate = max(1e-5, learning_rate / 1.5)
elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0:  # đổi quá ít
    learning_rate = min(1e-2, learning_rate * 1.5)
```

Xe tự đạp phanh khi vào cua gấp, tự đạp ga khi đường thẳng. **Đây là lý do `lr` trong
TensorBoard nhảy lung tung — bình thường, không phải bug.**

Hệ quả thực tế: chỉnh `learning_rate` gần như vô nghĩa (bị ghi đè sau vài iteration).
Muốn học nhanh/chậm hơn thì chỉnh **`desired_kl`**. Tăng lên 0.02 → học bạo hơn; giảm
xuống 0.005 → thận trọng hơn.

rsl_rl **không có** LR annealing tuyến tính. Adaptive thay thế nó.

---

## 5. Vòng đời một iteration

```
┌─ THU THẬP ────────────────────────────────────────────────────────┐
│  4096 env × 24 bước                                                │
│  mỗi bước: actor(obs, stochastic=True) → action                    │
│            env.step() → obs', reward, done                         │
│            check NaN                                               │
│            nếu done vì HẾT GIỜ: reward += γ·V(s)   ← quan trọng    │
│            lưu vào RolloutStorage                                  │
│  → 98.304 transitions                                              │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ TÍNH ADVANTAGE ──────────────────────────────────────────────────┐
│  V(s_last) = critic(obs_cuối)                                      │
│  duyệt ngược:  δ = r + γ·V(s') − V(s)                              │
│                A = δ + γ·λ·A                                       │
│  returns = A + V                                                   │
│  chuẩn hoá A trên toàn batch: (A − mean)/(std + 1e-8)              │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ TỐI ƯU: 5 epoch × 4 minibatch = 20 bước gradient ────────────────┐
│  shuffle bằng torch.randperm                                       │
│  cho mỗi minibatch:                                                │
│    ratio = exp(logπ_mới − logπ_cũ)                                 │
│    L_actor  = max(−A·ratio, −A·clip(ratio, 0.8, 1.2))              │
│    L_critic = max((V−R)², (V_clip−R)²)          [use_clipped]      │
│    L = L_actor + 1.0·L_critic − 0.01·entropy                       │
│                                                                    │
│    đo KL → tự chỉnh learning_rate                                  │
│    backward                                                        │
│    clip_grad_norm(actor, 1.0);  clip_grad_norm(critic, 1.0)        │
│    adam.step()                                                     │
└────────────────────────────────────────────────────────────────────┘
                              │
                       clear storage → lặp
```

Lưu ý kỹ thuật: grad norm được clip **riêng** cho actor và critic (không clip chung một
lần như CleanRL). Ảnh hưởng nhẹ, nhưng khác biệt có thật.

---

## 6. Kho "gia vị" của rsl_rl 5.0.1

Đối chiếu với checklist chuẩn ("The 37 Implementation Details of PPO", Huang et al.),
cộng nhóm riêng cho robotics.

### 6.1 Nhóm PPO cốt lõi

| Gia vị | Có? | Trong project | Ghi chú |
|---|:--:|---|---|
| GAE(λ) | ✅ | **bật** λ=0.95 | |
| Clipped surrogate | ✅ | **bật** 0.2 | |
| Clipped value loss | ✅ | **bật** | |
| Advantage normalization | ✅ | **bật** toàn batch | có tuỳ chọn `normalize_advantage_per_mini_batch` |
| Entropy bonus | ✅ | **bật** 0.01 | |
| Grad-norm clipping | ✅ | **bật** 1.0 | clip riêng actor/critic |
| Minibatch shuffle | ✅ | tự động | `torch.randperm` |
| Adaptive LR theo KL | ✅ | **bật** | đặc sản rsl_rl |
| LR annealing tuyến tính | ❌ | — | không có, adaptive thay thế |
| Orthogonal init | ⚠️ | **không chạy** | hàm có, không ai gọi |
| Value/return normalization | ❌ | — | không có |
| Reward clipping/scaling | ❌ | — | env tự lo |
| Chọn optimizer | ✅ | adam | adam/adamw/sgd/rmsprop |
| NaN check | ✅ | **bật** mặc định | `check_for_nan` |
| `clip_actions` ở wrapper | ✅ | None | env đã tự clamp ±3 |

### 6.2 Nhóm robotics — phần đáng giá

| Gia vị | Có? | Trong project | Vì sao quan trọng cho sim-to-real |
|---|:--:|:--:|---|
| **Time-limit bootstrapping** | ✅ | **BẬT tự động** | Khi episode kết thúc vì *hết giờ* (không phải ngã), cộng `γ·V(s)` vào reward ([`ppo.py:174-179`](file:///home/cat21/miniconda3/envs/isaacsim/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py)). Không có nó, robot học được "sống tới giây thứ 10 là bị phạt" — sai hoàn toàn. Rất nhiều lib PPO thiếu chỗ này. |
| **Asymmetric actor-critic** (`obs_groups`) | ✅ | ❌ **KHÔNG dùng** | Critic được nhìn thông tin đặc quyền (vận tốc thật, ma sát thật, lực tiếp xúc) mà Actor không có. Critic chấm chính xác → Actor học nhanh, mà Actor vẫn chỉ dùng cảm biến có thật trên robot. |
| **Symmetry augmentation / mirror loss** | ✅ | ❌ **KHÔNG dùng** | Robot 2 chân đối xứng trái-phải. Bật = nhân đôi dữ liệu miễn phí + ép dáng đi cân đối. Mittal et al. 2024, làm ra chính cho IsaacLab. |
| **Empirical obs normalization** | ✅ | ❌ tắt | Running mean/std tự học, **lưu trong checkpoint** → deploy sang robot thật không phải chép tay hằng số scale |
| **Recurrent policy** (LSTM/GRU) | ✅ | ❌ dùng MLP | `RslRlRNNModelCfg` + minibatch generator riêng có masking/padding |
| **Distillation** (teacher-student) | ✅ | ❌ | Thuật toán **riêng** (`RslRlDistillationRunnerCfg`). Train teacher với thông tin đặc quyền → chưng cất sang student chỉ dùng cảm biến thật. Công thức của ANYmal/Unitree. |
| **RND** (intrinsic reward) | ✅ | ❌ | Thưởng khám phá khi reward thưa |
| **State-dependent std** | ✅ | ❌ | `HeteroscedasticGaussianDistribution` |
| **CNN encoder** + `share_cnn_encoders` | ✅ | ❌ | cho depth camera / heightmap |
| **Multi-GPU** | ✅ | ❌ | all_reduce thủ công, sync cả KL |

### 6.3 Phán quyết

**Thư viện đủ đô. Rất đủ.** rsl_rl là codebase ETH Zurich dùng cho ANYmal và Unitree
dùng cho Go2/H1 — có nhiều policy đi bộ chạy trên phần cứng thật nhất hiện nay. Bốn thứ
then chốt cho sim-to-real đều có sẵn: **time-limit bootstrapping, asymmetric critic,
symmetry, distillation**.

**Không có lý do đổi sang skrl/SB3/CleanRL.**

Vấn đề: project đang dùng gần như **0%** trong số gia vị robotics. Config hiện tại là
vanilla PPO mặc định — đúng, chạy được, nhưng trống trơn.

---

## 7. Domain randomization hiện có

Phần này của env thực ra **khá mạnh** so với mặt bằng chung. Ghi lại để biết cái gì đã có.

### Randomize mỗi lần reset (per-env)

| Đại lượng | Dải | Code |
|---|---|---|
| Ma sát khớp | 0,300 – 0,500 | `self.frictions` |
| Giới hạn moment | 9,270 – 10,299 | `self.torques` |
| Damping khớp | 0,600 – 0,700 | `self.dampings` |
| Bias IMU roll | −0,10 – 0,15 rad | `imu_bias_range` |
| Bias IMU pitch | −0,42 – −0,12 rad | |
| Bias IMU yaw | −0,03 – 0,03 rad | |

### Nhiễu mỗi bước

| Đại lượng | Giá trị |
|---|---|
| Nhiễu orientation | σ = 0,04 rad |
| Nhiễu gyro | σ = 0,15 rad/s |
| Drift gyro | 1e-4 · randn · dt (tích luỹ) |
| Nhiễu servo | σ = 0,5° |
| Backlash (rơ bánh răng) | 2,5° |
| Delay actuator | 2–6 substep ⚠️ (xem §8.3) |
| Làm tròn obs | 4 chữ số thập phân |

Khi `domain_rand=False`: bias IMU cố định `[0, -0.193, 0]`, nhưng nhiễu và drift **vẫn chạy**.

### Chưa randomize

- Khối lượng / tâm khối các link
- Ma sát mặt đất (cố định static 2.0 / dynamic 2.5)
- Địa hình (chỉ mặt phẳng)
- Nhiễu loạn ngoài (push robot)
- Trễ observation (chỉ trễ actuator)

---

## 8. Lỗi và khoảng trống đã phát hiện

Rà soát env ngày 2026-08-16. Xếp theo mức ảnh hưởng.

### 8.1 `gear_position` và `last_direction` KHÔNG được reset — **nghiêm trọng**

Trong [`_reset_idx`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_walk10dof_env.py#L405) chỉ có:

```python
self.orient_h[env_ids] = 0.0
self.gyro_h[env_ids]   = 0.0
self.act_hist[env_ids, :] = self.base_pose[0]
self.cmd_actions[env_ids] = self.base_pose[0]
```

Thiếu `gear_position` và `last_direction`. Nhưng khớp vật lý **đã** được reset về
`default_joint_pos`.

Hệ quả: sang episode mới, mô hình backlash tin rằng bánh răng đang ở vị trí của episode
*trước*, trong khi khớp thật đã về 0. Vài bước đầu mỗi episode, backlash tính sai hoàn
toàn. Với 4096 env reset liên tục, sai số này hiện diện suốt quá trình train.

**Sửa** — thêm vào cuối `_reset_idx`:

```python
self.gear_position[env_ids]  = self.base_pose[0]
self.last_direction[env_ids] = 0.0
```

### 8.2 `act_hist` reset không khớp với lúc khởi tạo — **trung bình**

Hai đường code dùng hai công thức khác nhau:

| Nơi | Dòng | Công thức |
|---|---:|---|
| `__init__` | 213 | `(base_pose − servo_min)/(servo_max − servo_min)·2 − 1` (chuẩn hoá) |
| `_reset_idx` | 460 | `base_pose[0]` (**giá trị thô**, = 0) |

`base_pose` toàn 0° nhưng biên servo bất đối xứng, nên hai công thức lệch nhau:

| Khớp | Biên | Đúng | Khi reset |
|---|---|---:|---:|
| Bub / Hip / Twist | đối xứng | 0.000 | 0.000 ✓ |
| **Knee** | [−70, 5] | **0.867** | 0.000 ✗ |
| **Foot** | [−45, 30] | **0.200** | 0.000 ✗ |

Đầu mỗi episode, policy đọc lịch sử Knee/Foot sai. Một bước sau giá trị đúng tràn vào —
nó thấy một "cú giật" không có thật.

**Sửa** — dùng chung một hàm cho cả hai đường:

```python
def _normalize_pose(self, pose):
    return torch.clamp(
        (pose - self.servo_min) / (self.servo_max - self.servo_min) * 2 - 1, -1, 1
    )
```

### 8.3 `act_delay` là một số vô hướng dùng chung cho cả 4096 env — **trung bình**

```python
self.act_delay = torch.randint(low=..., high=..., size=(1,)).item()
```

`size=(1,)` → mọi env chia sẻ **đúng một** độ trễ tại mỗi bước, và nó đổi mới **mỗi bước**.

Ý định của domain randomization là mỗi robot có servo với đặc tính trễ *khác nhau và ổn
định trong episode*, để policy học chịu đựng cả dải trễ. Hiện tại nó chỉ thấy một nhiễu
trễ toàn cục — mất phần lớn giá trị.

**Sửa** — sample per-env trong `_reset_idx`, giữ cố định suốt episode:

```python
# trong _reset_idx
self.act_delay[env_ids] = torch.randint(
    self.cfg.actuator_delay_min, self.cfg.actuator_delay_max + 1,
    (len(env_ids),), device=self.device
)
```
`_apply_action` cũng phải đổi sang so sánh theo vector thay vì scalar.

### 8.4 `obs_groups` không khai báo → Critic mù ngang Actor — **nghiêm trọng về hiệu quả**

Env chỉ trả về một nhóm `{"policy": obs_buffer}`. Config không set `obs_groups`, nên
hàm `resolve_obs_groups` rơi vào nhánh mặc định: gán nhóm `"policy"` cho **cả** actor
lẫn critic.

Critic đang chấm điểm chỉ dựa trên 20 số IMU + 40 lệnh cũ. Nó không biết robot đang di
chuyển bao nhanh, khớp thật ở đâu, chân có chạm đất không.

Critic đoán mò → `advantage` nhiễu → Actor học chậm. Đây có thể là nguyên nhân chính
khiến `ep_len` mắc kẹt.

### 8.5 Thiếu command vận tốc trong observation — **nghiêm trọng về thiết kế**

Hướng đi nằm trong `act_direction` (dùng tính reward) nhưng policy không thấy. Robot
phải đoán mình đang được yêu cầu đi hướng nào. Mọi công thức locomotion chuẩn đều đưa
command vào obs.

---

## 9. Roadmap nâng cấp theo ưu tiên

Xếp theo tỉ lệ **lợi ích / công sức**. Làm từ trên xuống.

### Bậc 0 — sửa lỗi, không đổi shape, checkpoint cũ vẫn dùng được

1. **Reset `gear_position` + `last_direction`** (§8.1) — 2 dòng, rẻ nhất, ảnh hưởng mọi episode
2. **Thống nhất chuẩn hoá `act_hist`** (§8.2) — 1 hàm helper
3. **`act_delay` per-env** (§8.3) — ~5 dòng

Sau bậc này nên train lại từ đầu để đo lại baseline sạch.

### Bậc 1 — asymmetric critic (giữ nguyên Actor, checkpoint Actor vẫn khớp)

Env trả thêm một nhóm observation:

```python
# trong _get_observations
return {
    "policy": obs_buffer,                    # 60 chiều — GIỮ NGUYÊN
    "critic": privileged_buffer,             # thông tin đặc quyền
}
```

`privileged_buffer` nên chứa: `root_lin_vel_b`, `root_ang_vel_b`, `joint_pos` thật,
`joint_vel` thật, ma sát/damping/torque đã sample, lực tiếp xúc chân, chiều cao thân.

Config:

```python
obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
```

Actor không đổi shape → **checkpoint Actor cũ vẫn load được**, chỉ Critic phải học lại.

Tham chiếu: [`anymal_d/agents/rsl_rl_ppo_cfg.py:25`](file:///home/cat21/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py)

### Bậc 2 — command vận tốc vào observation

Thêm 2–3 chiều command (vx, vy, wz hoặc chỉ hướng) vào obs. Đây là thay đổi shape →
checkpoint cũ hỏng, nhưng gần như bắt buộc nếu muốn robot đi có chủ đích.

Nhân tiện cân nhắc thêm `joint_pos`/`joint_vel` thật vào actor obs — robot thật có
encoder, nên đây là thông tin **hợp lệ** cho sim-to-real, không phải đặc quyền.

### Bậc 3 — symmetry augmentation

Viết hàm hoán vị L↔R + đổi dấu các trục roll/yaw, rồi:

```python
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

algorithm = RslRlPpoAlgorithmCfg(
    ...,
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func=transformer_symmetry.compute_symmetric_states,
    ),
)
```

Chữ ký hàm:

```python
@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    # trả về (obs_aug, actions_aug), mỗi cái batch_size × num_aug
    # num_aug = 2 cho đối xứng gương trái-phải
```

Với robot 2 chân chỉ có **1 phép đối xứng** (gương trái-phải) → `num_aug = 2`. ANYmal
4 chân có 4 phép (gốc / trái-phải / trước-sau / chéo) → `num_aug = 4`.

Mẫu để copy: [`velocity/mdp/symmetry/anymal.py`](file:///home/cat21/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py)

Cần cẩn thận với layout obs: phải map đúng index của từng khớp trong cả 4 khung lịch sử.

### Bậc 4 — obs normalization

```python
actor  = RslRlMLPModelCfg(..., obs_normalization=True)
critic = RslRlMLPModelCfg(..., obs_normalization=True)
```

Đổi shape state_dict (thêm buffer `_mean`, `_var`, `_std`, `count`) → checkpoint cũ hỏng.
Bù lại: thống kê nằm trong checkpoint, deploy sang robot thật không cần chép tay hằng số.

### Bậc 5 — recurrent policy hoặc distillation

Chỉ làm khi các bậc trên đã ổn.

**Recurrent** — thay framestack 4 bằng LSTM, cửa sổ trí nhớ không giới hạn:

```python
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

actor = RslRlRNNModelCfg(
    hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1,
)
```

**Distillation** — teacher dùng full privileged obs, student chỉ dùng cảm biến thật:

```python
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlDistillationAlgorithmCfg

obs_groups = {"student": ["policy"], "teacher": ["policy", "critic"]}
algorithm = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=2, learning_rate=1e-3, gradient_length=15
)
```

Mẫu: [`anymal_d/agents/rsl_rl_distillation_cfg.py`](file:///home/cat21/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_distillation_cfg.py)

### Bậc 6 — domain randomization bổ sung

Nếu đã đi được trong sim mà chưa sang thật được: randomize khối lượng link, ma sát mặt
đất, thêm push ngẫu nhiên, thêm trễ observation.

---

## 10. Tra cứu nhanh

### Lệnh

```bash
cd transformer_nam

# Train
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 1500

# Play (checkpoint cần đường dẫn ĐẦY ĐỦ)
./run.sh scripts/rsl_rl/play.py --task Transformer-Walk10DOF-Direct-v0 --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/transformer_walk/<run>/model_1499_rslrl5.pt"

# Biểu đồ
tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006
```

### Con số cần nhớ

| Đại lượng | Giá trị |
|---|---:|
| Observation | 60 (20 IMU + 40 action, framestack **4**) |
| Action | 10 (delta góc, ±2°/bước) |
| Mạng | [256, 256, 128], ELU, ~230k tham số |
| Tần số điều khiển | 20 Hz |
| Episode | 200 bước = 10 s |
| Mẫu / iteration | 98.304 |
| Bước gradient / iteration | 20 |
| Chân trời γ | ~100 bước = 5 s |

### Bảng "muốn đổi X thì sửa ở đâu"

| Muốn | Sửa |
|---|---|
| Học nhanh/chậm hơn | `desired_kl` (**không** phải `learning_rate` — nó bị adaptive ghi đè) |
| Khám phá nhiều hơn | `entropy_coef` ↑, hoặc `init_std` ↑ |
| Nhìn xa hơn | `gamma` 0.99 → 0.995 |
| Nhiều dữ liệu mỗi lần học | `num_steps_per_env` ↑ (24 → 48) |
| Mạng to hơn | `hidden_dims` — **checkpoint cũ hỏng** |
| Framestack 4 → 8 | `orient_h`/`gyro_h`/`act_hist` shape + `num_observations` 60 → 120 — **checkpoint cũ hỏng** |
| Đổi shape nào cũng được mà giữ Actor | chỉ thêm nhóm `"critic"` vào `obs_groups` |

### Đọc TensorBoard

| Chỉ số | Ý nghĩa | Dấu hiệu tốt |
|---|---|---|
| `Train/mean_episode_length` | robot trụ được bao lâu | tăng đều, tiến tới 200 |
| `Loss/value` | Critic dự đoán sai bao nhiêu | giảm rồi ổn định |
| `Loss/surrogate` | — | dao động quanh 0, không phân kỳ |
| `Loss/learning_rate` | LR tự chỉnh | nhảy lung tung là **bình thường** |
| `Policy/mean_noise_std` | độ ngẫu nhiên | giảm dần từ 1.0 |
| `Train/mean_reward` | — | tăng, nhưng đọc kèm `episode_length` |

Nếu `mean_noise_std` tụt rất nhanh về ~0 mà `episode_length` không tăng → policy chốt
sớm vào một dáng tệ. Tăng `entropy_coef`.

### Đường dẫn file

| Nội dung | Đường dẫn |
|---|---|
| PPO config | `transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/agents/rsl_rl_ppo_cfg.py` |
| Env 10DOF | `.../tasks/direct/transformer_nam/transformer_walk10dof_env.py` |
| Env 10DOF6 | `.../tasks/direct/transformer_nam/transformer_walk10dof6_env.py` |
| Đăng ký task | `.../tasks/direct/transformer_nam/__init__.py` |
| Script train | `transformer_nam/scripts/rsl_rl/train.py` |
| Thư viện rsl_rl | `~/miniconda3/envs/isaacsim/lib/python3.12/site-packages/rsl_rl/` |
| Schema config IsaacLab | `~/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` |
| Mẫu ANYmal | `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/` |

### Đọc thêm

- **PPO gốc** — Schulman et al. 2017, *Proximal Policy Optimization Algorithms*
- **GAE** — Schulman et al. 2015, *High-Dimensional Continuous Control Using GAE*
- **Checklist implementation** — Huang et al., *The 37 Implementation Details of PPO*
- **Symmetry trong RL robot** — Mittal et al. 2024 (chính là tác giả `RslRlSymmetryCfg`)
- **rsl_rl gốc** — Rudin et al. 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL*
- Source code là tài liệu tốt nhất: `rsl_rl/algorithms/ppo.py` chỉ 545 dòng, đọc hết được
