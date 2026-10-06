# Gait Clock trong RL — Tổng hợp thảo luận

> File này tóm tắt những gì mình đã bàn về **đồng hồ nhịp bước** (gait clock) để train robot 2 chân bằng RL.
> Viết kiểu giải thích đơn giản, giữ thuật ngữ tiếng Anh, có trích dẫn dòng code để tự kiểm chứng.
> Ngày: 2026-10-06.

---

## 0. Bối cảnh của mình (robot + cách train)

- Robot **2 chân, 10 DOF** (degrees of freedom = số khớp cử động được).
- Servo **Feetech STS3120** (không tải ~132°/s) và **STS3215** (không tải ~270°/s), **điều khiển vị trí** (position control).
- Train bằng **Isaac Lab + rsl_rl PPO**.
- Policy: **MLP + 4 frame lịch sử** (200ms, 20Hz), obs ~60 chiều.
- Deploy: **ONNX trên Raspberry Pi**.
- Ưu tiên số 1: **bền chắc** (robust) — chịu được sai lệch sim-to-real, nhiễu, đẩy ngã.

**Hai thứ mình ĐÃ loại bỏ từ đầu:**
1. **KHÔNG dùng CPG + residual.**
2. **KHÔNG đưa quỹ đạo mẫu vào action.**

**Kế hoạch gait clock của mình:** pha `p ∈ [0,1]`, nhét `sin(2πp)` và `cos(2πp)` vào obs, thêm reward chạm đất theo pha.

---

## 1. Gait clock là gì (ELI5)

Tưởng tượng **cái máy đánh nhịp** (metronome) tích tắc đều. Mỗi vòng tích tắc = một bước đi.

- **Pha** `p` = cây kim chạy vòng tròn, từ 0 → 1 rồi lặp lại.
- `p=0`: đầu nhịp. `p=0.5`: giữa nhịp. `p→1`: cuối nhịp, sắp quay về 0.
- Hai chân **lệch nửa nhịp** (offset 0.5): chân trái đang chạm đất thì chân phải đang vung, và ngược lại.

**Vì sao cần nó?** Nếu không có nhịp, robot phải "tự mò" ra nhịp bước — chậm, hay ra dáng xấu (lê chân, nhảy cóc). Có đồng hồ, ta **ra lệnh nhịp** còn robot chỉ lo **làm sao bước cho khớp nhịp đó**.

**Vì sao encode bằng sin/cos chứ không nhét thẳng `p`?** Vì `p` nhảy đột ngột 0.99 → 0.00 (gãy khúc), mạng neuron ghét điểm gãy. `sin`/`cos` biến vòng tròn thành đường cong **liền mạch**, `p=0.99` và `p=0.01` nằm sát nhau. Hai số `(sin, cos)` xác định **duy nhất** một điểm trên vòng tròn, và đã nằm sẵn trong `[-1,1]` nên **không cần chuẩn hóa**.

---

## 2. Ba nguồn đã đọc + cơ chế của từng cái

### 2.1. unitree_rl_gym (bản Isaac Gym)

File: `unitree_rl_gym/legged_gym/envs/g1/g1_env.py`, `g1_config.py`

- **Tính pha** (`g1_env.py:66-80`): `period=0.8s`, `offset=0.5`.
  `phase = (episode_length_buf * dt) % period / period`.
  `phase_left = phase`, `phase_right = (phase + 0.5) % 1`.
  Reset pha là **ngầm**: đầu episode `episode_length_buf=0` → pha tự về 0.
- **Vào obs** (`g1_env.py:87-99`): chỉ lấy `sin/cos` của **pha trái** → **2 chiều**. (Chân phải không cần cho vào obs vì nó chỉ là trái + 0.5, mạng tự suy ra được.)
- **Reward pha** `_reward_contact` (`g1_env.py:124-133`): **XNOR**.
  ```python
  is_stance = self.leg_phase[:, i] < 0.55   # đồng hồ bảo "giờ nên chạm đất"
  contact   = contact_forces[..., 2] > 1    # cảm biến bảo "có chạm thật"
  res += ~(contact ^ is_stance)             # khớp nhau -> +1, lệch -> +0
  ```
  Ngưỡng `0.55 > 0.5` → cho ~10% **double support** (hai chân cùng chạm đất một lúc, giúp vững).
- **Lệnh vận tốc = 0:** KHÔNG có xử lý gì → robot vẫn **giậm chân tại chỗ** (march in place), vì reward pha vẫn ép nó theo nhịp.
- **Mạng:** `ActorCriticRecurrent`, `rnn_type='lstm'`, `rnn_hidden_size=64` → **LSTM**, KHÔNG phải MLP (`g1_config.py:97-105`).
- **Tần số:** `sim.dt=0.005`, `decimation=4` → điều khiển **50Hz**.
- **feet_air_time = 0.0** (`g1_config.py:81`) → tắt hẳn. Đây là **bằng chứng** rằng gait clock **thay thế** cơ chế air-time.

### 2.2. unitree_rl_lab (bản Isaac Lab — gần với setup của mình nhất)

File: `.../tasks/locomotion/mdp/observations.py`, `rewards.py`, `.../robots/g1/29dof/velocity_env_cfg.py`

- **Vào obs:** `gait_phase` ObsTerm trả 2 chiều `[sin, cos]` — NHƯNG đang bị **COMMENT OUT** trong cả nhóm policy lẫn critic (`velocity_env_cfg.py:203, 224`). Tức bản này **đang chạy không có clock trong obs**.
- **history_length = 5** (`velocity_env_cfg.py:206`) → frame-stacking → giống **MLP** (đúng kiểu của mình).
- **Reward pha** `feet_gait` (`rewards.py:174-200`): vẫn XNOR `reward += ~(is_stance ^ is_contact[:, i])`, `offset` là **list** `[0.0, 0.5]`, `threshold=0.55`, `period=0.8`.
  **Điểm quan trọng:** có **command gating** —
  ```python
  reward *= cmd_norm > 0.1   # lệnh vận tốc gần 0 -> reward pha = 0
  ```
  → cho phép robot **đứng yên** khi không có lệnh đi.
- **Lệnh = 0:** đứng yên (nhờ command gating ở trên + reward `stand_still`, `feet_contact_without_cmd`).

### 2.3. Humanoid-Gym (Gu et al. 2024)

File: `humanoid gym.pdf`

- **Clock vào obs:** `[sin(2πt/C_T), cos(2πt/C_T)]` = 2 chiều (Table I).
- **Frame Stack = 15** → **MLP** (Table II). Obs đơn = 47.
- **Reward 1 — Contact Pattern:** `φ(I_p(t) − I_d(t), ∞)` scale 1.0. `w=∞` → chỉ số cứng: khớp đúng pattern thì =1, sai thì =0. (Giống XNOR của unitree về tinh thần.)
- **Reward 2 — Joint Position Tracking:** `φ(θ − θ_target, 2)` scale **1.5**, trong đó **`θ_target` là quỹ đạo sin mẫu**. → **đây là "reference-in-reward"** (quỹ đạo mẫu nằm trong REWARD, không nằm trong action). Xem mục 4.
- **Lệnh = 0:** KHÔNG command-gating.
- **DR (domain randomization):** Table III có cả **System Delay [0,10]ms** — rất hợp với mục tiêu bền chắc của mình.
- **Tần số:** policy 100Hz / PD 1000Hz. Kernel: `φ(e,w) = exp(−w·‖e‖²)`.

### 2.4. Siekmann et al. 2021 — gốc rễ của mọi thứ

File: `periodic rew.pdf` ("Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition")

- **Công thức lõi:** `R_i = c_i · I_i(φ) · q_i(s)`
  - `I_i(φ)` = chỉ số pha (giai đoạn này là swing hay stance).
  - `q_i(s)` = đại lượng đo được: **lực chân** (foot force) hoặc **tốc độ chân** (foot speed).
  - `c_i` = hệ số phạt/thưởng.
- **Logic complementarity (bù trừ):**
  - **Swing (chân vung):** phạt **lực** (`c_swing_frc=−1`, `c_swing_spd=0`) → chân trên không nên có lực chạm.
  - **Stance (chân trụ):** phạt **tốc độ** (`c_stance_spd=−1`, `c_stance_frc=0`) → chân dưới không nên trượt.
  - **KHÔNG có quỹ đạo mẫu nào cả** → ít "cầm tay chỉ việc" nhất.
- **Von Mises smoothing** (tham số κ) làm mượt ranh giới swing/stance. Có tỉ lệ swing/stance `r`.
- **Clock vào obs:** `{sin(2π(φ+θ_left)/L), sin(2π(φ+θ_right)/L)}` — **chỉ 2 sin, KHÔNG có cos** + các tỉ lệ pha `r`.
- **Mạng:** **LSTM 2×128**. Output 30 = 10 joint pos + 10 P gain + 10 D gain.
- **Tần số:** policy 40Hz / PD 2000Hz.
- **Đứng yên:** ép **swing-ratio → 0** (cách nguyên lý nhất) + standing cost (Appendix).
- Deploy trên Cassie (blind), mirror loss để đối xứng.

---

## 3. "Cái thang cầm tay chỉ việc" (spectrum of hand-holding)

Xếp từ **ép mạnh** → **buông cho tự học**. Câu hỏi then chốt: **cái mẫu/nhịp đi vào ĐÂU, và ai vẽ ra dáng khớp?**

| Mức | Phương pháp | Cái mẫu đi vào đâu | Ai vẽ dáng khớp |
|---|---|---|---|
| 1 (ép nhất) | Replay thuần | = action luôn | Không ai, chép y |
| 2 | **CPG + residual** | ACTION (feedforward) | Phần lớn là CPG, policy chỉ sửa |
| 3 | **Humanoid-Gym** (sin-in-reward) | REWARD | **Policy tự vẽ hết**, sin nhắc bài |
| 4 | **unitree** (contact XNOR) | REWARD (chỉ timing) | Policy tự vẽ hết |
| 5 | **Siekmann** (lực/tốc độ) | REWARD (chỉ timing) | Policy tự vẽ hết |
| 6 (buông nhất) | air-time / emergent | Không có clock | Policy tự mò cả nhịp lẫn dáng |

→ Ba mức 3–4–5 đều "**policy tự vẽ dáng khớp**" — đó là **vùng mình muốn ở** (vì mình đã loại CPG+residual và reference-in-action).

---

## 4. CPG+residual ≠ reference-in-reward (chốt lại điểm hay nhầm)

Mình đã hỏi: "quỹ đạo mẫu trong Humanoid-Gym có phải CPG+residual không?" → **KHÔNG.** Khác ở **chỗ cái sin đi vào đâu.**

**CPG + residual (mức 2) — cái sin vào ACTION:**
```
lệnh gửi xuống khớp = sin_mẫu(pha)  +  policy_output(residual)
```
- Cái sin **trực tiếp quay khớp**. Policy chỉ học phần sửa nhỏ.
- Dù policy có "muốn" hay không, sin vẫn kéo chân đi. → Đây là thứ mình ĐÃ loại.

**Humanoid-Gym (mức 3) — cái sin vào REWARD:**
```
lệnh gửi xuống khớp = policy_output           # policy tự vẽ TOÀN BỘ
reward += điểm_nếu( khớp ≈ sin_mẫu )          # sin chỉ đứng ngoài chấm điểm
```
- Cái sin **không chạm vào khớp**. Policy **được phép cãi**, chỉ mất điểm.

**ELI5:**
- **CPG+residual** = xe tập đi có **bánh phụ tự đẩy** bạn theo đường; bạn chỉ lái nhẹ.
- **Humanoid-Gym** = cô giáo **làm mẫu rồi cho điểm** độ giống, nhưng **chân bạn tự bước**.
- **unitree/Siekmann** = cô giáo **không làm mẫu bước nào**, chỉ **vỗ nhịp**: "chân trái chạm phách này, chân phải phách kia".

**Lưu ý tinh tế:** cái `sin_mẫu` trong Humanoid-Gym thực chất **cũng là một CPG** (bộ dao động hình sin). Phần "CPG" giống nhau; cái khác là **có cộng vào action hay không**. Humanoid-Gym không cộng → không có phần "residual" → **không phải** kiến trúc CPG+residual.

---

## 5. Bảng so sánh 6 câu hỏi gốc

| | unitree_rl_gym | unitree_rl_lab | Humanoid-Gym | Siekmann 2021 |
|---|---|---|---|---|
| **(a) Tính pha** | period 0.8, offset 0.5, reset ngầm | như gym, offset list `[0,0.5]` | `t/C_T` | φ∈[0,1], offset `θ_L/θ_R`, Von Mises |
| **(b) Vào obs** | sin/cos pha trái = 2 chiều | 2 chiều (đang **comment out**) | sin/cos = 2 chiều | **2 sin (no cos)** + tỉ lệ r |
| **(c) Reward pha** | XNOR, ngưỡng 0.55 | XNOR + **command gating** | contact cứng `w=∞` + **sin-in-reward** | **lực/tốc độ bù trừ**, no reference |
| **(d) Lệnh = 0** | giậm chân (không xử lý) | **đứng yên** (gating) | không gating | **swing-ratio→0** |
| **(e) Deploy** | reconstruct `phase=count%period/period` | tương tự | 100Hz/1000Hz | 40Hz/2000Hz, blind |
| **(f) Mạng** | **LSTM 64** | MLP (history 5) | **MLP** (stack 15) | **LSTM 2×128** |

**Điểm mấu chốt về mạng:** clock là **tín hiệu phải quan sát được** (observable). Kể cả LSTM (nhớ được quá khứ) vẫn phải **nhét clock vào obs**, nếu không bài toán thành POMDP (robot không biết đang ở pha nào). → Mình dùng MLP + 4 frame thì **bắt buộc** clock nằm trong obs.

---

## 6. So với `_reward_feet_air_time`

File: `unitree_rl_gym/legged_gym/envs/base/legged_robot.py:703-713`
```python
rew_airTime = sum((feet_air_time - 0.5) * first_contact)
```
- **Không có clock.** Chỉ thưởng khi chân ở trên không ≥ 0.5s rồi mới chạm (first contact).
- Đây là cách **để gait tự nổi lên** (emergent) — mức 6, buông nhất.
- unitree **tắt nó** (`feet_air_time=0.0`) và thay bằng `contact` (clock). → Hai cái này **thay thế nhau**, không dùng chung.

---

## 7. Phân tích servo của mình (đã bàn)

- STS3120 ~132°/s, STS3215 ~270°/s (**không tải**). Có tải thực tế thường tụt còn ~70–90 / ~150–180°/s.
- Dù tụt, vẫn **nhanh gấp 2–4 lần** cái mình lo ban đầu (40°/s) → **period 0.8s của unitree là khả thi**, có thể kéo giãn ra **1.0–1.2s** cho an toàn.
- Vì servo **điều khiển vị trí**: mô hình torque-speed curve kiểu `UnitreeActuator` (`unitree_actuators.py`) **có thể lệch** với Feetech. Hướng hợp lý hơn: dùng **PD / ImplicitActuator** với `velocity_limit` + `stiffness` **đo từ step response thực**.
- (Phần model actuator Feetech đang **tạm gác** theo yêu cầu.)

### Ghi chú về `UnitreeActuator` (tham khảo)
`unitree_actuators.py` cài **đường cong torque-speed** đầy đủ: `X1`=tốc độ tại điểm gãy, `X2`=tốc độ không tải, `Y1`=mô-men cùng chiều, `Y2`=ngược chiều, `Fs/Fd`=ma sát tĩnh/động, `armature`. Đây là template nếu sau này muốn làm `FeetechActuatorCfg` (ví dụ STS3120 X2≈2.30 rad/s, STS3215 X2≈4.71 rad/s).

---

## 8. Gợi ý đặt vào Isaac Lab Manager (cho setup của mình)

- **ObservationTerm:** thêm `gait_phase` (2 chiều sin/cos) vào nhóm `policy` (và `critic` nếu dùng asymmetric). Chính là term đang bị comment ở `velocity_env_cfg.py:203`.
- **RewardTerm:** `feet_gait` (`rewards.py:174`) với `period`, `offset=[0.0,0.5]`, `threshold≈0.55`, `command_name="base_velocity"` (để có command gating → đứng yên được).

---

## 9. Những quyết định CÒN MỞ (cần chốt)

1. **Reward pha dùng loại nào?**
   - (A) **contact boolean / XNOR** (unitree) — đơn giản, dễ, nhưng thô (chỉ biết chạm/không).
   - (B) **lực/tốc độ bù trừ** (Siekmann) — mượt hơn, nguyên lý hơn, hợp "bền chắc", nhưng cần đọc được foot force/speed đáng tin trong sim.

2. **Xử lý đứng yên (lệnh = 0) thế nào?**
   - (A) **command gating** (unitree_rl_lab) — dễ cài.
   - (B) **swing-ratio → 0** (Siekmann) — nguyên lý hơn, đứng/đi liền mạch.

3. **Có cho phép "sin-in-reward" (Humanoid-Gym Joint Position Tracking) không?**
   - Nó KHÔNG phải CPG+residual, nhưng vẫn là "nhắc bài qua điểm thưởng".
   - Nếu triết lý của mình là "chỉ ra nhịp, để robot tự tìm dáng khớp" → **nên bỏ** cái này, chỉ giữ clock-timing kiểu unitree/Siekmann.

4. **(Tạm gác)** Model actuator Feetech: đo step response, `velocity_limit`, latency.

---

## 10. Nguồn & vị trí code

| Nội dung | File:dòng |
|---|---|
| Tính pha (gym) | `unitree_rl_gym/legged_gym/envs/g1/g1_env.py:66-80` |
| sin/cos vào obs (gym) | `.../g1_env.py:87-99` |
| `_reward_contact` XNOR | `.../g1_env.py:124-133` |
| Config gym (rewards, LSTM) | `.../g1/g1_config.py:68-105` |
| `_reward_feet_air_time` | `.../envs/base/legged_robot.py:703-713` |
| `_compute_torques` | `.../envs/base/legged_robot.py:308-330` |
| `gait_phase` ObsTerm (lab) | `unitree_rl_lab/.../mdp/observations.py:10-19` |
| `feet_gait` reward (lab) | `unitree_rl_lab/.../mdp/rewards.py:174-200` |
| Env cfg (lab, comment-out clock) | `unitree_rl_lab/.../robots/g1/29dof/velocity_env_cfg.py:203,224,206,299-308` |
| `UnitreeActuator` torque-speed | `unitree_rl_lab/.../assets/robots/unitree_actuators.py` |
| Humanoid-Gym | `humanoid gym.pdf` |
| Siekmann 2021 | `periodic rew.pdf` |
