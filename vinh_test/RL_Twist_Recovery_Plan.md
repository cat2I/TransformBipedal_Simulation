# Kế hoạch Huấn luyện RL: Phục hồi dáng đứng (Twist Recovery)

## 1. Mục tiêu (Objective)
Huấn luyện một policy Reinforcement Learning (RL) bằng `Gymnasium` và `MuJoCo` để điều khiển robot Bipedal chuyển từ tư thế vặn ngang (sau khi Get-up, khớp `Twist` ~1.51 rad) về tư thế đứng thẳng hướng tới trước (`Twist` = 0 rad). 
*Phương pháp:* Sử dụng dáng đi **"dậm chân tại chỗ" (Stomping / Step-turning)** để triệt tiêu ma sát tĩnh, thay vì vặn trực tiếp dưới đất dễ gây hỏng motor và trượt ngã.

---

## 2. Lựa chọn Thư viện Huấn luyện (RL Framework)
**Quyết định: Sử dụng Stable Baselines3 (SB3)** thay vì RSL_RL ở giai đoạn hiện tại.
*   **Lý do:** 
    *   Tương thích 100% với môi trường `gymnasium` CPU thông thường.
    *   Cực kỳ dễ code, dễ debug và thân thiện với người mới bắt đầu làm quen với RL logic.
    *   Giúp tập trung tối đa vào việc thiết kế Hàm thưởng (Reward) và Logic vật lý thay vì phải giải quyết các vấn đề phức tạp về Vector/Tensor trên GPU của RSL_RL.
*   *Lưu ý cho tương lai:* Khi logic môi trường đã hoàn hảo và cần đua tốc độ mô phỏng (train cực nhanh), có thể chuyển toàn bộ công thức Hàm thưởng sang RSL_RL + MuJoCo MJX (GPU).

---

## 3. Thiết kế Môi trường RL (Gymnasium Environment)

### A. Không gian quan sát (Observation Space)
Mạng nơ-ron sẽ nhận đầu vào là một mảng 1D (Vector) chứa trạng thái hiện tại của robot:
- **IMU:** Góc Roll, Pitch của phần thân (Base).
- **Khớp (Joints):** Vị trí hiện tại (Position) và Vận tốc (Velocity) của 10 động cơ.
- **Cảm biến chạm đất (Contacts):** Trạng thái chạm sàn của bàn chân trái và phải.

### B. Không gian hành động (Action Space)
Dùng **Residual Position Control** (Điều khiển vị trí thặng dư) để phù hợp với motor Feetech.
- Action Space: `Box(-1.0, 1.0, shape=(10,))`
- Ở mỗi step, mạng sẽ xuất ra một độ lệch nhỏ (Action). Góc cấp cho PID sẽ là: `Target_Position = Current_Position + Action * Scale` (Scale giúp giới hạn tốc độ xoay, làm chuyển động mượt mà).

### C. Khởi tạo & Kết thúc (Reset & Termination)
- **Hàm `reset()`:** Khởi tạo robot ở tư thế xoạc chân (`Twist` = 1.51 rad). Bắt buộc phải **cộng thêm nhiễu ngẫu nhiên (Uniform Noise)** vào các góc khớp và độ nghiêng thân để tạo tính Robustness (chống overfit).
- **Điều kiện `terminated` (Fail):** Nếu trục Z của phần thân rớt xuống dưới ngưỡng an toàn (ví dụ `< 0.35m`) hoặc góc nghiêng Roll/Pitch quá lớn (bị lật).
- **Điều kiện Thành công (Success):** Góc `abs(Twist) < 0.05 rad` duy trì liên tục trong vài chục steps.

---

## 4. Hàm Thưởng (Reward Function)
Sử dụng phương pháp **Thưởng liên tục (Dense Reward)** tính toán tại mỗi `step()`.
Tổng điểm thưởng = `[Thưởng Nhiệm vụ] + [Thưởng Sinh tồn] - [Điểm phạt]`

1. **Thưởng Nhiệm vụ (Twist Tracking):** Thưởng cực mạnh khi 2 khớp Twist tịnh tiến về 0. (Nên dùng hàm Exponential: `exp(-weight * error)` để robot tinh chỉnh chính xác ở các góc nhỏ).
2. **Thưởng Sinh tồn (Healthy Reward):** Được cộng điểm cố định mỗi bước nếu robot không ngã.
3. **Phạt định hình dáng đi (Gait Shaping Penalties):**
   - *Phạt No-Fly:* Phạt rất nặng nếu cả 2 chân đều mất tín hiệu chạm đất cùng lúc (cấm nhảy chồm lên).
   - *Phạt Air-Time:* Phạt nếu 1 chân ở trên không quá lâu (ép phải dậm chân dứt khoát).
   - *Phạt Control-Cost:* Phạt dựa trên bình phương của Action để robot chuyển động tiết kiệm năng lượng, ít giật cục.
   *(Lưu ý: Chấp nhận pha hỗ trợ kép - cả 2 chân chạm đất - để robot an toàn chuyển trọng tâm).*

---

## 4b. Khoảng cách Sim-to-Real — LÝ DO CỐT LÕI phải dùng RL

> Đây là phần quan trọng nhất. Nếu bỏ qua, policy sẽ chạy đẹp trong sim rồi chết
> trên phần cứng.

### Vấn đề đo được

Thử trong MuJoCo: xoay khớp Twist của **chân đang chịu tải** về 0. Kết quả — nó
xoay được dễ dàng, servo chỉ dùng 36–67% mô-men, kể cả khi tăng khối lượng base
gấp 3 và bật ma sát xoắn (`condim=6`, torsion 0.1). Tức là **sim nói phương án
xoay trực tiếp là ổn**.

Nhưng đó là ảo tưởng. MuJoCo mặc định bỏ qua 4 thứ, tất cả đều làm việc xoay
**khó hơn ngoài đời**:

| # | Sim bỏ qua | Hậu quả thật |
|---|---|---|
| 1 | `condim=3` — không có ma sát xoắn (torsion = 0.005 ≈ 0) | Đế cao su phẳng thật rất ghì khi xoay tại chỗ |
| 2 | Va chạm dùng vỏ lồi, bán kính đế nhỏ | Đế thật xòe rộng → cánh tay đòn ma sát lớn hơn nhiều |
| 3 | Không có đường cong mô-men–tốc độ | Servo Feetech thật quay nhanh là tụt lực (về 0 tại vận tốc giới hạn) |
| 4 | Không có stiction / backlash hộp số | Servo thật giật cục khi xoay chậm dưới tải |

→ Robot thật khó hơn hẳn con số 47% mà sim báo. **Trực giác "xoay dưới tải rất dễ
ngã" là đúng về vật lý; sim mặc định chỉ không thể hiện ra.**

### Kết luận cho thiết kế

1. **Không hard-code quỹ đạo xoay cố định.** Nó sẽ overfit vào cái sim lạc quan.
   Đây chính là lý do phải dùng RL: policy học chiến lược *bền* với 4 tham số
   không chắc ở trên.

2. **Env train BẮT BUỘC bật ma sát xoắn.** Trong `twist_env.py`, sau khi load
   model, đặt cho hai geom bàn chân:
   ```python
   for side in ("Footleft", "Footright"):
       bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side)
       for g in range(m.ngeom):
           if m.geom_bodyid[g] == bid:
               m.geom_condim[g] = 6          # thêm ma sát xoắn + lăn
               m.geom_friction[g, 1] = 0.05  # sẽ random, xem bảng dưới
   ```

3. **Domain Randomization — random mỗi tập `reset()`** (đây là thứ khiến policy
   sống sót sim-to-real):

   | Tham số | Khoảng random | Vì sao |
   |---|---|---|
   | Ma sát trượt (`friction[0]`) | 0.6 – 1.4 | sàn thật không biết trước |
   | Ma sát xoắn (`friction[1]`) | 0.02 – 0.12 | chính là thứ sim mặc định bỏ qua |
   | Khối lượng base (`body_mass`) | ×0.8 – ×1.5 | base thật nặng hơn CAD |
   | Trọng tâm base (`body_ipos`) | ±2 cm mỗi trục | lắp pin/mạch làm lệch CoM |
   | Trần mô-men Twist (`actfrcrange`) | ×0.7 – ×1.0 | mô phỏng servo yếu đi khi nóng/nhanh |

   Random rộng thì policy học chiến lược dồn-tải-rồi-xoay (thay vì xoay ẩu dưới
   tải), vì đó là cách duy nhất sống qua *mọi* biến thể.

### Bổ sung Reward cho chiến lược dồn tải

Ngoài các reward ở mục 4, thêm để ép robot học "đổ trọng tâm sang chân trụ rồi
mới xoay chân kia" — thay vì cố xoay cả hai chân đang chịu tải:

- **Thưởng dồn tải:** khi khớp Twist của một chân đang xoay mạnh (`|vận tốc Twist|`
  lớn), thưởng nếu **lực chạm sàn của chính chân đó thấp** (chân đã được nhấc bớt
  tải). Ép robot tự học: muốn xoay chân nào thì phải nhẹ tải chân đó trước.
- **Phạt xoay-dưới-tải:** phạt khi `|vận tốc Twist| × lực_chạm_sàn_cùng_chân` lớn
  — đúng cái tình huống mài đế xuống sàn làm hỏng servo / trượt ngã.

---

## 5. Cấu trúc Thư mục Dự án Triển khai
```text
transform_bipedal_rl/
│
├── assets/                  
│   └── Fulltrans_meshfixed.urdf  # File mô hình 3D và config vật lý
│
├── envs/                    
│   └── twist_env.py              # Class TwistRecoveryEnv kế thừa gymnasium.Env
│
├── train_ppo.py                  # Script dùng Stable Baselines3 (PPO) để train
│
└── play.py                       # Script load file .zip, chạy MuJoCo Viewer để xem kết quả
```
