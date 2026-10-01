# ALGO.md — Chọn thuật toán đi bộ cho OFFICIALdesign

Người lập: Claude (Tech Lead). Ngày: 2026-10-01.
File này giải thích **vì sao** chọn thuật toán nào và **cần làm gì**, theo phần cứng hiện có.
Việc code cụ thể sẽ tách thành từng đợt trong `PLAN.md` để agy làm.

---

## 0. Tóm tắt

- Lõi: **PPO + MLP** (đang dùng). Giữ nguyên.
- Thêm **gait clock**, rồi **CPG + residual RL**, để robot có nhịp bước ổn định và nhanh hơn.
- **ZMP** chỉ làm reward trong sim, không làm bộ điều khiển.
- Không dùng **whole-body MPC** và **SAC**.
- Nguồn chính của sim-to-real gap (khoảng cách giữa sim và robot thật) là **servo, độ trễ, hai Pi và hai IMU**. Phải đo phần cứng trước, rồi mới chỉnh sim theo số đo.

---

## 1. Phần cứng hiện có (không thay đổi được)

Robot có hai module chân riêng. Mỗi module từng là xe + tay máy. Hai module dock (ghép) lại thành biped.

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Số khớp | 10 (5 mỗi chân: Bub, Hip, rotate, Knee, Foot) | `assets/officialdesign/meta/calibration.json` |
| Khối lượng | 4.643 kg | `sim_handoff /SIM_HANDOFF.md` |
| Servo nặng | STS3095, khoảng 10.3 Nm stall: Bub, Hip, Knee | `calibration.json` |
| Servo nhẹ | STS3215, khoảng 2.94 Nm stall: rotate, Foot | `calibration.json` |
| IMU | 2 con, `IMUleft`/`IMUright`, gắn cứng vào Baselink, y = ±16.5 mm, cùng hướng | URDF |
| Máy tính | 2 Raspberry Pi, mỗi Pi lo 1 chân (1 bus servo + 1 IMU) | user |
| Policy | 20 Hz; 10 action tăng dần, mỗi bước tối đa ±2°; obs 60 số | `official_env.py` |

Giải thích các thuật ngữ trong bảng:
- **Servo bus** (Feetech STS): nhiều servo nối chung một dây tín hiệu. Mình chỉ ra lệnh "quay tới góc X". Bộ PID bên trong servo tự lo phần còn lại. **Không điều khiển lực (torque) trực tiếp được.**
- **Stall torque** (lực xoắn khi kẹt): lực lớn nhất servo tạo ra khi bị giữ đứng yên. Lúc đang quay, lực thực tế thấp hơn.
- **Obs 60 số** = 4 frame × 5 số IMU (roll, pitch, gyro x/y/z) + 4 frame × 10 lệnh góc đã gửi. Không có góc thật của khớp. Không có contact.

---

## 2. Chọn thuật toán

| Thuật toán | Nói đơn giản | Hợp không? | Lý do |
|---|---|---|---|
| **PPO + MLP** | Cho hàng nghìn robot ảo tập đi. Làm đúng được thưởng, ngã bị phạt. PPO (Proximal Policy Optimization) mỗi lần chỉ sửa policy một chút, nên học ổn định. MLP là mạng nơ-ron các lớp nối thẳng. | ✅ **Lõi** | Output là góc khớp, khớp với servo vị trí. Xuất ONNX nhẹ, chạy trên Pi được. |
| **SAC + MLP** | RL học lại từ kinh nghiệm cũ, tiết kiệm mẫu. | ❌ | Isaac chạy song song nên dư dữ liệu. Khi đó PPO nhanh và ổn định hơn. |
| **Whole-body MPC** | Mỗi vài ms, robot "nhìn trước tương lai" bằng mô hình vật lý rồi tính lực cho từng khớp. | ❌ | Cần điều khiển torque, mô hình khối lượng chính xác (CoM Hip chưa xác nhận) và vòng lặp ≥200 Hz. Không có cái nào. |
| **ZMP** | Giữ "điểm đè" của robot luôn nằm trong bàn chân thì không ngã. | ⚠️ Phụ | Online cần cảm biến lực bàn chân và CoM chính xác. Chỉ dùng trong sim (xem mục 4). |
| **CPG** | Bộ tạo nhịp: sinh sẵn quỹ đạo chân đều nhịp trái–phải, giống "bài nhảy mẫu". | ✅ Bổ trợ mạnh | Xem mục 3. |

**Hướng không chọn:** mỗi chân một policy riêng (decentralized / multi-agent). Biped cần hai chân phối hợp rất chặt để giữ thăng bằng, nên một policy trung tâm tốt hơn.

---

## 3. Gait clock và CPG + residual RL

### 3.1 Gait clock (đồng hồ nhịp bước)

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

### 3.2 CPG + residual RL (RL học phần sửa)

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

**Đối với OFFICIALdesign, mục này mới mô tả phương án:** chưa chốt bộ quỹ đạo hay bộ góc mẫu cho 10 khớp. Cần thiết kế và kiểm chứng phần đó khi triển khai B7. CPG không tự suy ra một dáng đi phù hợp chỉ từ số lượng khớp và đồng hồ.

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

---

## 4. ZMP: chỉ dùng trong sim

**ZMP** (Zero Moment Point) là điểm trên sàn nơi các lực làm robot lật triệt tiêu nhau. ZMP nằm trong vùng bàn chân chống (support polygon) thì robot không lật.

Dùng rẻ mà có lợi:
- **Làm reward:** sim biết chính xác CoM và điểm tiếp xúc, nên thưởng khi hình chiếu CoM/ZMP nằm trong vùng chống. Chỉ dùng lúc train, robot thật không cần cảm biến gì thêm.
- **(Tùy chọn) Quỹ đạo mẫu offline:** dùng LIPM (Linear Inverted Pendulum Model, coi robot như con lắc ngược) để sinh quỹ đạo cho CPG. Công sức nhiều, lợi thêm ít, nên để sau.

---

## 5. Servo và PID nội bộ

**Vấn đề:** policy học trong sim rằng "ra lệnh X thì khớp phản ứng kiểu Y". Servo thật lại phản ứng kiểu Z, và policy chưa từng gặp Z. Hiện `stiffness 60/40`, `damping 1.5/1.0`, `velocity 4.7 rad/s` và `delay 10 ms` đều là **số đoán** (`calibration.json` ghi "provisional").

Làm theo thứ tự:
1. **System identification** (nhận dạng hệ thống: đo hành vi thật rồi chỉnh sim cho khớp).
   - Gửi lệnh step (nhảy góc đột ngột, ví dụ 20–30°).
   - Đọc present position ở ≥200 Hz.
   - Đo **theo từng nhóm** (STS3095 và STS3215), cả **không tải** lẫn **đang chịu trọng lượng robot**.
   - Làm đúng thao tác đó trong Isaac, rồi chỉnh `stiffness`, `damping`, `effort_limit`, `velocity_limit`, `armature` và friction cho hai đường cong trùng nhau.
2. **Độ trễ:** mô phỏng đúng trễ đo được (xem mục 7).
3. **Domain randomization** (ngẫu nhiên hóa: mỗi robot ảo có tham số hơi khác nhau). Random Kp, Kd, trễ và ma sát khoảng ±20% quanh giá trị đo. Policy không phụ thuộc vào một bộ số chính xác.
4. **Để sau: actuator network** (mạng nơ-ron nhỏ học cách servo phản ứng; Isaac Lab có `ActuatorNetMLP`/`ActuatorNetLSTM`). STS chỉ trả về "load" với độ chính xác thấp, nên chỉ làm khi 3 bước trên chưa đủ.

Mẹo phụ:
- **Phạt `action_rate`** (mức thay đổi lệnh giữa hai bước) để lệnh mượt.
- Có thể giảm hệ số P trong EEPROM (bộ nhớ cấu hình) của servo cho khớp "mềm" hơn. **Đổi xong phải đo lại.**

---

## 6. Feedback góc servo

- STS **có** encoder từ (magnetic encoder, cảm biến đo góc bằng nam châm), đọc qua ô nhớ *Present Position*. Độ phân giải khoảng 0.09°.
- Nó "không đáng tin lắm" chủ yếu vì đường truyền, không phải vì encoder:
  - **Trễ:** đọc lần lượt từng servo trên bus.
  - **Mất gói:** timeout hoặc sai checksum (mã kiểm tra lỗi).
  - **Vận tốc tính từ hiệu góc rất nhiễu.**
- Cách dùng: đưa **góc** vào obs của actor. Trong sim cố ý làm "bẩn":
  - nhiễu khoảng 0.5–1°;
  - trễ 1–2 bước;
  - thỉnh thoảng giữ nguyên giá trị cũ (giả mất gói).
- **Không** đưa vận tốc khớp vào actor.
- Phương án an toàn hơn: chỉ đưa góc thật cho critic.
- Trước khi quyết, đo trên robot: đọc 5 servo mất bao nhiêu ms, tỉ lệ lỗi bao nhiêu %.

---

## 7. Contact: không có cảm biến tiếp đất

- Trong sim, contact sensor có sẵn và dùng để **tính reward** (air time, feet height, phạt trượt chân) và điều kiện kết thúc episode. Reward chỉ tồn tại lúc train, nên ngoài đời không cần cảm biến.
- **Quy tắc:** actor **không được** nhận contact. Env mới đang đúng: obs 60 số không có contact.
- **Asymmetric actor-critic** (actor và critic nhận input khác nhau): critic là "giám khảo", chỉ chạy lúc train, được xem thông tin đặc quyền (privileged information). Ví dụ: contact, vận tốc thân thật, ma sát, trễ đang random. Actor chỉ thấy thứ đo được ngoài đời. RSL-RL hỗ trợ việc này qua nhóm observation `critic`.
- Gait clock (mục 3.1) cung cấp lịch "chân nào **nên** chống, chân nào **nên** vung". Nó không đo được chân nào **đang thực sự** chạm đất; contact trong sim mới là thông tin dùng để kiểm tra robot có làm đúng lịch khi train hay không.

> Ghi chú: con số "44 chiều" trong tư vấn cũ là của env 6 khớp (`transformer_nam_env.py`: 20 IMU + 24 lịch sử action). Env mới là 60 chiều. Cả hai đều chỉ chứa thứ đo được.

---

## 8. Kiến trúc hai Pi

### 8.1 Hai giai đoạn
- **Giai đoạn 1:** laptop chạy policy, nối Ethernet tới cả hai Pi. Mỗi Pi đọc servo + IMU của chân mình, gửi lên, nhận lệnh, rồi ghi xuống servo.
- **Giai đoạn 2:** Pi A vừa chạy policy vừa điều khiển chân A. Pi B nối Pi A qua Ethernet, có dây **sync** và **GND chung**.

Đây là mẫu chuẩn "**bộ não trung tâm + các nút I/O phân tán**" (I/O node: bo mạch chỉ lo đọc cảm biến và điều khiển motor tại chỗ). Robot công nghiệp và ANYmal làm y như vậy, qua EtherCAT (một chuẩn Ethernet thời gian thực). Robot này không cần EtherCAT, chỉ cần làm đúng các điểm dưới đây.

### 8.2 Quy tắc thiết kế
1. **Dây sync làm "tiếng còi xuất phát".**
   - Master phát xung GPIO mỗi 50 ms.
   - Cả hai Pi **đọc cảm biến đúng lúc có xung**.
   - Nhờ vậy dữ liệu hai chân là cùng một khoảnh khắc, chính xác cỡ micro-giây. Tốt hơn đồng bộ đồng hồ qua mạng (PTP/chrony).
   - Giai đoạn 1: laptop không có GPIO, nên Pi A phát xung và Pi A làm "đồng hồ gốc".
2. **UDP, gói nhỏ cố định.**
   - UDP là kiểu gửi thẳng, không chờ xác nhận. Nhanh hơn TCP, và mất 1 gói không làm nghẽn các gói sau.
   - Mỗi gói gồm: `seq` (số thứ tự), timestamp, 5 góc servo, dữ liệu IMU, cờ lỗi.
   - Master chỉ dùng gói có `seq` đúng nhịp hiện tại.
3. **Giao thức giống hệt nhau cho hai giai đoạn.**
   - Pi B không cần biết đầu kia là laptop hay Pi A.
   - Chuyển giai đoạn chỉ đổi địa chỉ IP, không train lại.
   - MLP 60 input chạy bằng onnxruntime trên Pi mất dưới 1 ms.
4. **Watchdog** (bộ canh giờ): quá khoảng 100 ms không có lệnh mới thì Pi tự giữ tư thế an toàn.
5. **Lợi thế sẵn có:** hai bus servo chạy song song, mỗi bus chỉ 5 servo, nên đọc nhanh gấp đôi so với một bus 10 servo.

### 8.3 Ngân sách thời gian (ước tính, phải đo)
```
xung sync → đọc 5 servo + IMU (song song 2 Pi) → UDP lên → infer → UDP xuống → ghi servo
                 ~3–8 ms                          <1 ms    <1 ms   <1 ms       ~1–2 ms
tổng trễ đo được → servo nhận lệnh: khoảng 5–15 ms (chu kỳ 50 ms)
```

### 8.4 Đưa vào sim
- Random trễ **riêng cho từng chân** quanh giá trị đo.
- Thỉnh thoảng giả mất gói: chân đó giữ lệnh cũ.
- Hiện sim dùng một `actuator_delay_steps = 2` (10 ms) chung cho cả robot, và số này là giả định.

---

## 9. Hai IMU

`IMUleft` và `IMUright` đều gắn cứng vào Baselink, cách nhau 33 mm, cùng hướng. Hai con đo cùng một vật thể, nên gộp đơn giản.

Quy trình (chạy **cùng một đoạn code** ở cả sim và Pi):
1. **Hiệu chỉnh** offset gyro và hướng lắp của từng con khi robot đứng yên.
2. **Lấy trung bình gyro và accel thô** của hai con. Nhiễu độc lập nên giảm khoảng √2 lần. Lệch do khoảng cách 16.5 mm là không đáng kể.
3. Chạy **một** bộ lọc complementary hoặc Madgwick (thuật toán trộn gyro với accel để ra góc nghiêng ổn định) → roll, pitch.
4. **Obs giữ nguyên 5 số** (roll, pitch, gyro xyz). Không đưa cả hai IMU thô vào policy. Nếu đưa thì policy phụ thuộc vào việc cả hai con cùng sống.

Chẩn đoán:
- **Độ lệch giữa hai IMU** lớn nghĩa là một con hỏng, **hoặc mối dock giữa hai module đang rơ/uốn**. Rất đáng log lại.
- Một con chết thì chuyển sang dùng con còn lại. Có thể thỉnh thoảng giả "một IMU chết" trong sim.

Trong sim:
- Mô phỏng hai IMU với nhiễu và bias (độ lệch cố định) **độc lập**, rồi chạy đúng code gộp đó.
- Hiện env chỉ dùng `IMUleft`, lấy góc thật + nhiễu trắng, chưa mô phỏng bộ lọc.

---

## 10. Việc phải làm (theo thứ tự)

### A. Phần cứng / đo đạc — user làm
- [ ] A1. Step response servo STS3095 và STS3215, không tải và có tải, log present position ≥200 Hz.
- [ ] A2. Đo thời gian đọc 5 servo trên một bus, tỉ lệ lỗi đọc (%).
- [ ] A3. Dựng giao thức 2 Pi: xung sync GPIO, UDP + `seq`, watchdog.
- [ ] A4. Đo trễ end-to-end (từ xung sync đến lúc servo nhận lệnh), giai đoạn 1.
- [ ] A5. Hiệu chỉnh hai IMU. Log độ lệch giữa hai con lúc đứng và lúc lắc robot, để kiểm tra dock có rơ không.
- [ ] A6. Đo giới hạn cơ khí thật của từng khớp và điểm 0 servo. Xác nhận CoM Hip trái/phải.

### B. Sim — agy làm qua `PLAN.md` (sau khi có số đo A)
- [ ] B1. Cập nhật actuator theo A1. Delay theo A4, riêng từng chân, có random và mất gói.
- [ ] B2. Mô phỏng hai IMU + code gộp chung sim/Pi.
- [ ] B3. Domain randomization: Kp, Kd, trễ, ma sát, khối lượng khoảng ±20%.
- [ ] B4. Phạt `action_rate`.
- [ ] B5. Gait clock trong obs + reward theo pha.
- [ ] B6. Asymmetric critic (nhóm obs `critic`: contact, vận tốc thân, tham số random).
- [ ] B7. CPG + residual RL (policy chỉnh tần số, biên độ, phần sửa).
- [ ] B8. (Tùy chọn) Góc servo vào obs của actor, kèm nhiễu, trễ, mất gói.
- [ ] B9. (Tùy chọn) Reward CoM/ZMP nằm trong vùng chống.
- [ ] B10. (Để sau) Actuator network.

---

## 11. Giả định chưa xác nhận

| Giả định | Giá trị hiện tại | Cách xác nhận |
|---|---|---|
| Trễ actuator | 10 ms, chung | A4 |
| Kp / Kd servo | 60/1.5 (nặng), 40/1.0 (nhẹ) | A1 |
| Tốc độ servo | 4.7 rad/s | A1 |
| Torque | theo stall ghi trong handoff | A1 (có tải) |
| Giới hạn góc | "training envelope", chưa đo cơ khí | A6 |
| CoM Hip trái/phải | lệch khoảng 8 mm theo z | A6 / CAD |
| Mối dock cứng tuyệt đối | giả định cứng | A5 |
