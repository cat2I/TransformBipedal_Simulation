# Codex → Claude: observation, locomotion và giao tiếp phần cứng

Ngày tổng hợp: **2026-10-05**.

Tài liệu bàn giao nội dung đã thảo luận với user để Claude tiếp tục phân tích và lập kế hoạch. Đây là bản ghi nhận vấn đề và phương án, **không thay thế `PLAN.md`, không phải quyết định triển khai**. Trong cuộc trao đổi này Codex đọc code và tài liệu kỹ thuật; chưa chạy thử phần cứng, benchmark Pi, train lại hay sửa logic mô phỏng/nhúng.

## 1. Những kết luận chính

- Env mới tách giao diện điều khiển dùng chung khỏi task đi bộ, đồng thời có thay đổi logic so với env cũ; không phải chỉ di chuyển code sang file khác.
- Cả env cũ được chỉ định và env OFFICIALdesign hiện tại đều có frame stack 4 mẫu, nhưng chưa cung cấp observation riêng cho asymmetric critic.
- Frame stack phù hợp với locomotion. LSTM/GRU là phương án thử nghiệm, không mặc nhiên tốt hơn MLP + lịch sử.
- STS có phản hồi trạng thái khớp. Câu trước đây của Codex “thiếu phản hồi góc/tốc độ/tiếp xúc” nói về **đầu vào actor hiện tại**, không kết luận phần cứng không có encoder hay feedback.
- Với USB, 1 Mbps, 6 servo/bus, đọc feedback có triển vọng về băng thông. Chưa có căn cứ kết luận hệ đang sát ngưỡng trễ không chấp nhận được.
- `SYNC_READ` gom yêu cầu đọc, nhưng không bảo đảm tất cả cảm biến lấy mẫu đúng cùng thời điểm. Phải phân biệt thời điểm lấy mẫu, nhận gói và policy sử dụng dữ liệu.
- RL có thể học điều khiển với độ trễ có giới hạn; LL vẫn cần quản lý deadline, dữ liệu cũ và lỗi truyền thông. Huấn luyện không xóa trễ vật lý hay sửa một vòng đọc bị chặn vô hạn.

## 2. Phần cứng: thông tin user đã xác nhận và phần còn thiếu

### Đã xác nhận trong hội thoại

| Hạng mục | Thông tin |
|---|---|
| Bộ điều khiển LL | Raspberry Pi 5, xử lý IMU và động cơ |
| Mỗi chân | 1 Pi + 1 driver Waveshare serial bus + 6 servo STS |
| Pi ↔ driver | USB |
| Baudrate user đang sử dụng | 1.000.000 baud |
| Cách gửi lệnh hiện tại | `sync_write`, push lệnh, chưa chờ/khai thác feedback |
| Tần số inference | Chưa chốt; user cho biết có thể sửa thông số điều khiển sau |
| Servo được user nhắc đến | STS3120 và STS3215; chưa có danh sách model/firmware theo từng ID |

Nếu servo đang ở chế độ vị trí, “Pi push mù” không đồng nghĩa motor hoàn toàn hở vòng: servo vẫn có bộ điều khiển vị trí nội bộ. Feedback lên Pi phục vụ vòng điều khiển cấp policy. Chế độ vị trí và khả năng chỉnh PID được Feetech mô tả trong [tài liệu STS3215](https://www.feetechrc.com/20210430-56680.html).

### Cần làm rõ trước khi chốt hợp đồng sim ↔ nhúng

- SKU chính xác của driver Waveshare: Adapter (A), board ESP32 hay biến thể khác; không suy ra firmware/USB bridge chỉ từ tên “serial bus”.
- Mã nguồn LL, ngôn ngữ, phiên bản SDK và đường xử lý timeout đang chạy trên Pi.
- Payload thực tế của `sync_write`: chỉ vị trí hay cả gia tốc/thời gian/tốc độ.
- Mapping **12 servo phần cứng ↔ 10 action của sim hiện tại**. Chưa biết vai trò hai servo còn lại; không tự coi chúng là khớp cố định hay tự mở rộng action space.
- `calibration.json` hiện ghi nhóm heavy là **STS3095**, trong khi user nhắc STS3120. Cần xác nhận model, điện áp và khớp tương ứng trước khi dùng thông số actuator.
- Vị trí chạy policy, đường giao tiếp hai Pi và cơ chế đồng bộ hiện có chưa được xác nhận đầy đủ. Kiến trúc trong `ALGO.md` là nội dung cần đối chiếu, không coi là cấu hình phần cứng đã triển khai.

## 3. Env cũ và OFFICIALdesign hiện tại

Các file đã được đọc:

- [Env cũ](../isaac_rl/_archive/fulltrans/transformer_walk10dof_env.py).
- [Đăng ký OFFICIALdesign](../isaac_rl/bipedal/officialdesign/__init__.py).
- [Robot](../isaac_rl/bipedal/officialdesign/robot.py), [interface](../isaac_rl/bipedal/officialdesign/interface.py), [task walk](../isaac_rl/bipedal/officialdesign/task_walk.py), [PPO](../isaac_rl/bipedal/officialdesign/ppo.py).
- [PPO dùng chung](../isaac_rl/bipedal/_shared/ppo.py) và [calibration](../assets/officialdesign/meta/calibration.json).

### Cấu trúc và khác biệt hành vi

| Nội dung | Env cũ | OFFICIALdesign |
|---|---|---|
| Tổ chức | Một file chứa scene, obs/action, reset, reward, termination | `interface.py` dùng chung; `task_walk.py` chứa mục tiêu/reward/termination; `robot.py` chứa cấu hình robot |
| Asset | Fulltrans10DOF | OFFICIALdesign; đọc JSON và kiểm SHA-256 |
| Obs/action | 60/10 | 60/10, nhưng ý nghĩa mapping không mặc nhiên tương thích |
| Action | Clip `[-3,3]`, nhân `2/3` độ | Clip `[-1,1]`, nhân 2 độ |
| Lệnh góc tích lũy | `cmd_actions` có thể vượt giới hạn; clamp target cuối | Clamp ngay lệnh tích lũy, tránh windup |
| Mapping | Truyền target trực tiếp theo thứ tự đang dùng | Tìm joint theo tên, kiểm thứ tự và áp dụng dấu calibration |
| Actuator | Backlash 2,5°, noise, trễ ngẫu nhiên 10–30 ms | Trễ cố định 10 ms; chưa có backlash/noise actuator tương ứng |
| IMU | Baselink; bias, noise, drift | IMUleft; noise, không có bias/drift tương ứng |
| Vận tốc đích | Không có tốc độ đích cụ thể; thưởng tỷ lệ tiến theo +x | Bám 0,15 m/s theo trục x của thân |
| Độ cao đích | 0,43 m cố định | Từ spawn height và clearance của asset |
| Nhấc chân | Chỉ thưởng đúng một chân trên không | Thưởng đế chân gần 3,5 cm; cả hai chân bay vẫn được thưởng |
| Các khoản phạt bổ sung | Không có các khoản tương ứng | Trượt chân, effort, thay đổi action, lệch tư thế |
| Tổng reward | Chuẩn hóa trọng số | Tổng hệ số trực tiếp, nhân `step_dt` |
| Ngã | Thấp hơn 0,10 m hoặc nghiêng quá 0,95 rad | Thấp hơn 0,20 m hoặc nghiêng quá 0,90 rad |
| Episode / physics / policy | 10 s / 200 Hz / 20 Hz | 10 s / 200 Hz / 20 Hz |
| Số env mặc định | 4.096 | 256 |

Lưu ý cụ thể từ code:

- Env cũ tạo `act_direction` nhưng `velocity_reward()` không sử dụng biến này; thực tế vẫn thưởng `vx > 0`.
- Reward nhấc chân mới có `TODO`: lấy trung bình hai chân nên cả hai cùng bay đúng độ cao được thưởng hơn chỉ một chân. Không coi đây là reward luân phiên chân đã hoàn thiện.
- Interface mới chống dịch lịch sử hai lần trong cùng một bước và reset trạng thái lệnh/history đầy đủ hơn.
- Tài liệu module ghi mới smoke test 2 iterations, chưa có policy biết đi. Đây là trạng thái ghi trong repo, không phải kết quả vừa chạy lại trong cuộc trao đổi này.

### Asymmetric actor–critic

Env cũ được chỉ định chỉ trả `{"policy": obs_buffer}` và khai báo state riêng bằng 0. Env OFFICIALdesign cũng chỉ trả nhóm `policy`, `state_space = 0`. Cấu hình PPO đã đọc chưa bổ sung nhóm observation đặc quyền cho critic.

**Có mạng actor và critic riêng không đồng nghĩa asymmetric actor–critic.** Asymmetry cần critic nhận thông tin khác/thêm so với actor, chẳng hạn trạng thái simulator mà robot thật không đo được. Việc reward dùng contact, vận tốc hay torque thật trong sim không tự đưa chúng vào critic.

Phương án khả thi: actor dùng tín hiệu triển khai được; critic có thể nhận trạng thái đặc quyền khi train. Đây là lựa chọn độc lập với frame stack, LSTM và gait clock, chưa được triển khai trong đợt trao đổi này.

## 4. Frame stack, LSTM và khả năng chạy trên Pi 5

### Frame stack hiện tại

Observation gồm 4 mẫu `[roll, pitch, gx, gy, gz]` và 4 mẫu lệnh góc 10 khớp đã chuẩn hóa; lịch sử cũ → mới. Đó là **lịch sử lệnh góc**, chưa phải feedback encoder.

Với policy 20 Hz và lấy mẫu mỗi bước:

| Số mẫu K | Khoảng thời gian mẫu cũ nhất → mới nhất `(K-1)/20` | Số chiều nếu giữ 15 thành phần/mẫu |
|---|---:|---:|
| 1 | 0 ms | 15 |
| 2 | 50 ms | 30 |
| 4 | 150 ms | 60 |
| 8 | 350 ms | 120 |
| 16 | 750 ms | 240 |

Giảm lịch sử làm đầu vào nhỏ hơn nhưng mất bối cảnh chuyển động/đáp ứng actuator. Tăng lịch sử có thể giúp suy đoán trạng thái ẩn, đồng thời tăng chi phí và có thể làm học khó hơn. Không có kết luận 4 hay 8 mẫu tối ưu nếu chưa so sánh.

Stack không tự bắt robot đợi tích đủ K mẫu ở mỗi lần inference; buffer được cập nhật liên tục và có quy tắc khởi tạo khi reset.

Frame stack có dùng trong locomotion, không chỉ xử lý ảnh. Ví dụ [cấu hình Humanoid-Gym](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_config.py) dùng 15 frame actor, 3 frame critic, policy 100 Hz: lịch sử actor trải khoảng 140 ms, gần 4 mẫu/20 Hz của dự án. Cần so cả thời gian lịch sử, không chỉ số frame.

Các hướng khác đã đề cập: [RMA](https://ashish-kmr.github.io/rma-legged-robots/) dùng adaptation từ lịch sử; [Miki và cộng sự](https://arxiv.org/abs/2201.08117) dùng recurrent belief encoder. Chúng không chứng minh mọi bài toán locomotion cần thay frame stack bằng recurrent network.

### LSTM/GRU: phương án thử, chưa chốt nâng cấp

- Có lý do để thử khi actor quan sát không đầy đủ hoặc cần nhớ phản ứng actuator qua nhiều bước.
- Bộ nhớ không thể khôi phục chắc chắn một tín hiệu phần cứng không cung cấp, cũng không tự sửa reward sai.
- Nên giữ MLP + 4 frame làm baseline; thử 2/8 frame và mạng recurrent nhỏ với cùng ngân sách train, nhiều seed.
- Ví dụ phương án recurrent: 1 tầng LSTM, hidden size 64 hoặc 128, nhận mẫu hiện tại. Nếu đổi bộ observation bằng feedback khớp thì phải thiết kế lại input tương ứng, không mặc định còn 15 chiều.
- Khi triển khai LSTM phải truyền/lưu `hidden state` và `cell state`, reset đúng khi khởi động/reset task, xử lý mất dữ liệu theo cùng quy tắc với sim.
- Cần kiểm tra PPO sequence batching, mask/reset, play và export của đúng phiên bản đang cài. [RSL-RL hiện có RNN và ONNX export](https://leggedrobotics.github.io/rsl_rl/api/models.html), nhưng chưa kiểm thử đường recurrent trong dự án này.

Pi 5 có CPU Cortex-A76 4 nhân, 2,4 GHz theo [datasheet](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf). Actor recurrent nhỏ ở 20 Hz được đánh giá là khả thi về quy mô tính toán, **chưa có benchmark trên Pi của user**. Chỉ cần actor khi triển khai, không cần critic. Số tham số ít hơn không bảo đảm inference nhanh hơn do khác toán tử/runtime.

Mục tiêu inference dưới 5–10 ms đã được nêu như một mốc thử nghiệm, không phải số đo hay tiêu chuẩn ổn định chung. Cần đo p99 trong hệ thống chạy cả I/O và các tác vụ thực tế.

## 5. Gait clock và chuyển động không tuần hoàn

User đặt vấn đề đúng: một nhịp tuần hoàn cố định không mô tả đủ mọi chuyển động locomotion.

- Thêm `[sin(2πφ), cos(2πφ)]` vào obs chỉ cung cấp pha; không tự bắt chuyển động tuần hoàn.
- Reward ép chân chống/vung theo pha tạo ràng buộc hành vi mạnh hơn. Nếu trọng số quá lớn, lịch bước có thể xung đột với bước cứu thăng bằng hoặc tiếp xúc bất ngờ.
- Đi đều có thể hưởng lợi từ clock; đổi tốc độ, bắt đầu/dừng cần xử lý chuyển trạng thái; đứng dậy hay nhảy một lần cần tiến trình có điểm kết thúc hoặc pha theo sự kiện.
- Các phương án khả thi: clock đổi tần số, pause/reset pha, đồng bộ theo tiếp xúc, chuyển pha theo sự kiện, hoặc policy không dùng clock.

[Periodic Reward Composition](https://arxiv.org/abs/2011.01387) cho thấy thiết kế pha có thể bao phủ nhiều gait và cả đứng. Điều đó không chứng minh một clock cố định duy nhất đủ cho mọi task.

Đề xuất đã bàn: có thể thử clock cho task đi bộ, chưa nên bắt mọi task của robot dùng một lịch bước cứng. Nếu clock/mode trở thành input actor, vẫn phải định nghĩa trong hợp đồng sim ↔ nhúng. LSTM cung cấp bộ nhớ, clock cung cấp nhịp mong muốn; hai thứ không thay thế nhau.

## 6. Feedback servo và ước lượng tiếp xúc chân

### Dữ liệu đã xác minh từ tài liệu

- Feetech STS3215 liệt kê feedback vị trí, tốc độ, dòng, load, điện áp, nhiệt độ. Trong [bài kỹ thuật cho bản 7,4 V](https://www.feetechrc.com/20210430-56680.html), `load` được mô tả là duty cycle PWM điều khiển motor; dòng là trường riêng, ghi 6,5 mA/đơn vị. Không tự áp dụng hệ số đó cho mọi model/firmware.
- [Datasheet Feetech ST-3120-C001](https://cdn.shopify.com/s/files/1/0673/6848/5000/files/FEETECH-ST3120.pdf?v=1783131460), bản A/0 ngày 2026-01-19, cũng liệt kê vị trí/tốc độ/dòng/load. Đây là tài liệu hãng lưu tại nhà phân phối; cần đối chiếu chính xác mã servo user đang dùng.
- Chưa có bằng chứng các trường `load` đó là cảm biến mô-men trực tiếp ở đầu ra hộp số. Không đổi `load 50%` thành đúng 50% stall torque.

### Cách sử dụng khả thi

| Tín hiệu | Vai trò và điều kiện |
|---|---|
| Góc đo `q` | Phản hồi khớp trực tiếp; cần hiệu chuẩn zero, dấu, wrap, tỷ số truyền/cơ cấu nếu có |
| Tốc độ `dq` | Hữu ích cho động học; kiểm tra cách mã hóa, đơn vị, nhiễu và cập nhật thực |
| Dòng `I` | Thông tin effort/tải gián tiếp; không mặc định bằng mô-men ngoài ở khớp |
| Load/PWM | Tín hiệu bổ sung theo đúng ý nghĩa thanh ghi từng model |
| Contact ước lượng | Kết hợp nhiều tín hiệu, cần kiểm chứng báo nhầm/bỏ sót và trễ |

“Điện ngược” cần phân biệt: back-EMF gắn với tốc độ quay, còn dòng motor liên quan mô-men điện từ. Motor giữ tải khi đứng yên có thể có dòng trong khi back-EMF gần bằng 0. Quan hệ cơ bản được giải thích trong [tài liệu maxon](https://support.maxongroup.com/hc/en-us/articles/360013761160-Motor-data-and-simulation). Suy từ dòng servo sang mô-men ngoài còn phụ thuộc đo dòng, hộp số, ma sát và động lực học.

Chỉ đặt ngưỡng dòng cổ chân chưa đủ xác nhận tiếp xúc:

- Chân trên không vẫn có thể cần dòng lớn để tăng tốc hoặc thắng ma sát.
- Chân chịu tải nhưng đường lực đi gần trục cổ chân có thể tạo mô-men nhỏ quanh trục đó.

Phương án đã bàn: kết hợp dòng, `q`, `dq`, sai số góc lệnh–góc thực, IMU và nếu cần tín hiệu gối/hông; đánh giá bộ ước lượng bằng cảm biến tiếp xúc tạm thời. Nghiên cứu [phát hiện tiếp xúc từ cảm nhận nội tại](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.771415/full) cho thấy hướng phối hợp tín hiệu có cơ sở, không bảo đảm độ chính xác cho STS của dự án.

Ưu tiên đề xuất: khai thác/kiểm chứng `q`, `dq` trước khi đổi kiến trúc mạng; coi dòng/contact estimator là bước tiếp theo. Nếu actor dùng contact ước lượng ngoài đời, không train actor chỉ với contact lý tưởng rồi thay bằng estimator lúc triển khai.

## 7. SYNC_WRITE, SYNC_READ và ngân sách truyền

### Điều đã xác minh ở mức giao thức

Theo [giao thức Feetech/Waveshare, mục 1.3.6–1.3.7](https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf):

- `SYNC_WRITE` broadcast không có ACK từng servo; hàm báo gửi thành công không chứng minh từng khớp đã nhận/hoàn thành lệnh.
- `SYNC_READ` là giao dịch riêng; gửi một yêu cầu, nhận phản hồi theo thứ tự ID trong yêu cầu.
- Các servo trả lời nối tiếp trên bus half-duplex. Cần kiểm tra hỗ trợ trên đúng model/firmware.
- Tài liệu không cam kết chốt đồng thời mọi encoder/current register, cũng không cung cấp timestamp lấy mẫu chung.

Đọc feedback trong khi servo đang chuyển động là bình thường. Không cần đợi khớp tới đích, và không nên tự thêm `sleep` để “đợi feedback đúng”. Góc gần như chưa đổi ngay sau lệnh không nhất thiết là lỗi.

### Phép tính minh họa cho 6 servo, 1 Mbps, 8N1

Giả định dùng `SyncWritePosEx` ghi 7 byte dữ liệu/servo và đọc 15 byte/servo từ địa chỉ 56 đến 70 theo [SDK STS](https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.cpp). Map này cần kiểm tra tương thích model thực; đây là khối feedback runtime, không phải toàn bộ bảng cấu hình servo.

| Giao dịch | Byte trên dây | Thời gian truyền thuần |
|---|---:|---:|
| Sync write: `8 + 6 × (1 + 7)` | 56 | 0,56 ms |
| Sync read request: `8 + 6` | 14 | 0,14 ms |
| Sáu reply: `6 × (6 + 15)` | 126 | 1,26 ms |
| **Tổng** | **196** | **1,96 ms** |

Công thức: `T_wire = số_byte × 10 / baudrate`. Cấu trúc gói được đối chiếu với [SDK giao thức](https://github.com/ftservo/FTServo_Python/blob/main/scservo_sdk/protocol_packet_handler.py).

Nếu chỉ đọc vị trí+tốc độ, 4 byte/servo: lượt đọc có 74 byte = 0,74 ms; cộng cùng gói ghi trên thành 1,30 ms. Nếu gói ghi chỉ chứa vị trí thì lại ngắn hơn. Cần tính theo payload thật của LL.

**Các số trên là lower bound truyền thuần**, chưa gồm xử lý servo, khoảng nghỉ/chuyển hướng, USB, OS, parsing hoặc timeout. Chưa có số đo để nói một lượt bình thường trên Pi mất chính xác bao nhiêu ms.

1,96 ms chiếm khoảng 4% chu kỳ 20 Hz, 20% chu kỳ 100 Hz. Đây là tỷ lệ ngân sách giao tiếp, không phải phép chứng minh ổn định điều khiển. Hai bus độc lập có thể hoạt động song song, không bắt buộc cộng đôi thời gian bus khi tính hai chân.

### Timeout và USB

[`port_handler.py` của SDK Feetech](https://github.com/ftservo/FTServo_Python/blob/main/scservo_sdk/port_handler.py) tại thời điểm tra cứu đặt `LATENCY_TIMER = 50` ms và cộng vào deadline. Với 126 byte reply ở 1 Mbps, thời hạn khoảng **51,29 ms**. [Vòng đọc SDK](https://github.com/ftservo/FTServo_Python/blob/main/scservo_sdk/protocol_packet_handler.py) đợi đủ byte hoặc hết hạn.

- **Lượt đọc đủ dữ liệu không bắt buộc chờ 50 ms.**
- Thiếu một phản hồi có thể làm API chặn đến deadline, đủ trễ một tick 20 Hz.
- Serial `timeout=0` không tự biến hàm cấp cao thành bất đồng bộ.
- Cần kiểm tra SDK thật đang chạy; không kết luận mọi thư viện đều có hành vi/giá trị này.
- Chỉnh deadline cần dựa trên đo đạc và xử lý gói về muộn, không cắt tùy tiện rồi bỏ qua dữ liệu cũ lẫn vào giao dịch sau.

USB buffering/driver là phần cần đo. Nếu đúng Adapter (A), [schematic Waveshare](https://files.waveshare.com/wiki/Bus_Servo_Adapter_A/Bus_Servo_Adapter_A.pdf) ghi CH343P; chưa xác nhận board user đúng SKU này. Không áp dụng mẹo FTDI `latency_timer` cho mọi USB adapter, hoặc đồng nhất hằng timeout SDK với latency timer phần cứng.

## 8. Phương án LL và mô hình độ trễ trong sim

### Kiến trúc LL khả thi, chưa triển khai

1. Một worker sở hữu mỗi USB/bus, xếp lịch ghi, gửi read request, nhận dữ liệu và công bố snapshot.
2. Không để các thread read/write độc lập tự ý sử dụng cùng port: có thể xung đột giao dịch hoặc xóa buffer nhận.
3. Snapshot có `seq` của giao dịch phía host, giá trị từng ID, cờ hợp lệ/lỗi và timestamp nhận. Không coi `seq` tự tạo là ACK được servo phản hồi cho action tương ứng.
4. Policy lấy snapshot theo lịch inference; định nghĩa ngưỡng tuổi dữ liệu, cách xử lý thiếu mẫu và ngưỡng chuyển sang xử lý lỗi.
5. Có thể đọc feedback nhanh hơn inference, ví dụ thử read 100 Hz với policy 20/50 Hz. Không mặc định đọc ngay sau một lệnh rồi giữ mẫu gần hết chu kỳ dài.
6. Deadline cho bus phải hữu hạn; worker riêng tránh chặn policy trực tiếp nhưng không xóa trễ vật lý và không bảo đảm snapshot luôn mới.

Timestamp nhận chỉ cho biết dữ liệu đã đến host lúc nào. Nếu không biết thời điểm cập nhật thanh ghi, `now - receive_time` chỉ là thời gian lưu trên host, **chưa phải toàn bộ tuổi phép đo**. Hai Pi có đồng hồ riêng; cần đồng bộ/ước lượng offset khi so timestamp hoặc đo trễ một chiều giữa chúng.

### Các thành phần sim cần tách

| Thành phần | Mô phỏng điều gì |
|---|---|
| Trễ truyền lệnh | Lệnh cũ còn tác dụng trong khi lệnh mới chưa tới actuator |
| Đáp ứng actuator | PD, giới hạn torque/speed, gia tốc, friction/backlash nếu xác minh có ảnh hưởng |
| Cập nhật cảm biến | Nhịp cập nhật riêng, lượng tử hóa/lọc của vị trí, tốc độ, dòng, IMU |
| Trễ observation | Mẫu được lấy trước nhưng đến policy sau; có jitter |
| Skew | Lệch thời gian giữa khớp, IMU và hai chân |
| Mất/chậm gói | Giữ mẫu/lệnh, đánh dấu lỗi hoặc bỏ lệnh cũ theo đúng LL thực tế |

Mẫu trễ và lỗi có thể tương quan theo chân/bus, không nhất thiết là noise độc lập cho từng khớp. Nếu policy dùng cờ valid/tuổi dữ liệu thì cách mã hóa chúng cũng phải giống giữa sim và nhúng.

Hiện `actuator_delay_steps = 2` tại dt 5 ms chỉ giả định **10 ms trễ target**, chưa đại diện đầy đủ cho đường cảm biến và truyền thông. Không cộng gộp mù mọi độ trễ thành một con số này.

### RL “bù trễ” đến mức nào?

Policy có thể học sử dụng lịch sử để suy đoán trạng thái và chọn lệnh phù hợp với trễ, jitter và mất mẫu đã được mô phỏng. Trễ cố định hoặc có giới hạn thường dễ xử lý hơn trễ dài/thất thường.

Nhưng policy không thể biết chính xác một nhiễu mới xảy ra trước khi tín hiệu về đến nó; bộ nhớ không biến thông tin cũ thành phép đo mới. LL cần loại bỏ tắc nghẽn phần mềm có thể tránh được, còn RL học làm việc với phần trễ còn lại.

**Kết luận đã làm rõ với user:** chưa có bằng chứng độ trễ đang sát ngưỡng không chấp nhận được. 2 ms là truyền thuần; khoảng 51 ms là timeout khi thiếu dữ liệu; độ trễ vận hành bình thường chưa đo. Chu kỳ policy không phải ngưỡng ổn định tuyệt đối.

## 9. Đo gì để chốt mô hình và tần số?

Đây là đề xuất phép đo, chưa có kết quả pass/fail:

| Thử nghiệm | Kết quả cần lấy |
|---|---|
| Read một servo → đủ sáu servo | Model/firmware hỗ trợ, thời gian, tỷ lệ lỗi, đúng giá trị/đơn vị |
| Read + write đứng yên và chuyển động có tải | Độ trễ/jitter dưới hoạt động thực, tín hiệu dòng/tốc độ |
| Lịch giao tiếp thử 50/100/200 Hz | Throughput, deadline miss, mức dữ liệu lặp và mới |
| Thiếu một phản hồi trên bàn thử | Thời hạn chờ thật, ảnh hưởng tới lệnh kế tiếp, xử lý gói muộn |
| Hai Pi + IMU + inference cùng hoạt động | Tuổi obs, skew hai chân/IMU, trễ end-to-end |
| Đáp ứng actuator có/không tải | Độ trễ bắt đầu, đường đáp ứng, speed/effort, ảnh hưởng backlash |
| Nhận biết tiếp xúc | Báo nhầm, bỏ sót, trễ phát hiện so với dữ liệu đối chiếu |

Log timestamp gửi ở host, bus TX-end nếu đo được, nhận từng reply/đủ nhóm, và lúc policy dùng snapshot; cùng ID, số thứ tự và lỗi. Thống kê median, p95, p99, max, tỷ lệ mất gói và trễ deadline.

Logic analyzer trên bus giúp tách timing trên dây khỏi USB/host. Nó không tự cho biết timestamp lấy mẫu encoder hay thời điểm vòng PID nội bộ áp dụng target; cần tài liệu firmware hoặc phép đo đối chiếu để xác định thêm. Hàm `write()` trả về không chứng minh servo đã nhận lệnh.

Chốt tần số inference sau khi xem cả deadline giao tiếp, inference và động học robot; không chọn chỉ vì mạng chạy nhanh hoặc bus có throughput cao.

## 10. Các giả định trong repo cần Claude đối chiếu

Các mục sau là nhận xét để thảo luận; Codex chưa sửa tài liệu/code tương ứng:

- [ALGO.md](ALGO.md) §3.4 nói GPIO sync làm hai Pi đọc “cùng một khoảnh khắc, chính xác cỡ micro giây”. Xung chung chỉ hỗ trợ lịch yêu cầu; qua Linux/USB và register sampling chưa có bảo đảm đồng thời tới mức đó.
- Cùng mục tính 5 servo/bus, user xác nhận 6. Cần cập nhật mapping và ngân sách sau khi rõ chức năng từng servo.
- §3.2 vừa đề nghị đưa góc vào actor vừa nói phương án an toàn hơn chỉ đưa góc cho critic; cần chốt lựa chọn. Không mặc định cấm `dq` nếu chưa đánh giá feedback tốc độ thực.
- Noise góc 0,5–1°, trễ 1–2 bước, thời gian đọc 3–8 ms và các ngân sách tổng trong tài liệu hiện là đề xuất/ước lượng; không coi là số đo.
- Nhận xét “STS chỉ trả load” ở §3.1 cần đối chiếu lại với feedback position/speed/current đã tìm thấy; không vì có current mà coi đã có torque sensor chính xác.
- Calibration dùng STS3095 và các tham số provisional; cần xác nhận lại theo phần cứng user đang dùng, chưa thay tự động sang STS3120.

Hệ quả trực tiếp từ code cần lưu ý khi đổi tần số: action hiện tăng tối đa **2°/bước policy**. 20 Hz tương ứng tối đa 40°/s tăng lệnh, 50 Hz thành 100°/s nếu giữ nguyên scale. Đổi Hz còn đổi khoảng thời gian frame stack. Vì vậy “thông số có thể sửa sau” vẫn cần cập nhật nhất quán sim, policy và LL; không chỉ đổi tốc độ vòng lặp ngoài đời.

## 11. Những quyết định đề nghị thảo luận tiếp với Claude

1. Xác nhận danh sách servo/ID, mapping 12 servo ↔ asset 10 DOF, SKU driver và mã LL/SDK.
2. Lập kế hoạch đo giao tiếp và actuator, xác định deadline, quy tắc dữ liệu cũ và xử lý lỗi.
3. Chốt observation contract mới: có `q`, có `dq`, có dòng/load không; có thêm valid/age hay estimator không. Đây là thay đổi cần đồng bộ firmware/LL, export và checkpoint.
4. Chọn baseline MLP + lịch sử; cân nhắc asymmetric critic độc lập. Thử LSTM/GRU sau khi có tiêu chí so sánh rõ, không gộp nhiều thay đổi để khó biết nguyên nhân cải thiện.
5. Chốt scope của gait clock và cách bắt đầu/dừng/chuyển task; xử lý vấn đề reward khuyến khích hai chân cùng bay.
6. Lập mô hình delay/sampling/loss từ số đo, đánh dấu rõ mọi dải tạm và kiểm tra độ nhạy trước khi có đủ dữ liệu.
7. Đánh giá bằng thời gian trước khi ngã, sai số vận tốc, trượt chân, chịu nhiễu/trễ và latency inference; không chỉ nhìn tổng reward.

Không có quyết định cuối cùng về việc chuyển sang LSTM, thêm clock, thay reward, mở rộng số DOF hay tăng tần số. Theo workflow dự án, Claude tiếp tục đưa phạm vi và acceptance criteria vào `PLAN.md`; Codex thực thi trong phạm vi được giao.
