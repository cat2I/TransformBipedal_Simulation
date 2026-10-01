# Nhập khối lượng thực tế vào SolidWorks — OFFICIALdesign.SLDASM

Cập nhật: 2026-09-30. Mục đích: override khối lượng từng part bằng số cân thật, để SolidWorks tính lại trọng tâm và quán tính của robot (thân và 2 chân), rồi xuất lại cho mô phỏng (Fulltrans.csv / URDF).

Assembly: `New_base/OFFICIALdesign.SLDASM` (SolidWorks 2024 SP0.1). Vật liệu in là PETG.

---

## 1. Nguyên tắc

1. **Override áp dụng cho cả file part, không cho từng instance.** Mọi instance của cùng một file sẽ có cùng khối lượng.
2. **Servo dùng chung file cho thân và chân:**
   - `servoFeetechSTS3095.SLDPRT`: 6 cái (2 ở thân, 4 ở chân).
   - `STS3215_03a v1.SLDASM`: 11 cái (5 ở thân, 6 ở chân).

   Vì vậy, file servo chỉ nhập **khối lượng servo trần**. Phần ốc và dây của mỗi khâu dồn vào **part in 3D riêng của khâu đó**.
3. **Chân** được cân theo từng khâu, **đã gồm ốc và dây**. Cách tính: part in 3D = khối lượng khâu đo được − các servo trong khâu.
4. **Thân** được cân từng món, **chưa có ốc, chưa có dây** (trừ dock male có dây). Ốc và dây xử lý ở bước 5.
5. **Chỉ override Mass.** Không override CoM hay inertia. SolidWorks sẽ coi part là đồng chất và giữ trọng tâm hình học.

## 2. Khối lượng servo trần (đã cân)

| File | Nhập (g) | Ghi chú |
|---|---|---|
| `servoFeetechSTS3095.SLDPRT` (servo to) | **201,7** | Hiện đang là 194,5 |
| `STS3215_03a v1.SLDASM` (servo nhỏ) | **56,1** | Hiện đang là 55. Là file assembly: mở .SLDASM rồi override ở cấp assembly |

## 3. Bảng CHÂN (2 chân dùng chung file, mỗi chân 1089,7 g)

| Khâu | Đo (có ốc + dây) | Servo trong khâu | Part nhận phần còn lại | CAD hiện tại | **Nhập (g)** |
|---|---|---|---|---|---|
| Foot | 216,6 | 1× STS3215 (56,1) | `Foot30kg - Copy` | 75,0 | **112,3** |
| | | | `Jaw30kg` | 32,2 | **48,2** |
| Knee | 348,1 | STS3095 + STS3215 (257,8) | `Knee105+30kg` | 96,3 | **90,3** |
| Twist | 105,8 | không có | `Twist30kg` | 86,6 | **105,8** |
| Hip | 342,4 | STS3095 + STS3215 (257,8) | `Hip+twist_105kg` **và** `Hip+twist_105kg(2)` | 46,6 | **84,6** (cả 2 file) |
| Bub | 76,8 | không có | `Hip_105kg` | 50,1 | **76,8** |

- Ở khâu Foot, 216,6 − 56,1 = 160,5 g được chia cho Foot và Jaw theo tỉ lệ CAD 75 : 32,2.
- Instance trong assembly:
  - Servo hông STS3095: `-2`, `-5`. Servo gối STS3095: `-3`, `-4`.
  - STS3215 xoay hông: `-3`, `-10`. STS3215 cổ chân: `-4`, `-11`. STS3215 kẹp: `-5`, `-12`.

## 4. Bảng THÂN (chưa có ốc, chưa có dây)

| # | Đã cân | Component trong SW (số lượng) | CAD hiện tại | **Nhập (g)** | Trạng thái |
|---|---|---|---|---|---|
| 5 | plate 1.2: 96 | `Baseplate1.2` (1) | 252,5 | **96** | OK |
| 6 | plate 1.2.2: 106 | `Baseplate1.2.2` (1) | 252,5 | **106** | OK |
| 8 | plate 2.2.1: 133,7 | `Baseplate2.2.1` (1) | 195,9 | **133,7** | OK |
| 2 | plate 2.2.2: 121 | `Baseplate2.2.2` (1) | 183,8 | **121** | OK |
| 17 | servo to: 201,7 | `servoFeetechSTS3095` (thân: `-1`, `-6`) | 194,5 | **201,7** | OK (xem mục 2) |
| 13 | mount bub mới: 51,8 | `105motormount` (2) | 44,5 | **51,8** | OK |
| 12 | mount bánh xe mới: 18,4 | `drive_motor_mount_v2 (1)` (4 cái cạnh bánh: `-2`, `-3`, `-4`, `-6`) | **0** ⚠ | **18,4** | OK. Hiện CAD là 0 g |
| 10 | servo bánh xe + ngàm cũ: 73 (không dây), 81,7 (có 2 dây) | `STS3215` (bánh: `-1`, `-2`, `-7`, `-9`) | 55 | **56,1** | OK. Suy ra ngàm cũ = 16,9 g |
| 9 | dock male + servo (có dây): 94,2 | `Dokingm` (1) + `STS3215-8` | 23,7 | Dokingm **38,1** | OK |
| 15 | dock: 40,5 | `Dockingf` (1) | 30,0 | **40,5** | OK |
| 16 | male switch: 21,6 | `ngàm_đực_đực_switch` (1) | 31,6 | **21,6** | OK |
| 11 | imu + dây + ngàm dưới: 9,5 | `gá imu mới` (1) + `claude imu` (1) | 4,8 + 0,8 | gá **5,6**, claude imu **3,9** | OK. Gá 5,6 lấy theo slicer |
| 14 | bánh cả cụm: 138,5 | `Wheel` (4) + `servo_wheel_hub` (4) | 145 + 11,3 | Wheel **127,2**, hub **11,3** | **? Cần xác nhận** cụm đã gồm hub chưa |
| 7 | cam + ngàm: 31,2 | `cam` (1, đang ẩn) + `ngàm_đực_cam` (2) + `ngàm_đực_đực_cam` (1) | 6,2 + 31,1 ×2 + 30,9 | ? | **? Cần xác nhận** mỗi ngàm khoảng 31 g? Cam có lắp không? |
| 4 | pi + driver + case: 128,2 | `pi case top` (2), `bottomproto`, `basecomboproto`, `Waveshare_bus_servo_adapter` (2) | ≈ 61 tổng | ? | **? Cần xác nhận** 2 pi case = Pi + driver? Nếu đúng thì chia theo thể tích |
| 1 | ngàm cái: 20 | chưa tìm thấy trong CAD | – | ? | **? Cần xác nhận** part nào |
| 3 | bánh đa hướng: 11,3 | chưa rõ (Part1/2/3^OFFICIALdesign?) | – | ? | **? Cần xác nhận** part nào |

Tổng thân ước tính theo cách ghép trên: **≈ 2172 g**. Bảng 2 cũ ghi 2289,8 g, lệch khoảng 120 g. Lý do nhiều khả năng là số lượng của các dòng "?".

⚠ `drive_motor_mount_v2 (1)-5` nằm cạnh IMU, không nằm cạnh bánh, và `gá imu mới` đang tham chiếu in-context vào nó. Cần kiểm tra xem cái này có thật trên robot không, hay chỉ là part thừa.

### Trạng thái thân: ĐÃ NHẬP XONG (kiểm tra qua API ngày 2026-09-30)

Bạn đã gom các cụm thành subassembly ảo và override ở cấp subassembly. SolidWorks dùng đúng các số này: tổng assembly 4301,7 g, cộng tay ra khớp.

- `Pi case driver 1` và `pi case driver 2`: mỗi cái 128,2 g, mỗi cái gồm 1 pi case top + 1 case dưới + 1 Waveshare. Robot có **2 bộ Pi thật**.
- `imu 1` và `imu 2`: mỗi cái 11 g, mỗi cái gồm 1 gá + 1 claude imu + 1 nắp. Robot có **2 IMU thật**.
- `wheel hub 1–4`: mỗi cái 138,5 g. File `Wheel.SLDPRT` bên trong cũng đang để 138,5, nhưng không ảnh hưởng vì override của subassembly được ưu tiên.
- `bánh đa hướng` ×4 = 11,3. `ngàm_đực_cam` ×2 = 20 (thực ra là ngàm cái, đặt tên sai). `cam+ngàm` = 31,2. `docking male servo` = 94,2. `drive_motor_mount` ×5 = 18,4 (đã chuyển sang solid bằng Thicken bản copy, mate vẫn giữ nguyên).
- Dockingf để 41,0 (đo 40,5, bạn chấp nhận làm tròn). STS3215 đang để 56,0: nên đổi thành **56,1**, vì các số của chân đã tính theo 56,1.
- Thân trên SolidWorks ≈ **2385 g**. Khi chân nhập xong, tổng robot sẽ ≈ 2385 + 2 × 1089,7 ≈ **4565 g**.

### Trạng thái chân: ĐÃ NHẬP XONG (kiểm tra ngày 2026-09-30)

Bạn đã gom các khâu vào subassembly ảo: `foot 1/2` = 216,6, `knee 1/2` = 348,1, `hip 1/2` = 342,4, mỗi cái override ở cấp subassembly. Twist30kg = 105,8 và Hip_105kg = 76,8 vẫn override trực tiếp trên part.

- Cả 10 khâu (2 chân × 5 khâu) đều đúng số đo.
- Tổng robot **4565,1 g**, thân = 2385,7 g.
- CoM của robot (hệ tọa độ gốc của assembly, tư thế hiện tại): (−50,6; 62,8; 189,8) mm.

Lưu ý: các part in 3D **bên trong** subassembly chân vẫn để số cũ (ví dụ Knee 96,27, Foot 74,96). SolidWorks chia khối lượng của subassembly cho các part theo tỉ lệ khối lượng của chúng. Vì vậy phần ốc và dây bị rải cả sang servo, chứ không dồn hết vào part in 3D. Trọng tâm mỗi khâu lệch khoảng 1–2 mm so với cách dồn hết vào part in. Chấp nhận được.

**Khi tính khối lượng hoặc xuất URDF theo từng khâu, phải chọn cả subassembly** (`knee 1`…). Nếu chỉ chọn từng part bên trong, SolidWorks sẽ bỏ qua override của subassembly và ra sai khối lượng.

### Bù ốc và dây theo module (cân ngày 2026-09-30, cả 2 module đều có chân)

| Module | Cân thật | SolidWorks trước khi bù | Chênh | Bù vào baseplate |
|---|---|---|---|---|
| Male (2.2.1 + 1.2.2, gồm dock male, switch, cam+ngàm, imu 1, Pi 1, chân trái) | 2422,4 | 2336,1 | +86,3 | Baseplate1.2.2 **144,2**, Baseplate2.2.1 **181,8** |
| Female (2.2.2 + 1.2, gồm Dockingf, ngàm_đực_cam ×2, imu 2, Pi 2, chân phải) | 2214,0 | 2229,0 | −15,0 | Baseplate1.2 **89,4**, Baseplate2.2.2 **112,6** |

- Chênh lệch được chia cho 2 baseplate của mỗi module theo tỉ lệ khối lượng.
- Mức −15 g ở module female được coi là sai số cân: các con số lẻ đã được cộng dồn từ khoảng 20 lần cân.
- Đường nối giữa hai module: x = −47,8 mm trong hệ tọa độ của assembly.
- Tổng robot sau khi bù: **4636,4 g**.

### Xuất URDF: các giới hạn khớp đang là giá trị tạm (PHẢI ghi đè trong code)

Tất cả khớp revolute đang để giới hạn tạm: lower −3.1416, upper 3.1416, effort 10, velocity 1. Khi làm cấu hình Isaac Lab phải ghi đè bằng giá trị thật:
- **effort (stall torque):** servo STS3095 (105 kg·cm) ≈ 10,3 N·m, dùng cho các khớp Bub, Hip, Knee. Servo STS3215 (30 kg·cm) ≈ 2,94 N·m, dùng cho khớp rotate (twist) và Foot. Kiểm tra lại datasheet theo đúng điện áp đang dùng.
- **velocity:** khoảng 4,7 rad/s (≈ 0,22 s/60°). Kiểm tra lại datasheet.
- **lower/upper:** chưa đo. Đo góc tới lúc va chạm trong SolidWorks, hoặc lấy giới hạn cài trên servo thật. Góc 0 là tư thế hiện tại trong CAD.

Các thay đổi trong CAD phục vụ việc xuất (2026-09-30):
- Fix `imu 1` và `imu 2`.
- Tạo hệ tọa độ `Footleft` và `Footright` (nằm trên trục cổ chân, cùng hướng với Baselink), kèm các sketch `Foot*_origin_sketch`. Không xóa các sketch này.
- Hệ tọa độ `imu left` và `imuright` đặt tại tâm board, cùng hướng với Baselink (FLU).
- Joint fixed của IMU chọn một trục bất kỳ (Axis7/Axis8) để tool khỏi tự suy ra trục.

Quy ước trái/phải: tên link hiện đang đặt theo góc nhìn **đứng đối diện robot** (link "left" nằm ở phía y âm của Baselink). Chuẩn ROS/Isaac đặt tên theo góc nhìn của chính robot. Hãy thống nhất với code trước khi đổi tên.

### SW2URDF bỏ qua override → phải sửa inertial bằng script

SW2URDF tính khối lượng bằng hình học × mật độ, **bỏ qua mọi override** (cả part lẫn subassembly). Vì vậy khi xuất không cần sửa mass trong exporter. Thay vào đó, chạy trên Windows, lúc SolidWorks đang mở OFFICIALdesign:

```
python tools/urdf_inertials.py                      # in bảng + ghi tools/inertial_report.md/.csv
python tools/urdf_inertials.py "<pkg>/urdf/x.urdf"  # như trên + ghi đè <inertial> của 13 link (giữ .bak)
```

- Đã kiểm chứng: tổng khối lượng và CoM cộng từ 13 link khớp tuyệt đối với Mass Properties của SolidWorks (4643,0 g).
- Với Bubleft, quy về cùng khối lượng thì cả 6 thành phần inertia trùng với exporter.
- SolidWorks hiển thị tích quán tính theo "positive tensor notation". Script đã đổi dấu sang quy ước URDF (ixy = −Lxy).
- Báo cáo có thêm mục ổn định tĩnh. Ở tư thế CAD hiện tại, robot đứng trên 2 bàn chân, CoM cao 340 mm, cách mép vùng tựa khoảng 76 mm (xấp xỉ theo bounding box).

### Bản xuất ngày 2026-09-30 18:26

- Package: `D:\des\lab\Vinh&Minh\Transformer_VinhVersion\Transformer_VinhVersion\Transformer\OFFICIALdesign\`
- URDF: `urdf\OFFICIALdesign.urdf`. Meshes: `meshes\*.STL` (13 file). Số inertial đúng: `Transformer\tools\inertial_report.csv`.
- Lỗi trong URDF vừa xuất:
  - Mass và inertia là giá trị exporter tính theo hình học × mật độ (tổng 4,044 kg, đúng phải là 4,643 kg).
  - **Link Bubright có mass = 0, inertia = 0** (lỗi của exporter). Bắt buộc phải sửa, vì link khối lượng 0 sẽ làm hỏng mô phỏng vật lý.
  - Limit các khớp đang là giá trị tạm.
  - `config/joint_names_*.yaml` có một phần tử rỗng ở đầu danh sách.
  - `Baselink.STL` nặng khoảng 15 MB. Mesh va chạm cần đơn giản hóa (convex hull hoặc decomposition).
- Quy ước dấu: tích quán tính trong report đã theo quy ước URDF (ixy = −∫xy dm). Script đã chuyển sang dựng tensor từ trục chính và mômen chính, nên không còn phụ thuộc tùy chọn tensor notation của SolidWorks.

### Checklist sang Ubuntu / Isaac Lab
1. Copy nguyên package (`urdf/` + `meshes/`). Kiểm tra đường dẫn `package://`.
2. Chuyển URDF sang USD: tắt fix base, collision dùng convex hull hoặc decomposition.
3. Kiểm tra nhanh: tổng mass 4,643 kg, thả rơi không nổ, từng khớp quay đúng chiều.
4. Ghi đè trong code: giới hạn góc thật, effort/velocity, PD gain, armature, ma sát tiếp xúc.
5. Đối chiếu tên và dấu khớp với robot thật:
   - tên left/right đang theo góc nhìn người đứng đối diện robot,
   - trục Hip/Knee/Foot ngược dấu giữa hai bên,
   - khớp Foot có hệ tọa độ mới nên góc 0 có thể khác URDF cũ.

## 5. Ốc và dây của thân

1. Cân **cả thân đã lắp đủ**: có ốc, có dây, và cả các ốc M3×15 đang thiếu.
2. Tính k = cân thật / tổng khối lượng thân trên SolidWorks.
3. Nhân k **chỉ vào các part in 3D và baseplate của thân**, không nhân vào file servo (vì dùng chung với chân).

## 6. Các bước nhập trong SolidWorks

1. Trong cây OFFICIALdesign, chuột phải vào component → **Open Part**. Với STS3215 thì là **Open Assembly**.
2. **Tools → Evaluate → Mass Properties → Override Mass Properties…**
3. Tick **Override mass**, nhập số ở cột "Nhập". Để ý đơn vị (g hay kg), và không tick CoM hay inertia.
4. OK, rồi **Save**.
5. Quay về OFFICIALdesign, nhấn **Ctrl+Q** để rebuild toàn bộ.
6. Kiểm tra:
   - Mass Properties, chọn các component của 1 chân: tổng phải là **1089,7 g**.
   - Chọn từng khâu: Foot 216,6 / Knee 348,1 / Twist 105,8 / Hip 342,4 / Bub 76,8.
   - Không chọn gì: ra tổng robot, ≈ 2 × 1089,7 + khối lượng thân.
7. Xuất trọng tâm và quán tính: đặt coordinate system tại gốc base_link. Trong Mass Properties → Options chọn đơn vị kg và m, output theo coordinate system đó. Lấy bảng "Moments of inertia taken at the center of mass and aligned with the output coordinate system".

## 7. Tình trạng file (cần biết khi làm tiếp)

- `gá imu mới`: đã sửa hết lỗi qua API ngày 2026-09-29.
  - Xóa relation dangling ở Sketch3, 4, 5.
  - Đặt D1@Sketch1 thành driven.
  - Sketch2 được gán lại sketch plane vào mặt z = 10 mm của Boss-Extrude1, và đã tách khỏi Baseplate.
  - Kiểm tra lại xem đã **Save** chưa.
- Bản backup trước khi sửa: `%TEMP%\claude\...\scratchpad\backup\` (gồm `gá imu mới.SLDPRT` và `OFFICIALdesign.SLDASM`).
- Lỗi còn tồn tại: **`105motormount` Cut-Extrude11**, tham chiếu in-context tới Origin của `Hip_105kg<1>`. Chưa sửa.
- Trong CAD, nhiều part đã được override từ trước. Khi nhập, cứ ghi đè bằng số trong file này.
- Dữ liệu dạng bảng máy đọc được: `mass_log.csv` (cùng thư mục).
